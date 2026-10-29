package app.orubus.exmera.camera

import android.annotation.SuppressLint
import android.content.Context
import android.graphics.Matrix
import android.graphics.RectF
import android.graphics.SurfaceTexture
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraDevice
import android.hardware.camera2.CameraManager
import android.hardware.camera2.CameraMetadata
import android.hardware.camera2.CaptureRequest
import android.hardware.camera2.CaptureResult
import android.hardware.camera2.TotalCaptureResult
import android.media.ImageReader
import android.util.Log
import android.util.Size
import android.view.Surface
import android.view.TextureView
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

/**
 * PRO mode — the native Camera2 stack.
 *
 * Full manual control: sensor sensitivity, exposure time, focus distance,
 * white balance mode, zoom, and RAW_SENSOR DNG capture where the sensor
 * supports it (graceful JPEG fallback that goes through the same metadata
 * scrubber as every other mode). This is a real pipeline, not a demo:
 * one repeating preview request whose builder is mutated in place, plus
 * single-shot stills that reuse the current manual parameters.
 */
class ProCamera2(private val context: Context) {

    data class ProRanges(
        val isoMin: Int,
        val isoMax: Int,
        val expMinNs: Long,
        val expMaxNs: Long,
        val focusMin: Float,
        val focusMax: Float,
        val zoomMax: Float,
        val awbModes: List<Int>,
        val supportsRaw: Boolean,
        val previewSizes: List<Size>,
        val rawSize: Size?,
        val jpegSize: Size?,
        val sensorOrientation: Int
    )

    data class ProParams(
        val manualExposure: Boolean = false,
        val iso: Int = 400,
        val exposureNs: Long = 10_000_000L,
        val manualFocus: Boolean = false,
        val focusDistance: Float = 0f,
        val awbMode: Int = CameraMetadata.CONTROL_AWB_MODE_AUTO,
        val zoomRatio: Float = 1f,
        val rawEnabled: Boolean = true
    )

    private val _ranges = MutableStateFlow<ProRanges?>(null)
    val ranges: StateFlow<ProRanges?> = _ranges.asStateFlow()

    private val _params = MutableStateFlow(ProParams())
    val params: StateFlow<ProParams> = _params.asStateFlow()

    private val _ready = MutableStateFlow(false)
    val ready: StateFlow<Boolean> = _ready.asStateFlow()

    private val _message = MutableStateFlow("pro camera starting")
    val message: StateFlow<String> = _message.asStateFlow()

    private val handlerThread = android.os.HandlerThread("exmera-pro").apply { start() }
    private val handler = android.os.Handler(handlerThread.looper)
    private val saver: ExecutorService = Executors.newSingleThreadExecutor { r -> Thread(r, "exmera-pro-save") }

    private var device: CameraDevice? = null
    private var session: CameraCaptureSession? = null
    private var previewRequest: CaptureRequest.Builder? = null
    private var rawReader: ImageReader? = null
    private var jpegReader: ImageReader? = null
    private var characteristics: CameraCharacteristics? = null
    private var previewSurface: Surface? = null
    private val closed = AtomicBoolean(false)
    private val capturing = AtomicBoolean(false)

    @SuppressLint("MissingPermission")
    fun open(textureView: TextureView) {
        closed.set(false)
        val manager = context.getSystemService(CameraManager::class.java)
        val id = manager.cameraIdList.firstOrNull {
            manager.getCameraCharacteristics(it).get(CameraCharacteristics.LENS_FACING) == CameraCharacteristics.LENS_FACING_BACK
        } ?: run { _message.value = "no back camera"; return }
        val c = manager.getCameraCharacteristics(id)
        characteristics = c

        val map = c.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)
        val caps = c.get(CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES)?.toSet().orEmpty()
        val supportsRaw = map != null &&
            android.graphics.ImageFormat.RAW_SENSOR in map.outputFormats &&
            CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES_RAW in caps

        val isoRange = c.get(CameraCharacteristics.SENSOR_INFO_SENSITIVITY_RANGE)
        val expRange = c.get(CameraCharacteristics.SENSOR_INFO_EXPOSURE_TIME_RANGE)
        val focusMin = c.get(CameraCharacteristics.LENS_INFO_MINIMUM_FOCUS_DISTANCE) ?: 0f
        val zoomRange = c.get(CameraCharacteristics.CONTROL_ZOOM_RATIO_RANGE)
        val awb = c.get(CameraCharacteristics.CONTROL_AWB_AVAILABLE_MODES)?.toList()
            ?.filter { it in 0..7 } ?: listOf(CameraMetadata.CONTROL_AWB_MODE_AUTO)
        val previewSizes = map?.getOutputSizes(SurfaceTexture::class.java)?.sortedByDescending { it.width * it.height } ?: emptyList()
        val rawSize = if (supportsRaw) map?.getOutputSizes(android.graphics.ImageFormat.RAW_SENSOR)?.maxByOrNull { it.width * it.height } else null
        val jpegSize = map?.getOutputSizes(android.graphics.ImageFormat.JPEG)?.maxByOrNull { it.width * it.height }

        _ranges.value = ProRanges(
            isoMin = isoRange?.lower ?: 100, isoMax = isoRange?.upper ?: 3200,
            expMinNs = expRange?.lower ?: 10_000L, expMaxNs = expRange?.upper ?: 100_000_000L,
            focusMin = 0f, focusMax = focusMin,
            zoomMax = zoomRange?.upper ?: (c.get(CameraCharacteristics.SCALER_AVAILABLE_MAX_DIGITAL_ZOOM) ?: 1f),
            awbModes = awb,
            supportsRaw = supportsRaw,
            previewSizes = previewSizes,
            rawSize = rawSize,
            jpegSize = jpegSize,
            sensorOrientation = c.get(CameraCharacteristics.SENSOR_ORIENTATION) ?: 90
        )

        try {
            manager.openCamera(id, object : CameraDevice.StateCallback() {
                override fun onOpened(camera: CameraDevice) {
                    device = camera
                    attachPreview(textureView)
                }

                override fun onDisconnected(camera: CameraDevice) { camera.close() }
                override fun onError(camera: CameraDevice, error: Int) {
                    _message.value = "camera error $error"
                    camera.close()
                }
            }, handler)
        } catch (t: Throwable) {
            _message.value = "open failed: ${t.message}"
        }
    }

    /** Creates the session once the preview surface is available. */
    private fun attachPreview(textureView: TextureView) {
        if (closed.get()) return
        val st = textureView.surfaceTexture
        if (st == null) {
            textureView.surfaceTextureListener = object : TextureView.SurfaceTextureListener {
                override fun onSurfaceTextureAvailable(st: SurfaceTexture, w: Int, h: Int) {
                    textureView.surfaceTextureListener = null
                    attachPreview(textureView)
                }
                override fun onSurfaceTextureSizeChanged(st: SurfaceTexture, w: Int, h: Int) {}
                override fun onSurfaceTextureDestroyed(st: SurfaceTexture) = true
                override fun onSurfaceTextureUpdated(st: SurfaceTexture) {}
            }
            return
        }
        val sizes = _ranges.value?.previewSizes ?: emptyList()
        val chosen = sizes.firstOrNull { it.width <= 1920 && it.height <= 1440 } ?: sizes.lastOrNull() ?: Size(1280, 720)
        st.setDefaultBufferSize(chosen.width, chosen.height)
        val surface = Surface(st)
        previewSurface = surface
        configureTransform(textureView, chosen)

        val outputs = ArrayList<Surface>()
        outputs += surface
        val r = _ranges.value
        if (r?.rawSize != null) {
            rawReader = ImageReader.newInstance(r.rawSize!!.width, r.rawSize!!.height, android.graphics.ImageFormat.RAW_SENSOR, 2)
            rawReader?.setOnImageAvailableListener({ reader ->
                val image = reader.acquireLatestImage() ?: return@setOnImageAvailableListener
                val pending = pendingRawResult
                saver.execute {
                    try {
                        if (pending != null) writeDng(image, pending) else image.close()
                    } catch (t: Throwable) {
                        Log.e("ExmeraPro", "dng write failed", t)
                        image.close()
                    }
                }
            }, handler)
            outputs += rawReader!!.surface
        }
        if (r?.jpegSize != null) {
            jpegReader = ImageReader.newInstance(r.jpegSize!!.width, r.jpegSize!!.height, android.graphics.ImageFormat.JPEG, 2)
            jpegReader?.setOnImageAvailableListener({ reader ->
                val image = reader.acquireLatestImage() ?: return@setOnImageAvailableListener
                saver.execute {
                    try {
                        val buffer = image.planes[0].buffer
                        val bytes = ByteArray(buffer.remaining())
                        buffer.get(bytes)
                        image.close()
                        val rotation = jpegRotation()
                        val scrubbed = MetadataScrubber.scrubCapture(bytes, rotation)
                        if (scrubbed != null) {
                            MetadataScrubber.saveToGallery(context, scrubbed, MetadataScrubber.displayName("EXM", "jpg"), "image/jpeg")
                            _message.value = "saved · metadata scrubbed"
                        } else _message.value = "decode failed"
                    } catch (t: Throwable) {
                        Log.e("ExmeraPro", "jpeg save failed", t)
                        runCatching { image.close() }
                    }
                }
            }, handler)
            outputs += jpegReader!!.surface
        }

        device?.createCaptureSession(outputs, object : CameraCaptureSession.StateCallback() {
            override fun onConfigured(s: CameraCaptureSession) {
                if (closed.get()) return
                session = s
                startRepeating()
                _ready.value = true
                _message.value = if (r?.supportsRaw == true) "manual · RAW ${r.rawSize!!.width}×${r.rawSize!!.height}" else "manual · JPEG (no RAW sensor)"
            }

            override fun onConfigureFailed(s: CameraCaptureSession) {
                _message.value = "session configuration failed"
            }
        }, handler)
    }

    private var pendingRawResult: TotalCaptureResult? = null

    private fun startRepeating() {
        val dev = device ?: return
        val s = session ?: return
        val surface = previewSurface ?: return
        try {
            val builder = dev.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW)
            builder.addTarget(surface)
            applyParams(builder)
            previewRequest = builder
            s.setRepeatingRequest(builder.build(), null, handler)
        } catch (t: Throwable) {
            _message.value = "preview failed: ${t.message}"
        }
    }

    private fun applyParams(builder: CaptureRequest.Builder) {
        val p = _params.value
        if (p.manualExposure) {
            builder.set(CaptureRequest.CONTROL_MODE, CameraMetadata.CONTROL_MODE_OFF)
            builder.set(CaptureRequest.SENSOR_SENSITIVITY, p.iso)
            builder.set(CaptureRequest.SENSOR_EXPOSURE_TIME, p.exposureNs)
        } else {
            builder.set(CaptureRequest.CONTROL_MODE, CameraMetadata.CONTROL_MODE_AUTO)
        }
        if (p.manualFocus) {
            builder.set(CaptureRequest.CONTROL_AF_MODE, CameraMetadata.CONTROL_AF_MODE_OFF)
            builder.set(CaptureRequest.LENS_FOCUS_DISTANCE, p.focusDistance.coerceIn(0f, maxOf(0f, _ranges.value?.focusMax ?: 0f)))
        } else {
            builder.set(CaptureRequest.CONTROL_AF_MODE, CameraMetadata.CONTROL_AF_MODE_CONTINUOUS_PICTURE)
        }
        builder.set(CaptureRequest.CONTROL_AWB_MODE, p.awbMode)
        val zoomMax = _ranges.value?.zoomMax ?: 1f
        if (android.os.Build.VERSION.SDK_INT >= 30) {
            builder.set(CaptureRequest.CONTROL_ZOOM_RATIO, p.zoomRatio.coerceIn(1f, maxOf(1f, zoomMax)))
        } else {
            val c = characteristics
            val active = c?.get(CameraCharacteristics.SENSOR_INFO_ACTIVE_ARRAY_SIZE)
            if (active != null) {
                val zoom = p.zoomRatio.coerceIn(1f, maxOf(1f, zoomMax))
                val w = active.width() / zoom; val h = active.height() / zoom
                val left = active.left + (active.width() - w) / 2; val top = active.top + (active.height() - h) / 2
                builder.set(CaptureRequest.SCALER_CROP_REGION, android.graphics.Rect(left.toInt(), top.toInt(), (left + w).toInt(), (top + h).toInt()))
            }
        }
    }

    /** Applies new manual parameters to the repeating request (live, no session rebuild). */
    fun update(transform: (ProParams) -> ProParams) {
        _params.value = transform(_params.value)
        val builder = previewRequest ?: return
        val s = session ?: return
        applyParams(builder)
        runCatching { s.setRepeatingRequest(builder.build(), null, handler) }
            .onFailure { _message.value = "param failed: ${it.message}" }
    }

    /** Single still through the current manual parameters. RAW DNG when available+enabled. */
    fun captureStill() {
        val s = session ?: return
        val dev = device ?: return
        val p = _params.value
        val useRaw = p.rawEnabled && _ranges.value?.supportsRaw == true && rawReader != null

        val builder = dev.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW)
        if (useRaw) {
            builder.addTarget(rawReader!!.surface)
        } else if (jpegReader != null) {
            builder.addTarget(jpegReader!!.surface)
        } else {
            _message.value = "no still surface"
            return
        }
        applyParams(builder)

        try {
            s.capture(builder.build(), object : CameraCaptureSession.CaptureCallback() {
                override fun onCaptureCompleted(s: CameraCaptureSession, request: CaptureRequest, result: TotalCaptureResult) {
                    if (useRaw) {
                        pendingRawResult = result
                        _message.value = "writing DNG…"
                    }
                    // jpeg path completes via reader callback
                }
            }, handler)
        } catch (t: Throwable) {
            capturing.set(false)
            _message.value = "capture failed: ${t.message}"
        }
        capturing.set(false)
    }

    private fun writeDng(image: android.media.Image, result: TotalCaptureResult) {
        val c = characteristics ?: run { image.close(); return }
        val dng = android.hardware.camera2.DngCreator(c, result)
        dng.setOrientation(jpegRotation())
        val saved = runCatching {
            java.io.ByteArrayOutputStream().use { out ->
                dng.writeImage(out, image)
                val bytes = out.toByteArray()
                MetadataScrubber.saveToGallery(context, bytes, MetadataScrubber.displayName("EXM_RAW", "dng"), "image/x-adobe-dng")
            }
        }
        dng.close()
        image.close()
        pendingRawResult = null
        _message.value = if (saved.getOrNull() != null) "RAW DNG saved (no GPS written)" else "DNG failed: ${saved.exceptionOrNull()?.message}"
    }

    /** Orientation tag equivalent used for DNG; JPEG scrubber bakes pixels instead. */
    private fun jpegRotation(): Int {
        val sensor = _ranges.value?.sensorOrientation ?: 90
        val display = context.displayRotationDegrees()
        return if (context.isFrontCameraDefault()) (360 - ((sensor + display) % 360)) % 360 else (sensor - display + 360) % 360
    }

    private fun Context.displayRotationDegrees(): Int =
        (getSystemService(android.view.WindowManager::class.java)?.defaultDisplay?.rotation ?: Surface.ROTATION_0).let {
            when (it) {
                Surface.ROTATION_90 -> 90
                Surface.ROTATION_180 -> 180
                Surface.ROTATION_270 -> 270
                else -> 0
            }
        }

    private fun Context.isFrontCameraDefault(): Boolean = false

    /** Fits the landscape sensor buffer into a portrait 3:4 view without distortion. */
    private fun configureTransform(textureView: TextureView, size: Size) {
        val viewW = textureView.width
        val viewH = textureView.height
        if (viewW == 0 || viewH == 0) return
        val rotation = context.displayRotationDegrees()
        val matrix = Matrix()
        val rect = RectF(0f, 0f, size.width.toFloat(), size.height.toFloat())
        val viewRect = RectF(0f, 0f, viewW.toFloat(), viewH.toFloat())
        matrix.setRectToRect(rect, viewRect, Matrix.ScaleToFit.CENTER)
        matrix.postRotate(rotation.toFloat(), viewW / 2f, viewH / 2f)
        textureView.setTransform(matrix)
    }

    fun close() {
        closed.set(true)
        _ready.value = false
        runCatching { session?.close() }
        runCatching { device?.close() }
        runCatching { rawReader?.close() }
        runCatching { jpegReader?.close() }
        session = null
        device = null
        rawReader = null
        jpegReader = null
        previewSurface = null
        previewRequest = null
        handlerThread.quitSafely()
    }
}
