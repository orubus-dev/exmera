package app.orubus.exmera.camera

import android.content.ContentValues
import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraManager
import android.os.Build
import android.provider.MediaStore
import android.util.Log
import androidx.camera.core.Camera
import androidx.camera.core.CameraSelector
import androidx.camera.core.FocusMeteringAction
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.video.FallbackStrategy
import androidx.camera.video.MediaStoreOutputOptions
import androidx.camera.video.Quality
import androidx.camera.video.QualitySelector
import androidx.camera.video.Recorder
import androidx.camera.video.Recording
import androidx.camera.video.VideoCapture
import androidx.camera.video.VideoRecordEvent
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import studio.exmera.engine.core.MultiFrameBuffer
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

data class CameraCapabilities(
    val cameraId: String,
    val facing: Int,
    val sensorWidth: Int,
    val sensorHeight: Int,
    val maxDigitalZoom: Float,
    val supportsRaw: Boolean,
    val supportsManualExposure: Boolean,
    val supportsManualFocus: Boolean,
    val supportsOis: Boolean,
    val availableFps: List<Int>
)

data class CameraFrame(val image: ImageProxy, val rotationDegrees: Int, val timestampNs: Long)

data class RecordingState(val recording: Boolean, val uri: String? = null, val durationNs: Long = 0L, val error: String? = null)

enum class CameraMode { PRO, VIDEO, IMAGE, PORTRAIT, PANO }

/** CameraX stack for IMAGE / VIDEO / PORTRAIT / PANO. PRO runs on the Camera2 stack (ProCamera2). */
class ExmeraCameraController(
    private val context: Context,
    private val lifecycleOwner: LifecycleOwner
) {
    private val _capabilities = MutableStateFlow<CameraCapabilities?>(null)
    val capabilities: StateFlow<CameraCapabilities?> = _capabilities.asStateFlow()

    private val _isReady = MutableStateFlow(false)
    val isReady: StateFlow<Boolean> = _isReady.asStateFlow()

    private val _recording = MutableStateFlow(RecordingState(false))
    val recording: StateFlow<RecordingState> = _recording.asStateFlow()

    private val _busy = MutableStateFlow(false)
    val busy: StateFlow<Boolean> = _busy.asStateFlow()

    private val _boostState = MutableStateFlow("idle")
    val boostState: StateFlow<String> = _boostState.asStateFlow()

    private val _panoState = MutableStateFlow("idle")
    val panoState: StateFlow<String> = _panoState.asStateFlow()

    private val _lastShotSaved = MutableStateFlow<String?>(null)
    val lastShotSaved: StateFlow<String?> = _lastShotSaved.asStateFlow()

    private var provider: ProcessCameraProvider? = null
    private var camera: Camera? = null
    private var analysis: ImageAnalysis? = null
    private var capture: ImageCapture? = null
    private var videoCapture: VideoCapture<Recorder>? = null
    private var activeRecording: Recording? = null

    private val executor: ExecutorService = Executors.newFixedThreadPool(2)
    private val facingBack = AtomicBoolean(true)

    private val boostEngine = MultiFrameCaptureEngine(capacity = 8)
    private val panoFrames = ArrayList<PanoStitcher.GrayPlane>()
    private var panoActive = false
    private var panoLastAtMs = 0L
    private var lastFocusX = 0.5f
    private var lastFocusY = 0.45f

    fun bind(previewView: PreviewView, mode: CameraMode) {
        val future = ProcessCameraProvider.getInstance(context)
        future.addListener({
            runCatching {
                val p = future.get()
                provider = p
                _capabilities.value = CameraCapabilityRepository(context).discover(facing())
                val selector = if (facingBack.get()) CameraSelector.DEFAULT_BACK_CAMERA else CameraSelector.DEFAULT_FRONT_CAMERA

                val preview = Preview.Builder().build().also { it.surfaceProvider = previewView.surfaceProvider }
                p.unbindAll()

                val useCases = ArrayList<androidx.camera.core.UseCase>()
                useCases += preview
                if (mode != CameraMode.VIDEO) {
                    useCases += ImageCapture.Builder()
                        .setCaptureMode(ImageCapture.CAPTURE_MODE_MAXIMIZE_QUALITY)
                        .build()
                        .also { capture = it }
                }
                if (mode == CameraMode.VIDEO) {
                    val quality = QualitySelector.fromOrderedList(
                        listOf(Quality.UHD, Quality.FHD, Quality.HD, Quality.SD),
                        FallbackStrategy.lowerQualityOrHigherThan(Quality.HD)
                    )
                    videoCapture = VideoCapture.withOutput(Recorder.Builder().setQualitySelector(quality).build())
                    useCases += videoCapture!!
                }
                if (mode == CameraMode.IMAGE || mode == CameraMode.PANO) {
                    useCases += ImageAnalysis.Builder()
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                        .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_YUV_420_888)
                        .build()
                        .also { a ->
                            analysis = a
                            a.setAnalyzer(executor) { proxy -> routeFrame(proxy, mode) }
                        }
                }
                camera = p.bindToLifecycle(lifecycleOwner, selector, *useCases.toTypedArray())
                _isReady.value = true
            }.onFailure { Log.e("ExmeraCamera", "bind failed", it) }
        }, ContextCompat.getMainExecutor(context))
    }

    private fun facing(): Int =
        if (facingBack.get()) CameraCharacteristics.LENS_FACING_BACK else CameraCharacteristics.LENS_FACING_FRONT

    /** Toggles front/back; the preview host rebinds afterwards. */
    fun flip() {
        facingBack.set(!facingBack.get())
    }

    /** Releases the camera without destroying the controller (mode switch to PRO). */
    fun pause() {
        panoActive = false
        boostActive = false
        provider?.unbindAll()
        _isReady.value = false
    }

    private fun routeFrame(proxy: ImageProxy, mode: CameraMode) {
        when {
            panoActive && mode == CameraMode.PANO -> {
                val now = System.currentTimeMillis()
                if (now - panoLastAtMs >= 350 && panoFrames.size < 10) {
                    panoLastAtMs = now
                    panoFrames += lumaPlane(proxy, 480)
                    _panoState.value = "sweep · ${panoFrames.size} frames"
                }
                proxy.close()
            }
            boostActive && mode == CameraMode.IMAGE -> {
                if (!boostEngine.accept(proxy)) proxy.close()
            }
            else -> proxy.close()
        }
    }

    private var boostActive = false

    fun flipCamera(previewView: PreviewView, mode: CameraMode) {
        facingBack.set(!facingBack.get())
        bind(previewView, mode)
    }

    fun zoom(ratio: Float) {
        val c = _capabilities.value ?: return
        camera?.cameraControl?.setZoomRatio(ratio.coerceIn(1f, maxOf(1f, c.maxDigitalZoom)))
    }

    fun focusAt(view: PreviewView, vx: Float, vy: Float) {
        lastFocusX = if (view.width > 0) vx / view.width else 0.5f
        lastFocusY = if (view.height > 0) vy / view.height else 0.45f
        val point = view.meteringPointFactory.createPoint(vx, vy)
        camera?.cameraControl?.startFocusAndMetering(
            FocusMeteringAction.Builder(point, FocusMeteringAction.FLAG_AF or FocusMeteringAction.FLAG_AE)
                .disableAutoCancel().build()
        )
    }

    fun setFlash(mode: Int) {
        capture?.flashMode = mode
    }

    /* ---------------- IMAGE / PORTRAIT ---------------- */

    /**
     * Captures, scrubs ALL metadata, and saves to Pictures/exmera.
     * In PORTRAIT mode the software bokeh pass runs before the save.
     */
    fun captureStill(
        mode: CameraMode,
        portraitStrength: Int,
        onDone: (Boolean, String) -> Unit
    ) {
        val ic = capture
        if (ic == null || _busy.value) {
            onDone(false, "camera busy")
            return
        }
        _busy.value = true
        ic.takePicture(executor, object : ImageCapture.OnImageCapturedCallback() {
            override fun onCaptureSuccess(image: ImageProxy) {
                executor.execute {
                    var ok = false
                    var message = ""
                    try {
                        val buffer = image.planes[0].buffer
                        val bytes = ByteArray(buffer.remaining())
                        buffer.get(bytes)
                        val rotation = image.imageInfo.rotationDegrees
                        image.close()

                        if (!MetadataScrubber.isJpeg(bytes)) {
                            message = "unsupported capture payload"
                        } else {
                            val name = MetadataScrubber.displayName("EXM", "jpg")
                            if (mode == CameraMode.PORTRAIT) {
                                val src = BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
                                val upright = MetadataScrubber.rotate(src, rotation)
                                val bokeh = Bokeh.apply(upright, lastFocusX, lastFocusY, portraitStrength)
                                val data = MetadataScrubber.bitmapToJpeg(bokeh)
                                bokeh.recycle()
                                if (upright !== src) upright.recycle()
                                src.recycle()
                                val uri = MetadataScrubber.saveToGallery(context, data, name, "image/jpeg")
                                ok = uri != null
                                message = if (ok) "portrait saved · metadata scrubbed" else "save failed"
                            } else {
                                val scrubbed = MetadataScrubber.scrubCapture(bytes, rotation)
                                if (scrubbed == null) {
                                    message = "decode failed"
                                } else {
                                    val uri = MetadataScrubber.saveToGallery(context, scrubbed, name, "image/jpeg")
                                    ok = uri != null
                                    message = if (ok) "saved · metadata scrubbed" else "save failed"
                                }
                            }
                            _lastShotSaved.value = name
                        }
                    } catch (t: Throwable) {
                        message = "capture error: ${t.message}"
                    } finally {
                        _busy.value = false
                        onDone(ok, message)
                    }
                }
            }

            override fun onError(exception: ImageCaptureException) {
                _busy.value = false
                onDone(false, "capture failed: ${exception.message}")
            }
        })
    }

    /* ---------------- IMAGE boost (multi-frame fusion) ---------------- */

    fun startBoost(targetFrames: Int = 5) {
        if (boostActive) return
        boostActive = true
        boostEngine.start(targetFrames)
        _boostState.value = "capturing"
    }

    fun hasBoostFrames(): Boolean = boostEngine.state.value == MultiFrameCaptureEngine.State.COMPLETE

    fun finishBoost(onDone: (Boolean, String) -> Unit) {
        if (!boostActive) {
            onDone(false, "boost not running")
            return
        }
        boostActive = false
        _boostState.value = "processing"
        kotlinx.coroutines.CoroutineScope(Dispatchers.Default).launch {
            var ok = false
            var message = ""
            try {
                val bitmap = boostEngine.processToBitmap()
                if (bitmap == null) {
                    message = "fusion failed"
                } else {
                    val data = MetadataScrubber.bitmapToJpeg(bitmap)
                    bitmap.recycle()
                    val name = MetadataScrubber.displayName("EXM_BOOST", "jpg")
                    val uri = MetadataScrubber.saveToGallery(context, data, name, "image/jpeg")
                    ok = uri != null
                    message = if (ok) "boost saved · metadata scrubbed" else "save failed"
                }
            } catch (t: Throwable) {
                message = "boost error: ${t.message}"
            } finally {
                _boostState.value = if (ok) "done" else "failed"
                onDone(ok, message)
            }
        }
    }

    fun cancelBoost() {
        boostActive = false
        boostEngine.reset()
        _boostState.value = "idle"
    }

    /* ---------------- PANO ---------------- */

    fun startPano() {
        if (panoActive) return
        panoFrames.clear()
        panoLastAtMs = 0L
        panoActive = true
        _panoState.value = "sweep · ${panoFrames.size} frames"
    }

    fun finishPano(onDone: (Boolean, String) -> Unit) {
        if (!panoActive) {
            onDone(false, "panorama not running")
            return
        }
        panoActive = false
        val frames = ArrayList(panoFrames)
        panoFrames.clear()
        if (frames.size < 2) {
            _panoState.value = "idle"
            onDone(false, "need at least 2 frames — sweep slower")
            return
        }
        _panoState.value = "stitching ${frames.size} frames"
        kotlinx.coroutines.CoroutineScope(Dispatchers.Default).launch {
            var ok = false
            var message = ""
            try {
                val result = withContext(Dispatchers.Default) { PanoStitcher.stitch(frames) }
                if (result == null) {
                    message = "stitch failed"
                } else {
                    val bitmap = grayToBitmap(result)
                    val scaled = if (bitmap.width < 2000) {
                        val target = minOf(2000, bitmap.width * 3)
                        Bitmap.createScaledBitmap(bitmap, target, bitmap.height * target / bitmap.width, true)
                    } else bitmap
                    val data = MetadataScrubber.bitmapToJpeg(scaled)
                    if (scaled !== bitmap) scaled.recycle()
                    bitmap.recycle()
                    val name = MetadataScrubber.displayName("EXM_PANO", "jpg")
                    val uri = MetadataScrubber.saveToGallery(context, data, name, "image/jpeg")
                    ok = uri != null
                    message = if (ok) "panorama saved · ${result.width}×${result.height} · metadata scrubbed" else "save failed"
                }
            } catch (t: Throwable) {
                message = "pano error: ${t.message}"
            } finally {
                _panoState.value = if (ok) "done" else "failed"
                onDone(ok, message)
            }
        }
    }

    fun cancelPano() {
        panoActive = false
        panoFrames.clear()
        _panoState.value = "idle"
    }

    /* ---------------- VIDEO ---------------- */

    fun startRecording() {
        val vc = videoCapture ?: return
        if (activeRecording != null) return
        val values = ContentValues().apply {
            put(MediaStore.Video.Media.DISPLAY_NAME, MetadataScrubber.displayName("EXM_V", "mp4"))
            put(MediaStore.Video.Media.MIME_TYPE, "video/mp4")
            if (Build.VERSION.SDK_INT >= 29) put(MediaStore.Video.Media.RELATIVE_PATH, "Movies/exmera")
        }
        val output = MediaStoreOutputOptions.Builder(context.contentResolver, MediaStore.Video.Media.EXTERNAL_CONTENT_URI)
            .setContentValues(values).build()
        activeRecording = vc.output.prepareRecording(context, output)
            .start(ContextCompat.getMainExecutor(context)) { event ->
                when (event) {
                    is VideoRecordEvent.Start -> _recording.value = RecordingState(true)
                    is VideoRecordEvent.Finalize -> {
                        activeRecording = null
                        if (event.hasError()) {
                            _recording.value = RecordingState(false, error = "recording failed: ${event.error}")
                        } else {
                            _recording.value = RecordingState(
                                false,
                                event.outputResults.outputUri.toString(),
                                event.recordingStats.recordedDurationNanos
                            )
                        }
                    }
                    else -> {}
                }
            }
    }

    fun stopRecording() {
        activeRecording?.stop()
    }

    fun shutdown() {
        runCatching { activeRecording?.stop() }
        activeRecording = null
        boostEngine.close()
        provider?.unbindAll()
        analysis?.clearAnalyzer()
        analysis = null
        capture = null
        videoCapture = null
        camera = null
        _isReady.value = false
        executor.shutdownNow()
    }

    companion object {
        /** Renders a stitched luma plane as a grayscale bitmap. */
        fun grayToBitmap(result: PanoStitcher.StitchResult): Bitmap {
            val w = result.width; val h = result.height
            val pixels = IntArray(w * h)
            for (i in pixels.indices) {
                val v = (result.data[i].coerceIn(0f, 1f) * 255f).toInt()
                pixels[i] = (0xFF shl 24) or (v shl 16) or (v shl 8) or v
            }
            val out = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
            out.setPixels(pixels, 0, w, 0, 0, w, h)
            return out
        }

        /** Samples the Y plane into a bounded luma plane for stitching. */
        fun lumaPlane(proxy: ImageProxy, maxW: Int): PanoStitcher.GrayPlane {
            val plane = proxy.planes[0]
            val srcW = proxy.width; val srcH = proxy.height
            val scale = minOf(1f, maxW.toFloat() / maxOf(srcW, srcH))
            val w = maxOf(1, (srcW * scale).toInt())
            val h = maxOf(1, (srcH * scale).toInt())
            val data = FloatArray(w * h)
            val rowStride = plane.rowStride
            val buffer = plane.buffer
            for (y in 0 until h) {
                val sy = (y / scale).toInt().coerceAtMost(srcH - 1)
                var base = sy * rowStride
                for (x in 0 until w) {
                    val sx = (x / scale).toInt().coerceAtMost(srcW - 1)
                    data[y * w + x] = (buffer.get(base + sx).toInt() and 0xFF) / 255f
                }
            }
            var p = PanoStitcher.GrayPlane(w, h, data)
            when (proxy.imageInfo.rotationDegrees) {
                90 -> p = p.rotate90cw()
                180 -> {
                    val out = FloatArray(w * h)
                    for (y in 0 until h) for (x in 0 until w) out[y * w + x] = data[(h - 1 - y) * w + (w - 1 - x)]
                    p = PanoStitcher.GrayPlane(w, h, out)
                }
                270 -> {
                    p = p.rotate90cw().rotate90cw().rotate90cw()
                }
            }
            return p
        }
    }
}

class CameraCapabilityRepository(private val context: Context) {
    fun discover(facing: Int = CameraCharacteristics.LENS_FACING_BACK): CameraCapabilities? {
        val manager = context.getSystemService(CameraManager::class.java) ?: return null
        val id = manager.cameraIdList.firstOrNull {
            manager.getCameraCharacteristics(it).get(CameraCharacteristics.LENS_FACING) == facing
        } ?: return null
        val c = manager.getCameraCharacteristics(id)
        val map = c.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP) ?: return null
        val pixel = c.get(CameraCharacteristics.SENSOR_INFO_PIXEL_ARRAY_SIZE)
        val fps = c.get(CameraCharacteristics.CONTROL_AE_AVAILABLE_TARGET_FPS_RANGES)
            ?.map { it.upper }?.distinct()?.sorted() ?: emptyList()
        val formats = map.outputFormats.toSet()
        val caps = c.get(CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES)?.toSet().orEmpty()
        return CameraCapabilities(
            cameraId = id,
            facing = facing,
            sensorWidth = pixel?.width ?: 0,
            sensorHeight = pixel?.height ?: 0,
            maxDigitalZoom = c.get(CameraCharacteristics.SCALER_AVAILABLE_MAX_DIGITAL_ZOOM) ?: 1f,
            supportsRaw = ImageFormat.RAW_SENSOR in formats &&
                CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES_RAW in caps,
            supportsManualExposure = c.get(CameraCharacteristics.CONTROL_AE_AVAILABLE_MODES)
                ?.contains(CameraCharacteristics.CONTROL_AE_MODE_OFF) == true,
            supportsManualFocus = c.get(CameraCharacteristics.CONTROL_AF_AVAILABLE_MODES)
                ?.contains(CameraCharacteristics.CONTROL_AF_MODE_OFF) == true,
            supportsOis = c.get(CameraCharacteristics.LENS_INFO_AVAILABLE_OPTICAL_STABILIZATION)
                ?.contains(CameraCharacteristics.LENS_OPTICAL_STABILIZATION_MODE_ON) == true,
            availableFps = fps
        )
    }
}
