package app.orubus.exmera.ui

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.platform.LocalHapticFeedback
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import app.orubus.exmera.camera.CameraMode
import app.orubus.exmera.camera.ExmeraCameraController
import app.orubus.exmera.camera.ProCamera2
import androidx.camera.core.ImageCapture

private val MODE_ORDER = listOf(CameraMode.PRO, CameraMode.VIDEO, CameraMode.IMAGE, CameraMode.PORTRAIT, CameraMode.PANO)

fun modeLabel(m: CameraMode): String = when (m) {
    CameraMode.PRO -> "PRO"
    CameraMode.VIDEO -> "VIDEO"
    CameraMode.IMAGE -> "IMAGE"
    CameraMode.PORTRAIT -> "PORTRAIT"
    CameraMode.PANO -> "PANO"
}

/** The exmera client screen — monochrome liquid glass over a full-bleed preview. */
@Composable
fun ExmeraApp(profile: String) {
    var mode by remember { mutableStateOf(CameraMode.IMAGE) }
    var status by remember { mutableStateOf("exmera · $profile · ships clean, packs optional") }
    var sheet by remember { mutableStateOf<String?>(null) } // "packs" | "privacy"
    var flashMode by remember { mutableStateOf(ImageCapture.FLASH_MODE_OFF) }
    var portraitStrength by remember { mutableStateOf(5) }
    var flipToken by remember { mutableStateOf(0) }
    val haptics = LocalHapticFeedback.current

    val context = androidx.compose.ui.platform.LocalContext.current
    val lifecycleOwner = androidx.lifecycle.compose.LocalLifecycleOwner.current
    val controller = remember { ExmeraCameraControllerHolder() }
    val pro = remember { ProCamera2Holder() }

    DisposableEffect(Unit) {
        onDispose { controller.dispose(); pro.dispose() }
    }

    Box(Modifier.fillMaxSize().background(Color.Black)) {
        if (mode == CameraMode.PRO) {
            ProPreviewHost(pro, Modifier.fillMaxSize())
        } else {
            CameraXPreviewHost(controller, mode, Modifier.fillMaxSize(), flipToken)
        }

        // top bar
        GlassPanel(
            Modifier
                .align(Alignment.TopCenter)
                .padding(top = 44.dp, start = 16.dp, end = 16.dp)
                .fillMaxWidth()
                .height(46.dp),
            corner = 20.dp
        ) {
            Row(
                Modifier.fillMaxSize().padding(horizontal = 14.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text("exmera", color = Mono.ink, fontSize = 15.sp)
                Spacer(Modifier.width(8.dp))
                Text(modeLabel(mode), color = Mono.inkDim, fontSize = 10.sp)
                Spacer(Modifier.weight(1f))
                if (mode != CameraMode.PRO && mode != CameraMode.VIDEO) {
                    GlassChip(
                        label = when (flashMode) {
                            ImageCapture.FLASH_MODE_ON -> "flash on"
                            ImageCapture.FLASH_MODE_AUTO -> "flash auto"
                            else -> "flash off"
                        },
                        selected = false
                    ) {
                        flashMode = when (flashMode) {
                            ImageCapture.FLASH_MODE_OFF -> ImageCapture.FLASH_MODE_AUTO
                            ImageCapture.FLASH_MODE_AUTO -> ImageCapture.FLASH_MODE_ON
                            else -> ImageCapture.FLASH_MODE_OFF
                        }
                        controller.get()?.setFlash(flashMode)
                    }
                    Spacer(Modifier.width(8.dp))
                }
                MonoIcon(
                    "packs",
                    Modifier.size(22.dp).pointerInput(sheet) { detectTapGestures { sheet = "packs" } }
                )
                Spacer(Modifier.width(14.dp))
                MonoIcon(
                    "privacy",
                    Modifier.size(22.dp).pointerInput(sheet) { detectTapGestures { sheet = "privacy" } }
                )
            }
        }

        // right rail: zoom + flip (CameraX modes)
        if (mode != CameraMode.PRO) {
            Column(
                Modifier
                    .align(Alignment.CenterEnd)
                    .padding(end = 12.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                listOf(1f, 2f, 5f).forEach { z ->
                    GlassChip("·${z.toInt()}x", selected = false) { controller.get()?.zoom(z) }
                }
                GlassPanel(corner = 999.dp, modifier = Modifier.size(44.dp)) {
                    MonoIcon(
                        "flip",
                        Modifier.size(22.dp).align(Alignment.Center).pointerInput(mode) {
                            detectTapGestures {
                                controller.get()?.flip()
                                flipToken++
                            }
                        }
                    )
                }
            }
        }

        // bottom: controls, status, shutter, carousel
        Column(
            Modifier
                .align(Alignment.BottomCenter)
                .fillMaxWidth()
                .padding(bottom = 26.dp)
        ) {
            ModeControls(mode, controller, pro, portraitStrength) { portraitStrength = it }
            Spacer(Modifier.height(10.dp))
            Text(
                status,
                color = Mono.inkDim,
                fontSize = 11.sp,
                modifier = Modifier.align(Alignment.CenterHorizontally).padding(horizontal = 24.dp)
            )
            Spacer(Modifier.height(12.dp))
            Row(
                Modifier.fillMaxWidth().padding(horizontal = 26.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                Box(Modifier.weight(1f))
                ShutterButton(
                    modifier = Modifier.size(76.dp),
                    recording = mode == CameraMode.VIDEO && controller.get()?.recording?.value?.recording == true
                ) {
                    haptics.performHapticFeedback(androidx.compose.ui.hapticfeedback.HapticFeedbackType.LongPress)
                    val c = controller.get()
                    when (mode) {
                        CameraMode.PRO -> {
                            pro.get()?.captureStill()
                            status = "pro capture · manual parameters"
                        }
                        CameraMode.VIDEO -> {
                            if (c == null) return@ShutterButton
                            if (c.recording.value.recording) {
                                c.stopRecording(); status = "video saved · no location written"
                            } else {
                                c.startRecording(); status = "recording…"
                            }
                        }
                        CameraMode.IMAGE -> {
                            if (c == null || c.busy.value) return@ShutterButton
                            if (c.boostState.value == "capturing" || c.hasBoostFrames()) {
                                c.finishBoost { _, msg -> status = msg }
                            } else {
                                c.captureStill(CameraMode.IMAGE, 0) { _, msg -> status = msg }
                            }
                        }
                        CameraMode.PORTRAIT -> {
                            if (c == null || c.busy.value) return@ShutterButton
                            c.captureStill(CameraMode.PORTRAIT, portraitStrength) { _, msg -> status = msg }
                        }
                        CameraMode.PANO -> {
                            if (c == null) return@ShutterButton
                            if (!c.panoState.value.startsWith("sweep")) {
                                c.startPano(); status = "sweep sideways, tap again to stitch"
                            } else {
                                c.finishPano { _, msg -> status = msg }
                            }
                        }
                    }
                }
                Box(Modifier.weight(1f))
            }
            Spacer(Modifier.height(14.dp))
            ModeCarousel(mode, Modifier.fillMaxWidth()) { m ->
                if (m != mode) {
                    mode = m
                    status = when (m) {
                        CameraMode.PRO -> "manual · ISO · shutter · focus · WB · RAW"
                        CameraMode.VIDEO -> "video · mp4 · location never written"
                        CameraMode.IMAGE -> "image · auto metadata scrub"
                        CameraMode.PORTRAIT -> "portrait · software bokeh · depth pack ready"
                        CameraMode.PANO -> "panorama · capture then stitch on device"
                    }
                }
            }
        }

        sheet?.let {
            when (it) {
                "packs" -> PacksSheet(onDismiss = { sheet = null }, onStatus = { status = it })
                "privacy" -> PrivacySheet(onDismiss = { sheet = null })
            }
        }
    }
}

/** Mode-specific glass controls above the shutter row. */
@Composable
private fun ModeControls(
    mode: CameraMode,
    controller: ExmeraCameraControllerHolder,
    pro: ProCamera2Holder,
    portrait: Int,
    onPortraitChange: (Int) -> Unit
) {
    when (mode) {
        CameraMode.PRO -> ProControls(pro)
        CameraMode.PORTRAIT -> GlassPanel(
            Modifier.padding(horizontal = 16.dp).fillMaxWidth(), corner = 20.dp
        ) {
            Column(Modifier.padding(horizontal = 16.dp, vertical = 8.dp)) {
                MonoSlider(
                    value = portrait / 10f,
                    onValueChange = { onPortraitChange((it * 10f).toInt().coerceIn(1, 10)) },
                    label = "bokeh",
                    valueText = "strength $portrait"
                )
            }
        }
        CameraMode.IMAGE -> {
            val c = controller.get()
            val state = c?.boostState?.value ?: "idle"
            val done = c?.hasBoostFrames() == true
            GlassPanel(
                Modifier.padding(horizontal = 16.dp).fillMaxWidth(), corner = 20.dp
            ) {
                Row(Modifier.padding(horizontal = 14.dp, vertical = 8.dp), verticalAlignment = Alignment.CenterVertically) {
                    Text("multi-frame boost", color = Mono.inkDim, fontSize = 11.sp, modifier = Modifier.weight(1f))
                    GlassChip(
                        label = when {
                            state == "capturing" -> "capturing · shutter fuses"
                            state == "processing" -> "fusing…"
                            state == "done" -> "saved"
                            done -> "frames ready · shutter fuses"
                            else -> "off · tap to capture 5"
                        },
                        selected = state == "capturing" || done
                    ) {
                        when {
                            state == "capturing" -> {}
                            done -> c?.finishBoost { _, msg -> }
                            else -> c?.startBoost(5)
                        }
                    }
                }
            }
        }
        else -> {}
    }
}

@Composable
private fun ModeCarousel(mode: CameraMode, modifier: Modifier, onSelect: (CameraMode) -> Unit) {
    LazyRow(
        modifier = modifier,
        horizontalArrangement = Arrangement.spacedBy(6.dp),
        contentPadding = androidx.compose.foundation.layout.PaddingValues(horizontal = 18.dp)
    ) {
        items(MODE_ORDER) { m ->
            GlassChip(modeLabel(m), selected = m == mode) { onSelect(m) }
        }
    }
}

@Composable
private fun ShutterButton(modifier: Modifier, recording: Boolean, onShutter: () -> Unit) {
    Box(
        modifier.pointerInput(recording) { detectTapGestures { onShutter() } },
        contentAlignment = Alignment.Center
    ) {
        Canvas(Modifier.fillMaxSize()) {
            drawCircle(
                Mono.ink,
                radius = size.minDimension / 2,
                style = androidx.compose.ui.graphics.drawscope.Stroke(width = 3.dp.toPx())
            )
            val r = size.minDimension / 2 * if (recording) 0.42f else 0.68f
            drawCircle(Mono.ink, radius = r)
        }
    }
}

/** Holds the CameraX controller across recompositions without leaking previews. */
class ExmeraCameraControllerHolder {
    private var impl: ExmeraCameraController? = null

    fun getOrCreate(context: android.content.Context, owner: androidx.lifecycle.LifecycleOwner): ExmeraCameraController {
        if (impl == null) impl = ExmeraCameraController(context, owner)
        return impl!!
    }

    fun get(): ExmeraCameraController? = impl

    fun dispose() {
        impl?.shutdown()
        impl = null
    }
}

/** Holds the Pro stack across recompositions. */
class ProCamera2Holder {
    private var impl: ProCamera2? = null

    fun getOrCreate(context: android.content.Context): ProCamera2 {
        if (impl == null) impl = ProCamera2(context)
        return impl!!
    }

    fun get(): ProCamera2? = impl

    fun dispose() {
        impl?.close()
        impl = null
    }
}
