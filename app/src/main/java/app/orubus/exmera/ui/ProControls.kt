package app.orubus.exmera.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import android.hardware.camera2.CameraMetadata
import app.orubus.exmera.camera.ProCamera2
import kotlin.math.ln
import kotlin.math.pow
import kotlin.math.roundToInt

/** PRO mode glass panel: manual exposure, focus, white balance, RAW. */
@Composable
fun ProControls(holder: ProCamera2Holder) {
    val pro = holder.get() ?: return
    val ranges by pro.ranges.collectAsState()
    val params by pro.params.collectAsState()
    val r = ranges ?: return

    GlassPanel(Modifier.padding(horizontal = 16.dp).fillMaxWidth(), corner = 20.dp, strong = true) {
        Column(Modifier.padding(horizontal = 14.dp, vertical = 10.dp)) {
            Row(horizontalArrangement = Arrangement.spacedBy(6.dp), modifier = Modifier.fillMaxWidth()) {
                GlassChip(
                    "EXP" + if (params.manualExposure) " on" else "",
                    selected = params.manualExposure
                ) { pro.update { it.copy(manualExposure = !it.manualExposure) } }
                GlassChip(
                    "FOCUS" + if (params.manualFocus) " on" else "",
                    selected = params.manualFocus
                ) { pro.update { it.copy(manualFocus = !it.manualFocus) } }
                GlassChip(
                    if (r.supportsRaw) (if (params.rawEnabled) "RAW on" else "RAW off") else "no RAW",
                    selected = params.rawEnabled && r.supportsRaw
                ) {
                    if (r.supportsRaw) pro.update { it.copy(rawEnabled = !it.rawEnabled) }
                }
            }

            if (params.manualExposure) {
                val isoSpan = (r.isoMax - r.isoMin).coerceAtLeast(1)
                MonoSlider(
                    value = (params.iso - r.isoMin).toFloat() / isoSpan,
                    onValueChange = { v ->
                        pro.update { it.copy(manualExposure = true, iso = (r.isoMin + v * isoSpan).roundToInt()) }
                    },
                    label = "ISO",
                    valueText = "${params.iso}"
                )
                val expMin = r.expMinNs.toFloat().coerceAtLeast(1f)
                val expMax = r.expMaxNs.toFloat()
                val logRatio = if (expMax > expMin) ln(expMax / expMin) else 1f
                val expV = if (params.exposureNs > 0) (ln(params.exposureNs.toFloat() / expMin) / logRatio).coerceIn(0f, 1f) else 0f
                MonoSlider(
                    value = expV,
                    onValueChange = { v ->
                        val ns = (expMin * (expMax / expMin).pow(v)).toLong().coerceIn(r.expMinNs, r.expMaxNs)
                        pro.update { it.copy(manualExposure = true, exposureNs = ns) }
                    },
                    label = "shutter",
                    valueText = formatShutter(params.exposureNs)
                )
            }
            if (params.manualFocus) {
                MonoSlider(
                    value = (params.focusDistance / r.focusMax.coerceAtLeast(0.01f)).coerceIn(0f, 1f),
                    onValueChange = { v ->
                        pro.update { it.copy(manualFocus = true, focusDistance = v * r.focusMax) }
                    },
                    label = "focus",
                    valueText = if (params.focusDistance <= 0.01f) "∞" else "%.2f m".format(1f / params.focusDistance)
                )
            }
            Row(
                horizontalArrangement = Arrangement.spacedBy(6.dp),
                modifier = Modifier.fillMaxWidth().padding(top = 4.dp)
            ) {
                r.awbModes.take(6).forEach { mode ->
                    GlassChip(awbLabel(mode), selected = params.awbMode == mode) {
                        pro.update { it.copy(awbMode = mode) }
                    }
                }
            }
            Text(
                if (r.supportsRaw) "raw ${r.rawSize?.width}×${r.rawSize?.height} · dng, no GPS" else "manual on jpeg · sensor lacks raw",
                color = Mono.inkFaint,
                fontSize = 9.sp,
                modifier = Modifier.padding(top = 6.dp)
            )
        }
    }
}

private fun formatShutter(ns: Long): String {
    val s = ns / 1_000_000_000.0
    return if (s < 1f) "1/${(1 / s).roundToInt()} s" else "%.1f s".format(s)
}

private fun awbLabel(mode: Int): String = when (mode) {
    CameraMetadata.CONTROL_AWB_MODE_AUTO -> "auto"
    CameraMetadata.CONTROL_AWB_MODE_INCANDESCENT -> "tungsten"
    CameraMetadata.CONTROL_AWB_MODE_FLUORESCENT -> "fluor."
    CameraMetadata.CONTROL_AWB_MODE_WARM_FLUORESCENT -> "warm"
    CameraMetadata.CONTROL_AWB_MODE_DAYLIGHT -> "daylight"
    CameraMetadata.CONTROL_AWB_MODE_CLOUDY_DAYLIGHT -> "cloudy"
    CameraMetadata.CONTROL_AWB_MODE_TWILIGHT -> "twilight"
    CameraMetadata.CONTROL_AWB_MODE_SHADE -> "shade"
    else -> "wb$mode"
}
