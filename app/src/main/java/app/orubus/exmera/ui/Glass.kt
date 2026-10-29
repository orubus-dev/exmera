package app.orubus.exmera.ui

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.gestures.detectDragGestures
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import kotlin.math.roundToInt

/**
 * Monochrome liquid glass design system.
 *
 * One palette (black glass, white ink), one shape language (deep rounded
 * corners), one material: translucent panels with a hairline border and a
 * top-edge specular highlight — the macOS liquid-glass read, no color anywhere.
 */
object Mono {
    val ink = Color.White
    val inkDim = Color.White.copy(alpha = 0.55f)
    val inkFaint = Color.White.copy(alpha = 0.32f)
    val glass = Color.Black.copy(alpha = 0.42f)
    val glassStrong = Color.Black.copy(alpha = 0.58f)
    val hairline = Color.White.copy(alpha = 0.14f)
    val sheen = Color.White.copy(alpha = 0.10f)
}

/** A liquid glass panel: translucent black, hairline border, top sheen. */
@Composable
fun GlassPanel(
    modifier: Modifier = Modifier,
    corner: Dp = 24.dp,
    strong: Boolean = false,
    content: @Composable androidx.compose.foundation.layout.BoxScope.() -> Unit
) {
    Box(
        modifier
            .clip(RoundedCornerShape(corner))
            .background(if (strong) Mono.glassStrong else Mono.glass)
            .border(1.dp, Mono.hairline, RoundedCornerShape(corner))
    ) {
        // top-edge sheen
        Box(
            Modifier
                .fillMaxWidth()
                .height(corner)
                .background(
                    Brush.verticalGradient(listOf(Mono.sheen, Color.Transparent))
                )
        )
        content()
    }
}

/** Small glass pill with a text label — mode chips, toggles, zoom steps. */
@Composable
fun GlassChip(
    label: String,
    selected: Boolean,
    modifier: Modifier = Modifier,
    onClick: () -> Unit
) {
    Box(
        modifier
            .clip(RoundedCornerShape(999.dp))
            .background(if (selected) Color.White.copy(alpha = 0.16f) else Mono.glass)
            .border(
                1.dp,
                if (selected) Color.White.copy(alpha = 0.5f) else Mono.hairline,
                RoundedCornerShape(999.dp)
            )
            .pointerInput(selected, onClick) { detectTapGestures { onClick() } }
            .padding(horizontal = 14.dp, vertical = 7.dp)
    ) {
        Text(
            label,
            color = if (selected) Mono.ink else Mono.inkDim,
            fontSize = 12.sp
        )
    }
}

/** Vertical or horizontal mono slider rendered on canvas — thin track, round knob. */
@Composable
fun MonoSlider(
    value: Float,
    onValueChange: (Float) -> Unit,
    modifier: Modifier = Modifier,
    label: String = "",
    valueText: String = "",
    enabled: Boolean = true
) {
    androidx.compose.foundation.layout.Column(modifier = modifier) {
        if (label.isNotEmpty() || valueText.isNotEmpty()) {
            androidx.compose.foundation.layout.Row(
                Modifier.fillMaxWidth(),
                horizontalArrangement = androidx.compose.foundation.layout.Arrangement.SpaceBetween
            ) {
                Text(label, color = Mono.inkDim, fontSize = 10.sp)
                Text(valueText, color = Mono.ink, fontSize = 10.sp)
            }
        }
        Box(
            Modifier
                .fillMaxWidth()
                .height(28.dp)
                .pointerInput(enabled) {
                    if (!enabled) return@pointerInput
                    detectTapGestures { offset ->
                        onValueChange((offset.x / size.width.coerceAtLeast(1)).coerceIn(0f, 1f))
                    }
                }
                .pointerInput(enabled) {
                    if (!enabled) return@pointerInput
                    detectDragGestures { change, _ ->
                        change.consume()
                        onValueChange((change.position.x / size.width.coerceAtLeast(1)).coerceIn(0f, 1f))
                    }
                }
        ) {
            Canvas(Modifier.fillMaxWidth().height(28.dp).padding(vertical = 12.dp)) {
                val y = size.height / 2
                drawLine(
                    Mono.inkFaint,
                    Offset(0f, y),
                    Offset(size.width, y),
                    strokeWidth = 2.dp.toPx(),
                    cap = StrokeCap.Round
                )
                drawCircle(
                    Mono.ink,
                    radius = 7.dp.toPx(),
                    center = Offset(size.width * value.coerceIn(0f, 1f), y)
                )
                drawCircle(
                    Color.Black.copy(alpha = 0.5f),
                    radius = 7.dp.toPx(),
                    center = Offset(size.width * value.coerceIn(0f, 1f), y),
                    style = Stroke(width = 1.dp.toPx())
                )
            }
        }
    }
}

/** Minimal monochrome vector icons drawn in-place (no icon dependency). */
@Composable
fun MonoIcon(kind: String, modifier: Modifier = Modifier, tint: Color = Mono.ink) {
    Canvas(modifier) {
        val w = size.width; val h = size.height
        val stroke = Stroke(width = w.coerceAtMost(h) / 14f, cap = StrokeCap.Round)
        when (kind) {
            "packs" -> { // stacked squares
                val s = w * 0.52f; val o = w * 0.14f
                drawRoundRect(tint, Offset(o, o + h * 0.28f), androidx.compose.ui.geometry.Size(s, s), stroke)
                drawRoundRect(tint, Offset(o + w * 0.16f, o), androidx.compose.ui.geometry.Size(s, s), stroke)
            }
            "privacy" -> { // shield
                val path = Path().apply {
                    moveTo(w / 2, h * 0.1f)
                    lineTo(w * 0.85f, h * 0.25f)
                    lineTo(w * 0.85f, h * 0.55f)
                    quadraticBezierTo(w * 0.85f, h * 0.8f, w / 2, h * 0.92f)
                    quadraticBezierTo(w * 0.15f, h * 0.8f, w * 0.15f, h * 0.55f)
                    lineTo(w * 0.15f, h * 0.25f)
                    close()
                }
                drawPath(path, tint, style = stroke)
            }
            "flip" -> { // circular arrows
                drawArc(tint, -60f, 280f, false, style = stroke)
                val tip = Path().apply {
                    moveTo(w * 0.86f, h * 0.16f)
                    lineTo(w * 0.70f, h * 0.10f)
                    lineTo(w * 0.78f, h * 0.30f)
                    close()
                }
                drawPath(tip, tint)
            }
            "flash" -> {
                val path = Path().apply {
                    moveTo(w * 0.55f, h * 0.08f)
                    lineTo(w * 0.30f, h * 0.55f)
                    lineTo(w * 0.50f, h * 0.55f)
                    lineTo(w * 0.45f, h * 0.92f)
                    lineTo(w * 0.72f, h * 0.42f)
                    lineTo(w * 0.52f, h * 0.42f)
                    close()
                }
                drawPath(path, tint, style = stroke)
            }
            "grid" -> {
                drawLine(tint, Offset(w / 2, h * 0.12f), Offset(w / 2, h * 0.88f), strokeWidth = stroke.width, cap = StrokeCap.Round)
                drawLine(tint, Offset(w * 0.12f, h / 2), Offset(w * 0.88f, h / 2), strokeWidth = stroke.width, cap = StrokeCap.Round)
            }
            "close" -> {
                drawLine(tint, Offset(w * 0.2f, h * 0.2f), Offset(w * 0.8f, h * 0.8f), strokeWidth = stroke.width, cap = StrokeCap.Round)
                drawLine(tint, Offset(w * 0.8f, h * 0.2f), Offset(w * 0.2f, h * 0.8f), strokeWidth = stroke.width, cap = StrokeCap.Round)
            }
            "boost" -> { // spark
                val path = Path().apply {
                    moveTo(w * 0.5f, h * 0.06f)
                    lineTo(w * 0.62f, h * 0.40f)
                    lineTo(w * 0.96f, h * 0.50f)
                    lineTo(w * 0.62f, h * 0.60f)
                    lineTo(w * 0.5f, h * 0.94f)
                    lineTo(w * 0.38f, h * 0.60f)
                    lineTo(w * 0.04f, h * 0.50f)
                    lineTo(w * 0.38f, h * 0.40f)
                    close()
                }
                drawPath(path, tint, style = stroke)
            }
        }
    }
}

fun formatSeconds(ns: Long): String {
    val s = ns / 1_000_000_000L
    return "%02d:%02d".format(s / 60, s % 60)
}

fun clamp01(v: Float): Float = v.coerceIn(0f, 1f)

fun percent(v: Float): Int = (v * 100).roundToInt()
