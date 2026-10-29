package app.orubus.exmera.camera

import kotlin.math.PI
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/** Classic stack blur over ARGB pixels — pure Kotlin, deterministic, allocation-free per pass. */
object StackBlur {

    fun blur(pixels: IntArray, w: Int, h: Int, radius: Int) {
        require(w > 0 && h > 0) { "empty image" }
        val r = max(0, radius)
        if (r == 0) return
        val div = 2 * r + 1
        val temp = IntArray(w * h)

        // horizontal pass
        for (y in 0 until h) {
            val row = y * w
            var sr = 0; var sg = 0; var sb = 0
            for (i in -r..r) {
                val p = pixels[row + min(w - 1, max(0, i))]
                sr += (p shr 16) and 0xFF; sg += (p shr 8) and 0xFF; sb += p and 0xFF
            }
            for (x in 0 until w) {
                temp[row + x] = (0xFF shl 24) or ((sr / div) shl 16) or ((sg / div) shl 8) or (sb / div)
                val outX = min(w - 1, max(0, x - r))
                val inX = min(w - 1, max(0, x + r + 1))
                val outP = pixels[row + outX]; val inP = pixels[row + inX]
                sr += ((inP shr 16) and 0xFF) - ((outP shr 16) and 0xFF)
                sg += ((inP shr 8) and 0xFF) - ((outP shr 8) and 0xFF)
                sb += (inP and 0xFF) - (outP and 0xFF)
            }
        }

        // vertical pass
        for (x in 0 until w) {
            var sr = 0; var sg = 0; var sb = 0
            for (i in -r..r) {
                val p = temp[min(h - 1, max(0, i)) * w + x]
                sr += (p shr 16) and 0xFF; sg += (p shr 8) and 0xFF; sb += p and 0xFF
            }
            for (y in 0 until h) {
                pixels[y * w + x] = (0xFF shl 24) or ((sr / div) shl 16) or ((sg / div) shl 8) or (sb / div)
                val outY = min(h - 1, max(0, y - r))
                val inY = min(h - 1, max(0, y + r + 1))
                val outP = temp[outY * w + x]; val inP = temp[inY * w + x]
                sr += ((inP shr 16) and 0xFF) - ((outP shr 16) and 0xFF)
                sg += ((inP shr 8) and 0xFF) - ((outP shr 8) and 0xFF)
                sb += (inP and 0xFF) - (outP and 0xFF)
            }
        }
    }
}

/**
 * Portrait bokeh without any model: a feathered elliptical in-focus region
 * over a fully blurred copy, blended per pixel. The mask center follows the
 * tap-to-focus point so the subject stays where the user asked for focus.
 *
 * When a depth EXM pack lands, this same entry point upgrades to true depth
 * bokeh — the UI contract does not change.
 */
object Bokeh {

    /**
     * @param focusCx focus point x in [0,1] (normalized image coordinates)
     * @param focusCy focus point y in [0,1]
     * @param strength 1..10 blur radius scale in the out-of-focus region
     */
    fun apply(
        src: android.graphics.Bitmap,
        focusCx: Float = 0.5f,
        focusCy: Float = 0.45f,
        strength: Int = 5
    ): android.graphics.Bitmap {
        val w = src.width; val h = src.height
        val pixels = IntArray(w * h)
        src.getPixels(pixels, 0, w, 0, 0, w, h)

        val blurred = pixels.copyOf()
        val radius = (min(w, h) / 140f * strength.coerceIn(1, 10)).roundToInt().coerceIn(1, 40)
        StackBlur.blur(blurred, w, h, radius)

        // Elliptical mask: sharp inside, gaussian feather to full blur outside.
        val cx = focusCx.coerceIn(0f, 1f) * w
        val cy = focusCy.coerceIn(0f, 1f) * h
        val rx = w * 0.34f
        val ry = h * 0.30f
        val feather = 0.55f // fraction of the radius over which sharpness falls off
        val out = IntArray(w * h)
        for (y in 0 until h) {
            val dy = (y - cy) / ry
            for (x in 0 until w) {
                val dx = (x - cx) / rx
                val d = kotlin.math.sqrt(dx * dx + dy * dy)
                val t = sharpness(d, feather)
                val i = y * w + x
                if (t >= 0.999f) { out[i] = pixels[i]; continue }
                if (t <= 0.001f) { out[i] = blurred[i]; continue }
                val p = pixels[i]; val b = blurred[i]
                val a = (t * 255f).roundToInt()
                out[i] = blend(p, b, a)
            }
        }

        val result = android.graphics.Bitmap.createBitmap(w, h, android.graphics.Bitmap.Config.ARGB_8888)
        result.setPixels(out, 0, w, 0, 0, w, h)
        return result
    }

    /** 1.0 inside the ellipse, smooth gaussian falloff across the feather band. */
    internal fun sharpness(d: Float, feather: Float): Float {
        if (d <= 1f) return 1f
        val t = (d - 1f) / max(0.05f, feather)
        return exp(-0.5f * t * t * 2.0f).coerceIn(0f, 1f)
    }

    private fun blend(sharp: Int, blurred: Int, alpha255: Int): Int {
        val inv = 255 - alpha255
        val r = (((sharp shr 16) and 0xFF) * alpha255 + ((blurred shr 16) and 0xFF) * inv) / 255
        val g = (((sharp shr 8) and 0xFF) * alpha255 + ((blurred shr 8) and 0xFF) * inv) / 255
        val b = ((sharp and 0xFF) * alpha255 + (blurred and 0xFF) * inv) / 255
        return (0xFF shl 24) or (r shl 16) or (g shl 8) or b
    }
}
