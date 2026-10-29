package app.orubus.exmera

import app.orubus.exmera.camera.Bokeh
import app.orubus.exmera.camera.MetadataScrubber
import app.orubus.exmera.camera.StackBlur
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class ProcessingPolicyTest {

    @Test
    fun `jpeg magic detection is exact`() {
        val jpegHeader = byteArrayOf(0xFF.toByte(), 0xD8.toByte(), 0xFF.toByte(), 0xE0.toByte())
        assertTrue(MetadataScrubber.isJpeg(jpegHeader))
        assertFalse(MetadataScrubber.isJpeg(byteArrayOf(0x89.toByte(), 0x50.toByte()))) // png
        assertFalse(MetadataScrubber.isJpeg(ByteArray(0)))
        assertFalse(MetadataScrubber.isJpeg(byteArrayOf(0xFF.toByte(), 0xD9.toByte())))
    }

    @Test
    fun `scrub policy defaults are privacy-first`() {
        val policy = MetadataScrubber.Policy()
        assertTrue(policy.scrubEnabled, "auto scrubbing must default ON")
        assertTrue(policy.keepRawDng, "raw negatives stay technical by default")
    }

    @Test
    fun `stack blur preserves dimensions and bounds`() {
        val w = 32; val h = 24
        val pixels = IntArray(w * h) { (it * 7919) and 0xFFFFFF or (0xFF shl 24) }
        val copy = pixels.copyOf()
        StackBlur.blur(pixels, w, h, 4)
        assertEquals(copy.size, pixels.size)
        // every pixel stays opaque ARGB
        assertTrue(pixels.all { (it ushr 24) == 0xFF })
        // blurred image must differ from the source somewhere (blur actually ran)
        assertTrue(pixels.indices.any { pixels[it] != copy[it] })
    }

    @Test
    fun `stack blur with zero radius is a no-op`() {
        val pixels = IntArray(100) { it }
        val copy = pixels.copyOf()
        StackBlur.blur(pixels, 10, 10, 0)
        assertTrue(pixels.contentEquals(copy))
    }

    @Test
    fun `bokeh sharpness falls off outside the ellipse`() {
        assertTrue(Bokeh.sharpness(0.5f, 0.55f) >= 0.999f, "inside must be fully sharp")
        assertTrue(Bokeh.sharpness(1.0f, 0.55f) >= 0.999f, "ellipse edge is sharp")
        assertTrue(Bokeh.sharpness(2.0f, 0.55f) < 0.6f, "beyond feather must be mostly blurred")
        assertTrue(Bokeh.sharpness(6.0f, 0.55f) <= 0.05f, "far field must be fully blurred")
        // monotonic decay
        assertTrue(Bokeh.sharpness(1.2f, 0.55f) > Bokeh.sharpness(2.2f, 0.55f))
    }
}
