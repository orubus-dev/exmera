package app.orubus.exmera

import app.orubus.exmera.camera.PanoStitcher
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

class PanoStitcherTest {

    /** Aperiodic deterministic scene content: alignment is unambiguous. */
    private fun scene(w: Int, h: Int, xShift: Int): PanoStitcher.GrayPlane {
        val data = FloatArray(w * h)
        for (y in 0 until h) {
            for (x in 0 until w) {
                val sx = x + xShift
                val hash = abs((sx * 2654435761L + y * 40503L) % 1048576L).toInt()
                data[y * w + x] = hash / 1048576f * 0.8f + 0.1f
            }
        }
        return PanoStitcher.GrayPlane(w, h, data)
    }

    @Test
    fun `single frame passes through`() {
        val p = scene(64, 48, 0)
        val r = PanoStitcher.stitch(listOf(p))!!
        assertEquals(64, r.width)
        assertEquals(1, r.placements.size)
    }

    @Test
    fun `two frames align at the known offset`() {
        val w = 200; val h = 120
        val trueShift = 120 // frame B starts at scene column 120
        val a = scene(w, h, 0)
        val b = scene(w, h, trueShift)
        val r = PanoStitcher.stitch(listOf(a, b))!!
        assertEquals(2, r.placements.size)
        val dx = r.placements[1].x - r.placements[0].x
        assertTrue(abs(dx - trueShift) <= 8, "aligned dx=$dx, true=$trueShift")
        assertTrue(r.width in (w + trueShift - 8)..(w + trueShift + 8), "width ${r.width}")
    }

    @Test
    fun `three frames stay monotonic`() {
        val w = 160; val h = 100
        val frames = listOf(
            scene(w, h, 0),
            scene(w, h, 110),
            scene(w, h, 165)
        )
        val r = PanoStitcher.stitch(frames)!!
        assertEquals(3, r.placements.size)
        val xs = r.placements.map { it.x }
        assertTrue(xs[1] > xs[0] && xs[2] > xs[1], "placements not monotonic: $xs")
        assertTrue(r.width > w)
    }

    @Test
    fun `empty list yields null`() {
        assertEquals(null, PanoStitcher.stitch(emptyList()))
    }

    @Test
    fun `scaled and rotated planes keep dimensions sane`() {
        val p = scene(480, 640, 0)
        val q = p.scaled(0.25f)
        assertEquals(120, q.width)
        assertEquals(160, q.height)
        val r = p.rotate90cw()
        assertEquals(640, r.width)
        assertEquals(480, r.height)
        assertEquals(p.at(0, 0), r.at(r.width - 1, 0), 1e-6f)
    }

    @Test
    fun `blend output stays finite and bounded`() {
        val a = PanoStitcher.GrayPlane(40, 20, FloatArray(40 * 20) { 1f })
        val b = PanoStitcher.GrayPlane(40, 20, FloatArray(40 * 20) { 0f })
        val r = PanoStitcher.stitch(listOf(a, b))
        assertNotNull(r)
        assertTrue(r.data.all { it.isFinite() && it in 0f..1f })
    }
}
