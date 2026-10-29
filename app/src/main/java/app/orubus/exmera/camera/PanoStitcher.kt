package app.orubus.exmera.camera

import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min

/**
 * Horizontal panorama stitching without OpenCV and without any model.
 *
 * Coarse-to-fine block matching: mean-absolute-difference template match on a
 * quarter-scale luma pyramid to lock the search neighborhood, then a refinement
 * pass at full scale. Frames are blended across their overlap with a linear
 * feather so seams are invisible on real sweeps.
 *
 * Pure Kotlin on plain float luma planes — unit-testable on the JVM, runs in
 * tens of milliseconds per frame pair on a budget phone.
 */
object PanoStitcher {

    class GrayPlane(val width: Int, val height: Int, val data: FloatArray) {
        init { require(width > 0 && height > 0 && data.size >= width * height) { "bad plane" } }

        fun at(x: Int, y: Int): Float = data[y * width + x]

        fun scaled(factor: Float): GrayPlane {
            val nw = max(8, (width * factor).toInt())
            val nh = max(8, (height * factor).toInt())
            val out = FloatArray(nw * nh)
            for (y in 0 until nh) {
                val sy = min(height - 1, (y / factor).toInt())
                for (x in 0 until nw) {
                    val sx = min(width - 1, (x / factor).toInt())
                    out[y * nw + x] = data[sy * width + sx]
                }
            }
            return GrayPlane(nw, nh, out)
        }

        fun rotate90cw(): GrayPlane {
            val out = FloatArray(width * height)
            for (y in 0 until height) for (x in 0 until width) {
                out[x * height + (height - 1 - y)] = data[y * width + x]
            }
            return GrayPlane(height, width, out)
        }
    }

    data class Placement(val index: Int, val x: Int, val y: Int)

    data class StitchResult(
        val width: Int,
        val height: Int,
        val placements: List<Placement>,
        val data: FloatArray
    )

    data class Offset(val dx: Int, val dy: Int, val score: Double)

    /**
     * Stitches left-to-right. Returns null when fewer than one usable frame.
     * Frames keep their own resolution; the canvas is their union with drift.
     */
    fun stitch(frames: List<GrayPlane>): StitchResult? {
        if (frames.isEmpty()) return null
        if (frames.size == 1) {
            return StitchResult(frames[0].width, frames[0].height, listOf(Placement(0, 0, 0)), frames[0].data.copyOf())
        }

        data class Placed(val plane: GrayPlane, val x: Int, val y: Int)
        val placed = ArrayList<Placed>(frames.size)
        placed += Placed(frames[0], 0, 0)

        for (i in 1 until frames.size) {
            val prev = placed.last()
            val off = align(prev.plane, frames[i]) ?: Offset(prev.plane.width / 2, 0, Double.MAX_VALUE)
            val x = prev.x + off.dx
            val y = prev.y + off.dy
            placed += Placed(frames[i], x, y)
        }

        val minX = placed.minOf { it.x }; val maxX = placed.maxOf { it.x + it.plane.width }
        val minY = placed.minOf { it.y }; val maxY = placed.maxOf { it.y + it.plane.height }
        val cw = maxX - minX; val ch = maxY - minY
        require(cw in 1..20000 && ch in 1..20000) { "canvas out of bounds: ${cw}x${ch}" }

        val canvas = FloatArray(cw * ch)
        val weight = FloatArray(cw * ch)

        placed.forEachIndexed { idx, p ->
            val pl = p.plane
            val ox = p.x - minX; val oy = p.y - minY
            for (y in 0 until pl.height) {
                val cy = oy + y
                for (x in 0 until pl.width) {
                    val cx = ox + x
                    val ci = cy * cw + cx
                    // linear feather on left/right edges of this frame
                    val edge = min(1f, min(x + 1f, (pl.width - x).toFloat()) / 24f)
                    val v = pl.data[y * pl.width + x]
                    canvas[ci] += v * edge
                    weight[ci] += edge
                }
            }
            if (idx == 0) { /* weight covers everything frame 0 touches */ }
        }

        for (i in canvas.indices) {
            if (weight[i] > 0f) canvas[i] /= weight[i]
        }

        return StitchResult(cw, ch, placed.mapIndexed { i, p -> Placement(index = i, x = p.x - minX, y = p.y - minY) }, canvas)
    }

    /**
     * Finds where [next] begins inside [prev]: returns the horizontal step from
     * prev's left edge to next's left edge (0..prev.width-1) plus vertical drift.
     */
    internal fun align(prev: GrayPlane, next: GrayPlane): Offset? {
        val coarse = 0.25f
        val pC = prev.scaled(coarse)
        val nC = next.scaled(coarse)
        val rough = search(pC, nC, coarse, dxMin = 0, dxMax = pC.width - 8, dyMax = max(4, pC.height / 16)) ?: return null

        val fullDx = (rough.dx / coarse).toInt()
        val fullDy = (rough.dy / coarse).toInt()
        val fine = search(prev, next, 1f,
            dxMin = max(0, fullDx - 8), dxMax = min(prev.width - 1, fullDx + 8),
            dyMax = max(4, next.height / 12), dyHint = fullDy
        ) ?: rough
        return fine
    }

    private fun search(prev: GrayPlane, next: GrayPlane, scale: Float,
                       dxMin: Int, dxMax: Int, dyMax: Int, dyHint: Int = 0): Offset? {
        val tplW = max(8, (next.width * scale * 0.15f).toInt())
        val tplH = max(8, (next.height * scale * 0.5f).toInt())
        if (tplW >= next.width || tplH >= next.height) return null

        val bandY = (next.height - tplH) / 2
        var best: Offset? = null
        var bestScore = Double.MAX_VALUE

        for (dx in dxMin..dxMax) {
            for (dy in (dyHint - dyMax).coerceAtLeast(-dyMax)..(dyHint + dyMax).coerceAtMost(dyMax)) {
                // next's column 0 lands at prev column dx; overlap starts at dx
                var acc = 0.0; var n = 0
                for (y in 0 until tplH) {
                    val py = bandY + y + dy; val ny = bandY + y
                    if (py < 0 || py >= prev.height) continue
                    for (x in 0 until tplW) {
                        val px = dx + x
                        if (px < 0 || px >= prev.width) continue
                        acc += abs(prev.at(px, py) - next.at(x, ny))
                        n++
                    }
                }
                if (n < tplW * tplH / 2) continue
                val score = acc / n
                // primary: MAD. tie-break: prefer zero vertical drift, then more
                // canvas coverage — protects against self-similar textures (sky,
                // walls) where several offsets score identically.
                val better = score < bestScore ||
                    (score == bestScore && best != null &&
                        (abs(dy) < abs(best.dy) || (abs(dy) == abs(best.dy) && dx > best.dx)))
                if (better) {
                    bestScore = score
                    best = Offset(dx, dy, score)
                }
            }
        }
        return best
    }
}
