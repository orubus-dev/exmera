package studio.exmera.exm

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class ExmBenchmarkTest {

    /** Mixed-entropy payload: compressible runs + pseudo-random noise, like real weights. */
    private fun mixedPayload(size: Int): ByteArray = ByteArray(size) { i ->
        if (i % 97 < 64) ((i / 97) % 251).toByte() // compressible band
        else ((i * 31 + 7) % 251).toByte()         // noise band
    }

    @Test
    fun reportsAreSaneAndIntegrityHolds() {
        val payload = mixedPayload(300_000)
        val reports = ExmBenchmark.run(
            payload,
            configs = listOf(
                ExmBenchmark.Config("64k", 64L * 1024L),
                ExmBenchmark.Config("128k", 128L * 1024L)
            )
        )
        assertEquals(2, reports.size)
        for (r in reports) {
            assertEquals(payload.size.toLong(), r.originalBytes)
            assertTrue(r.storedBytes > 0 && r.storedBytes < r.originalBytes * 2) { "stored size implausible: $r" }
            assertTrue(r.ratio > 0.0) { "ratio must be positive: $r" }
            assertTrue(r.chunks >= 1) { "must have chunks: $r" }
            assertTrue(r.integrity) { "integrity must hold: $r" }
            assertTrue(r.writeMs >= 0.0 && r.headerMs >= 0.0 && r.assembleMs >= 0.0)
        }
        assertEquals(5, reports[0].chunks) // ceil(300000/65536) = 5
        assertEquals(3, reports[1].chunks) // ceil(300000/131072) = 3
    }

    @Test
    fun declaredTensorBenchmarkKeepsCoverage() {
        val payload = mixedPayload(8_192)
        val tensors = listOf(
            ExmChunked.TensorSpec("enc", "int8", intArrayOf(4_096)),
            ExmChunked.TensorSpec("dec", "int8", intArrayOf(4_096))
        )
        val report = ExmBenchmark.runOne(payload, ExmBenchmark.Config("2k-tensor", 2_048L, tensors))
        assertEquals(4, report.chunks) // 2 tensors × 2 spans each
        assertTrue(report.integrity)
    }

    @Test
    fun renderProducesAnAlignedTable() {
        val payload = mixedPayload(50_000)
        val table = ExmBenchmark.render(ExmBenchmark.run(payload, configs = listOf(ExmBenchmark.Config("only", 16_384L))))
        val lines = table.lines()
        assertEquals(2, lines.size)
        assertTrue(lines[0].contains("original"))
        assertTrue(lines[1].contains("ok"))
    }
}
