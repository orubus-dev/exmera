package studio.exmera.exm

import java.io.ByteArrayOutputStream
import java.io.File
import java.io.RandomAccessFile

/**
 * Compression benchmark harness for EXM containers.
 *
 * Everything is measured on real container artifacts — never assumed. A run
 * writes a chunked v3 container to disk, parses the header back, assembles
 * and verifies the payload, and reports sizes, ratio and wall-clock timings.
 *
 * Quality (PSNR/SSIM against a reference) is intentionally out of scope here:
 * it belongs to the model benchmark harness, which needs real weights.
 */
object ExmBenchmark {

    data class Config(
        val label: String,
        val chunkBytes: Long,
        val tensors: List<ExmChunked.TensorSpec> = emptyList()
    )

    data class Report(
        val label: String,
        val chunkBytes: Long,
        val originalBytes: Long,
        val storedBytes: Long,
        val chunks: Int,
        val ratio: Double,
        val writeMs: Double,
        val headerMs: Double,
        val assembleMs: Double,
        val integrity: Boolean
    )

    /** Default sweep: small (RAM-tight devices), medium, large chunks. */
    fun defaultConfigs(): List<Config> = listOf(
        Config("256k-chunks", 256L * 1024L),
        Config("1m-chunks", 1024L * 1024L),
        Config("4m-chunks", ExmChunked.DEFAULT_CHUNK_BYTES)
    )

    fun run(
        payload: ByteArray,
        configs: List<Config> = defaultConfigs(),
        quantization: ExmFormat.Quantization = ExmFormat.Quantization.FP16
    ): List<Report> = configs.map { runOne(payload, it, quantization) }

    fun runOne(payload: ByteArray, config: Config, quantization: ExmFormat.Quantization): Report {
        require(payload.size.toLong() <= ExmFormat.MAX_PAYLOAD_BYTES)
        val manifest = ExmFormat.Manifest(
            id = "bench-model",
            version = "0.0.0",
            name = "benchmark",
            quantization = quantization,
            backends = setOf(ExmFormat.Backend.CPU),
            sha256 = ExmReader.sha256(payload),
            sizeBytes = payload.size.toLong(),
            license = "benchmark-only"
        )

        val container = File.createTempFile("exm-bench", ".exm")
        container.deleteOnExit()
        try {
            val (writeMs, _) = timedMs {
                val bytes = ByteArrayOutputStream()
                    .also { ExmChunked.write(it, manifest, config.tensors, payload, config.chunkBytes) }
                    .toByteArray()
                container.writeBytes(bytes)
            }

            val storedBytes = container.length()
            val (headerMs, chunkCount) = timedMs {
                RandomAccessFile(container, "r").use { raf ->
                    val reader = ExmChunked.open(raf.channel.position(0))
                    val count = reader.chunkCount
                    reader.close()
                    count
                }
            }

            val (assembleMs, integrity) = timedMs {
                RandomAccessFile(container, "r").use { raf ->
                    val reader = ExmChunked.open(raf.channel.position(0))
                    val assembled = reader.assemble()
                    reader.close()
                    assembled.contentEquals(payload)
                }
            }

            return Report(
                label = config.label,
                chunkBytes = config.chunkBytes,
                originalBytes = payload.size.toLong(),
                storedBytes = storedBytes,
                chunks = chunkCount,
                ratio = if (payload.isEmpty()) 0.0 else storedBytes.toDouble() / payload.size,
                writeMs = writeMs,
                headerMs = headerMs,
                assembleMs = assembleMs,
                integrity = integrity
            )
        } finally {
            container.delete()
        }
    }

    /** Renders an aligned comparison table for CI logs and docs. */
    fun render(reports: List<Report>): String {
        val header = String.format(
            "%-14s %10s %12s %12s %8s %8s %10s %10s %10s %9s",
            "chunks", "chunkSize", "original", "stored", "ratio", "chunks#", "write ms", "header ms", "assemble", "integrity"
        )
        val rows = reports.map {
            String.format(
                "%-14s %10s %12d %12d %8.3f %8d %10.1f %10.1f %10.1f %9s",
                it.label, humanBytes(it.chunkBytes), it.originalBytes, it.storedBytes,
                it.ratio, it.chunks, it.writeMs, it.headerMs, it.assembleMs,
                if (it.integrity) "ok" else "FAIL"
            )
        }
        return (listOf(header) + rows).joinToString("\n")
    }

    private fun humanBytes(bytes: Long): String = when {
        bytes >= 1024L * 1024L -> "${bytes / (1024L * 1024L)}m"
        bytes >= 1024L -> "${bytes / 1024L}k"
        else -> "$bytes"
    }

    private inline fun <T> timedMs(block: () -> T): Pair<Double, T> {
        val start = System.nanoTime()
        val value = block()
        return (System.nanoTime() - start) / 1_000_000.0 to value
    }
}
