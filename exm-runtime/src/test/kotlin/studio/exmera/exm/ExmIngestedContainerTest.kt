package studio.exmera.exm

import java.io.File
import java.io.RandomAccessFile
import java.security.MessageDigest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * Cross-language contract test: this container was produced by
 * tools/onnx_to_exm.py (Python ONNX ingestion) from the ONNX Model Zoo's
 * super-resolution-10 (sub-pixel CNN 2016, MIT). The Kotlin runtime must be
 * able to open it header-only, stream every tensor, verify integrity, and
 * rebuild it byte-for-byte — proving Python and Kotlin writers encode the
 * exact same deterministic container.
 */
class ExmIngestedContainerTest {

    private fun openFixture(): Pair<ExmChunked.Reader, ByteArray> {
        val bytes = javaClass.classLoader
            .getResourceAsStream("ingested/super-resolution-10.exm")
            ?.readBytes()
            ?: error("fixture ingested/super-resolution-10.exm missing from test resources")
        val tmp = File.createTempFile("ingested", ".exm").apply {
            deleteOnExit(); writeBytes(bytes)
        }
        val raf = RandomAccessFile(tmp, "r")
        return ExmChunked.open(raf.channel.position(0)) to bytes
    }

    private inline fun <T> withReader(block: (ExmChunked.Reader) -> T): T {
        val (reader, _) = openFixture()
        try {
            return block(reader)
        } finally {
            reader.close()
        }
    }

    @Test
    fun `header-only open exposes the ingested manifest`() {
        withReader {
            val m = it.header.manifest
            assertEquals("pipeline/super-resolution-10", m.id)
            assertEquals("1.0.0", m.version)
            assertEquals("Super Resolution (sub-pixel CNN 2016)", m.name)
            assertEquals("MIT", m.license)
            assertEquals(ExmFormat.Quantization.FP32, m.quantization)
            assertEquals(setOf(ExmFormat.Backend.CPU), m.backends)
            assertEquals(ExmFormat.Compression.NONE, m.compression)
            assertEquals(241_467L, m.sizeBytes)
            assertEquals(m.sizeBytes, m.originalSizeBytes)
            assertEquals("4f0975568d6f47014435e70b458d04470eac2513e695b0bd6eaffdae2bc29c71", m.sha256)
            assertEquals(11, it.header.tensors.size)
            assertEquals(11, it.chunkCount)
            assertEquals(ExmChunked.DEFAULT_CHUNK_BYTES, it.header.chunkSize)
        }
    }

    @Test
    fun `graph tensor is present as reserved metadata`() {
        withReader {
            val graph = it.header.tensors.firstOrNull { t -> t.name == "@graph" }
            assertTrue(graph != null, "@graph tensor missing")
            assertEquals("bytes", graph!!.dtype)
            assertEquals(listOf(2_759), graph.shape)
            val blob = it.readTensor("@graph").single()
            assertEquals(2_759, blob.size)
            // deterministic JSON, self-describing
            val json = String(blob, Charsets.UTF_8)
            assertTrue(json.contains("\"format\":\"exm-graph\""))
            assertTrue(json.contains("\"byteOrder\":\"little\""))
            assertTrue(json.contains("\"op\":\"Conv\""))
        }
    }

    @Test
    fun `every tensor streams independently with exact byte totals`() {
        withReader {
            var total = 0L
            for (t in it.header.tensors) {
                val parts = it.readTensor(t.name)
                val bytes = parts.sumOf { p -> p.size.toLong() }
                val expected = ExmChunked.tensorBytes(
                    ExmChunked.TensorSpec(t.name, t.dtype, t.shape.toIntArray())
                )
                assertEquals(expected, bytes, "tensor ${t.name} streamed size")
                total += bytes
            }
            assertEquals(241_467L, total)
        }
    }

    @Test
    fun `assemble verifies chunk digests and manifest digest`() {
        withReader {
            val payload = it.assemble()
            assertEquals(241_467, payload.size)
            val digest = MessageDigest.getInstance("SHA-256").digest(payload)
                .joinToString("") { b -> "%02x".format(b) }
            assertEquals(it.header.manifest.sha256, digest)
        }
    }

    @Test
    fun `kotlin writer rebuilds the python container byte-for-byte`() {
        withReader { reader ->
            val payload = reader.assemble()
            val specs = reader.header.tensors.map { t ->
                ExmChunked.TensorSpec(t.name, t.dtype, t.shape.toIntArray(), t.quantization)
            }
            val rebuilt = java.io.ByteArrayOutputStream()
                .also { out ->
                    ExmChunked.write(out, reader.header.manifest, specs, payload, reader.header.chunkSize)
                }
                .toByteArray()
            val (_, original) = openFixture()
            assertTrue(rebuilt.contentEquals(original), "Kotlin rebuild differs from Python container")
        }
    }

    @Test
    fun `unknown tensor names are rejected on the streaming path`() {
        withReader {
            assertFailsWith<IllegalArgumentException> { it.readTensor("nope") }
        }
    }
}
