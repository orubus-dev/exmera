package studio.exmera.exm

import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import java.io.EOFException
import java.io.File
import java.io.RandomAccessFile
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class ExmChunkedTest {

    private fun options() = ExmCompileOptions(license = "Apache-2.0")

    private fun pattern(size: Int, seed: Int = 7): ByteArray =
        ByteArray(size) { i -> ((i * 31 + seed * 17 + (i % 7)) % 251).toByte() }

    private fun <T> withTempContainer(bytes: ByteArray, block: (ExmChunked.Reader) -> T): T {
        val f = File.createTempFile("exm-test", ".exm")
        f.deleteOnExit()
        f.writeBytes(bytes)
        val raf = RandomAccessFile(f, "r")
        val reader = ExmChunked.open(raf.channel.position(0))
        try {
            return block(reader)
        } finally {
            reader.close()
            f.delete()
        }
    }

    @Test
    fun anonymousRoundTripSingleChunk() {
        val payload = pattern(10_000)
        val container = ExmPackCompiler().compileChunked("m", "1.0", "M", payload, options())
        withTempContainer(container) { reader ->
            assertEquals(1, reader.chunkCount)
            assertEquals(payload.size.toLong(), reader.header.manifest.sizeBytes)
            assertEquals("m", reader.header.manifest.id)
            assertContentEquals(payload, reader.assemble())
        }
    }

    @Test
    fun declaredTensorsChunkIndependentlyAndStreamByTensor() {
        val payload = pattern(3_000)
        val tensors = listOf(
            ExmChunked.TensorSpec("w1", "int8", intArrayOf(1_000)),
            ExmChunked.TensorSpec("w2", "fp16", intArrayOf(500)),
            ExmChunked.TensorSpec("w3", "bytes", intArrayOf(1_000))
        )
        val container = ExmPackCompiler().compileChunked(
            "m", "1.0", "M", payload, options(), tensors, chunkBytes = 400L
        )
        withTempContainer(container) { reader ->
            // each tensor spans ceil(1000/400)=3 chunks; 9 chunks total
            assertEquals(9, reader.chunkCount)
            assertEquals(3, reader.header.tensors.size)
            assertEquals(0, reader.header.tensors[0].chunkStart)
            assertEquals(3, reader.header.tensors[1].chunkStart)

            val w2 = reader.readTensor("w2").reduce { a, b -> a + b }
            assertContentEquals(payload.copyOfRange(1_000, 2_000), w2)

            val w3 = reader.readTensor("w3").reduce { a, b -> a + b }
            assertContentEquals(payload.copyOfRange(2_000, 3_000), w3)

            assertContentEquals(payload, reader.assemble())
        }
    }

    @Test
    fun writerIsDeterministic() {
        val payload = pattern(50_000, seed = 3)
        val tensors = listOf(
            ExmChunked.TensorSpec("a", "int8", intArrayOf(30_000)),
            ExmChunked.TensorSpec("b", "int8", intArrayOf(20_000))
        )
        val c1 = ExmPackCompiler().compileChunked("m", "1.0", "M", payload, options(), tensors, chunkBytes = 8_192L)
        val c2 = ExmPackCompiler().compileChunked("m", "1.0", "M", payload, options(), tensors, chunkBytes = 8_192L)
        assertContentEquals(c1, c2)
    }

    @Test
    fun corruptChunkByteFailsIntegrityOnRead() {
        val payload = pattern(5_000)
        val container = ExmPackCompiler().compileChunked("m", "1.0", "M", payload, options())
        // flip one byte in the data region (last 100 bytes of the file)
        val corrupted = container.copyOf()
        corrupted[corrupted.size - 10] = (corrupted[corrupted.size - 10].toInt() xor 0x55).toByte()
        withTempContainer(corrupted) { reader ->
            // header parses fine (indexes are intact) — the digest catches the data damage
            assertEquals(1, reader.chunkCount)
            val e = assertFailsWith<IllegalStateException> { reader.readChunk(0) }
            assertTrue(e.message!!.contains("integrity"))
            val e2 = assertFailsWith<IllegalStateException> { reader.assemble() }
            assertTrue(e2.message!!.contains("integrity"))
        }
    }

    @Test
    fun truncatedContainerThrowsEofOnChunkRead() {
        val payload = pattern(50_000)
        val container = ExmPackCompiler().compileChunked("m", "1.0", "M", payload, options(), chunkBytes = 8_192L)
        val truncated = container.copyOfRange(0, container.size - 500)
        withTempContainer(truncated) { reader ->
            // header is intact (it sits before the data), the last chunk is cut short
            assertFailsWith<EOFException> { reader.assemble() }
        }
    }

    @Test
    fun emptyPayloadProducesValidContainer() {
        val container = ExmPackCompiler().compileChunked("m", "1.0", "M", ByteArray(0), options())
        withTempContainer(container) { reader ->
            assertEquals(1, reader.chunkCount)
            assertEquals(0L, reader.header.manifest.sizeBytes)
            assertContentEquals(ByteArray(0), reader.assemble())
        }
    }

    @Test
    fun chunkAlignedBoundariesRoundTrip() {
        // payload exactly == 2 × chunkBytes with two aligned tensors → 1 chunk each
        val payload = pattern(2_048)
        val tensors = listOf(
            ExmChunked.TensorSpec("x", "int8", intArrayOf(1_024)),
            ExmChunked.TensorSpec("y", "int8", intArrayOf(1_024))
        )
        val container = ExmPackCompiler().compileChunked("m", "1.0", "M", payload, options(), tensors, chunkBytes = 1_024L)
        withTempContainer(container) { reader ->
            assertEquals(2, reader.chunkCount)
            assertContentEquals(payload.copyOfRange(0, 1_024), reader.readTensor("x").single())
            assertContentEquals(payload.copyOfRange(1_024, 2_048), reader.readTensor("y").single())
        }
    }

    @Test
    fun tensorCoverageMismatchIsRejected() {
        val payload = pattern(1_000)
        val tensors = listOf(ExmChunked.TensorSpec("w", "int8", intArrayOf(400))) // covers 400 of 1000
        val e = assertFailsWith<IllegalArgumentException> {
            ExmPackCompiler().compileChunked("m", "1.0", "M", payload, options(), tensors)
        }
        assertTrue(e.message!!.contains("coverage"))
    }

    @Test
    fun unsupportedDtypeIsRejected() {
        val spec = ExmChunked.TensorSpec("w", "q9", intArrayOf(4))
        val e = assertFailsWith<IllegalArgumentException> { ExmChunked.tensorBytes(spec) }
        assertTrue(e.message!!.contains("dtype"))
    }

    @Test
    fun manifestDigestIsEnforcedAtWrite() {
        val payload = pattern(1_000)
        val badManifest = ExmFormat.Manifest(
            id = "m", version = "1.0", name = "M",
            quantization = ExmFormat.Quantization.INT8,
            backends = setOf(ExmFormat.Backend.CPU),
            sha256 = "deadbeef", sizeBytes = payload.size.toLong()
        )
        assertFailsWith<IllegalArgumentException> {
            ExmChunked.write(ByteArrayOutputStream(), badManifest, emptyList(), payload)
        }
    }

    @Test
    fun legacyV2ContainerStillRoundTrips() {
        val payload = pattern(2_000)
        val manifest = ExmFormat.Manifest(
            id = "legacy", version = "1.0", name = "L",
            quantization = ExmFormat.Quantization.FP32,
            backends = setOf(ExmFormat.Backend.CPU),
            sha256 = ExmReader.sha256(payload), sizeBytes = payload.size.toLong()
        )
        val bytes = ByteArrayOutputStream().also { ExmContainer.write(it, manifest, payload) }.toByteArray()
        val (readManifest, readPayload) = ExmContainer.read(ByteArrayInputStream(bytes))
        assertEquals("legacy", readManifest.id)
        assertContentEquals(payload, readPayload)
    }
}
