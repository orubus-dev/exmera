package studio.exmera.exm

import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import java.io.DataInputStream
import java.io.DataOutputStream
import java.io.EOFException
import java.io.InputStream
import java.io.OutputStream
import java.nio.ByteBuffer
import java.nio.channels.SeekableByteChannel
import java.security.MessageDigest
import java.util.zip.Deflater
import java.util.zip.DeflaterOutputStream
import java.util.zip.InflaterInputStream
import kotlin.math.ceil

/**
 * EXM chunked container (format v3) — the canonical encoder/decoder.
 *
 * A model is stored as an ordered list of independently compressed chunks,
 * each carrying its own size + SHA-256, preceded by a manifest, a tensor
 * index and a chunk index. The runtime can therefore:
 *   - open a container and read only the header (no weight I/O),
 *   - stream exactly the chunks an execution plan needs,
 *   - verify integrity per chunk instead of trusting the file as a whole,
 *   - assemble the original payload and check the manifest digest.
 *
 * Layout (all integers big-endian):
 *   magic  "EXM1" | version i32 = 3 | flags i32 (bit0 = chunked)
 *   manifest  (same field order as v2, see ExmContainer)
 *   chunkSize i64 (max original bytes per chunk; last chunk may be smaller)
 *   tensorCount i32; per tensor:
 *       name str | dtype str | shapeCount i32 + shape i32* |
 *       quantization i32 (-1 = inherit manifest) | chunkStart i32 | chunkCount i32
 *   chunkCount i32; per chunk:
 *       storedSize i64 | originalSize i64 | codec i32 | sha256 (64 ASCII bytes)
 *   chunk payloads in index order (storedSize bytes each)
 *
 * Deterministic: identical manifest/tensors/payload/chunkBytes produce
 * byte-identical containers.
 */
object ExmChunked {

    const val VERSION = 3
    const val FLAG_CHUNKED = 1
    const val DEFAULT_CHUNK_BYTES: Long = 4L * 1024L * 1024L
    private const val MAX_CHUNKS = 10_000_000

    /** Declared tensor metadata. Byte size = dtype × shape; ranges are resolved by the writer. */
    data class TensorSpec(
        val name: String,
        val dtype: String,
        val shape: IntArray,
        val quantization: ExmFormat.Quantization? = null
    )

    data class TensorEntry(
        val name: String,
        val dtype: String,
        val shape: List<Int>,
        val quantization: ExmFormat.Quantization?,
        val chunkStart: Int,
        val chunkCount: Int
    )

    data class ChunkEntry(
        val storedSize: Long,
        val originalSize: Long,
        val codec: ExmFormat.Compression,
        val sha256: String
    )

    data class ChunkedManifest(
        val manifest: ExmFormat.Manifest,
        val chunkSize: Long,
        val tensors: List<TensorEntry>,
        val chunks: List<ChunkEntry>
    ) {
        val chunkCount: Int get() = chunks.size
    }

    /* ---------------- writing (packaging path) ---------------- */

    /**
     * Encodes a chunked v3 container. Compresses with DEFLATE(BEST_SPEED) per chunk —
     * chosen for mobile-first packaging where the compressor may run on-device.
     */
    fun write(
        output: OutputStream,
        manifest: ExmFormat.Manifest,
        tensors: List<TensorSpec>,
        payload: ByteArray,
        chunkBytes: Long = DEFAULT_CHUNK_BYTES
    ): ChunkedManifest {
        require(manifest.id.isNotBlank()) { "manifest id must not be blank" }
        require(payload.size.toLong() <= ExmFormat.MAX_PAYLOAD_BYTES) { "payload exceeds EXM limit" }
        require(chunkBytes > 0) { "chunkBytes must be positive" }
        require(manifest.sha256 == ExmReader.sha256(payload)) { "manifest sha256 does not describe the payload" }
        require(manifest.compression == ExmFormat.Compression.NONE) { "v3 compresses per chunk; manifest compression must be NONE" }

        // 1. Resolve declared tensors onto contiguous byte ranges.
        val ranges = resolveRanges(tensors, payload.size.toLong())

        // 2. Chunk each tensor's byte range independently — chunks never straddle
        //    tensors, so the runtime can stream one tensor without touching others.
        val chunkEntries = ArrayList<ChunkEntry>()
        val storedChunks = ArrayList<ByteArray>()
        for (r in ranges) {
            var offset = r.start
            val end = r.start + r.size
            while (offset < end) {
                val stop = minOf(offset + chunkBytes, end)
                val original = payload.copyOfRange(offset.toInt(), stop.toInt())
                val stored = deflate(original)
                chunkEntries += ChunkEntry(stored.size.toLong(), original.size.toLong(), ExmFormat.Compression.DEFLATE, sha256(stored))
                storedChunks += stored
                offset = stop
            }
        }
        if (chunkEntries.isEmpty()) { // empty payload: one empty chunk keeps chunkCount >= 1
            val stored = deflate(ByteArray(0))
            chunkEntries += ChunkEntry(stored.size.toLong(), 0L, ExmFormat.Compression.DEFLATE, sha256(stored))
            storedChunks += stored
        }

        // 3. Map each tensor onto the chunk grid (spans were produced per tensor above).
        val entries = ArrayList<TensorEntry>(ranges.size)
        var chunkCursor = 0
        for (r in ranges) {
            val spans = ceil(r.size.toDouble() / chunkBytes).toInt().coerceAtLeast(1)
            entries += TensorEntry(
                r.spec.name, r.spec.dtype, r.spec.shape.toList(), r.spec.quantization,
                chunkCursor, spans
            )
            chunkCursor += spans
        }
        if (entries.isEmpty() && chunkEntries.isNotEmpty()) {
            entries += TensorEntry("", "bytes", emptyList(), null, 0, chunkEntries.size)
        }
        check(chunkCursor == chunkEntries.size || entries.size == 1) {
            "internal chunk alignment mismatch: tensors span $chunkCursor chunks, payload produced ${chunkEntries.size}"
        }

        // 4. Serialize — every index field is known up front, so output is deterministic.
        DataOutputStream(output).use { out ->
            out.writeBytes(ExmFormat.MAGIC)
            out.writeInt(VERSION)
            out.writeInt(FLAG_CHUNKED)
            writeManifest(out, manifest)
            out.writeLong(chunkBytes)

            out.writeInt(entries.size)
            for (t in entries) {
                writeString(out, t.name)
                writeString(out, t.dtype)
                out.writeInt(t.shape.size)
                for (d in t.shape) out.writeInt(d)
                out.writeInt(t.quantization?.ordinal ?: -1)
                out.writeInt(t.chunkStart)
                out.writeInt(t.chunkCount)
            }

            out.writeInt(chunkEntries.size)
            for (c in chunkEntries) {
                out.writeLong(c.storedSize)
                out.writeLong(c.originalSize)
                out.writeInt(c.codec.ordinal)
                writeHex64(out, c.sha256)
            }

            for (s in storedChunks) out.write(s)
        }
        return ChunkedManifest(manifest, chunkBytes, entries, chunkEntries)
    }

    /** Byte range each declared tensor occupies inside the payload, in declaration order. */
    private data class ByteRange(val spec: TensorSpec, val start: Long, val size: Long)

    private fun resolveRanges(tensors: List<TensorSpec>, payloadSize: Long): List<ByteRange> {
        if (tensors.isEmpty()) {
            // Anonymous whole-payload tensor keeps the container valid while no
            // model-graph parser exists yet (ONNX ingestion is a later phase).
            return if (payloadSize > 0L) listOf(ByteRange(TensorSpec("", "bytes", IntArray(0)), 0L, payloadSize)) else emptyList()
        }
        val ranges = ArrayList<ByteRange>(tensors.size)
        var cursor = 0L
        for (t in tensors) {
            require(t.name.isNotBlank()) { "tensor name must not be blank" }
            val bytes = tensorBytes(t)
            require(bytes > 0) { "tensor ${t.name}: byte size must be positive" }
            require(cursor + bytes <= payloadSize) { "tensor ${t.name}: range [$cursor,${cursor + bytes}) exceeds payload $payloadSize" }
            ranges += ByteRange(t, cursor, bytes)
            cursor += bytes
        }
        require(cursor == payloadSize) { "tensor coverage $cursor != payload $payloadSize" }
        return ranges
    }

    /* ---------------- reading (runtime path) ---------------- */

    /** Opens a container and reads only header + indexes. No chunk bytes are touched. */
    fun open(channel: SeekableByteChannel): Reader = Reader.create(channel)

    class Reader private constructor(
        private val channel: SeekableByteChannel,
        val header: ChunkedManifest,
        private val dataStart: Long
    ) {
        companion object {
            internal fun create(channel: SeekableByteChannel): Reader {
                channel.position(0)
                val stream = ChunkChannelStream(channel, channel.size())
                val header = readHeader(DataInputStream(stream))
                return Reader(channel, header, stream.consumed())
            }
        }

        val chunkCount: Int get() = header.chunkCount

        private val chunkOffsets: List<Long> = buildList(header.chunkCount) {
            var acc = dataStart
            for (c in header.chunks) {
                add(acc)
                acc += c.storedSize
            }
        }

        fun readChunkStored(index: Int): ByteArray {
            header.chunks.getOrNull(index) ?: throw IndexOutOfBoundsException("chunk $index of ${header.chunkCount}")
            val c = header.chunks[index]
            return readAt(chunkOffsets[index], c.storedSize)
        }

        fun readChunk(index: Int): ByteArray {
            val c = header.chunks.getOrNull(index) ?: throw IndexOutOfBoundsException("chunk $index of ${header.chunkCount}")
            val stored = readChunkStored(index)
            require(sha256(stored) == c.sha256) { "chunk $index integrity check failed" }
            val original = when (c.codec) {
                ExmFormat.Compression.DEFLATE -> inflate(stored, c.originalSize)
                ExmFormat.Compression.NONE -> stored
            }
            require(original.size.toLong() == c.originalSize) { "chunk $index size mismatch" }
            return original
        }

        /** Assembles the full original payload, verifying every chunk and the manifest digest. */
        fun assemble(): ByteArray {
            val out = ByteArrayOutputStream(header.manifest.sizeBytes.coerceAtMost(Int.MAX_VALUE.toLong()).toInt())
            for (i in 0 until chunkCount) out.write(readChunk(i))
            val payload = out.toByteArray()
            require(payload.size.toLong() == header.manifest.sizeBytes) { "assembled size mismatch" }
            require(sha256(payload) == header.manifest.sha256) { "manifest integrity check failed" }
            return payload
        }

        /** Reads only the chunks of one tensor, in order — the streaming execution path. */
        fun readTensor(name: String): List<ByteArray> {
            val t = header.tensors.firstOrNull { it.name == name }
                ?: throw IllegalArgumentException("unknown tensor $name")
            return (t.chunkStart until t.chunkStart + t.chunkCount).map { readChunk(it) }
        }

        fun close() = channel.close()

        private fun readAt(position: Long, size: Long): ByteArray {
            require(size in 0..ExmFormat.MAX_PAYLOAD_BYTES) { "invalid read size" }
            val buffer = ByteBuffer.allocate(size.toInt())
            var read = 0L
            channel.position(position)
            while (read < size) {
                val n = channel.read(buffer)
                if (n < 0) throw EOFException("container truncated at $position (+$read/$size)")
                read += n
            }
            return buffer.array()
        }
    }

    /* ---------------- header parsing ---------------- */

    private fun readHeader(input: DataInputStream): ChunkedManifest {
        val magic = ByteArray(4); input.readFully(magic)
        require(String(magic, Charsets.US_ASCII) == ExmFormat.MAGIC) { "Invalid EXM magic" }
        val version = input.readInt()
        require(version == VERSION) { "Unsupported EXM version $version (expected $VERSION)" }
        val flags = input.readInt()
        require(flags and FLAG_CHUNKED != 0) { "not a chunked EXM container" }

        val manifest = readManifest(input)
        val chunkSize = input.readLong()
        require(chunkSize > 0) { "invalid chunkSize" }

        val tensorCount = input.readInt()
        require(tensorCount in 0..1_000_000) { "invalid tensor count" }
        val tensors = ArrayList<TensorEntry>(tensorCount)
        repeat(tensorCount) {
            val name = readString(input)
            val dtype = readString(input)
            val dims = input.readInt()
            require(dims in 0..64) { "invalid tensor rank" }
            val shape = ArrayList<Int>(dims)
            repeat(dims) { shape.add(input.readInt()) }
            val q = input.readInt()
            val quant = if (q < 0) null else ExmFormat.Quantization.entries.getOrNull(q)
                ?: throw IllegalStateException("invalid tensor quantization")
            val chunkStart = input.readInt()
            val chunkCount = input.readInt()
            tensors += TensorEntry(name, dtype, shape, quant, chunkStart, chunkCount)
        }

        val chunkCount = input.readInt()
        require(chunkCount in 1..MAX_CHUNKS) { "invalid chunk count" }
        val chunks = ArrayList<ChunkEntry>(chunkCount)
        repeat(chunkCount) {
            val stored = input.readLong()
            val original = input.readLong()
            val codec = ExmFormat.Compression.entries.getOrNull(input.readInt())
                ?: throw IllegalStateException("invalid chunk codec")
            val hex = ByteArray(64); input.readFully(hex)
            chunks += ChunkEntry(stored, original, codec, String(hex, Charsets.US_ASCII))
        }

        // Structural sanity before any chunk byte is read.
        var totalOriginal = 0L
        var totalStored = 0L
        for (c in chunks) {
            require(c.storedSize >= 0 && c.originalSize >= 0) { "negative chunk size" }
            require(c.storedSize <= ExmFormat.MAX_PAYLOAD_BYTES && c.originalSize <= ExmFormat.MAX_PAYLOAD_BYTES) { "chunk exceeds format limit" }
            require(c.sha256.length == 64 && c.sha256.all { it.isDigit() || it in 'a'..'f' }) { "malformed chunk digest" }
            totalOriginal += c.originalSize
            totalStored += c.storedSize
        }
        require(totalOriginal == manifest.sizeBytes) { "chunk index does not cover manifest size" }
        require(totalStored <= ExmFormat.MAX_PAYLOAD_BYTES) { "stored payload exceeds format limit" }
        validateTensorChunkRanges(tensors, chunks, chunkSize)
        return ChunkedManifest(manifest, chunkSize, tensors, chunks)
    }

    /** Validates tensor chunk ranges against dtype×shape byte sizes and the chunk grid. */
    private fun validateTensorChunkRanges(tensors: List<TensorEntry>, chunks: List<ChunkEntry>, chunkSize: Long) {
        if (tensors.isEmpty()) return
        if (tensors.size == 1 && tensors[0].name.isEmpty()) {
            require(tensors[0].chunkStart == 0 && tensors[0].chunkCount == chunks.size) { "anonymous tensor must own every chunk" }
            return
        }
        var expectedStart = 0
        for (t in tensors) {
            require(t.chunkStart == expectedStart) { "tensor ${t.name}: non-contiguous chunk ranges" }
            val bytes = tensorBytes(TensorSpec(t.name, t.dtype, t.shape.toIntArray(), t.quantization))
            val spans = ceil(bytes.toDouble() / chunkSize).toInt().coerceAtLeast(1)
            require(t.chunkCount == spans) { "tensor ${t.name}: chunk count ${t.chunkCount} != expected $spans" }
            var acc = 0L
            for (i in t.chunkStart until t.chunkStart + t.chunkCount) acc += chunks[i].originalSize
            require(acc == bytes) { "tensor ${t.name}: chunks cover $acc != $bytes" }
            expectedStart += t.chunkCount
        }
        require(expectedStart == chunks.size) { "tensor ranges do not cover all chunks" }
    }

    /* ---------------- shared codecs ---------------- */

    /** Byte size of a tensor from dtype × shape. */
    fun tensorBytes(t: TensorSpec): Long {
        val dtype = t.dtype.lowercase()
        val count = if (t.shape.isEmpty()) 1L else t.shape.fold(1L) { a, b -> a * b.toLong() }
        return when (dtype) {
            "fp32", "float32" -> count * 4
            "fp16", "float16" -> count * 2
            "int8" -> count
            "int4" -> ceil(count / 2.0).toLong()
            "bytes" -> count
            else -> throw IllegalArgumentException("unsupported dtype ${t.dtype}")
        }
    }

    internal fun writeManifest(out: DataOutputStream, m: ExmFormat.Manifest) {
        writeString(out, m.id)
        writeString(out, m.version)
        writeString(out, m.name)
        out.writeInt(m.quantization.ordinal)
        out.writeInt(m.backends.fold(0) { a, b -> a or (1 shl b.ordinal) })
        out.writeInt(m.compression.ordinal)
        writeString(out, m.sha256)
        out.writeLong(m.sizeBytes)
        out.writeLong(m.originalSizeBytes)
        out.writeInt(m.minRamMb)
        out.writeInt(m.minAndroid)
        writeString(out, m.license)
    }

    internal fun readManifest(input: DataInputStream): ExmFormat.Manifest {
        val id = readString(input); val version = readString(input); val name = readString(input)
        val q = ExmFormat.Quantization.entries.getOrNull(input.readInt()) ?: throw IllegalStateException("Invalid quantization")
        val mask = input.readInt()
        val backends = ExmFormat.Backend.entries.filter { mask and (1 shl it.ordinal) != 0 }.toSet()
        val compression = ExmFormat.Compression.entries.getOrNull(input.readInt()) ?: throw IllegalStateException("Invalid compression")
        val sha = readString(input)
        val storedSize = input.readLong()
        val originalSize = input.readLong()
        val minRam = input.readInt()
        val minAndroid = input.readInt()
        val license = readString(input)
        return ExmFormat.Manifest(id, version, name, q, backends, sha, storedSize, minRam, minAndroid, license, compression, originalSize)
    }

    private fun writeString(out: DataOutputStream, value: String) {
        val bytes = value.toByteArray(Charsets.UTF_8)
        require(bytes.size <= ExmFormat.MAX_STRING_BYTES) { "string too long" }
        out.writeInt(bytes.size)
        out.write(bytes)
    }

    private fun readString(input: DataInputStream): String {
        val size = input.readInt()
        require(size in 0..ExmFormat.MAX_STRING_BYTES) { "invalid string length" }
        val bytes = ByteArray(size); input.readFully(bytes)
        return String(bytes, Charsets.UTF_8)
    }

    private fun writeHex64(out: DataOutputStream, sha: String) {
        val b = sha.toByteArray(Charsets.US_ASCII)
        require(b.size == 64) { "expected 64-char hex digest" }
        out.write(b)
    }

    /* ---------------- chunk codec + hashing ---------------- */

    fun deflate(data: ByteArray): ByteArray {
        val out = ByteArrayOutputStream(data.size / 2)
        DeflaterOutputStream(out, Deflater(Deflater.BEST_SPEED, false)).use { it.write(data) }
        return out.toByteArray()
    }

    fun inflate(data: ByteArray, expected: Long): ByteArray {
        require(expected in 0..ExmFormat.MAX_PAYLOAD_BYTES)
        val out = ByteArrayOutputStream(expected.coerceAtMost(64L * 1024L * 1024L).toInt())
        InflaterInputStream(ByteArrayInputStream(data)).use { input ->
            val buffer = ByteArray(1 shl 16)
            while (true) {
                val n = input.read(buffer)
                if (n < 0) break
                out.write(buffer, 0, n)
            }
        }
        return out.toByteArray()
    }

    fun sha256(data: ByteArray): String =
        MessageDigest.getInstance("SHA-256").digest(data).joinToString("") { "%02x".format(it) }

    /**
     * Stream over a SeekableByteChannel so header parsing can use DataInputStream.
     * Tracks consumed bytes so the caller knows where the data region begins.
     */
    private class ChunkChannelStream(private val channel: SeekableByteChannel, private val size: Long) : InputStream() {
        private var pos: Long = 0

        fun consumed(): Long = pos

        override fun read(): Int {
            val single = ByteArray(1)
            return if (read(single, 0, 1) == 1) single[0].toInt() and 0xff else -1
        }

        override fun read(b: ByteArray, off: Int, len: Int): Int {
            if (len == 0) return 0
            if (pos >= size) return -1
            val capped = minOf(len.toLong(), size - pos).toInt()
            val buffer = ByteBuffer.allocate(capped)
            var got = 0L
            while (got < capped) {
                channel.position(pos + got)
                val n = channel.read(buffer)
                if (n < 0) break
                got += n
            }
            if (got <= 0L) return -1
            System.arraycopy(buffer.array(), 0, b, off, got.toInt())
            pos += got
            return got.toInt()
        }
    }
}
