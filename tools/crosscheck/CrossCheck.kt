import java.io.ByteArrayOutputStream
import java.io.File
import java.io.RandomAccessFile
import studio.exmera.exm.ExmChunked
import studio.exmera.exm.ExmFormat

/**
 * Cross-language contract check: containers written by tools/onnx_to_exm.py
 * (Python) must be readable AND byte-reproducible by the Kotlin runtime.
 *
 * Modes:
 *   read <container.exm>            — header-only open, tensor streaming, full assemble
 *   roundtrip <container.exm>       — read + rebuild with ExmChunked.write, byte-compare
 */
fun main(args: Array<String>) {
    val mode = args.getOrElse(0) { error("usage: read|roundtrip <container.exm>") }
    val path = File(args.getOrElse(1) { error("container path required") })
    require(path.isFile) { "no such file: $path" }

    RandomAccessFile(path, "r").use { raf ->
        val t0 = System.nanoTime()
        val reader = ExmChunked.open(raf.channel.position(0))
        val openMs = (System.nanoTime() - t0) / 1_000_000.0
        val m = reader.header.manifest
        println("file=$path")
        println("headerOnlyOpenMs=$openMs")
        println("id=${m.id} version=${m.version} name=${m.name}")
        println("quantization=${m.quantization} backends=${m.backends} compression=${m.compression}")
        println("sha256=${m.sha256} sizeBytes=${m.sizeBytes} originalSizeBytes=${m.originalSizeBytes}")
        println("minRam=${m.minRamMb} minAndroid=${m.minAndroid} license=${m.license}")
        println("tensors=${reader.header.tensors.size} chunks=${reader.chunkCount} chunkBytes=${reader.header.chunkSize}")
        val graph = reader.header.tensors.firstOrNull { it.name == "@graph" }
        println("graphTensor=" + (graph?.let { "${it.dtype} ${it.shape} chunks=${it.chunkCount}" } ?: "MISSING"))

        // per-tensor streaming: read every tensor independently, check byte totals
        var streamed = 0L
        val t1 = System.nanoTime()
        for (t in reader.header.tensors) {
            val parts = reader.readTensor(t.name)
            val bytes = parts.sumOf { it.size.toLong() }
            val expected = ExmChunked.tensorBytes(
                ExmChunked.TensorSpec(t.name, t.dtype, t.shape.toIntArray())
            )
            check(bytes == expected) { "tensor ${t.name}: streamed $bytes != $expected" }
            streamed += bytes
        }
        val streamMs = (System.nanoTime() - t1) / 1_000_000.0
        println("streamedAllTensorsBytes=$streamed in ${streamMs}ms")

        // full assemble + manifest digest
        val t2 = System.nanoTime()
        val payload = reader.assemble()
        val assembleMs = (System.nanoTime() - t2) / 1_000_000.0
        println("assemble=${payload.size}B in ${assembleMs}ms integrity=ok")

        if (mode == "roundtrip") {
            val specs = reader.header.tensors.map {
                ExmChunked.TensorSpec(it.name, it.dtype, it.shape.toIntArray(), it.quantization)
            }
            val t3 = System.nanoTime()
            val rebuilt = ByteArrayOutputStream()
                .also { ExmChunked.write(it, m, specs, payload, reader.header.chunkSize) }
                .toByteArray()
            val writeMs = (System.nanoTime() - t3) / 1_000_000.0
            val identical = rebuilt.contentEquals(path.readBytes())
            println("kotlinRewriteBytes=${rebuilt.size} in ${writeMs}ms")
            println("crossLanguageByteIdentical=$identical")
            check(identical) { "Kotlin rebuild differs from Python container" }
        }
        reader.close()
        println("RESULT=PASS")
    }
}
