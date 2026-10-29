package studio.exmera.exm

/**
 * Compiler boundary for future ONNX/TFLite/PyTorch conversion.
 * Phase 3-5 intentionally keeps model transformation separate from the runtime.
 */
data class ExmCompileOptions(
    val quantization: ExmFormat.Quantization = ExmFormat.Quantization.INT8,
    val backends: Set<ExmFormat.Backend> = setOf(ExmFormat.Backend.CPU),
    val minRamMb: Int = 2048,
    val minAndroid: Int = 26,
    val license: String = "UNKNOWN"
)

class ExmPackCompiler {
    fun compile(
        id: String,
        version: String,
        name: String,
        sourcePayload: ByteArray,
        options: ExmCompileOptions
    ): ByteArray {
        // This is a packaging compiler today. Quantization/pruning passes plug in here later.
        val manifest = ExmFormat.Manifest(
            id = id, version = version, name = name,
            quantization = options.quantization, backends = options.backends,
            sha256 = ExmReader.sha256(sourcePayload), sizeBytes = sourcePayload.size.toLong(),
            minRamMb = options.minRamMb, minAndroid = options.minAndroid, license = options.license
        )
        return java.io.ByteArrayOutputStream().also { ExmContainer.write(it, manifest, sourcePayload) }.toByteArray()
    }

    /**
     * Canonical path: chunked v3 container. Tensors, when declared, must exactly
     * cover the payload; otherwise the payload is stored as one anonymous tensor.
     * Chunks never straddle tensors, per-chunk digests are verified on read, and
     * the runtime can stream any tensor without loading the rest of the model.
     */
    fun compileChunked(
        id: String,
        version: String,
        name: String,
        sourcePayload: ByteArray,
        options: ExmCompileOptions,
        tensors: List<ExmChunked.TensorSpec> = emptyList(),
        chunkBytes: Long = ExmChunked.DEFAULT_CHUNK_BYTES
    ): ByteArray {
        val manifest = ExmFormat.Manifest(
            id = id, version = version, name = name,
            quantization = options.quantization, backends = options.backends,
            sha256 = ExmReader.sha256(sourcePayload), sizeBytes = sourcePayload.size.toLong(),
            minRamMb = options.minRamMb, minAndroid = options.minAndroid, license = options.license,
            compression = ExmFormat.Compression.NONE,
            originalSizeBytes = sourcePayload.size.toLong()
        )
        return java.io.ByteArrayOutputStream()
            .also { ExmChunked.write(it, manifest, tensors, sourcePayload, chunkBytes) }
            .toByteArray()
    }
}
