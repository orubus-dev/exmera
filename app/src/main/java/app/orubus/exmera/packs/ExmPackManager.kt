package app.orubus.exmera.packs

import android.content.Context
import studio.exmera.exm.ExmChunked
import java.io.File
import java.io.FileInputStream
import java.io.InputStream

/**
 * Model packs — the EXM v3 runtime entry point on device.
 *
 * The client ships with ZERO models. Everything learned is delivered as a
 * .exm pack: imported through the system file picker, header-validated on
 * arrival (magic, manifest, tensor/chunk indexes), integrity-verified chunk
 * by chunk, then stored under filesDir/packs. The header-only open contract
 * means listing installed packs never touches weight bytes.
 */
class ExmPackManager(private val root: File) {

    init {
        if (!root.exists()) root.mkdirs()
    }

    data class PackInfo(
        val fileName: String,
        val id: String,
        val version: String,
        val name: String,
        val quantization: String,
        val backends: List<String>,
        val license: String,
        val sizeBytes: Long,
        val originalSizeBytes: Long,
        val minRamMb: Int,
        val minAndroid: Int,
        val tensorCount: Int,
        val chunkCount: Int,
        val chunkSize: Long,
        val hasGraph: Boolean,
        val integrityVerified: Boolean
    )

    /** Imports a .exm from [source]. Throws IllegalArgumentException with a readable reason. */
    fun importFrom(source: InputStream, suggestedName: String): PackInfo {
        val tmp = File(root, "incoming_${System.currentTimeMillis()}_$suggestedName")
        try {
            source.use { input -> tmp.outputStream().use { input.copyTo(it) } }
            val info = openInfo(tmp)
                ?: throw IllegalArgumentException("not a valid EXM container: $suggestedName")
            // full per-chunk integrity pass on arrival — trust nothing until proven
            var verified = true
            FileInputStream(tmp).channel.use { ch ->
                val reader = ExmChunked.open(ch)
                try {
                    for (i in 0 until reader.chunkCount) {
                        // readChunk verifies SHA-256 and inflates; throws on corruption
                        reader.readChunk(i)
                    }
                } finally {
                    reader.close()
                }
            }
            val dest = File(root, "${info.id.replace('/', '_')}@${info.version}.exm")
            if (dest.exists()) dest.delete()
            if (!tmp.renameTo(dest)) {
                tmp.copyTo(dest, overwrite = true)
                tmp.delete()
            }
            return info.copy(fileName = dest.name, integrityVerified = verified)
        } catch (t: Throwable) {
            tmp.delete()
            throw t
        }
    }

    fun list(): List<PackInfo> =
        root.listFiles { f -> f.isFile && f.name.endsWith(".exm") }
            ?.mapNotNull { openInfo(it) }
            ?.sortedBy { it.id }
            ?: emptyList()

    fun remove(fileName: String): Boolean {
        if (!fileName.endsWith(".exm") || fileName.contains("/")) return false
        return File(root, fileName).delete()
    }

    fun packFile(fileName: String): File = File(root, fileName)

    /** Header-only open: no weight bytes are read. Returns null for non-EXM files. */
    fun openInfo(file: File): PackInfo? {
        if (!file.exists() || file.length() == 0L) return null
        return try {
            FileInputStream(file).channel.use { ch ->
                val reader = ExmChunked.open(ch)
                try {
                    val m = reader.header.manifest
                    PackInfo(
                        fileName = file.name,
                        id = m.id,
                        version = m.version,
                        name = m.name,
                        quantization = m.quantization.name,
                        backends = m.backends.map { it.name },
                        license = m.license,
                        sizeBytes = m.sizeBytes,
                        originalSizeBytes = m.originalSizeBytes,
                        minRamMb = m.minRamMb,
                        minAndroid = m.minAndroid,
                        tensorCount = reader.header.tensors.size,
                        chunkCount = reader.header.chunkCount,
                        chunkSize = reader.header.chunkSize,
                        hasGraph = reader.header.tensors.any { it.name == "@graph" },
                        integrityVerified = false
                    )
                } finally {
                    reader.close()
                }
            }
        } catch (t: Throwable) {
            null
        }
    }

    companion object {
        fun defaultRoot(context: Context): File = File(context.filesDir, "packs")

        /** Human summary used by the packs sheet. */
        fun summarize(info: PackInfo): String = buildString {
            append(info.name)
            append(" · ").append(info.quantization)
            append(" · ").append(formatBytes(info.sizeBytes))
            append(" stored")
            if (info.originalSizeBytes > 0 && info.originalSizeBytes != info.sizeBytes) {
                append(" (of ").append(formatBytes(info.originalSizeBytes)).append(")")
            }
            append(" · ").append(info.tensorCount).append(" tensors / ").append(info.chunkCount).append(" chunks")
            if (info.hasGraph) append(" · graph ready")
            append(" · license: ").append(info.license)
            append(" · backends: ").append(info.backends.joinToString("/").ifEmpty { "unspecified" })
        }

        fun formatBytes(bytes: Long): String = when {
            bytes >= 1L shl 30 -> "%.1f GB".format(bytes.toDouble() / (1L shl 30))
            bytes >= 1L shl 20 -> "%.1f MB".format(bytes.toDouble() / (1L shl 20))
            bytes >= 1L shl 10 -> "%.1f KB".format(bytes.toDouble() / (1L shl 10))
            else -> "$bytes B"
        }
    }
}
