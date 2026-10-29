package app.orubus.exmera

import app.orubus.exmera.packs.ExmPackManager
import studio.exmera.exm.ExmChunked
import studio.exmera.exm.ExmFormat
import studio.exmera.exm.ExmReader
import java.io.ByteArrayOutputStream
import java.io.File
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

class ExmPackManagerTest {

    private fun tempDir(): File = createTempDir(prefix = "exm-packs-test")

    private fun buildContainer(chunkBytes: Long = 1024L): ByteArray {
        val payload = ByteArray(5000) { (it % 251).toByte() }
        val manifest = ExmFormat.Manifest(
            id = "test/tiny-model",
            version = "1.0.0",
            name = "Tiny Test Model",
            quantization = ExmFormat.Quantization.FP16,
            backends = setOf(ExmFormat.Backend.CPU),
            sha256 = ExmReader.sha256(payload),
            sizeBytes = payload.size.toLong(),
            minRamMb = 512,
            minAndroid = 26,
            license = "MIT"
        )
        val tensors = listOf(
            ExmChunked.TensorSpec("weights.0", "fp32", intArrayOf(500), null),
            ExmChunked.TensorSpec("weights.1", "fp32", intArrayOf(750), null)
        )
        val out = ByteArrayOutputStream()
        ExmChunked.write(out, manifest, tensors, payload, chunkBytes)
        return out.toByteArray()
    }

    @Test
    fun `import validates header and installs into root`() {
        val root = tempDir()
        val manager = ExmPackManager(root)
        val bytes = buildContainer()
        val info = manager.importFrom(bytes.inputStream(), "tiny.exm")
        assertEquals("test/tiny-model", info.id)
        assertEquals("1.0.0", info.version)
        assertEquals("FP16", info.quantization)
        assertEquals("MIT", info.license)
        assertEquals(5000L, info.originalSizeBytes)
        assertTrue(info.chunkCount >= 1)
        assertTrue(info.integrityVerified, "per-chunk verification must run on import")
        assertTrue(info.fileName.endsWith(".exm"))
        assertEquals(listOf(info.fileName), manager.list().map { it.fileName })
        // stored file name carries id + version
        assertTrue(info.fileName.contains("test_tiny-model@1.0.0"))
    }

    @Test
    fun `import rejects a non-exm file with a readable error`() {
        val root = tempDir()
        val manager = ExmPackManager(root)
        val garbage = "this is definitely not a model container".toByteArray()
        val ex = assertFailsWith<IllegalArgumentException> {
            manager.importFrom(garbage.inputStream(), "fake.exm")
        }
        assertTrue(ex.message!!.contains("not a valid EXM container"))
        // nothing left behind
        assertTrue(root.listFiles()!!.none { it.name.startsWith("incoming_") })
    }

    @Test
    fun `corrupted chunk fails import verification`() {
        val root = tempDir()
        val manager = ExmPackManager(root)
        val bytes = buildContainer()
        // flip a byte inside the first chunk payload (container tail)
        val corrupted = bytes.copyOf()
        corrupted[corrupted.size - 10] = (corrupted[corrupted.size - 10].toInt() xor 0x5A).toByte()
        assertFailsWith<Throwable> { manager.importFrom(corrupted.inputStream(), "broken.exm") }
        // failed import must not leave a pack behind
        assertTrue(manager.list().isEmpty())
    }

    @Test
    fun `header-only open never verifies and never installs`() {
        val root = tempDir()
        val manager = ExmPackManager(root)
        val file = File(root, "probe.exm")
        file.writeBytes(buildContainer())
        val info = manager.openInfo(file)
        assertNotNull(info)
        assertEquals("test/tiny-model", info.id)
        assertTrue(!info.integrityVerified, "header open must not imply chunk verification")
    }

    @Test
    fun `remove deletes by safe file name only`() {
        val root = tempDir()
        val manager = ExmPackManager(root)
        val info = manager.importFrom(buildContainer().inputStream(), "tiny.exm")
        assertTrue(!manager.remove("../escape"), "path traversal must be rejected")
        assertTrue(manager.remove(info.fileName))
        assertTrue(manager.list().isEmpty())
    }

    @Test
    fun `summarize mentions size tensors license and graph state`() {
        val root = tempDir()
        val manager = ExmPackManager(root)
        val info = manager.importFrom(buildContainer().inputStream(), "tiny.exm")
        val s = ExmPackManager.summarize(info)
        assertTrue(s.contains("Tiny Test Model"))
        assertTrue(s.contains("FP16"))
        assertTrue(s.contains("MIT"))
        assertTrue(s.contains("graph ready").not()) // this container has no @graph tensor
    }

    @Test
    fun `formatBytes scales correctly`() {
        assertEquals("512 B", ExmPackManager.formatBytes(512))
        assertEquals("1.0 KB", ExmPackManager.formatBytes(1024))
        assertEquals("1.0 MB", ExmPackManager.formatBytes(1024 * 1024))
    }
}
