package app.orubus.exmera.camera

import android.content.ContentValues
import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.net.Uri
import android.os.Build
import android.os.Environment
import android.provider.MediaStore
import java.io.File
import java.io.OutputStream

/**
 * Auto metadata scrubbing.
 *
 * Every still exmera delivers to the gallery is a fresh pixel-only re-encode:
 * the original capture bytes (EXIF, GPS, device fingerprints, maker notes,
 * thumbnails, ICC payloads) are decoded to pixels and re-compressed, so no
 * metadata block survives. Rotation is baked into the pixels because a
 * scrubbed file can not rely on an orientation tag to display correctly.
 *
 * Video never receives location metadata in the first place (nothing writes
 * it), so the video path is clean by construction.
 */
object MetadataScrubber {

    const val QUALITY = 95

    data class Policy(
        val scrubEnabled: Boolean = true,
        val keepRawDng: Boolean = true
    )

    fun isJpeg(bytes: ByteArray): Boolean =
        bytes.size >= 3 && bytes[0] == 0xFF.toByte() && bytes[1] == 0xD8.toByte() && bytes[2] == 0xFF.toByte()

    /** Decodes [bytes] and re-encodes pixels only. Returns null when decoding fails. */
    fun scrubToJpegBytes(bytes: ByteArray, quality: Int = QUALITY): ByteArray? {
        val src = BitmapFactory.decodeByteArray(bytes, 0, bytes.size) ?: return null
        val out = java.io.ByteArrayOutputStream(src.byteCount.coerceAtMost(64 shl 20))
        src.compress(Bitmap.CompressFormat.JPEG, quality, out)
        src.recycle()
        return out.toByteArray()
    }

    /** Rotates [bitmap] by [degrees] (baked orientation, EXIF-free display correctness). */
    fun rotate(bitmap: Bitmap, degrees: Int): Bitmap {
        if (degrees == 0) return bitmap
        val m = Matrix().apply { postRotate(degrees.toFloat()) }
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, m, true)
    }

    /** Full scrub pipeline for a CameraX ImageProxy JPEG payload. */
    fun scrubCapture(bytes: ByteArray, rotationDegrees: Int, quality: Int = QUALITY): ByteArray? {
        val src = BitmapFactory.decodeByteArray(bytes, 0, bytes.size) ?: return null
        val upright = rotate(src, rotationDegrees)
        val out = java.io.ByteArrayOutputStream()
        upright.compress(Bitmap.CompressFormat.JPEG, quality, out)
        if (upright !== src) upright.recycle()
        src.recycle()
        return out.toByteArray()
    }

    /**
     * Saves pixels-only media into Pictures/exmera (or Movies/exmera for video URIs
     * handled elsewhere). DNG negatives are exempt from scrubbing by policy — a RAW
     * file without its technical tags is corrupt — but DNGs never carry GPS here
     * because exmera does not write it.
     */
    fun saveToGallery(
        context: Context,
        data: ByteArray,
        displayName: String,
        mimeType: String
    ): Uri? {
        return if (Build.VERSION.SDK_INT >= 29) {
            val values = ContentValues().apply {
                put(MediaStore.MediaColumns.DISPLAY_NAME, displayName)
                put(MediaStore.MediaColumns.MIME_TYPE, mimeType)
                put(MediaStore.MediaColumns.RELATIVE_PATH, relativePathFor(mimeType))
                put(MediaStore.MediaColumns.IS_PENDING, 1)
            }
            val uri = context.contentResolver.insert(collectionFor(mimeType), values) ?: return null
            runCatching {
                context.contentResolver.openOutputStream(uri)?.use { it.write(data) }
                    ?: throw IllegalStateException("output stream unavailable")
                val done = ContentValues().apply { put(MediaStore.MediaColumns.IS_PENDING, 0) }
                context.contentResolver.update(uri, done, null, null)
            }.onFailure {
                context.contentResolver.delete(uri, null, null)
                return null
            }
            uri
        } else {
            val dir = File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES), "exmera")
            if (!dir.exists()) dir.mkdirs()
            val file = File(dir, displayName)
            file.outputStream().use { it.write(data) }
            android.media.MediaScannerConnection.scanFile(context, arrayOf(file.path), arrayOf(mimeType), null)
            Uri.fromFile(file)
        }
    }

    fun relativePathFor(mimeType: String): String =
        if (mimeType.startsWith("video/")) "Movies/exmera" else "Pictures/exmera"

    fun collectionFor(mimeType: String): Uri =
        if (mimeType.startsWith("video/")) MediaStore.Video.Media.EXTERNAL_CONTENT_URI
        else MediaStore.Images.Media.EXTERNAL_CONTENT_URI

    fun displayName(prefix: String, extension: String, atMs: Long = System.currentTimeMillis()): String =
        "${prefix}_${atMs}.$extension"

    /** Convenience for callers that already hold a Bitmap (portrait, boost, panorama). */
    fun bitmapToJpeg(bitmap: Bitmap, quality: Int = QUALITY): ByteArray {
        val out = java.io.ByteArrayOutputStream()
        bitmap.compress(Bitmap.CompressFormat.JPEG, quality, out)
        return out.toByteArray()
    }
}
