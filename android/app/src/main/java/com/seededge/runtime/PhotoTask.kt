package com.seededge.runtime

import android.graphics.Bitmap
import android.graphics.Canvas
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.security.MessageDigest
import java.util.UUID

/**
 * Task-local helpers for the Phase 18D single-photo-tile flow.
 *
 * Kotlin owns Photo Picker import, EXIF-corrected RGB8 buffer creation, task
 * storage under files/tasks/<task_id>/tiles/tile_0000, PNG encoding and identity
 * hashes.  The C++ side owns tile planning, padding, model execution and
 * status.json; it never accepts a Phase 18C.4 checkpoint from this flow.
 */
object PhotoTask {
    const val REQUEST_SCHEMA_VERSION = 1
    const val MODEL_VERSION = "phase18d-android-v1"
    const val TILE_SIZE = 128

    /** Returns a task id matching the native `[A-Za-z0-9_-]+` contract. */
    fun newTaskId(): String =
        "photo_" + System.currentTimeMillis() + "_" + UUID.randomUUID().toString().replace("-", "").take(8)

    /** Native request parsing accepts only `[A-Za-z0-9_-]+` for task ids. */
    fun isSafeTaskId(taskId: String): Boolean =
        taskId.isNotEmpty() && taskId.all { it in 'a'..'z' || it in 'A'..'Z' || it in '0'..'9' || it == '_' || it == '-' }

    fun sha256Hex(bytes: ByteArray): String =
        MessageDigest.getInstance("SHA-256").digest(bytes).joinToString("") { "%02x".format(it) }

    fun sha256Hex(file: File): String {
        val digest = MessageDigest.getInstance("SHA-256")
        FileInputStream(file).use { input ->
            val buffer = ByteArray(64 * 1024)
            while (true) {
                val read = input.read(buffer)
                if (read < 0) break
                digest.update(buffer, 0, read)
            }
        }
        return digest.digest().joinToString("") { "%02x".format(it) }
    }

    /** files/tasks/<task_id> — the native task-root contract. */
    fun taskRoot(root: File, taskId: String): File = File(root, "tasks/$taskId")

    /** files/tasks/<task_id>/tiles/tile_0000 — the only tile this flow runs. */
    fun tileRoot(root: File, taskId: String): File = File(taskRoot(root, taskId), "tiles/tile_0000")

    fun buildRequestJson(
        taskId: String,
        modelVersion: String,
        sourceImageSha256: String,
        normalizedImageSha256: String,
        manifestSha256: String
    ): String {
        val quoted = { value: String ->
            "\"" + value.replace("\\", "\\\\").replace("\"", "\\\"") + "\""
        }
        return "{" +
            "\"schema_version\":$REQUEST_SCHEMA_VERSION," +
            "\"task_id\":" + quoted(taskId) + "," +
            "\"model_version\":" + quoted(modelVersion) + "," +
            "\"source_image_sha256\":" + quoted(sourceImageSha256) + "," +
            "\"normalized_image_sha256\":" + quoted(normalizedImageSha256) + "," +
            "\"model_manifest_sha256\":" + quoted(manifestSha256) + "," +
            "\"tile_index\":0,\"tile_size\":$TILE_SIZE,\"overlap\":32,\"stride\":96," +
            "\"temporal_frames\":5,\"target_frame_index\":0" +
            "}"
    }

    /** Converts an ARGB_8888 bitmap into contiguous RGB8 HWC bytes (row-major). */
    fun bitmapToRgb8(bitmap: Bitmap): ByteArray {
        val width = bitmap.width
        val height = bitmap.height
        val pixels = IntArray(width * height)
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height)
        val rgb = ByteArray(width * height * 3)
        var index = 0
        for (pixel in pixels) {
            rgb[index++] = (pixel shr 16 and 0xFF).toByte()
            rgb[index++] = (pixel shr 8 and 0xFF).toByte()
            rgb[index++] = (pixel and 0xFF).toByte()
        }
        return rgb
    }

    /** Encodes a contiguous RGB8 file as a PNG of exactly width*height. */
    fun writeRgbPng(rgbFile: File, pngFile: File, width: Int, height: Int): Boolean {
        val rgb = rgbFile.readBytes()
        if (rgb.size != width * height * 3) return false
        val rgba = ByteArray(width * height * 4)
        for (pixel in 0 until width * height) {
            rgba[pixel * 4] = rgb[pixel * 3]
            rgba[pixel * 4 + 1] = rgb[pixel * 3 + 1]
            rgba[pixel * 4 + 2] = rgb[pixel * 3 + 2]
            rgba[pixel * 4 + 3] = 0xff.toByte()
        }
        val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        bitmap.copyPixelsFromBuffer(java.nio.ByteBuffer.wrap(rgba))
        pngFile.outputStream().use { stream ->
            return bitmap.compress(Bitmap.CompressFormat.PNG, 100, stream)
        }
    }

    /** Combines input and output PNGs side by side onto one canvas. */
    fun writeSideBySide(inputPng: File, outputPng: File, outFile: File): Boolean {
        val left = android.graphics.BitmapFactory.decodeFile(inputPng.path) ?: return false
        val right = android.graphics.BitmapFactory.decodeFile(outputPng.path) ?: return false
        val width = left.width + right.width
        val height = maxOf(left.height, right.height)
        val canvasBitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(canvasBitmap)
        canvas.drawBitmap(left, 0f, 0f, null)
        canvas.drawBitmap(right, left.width.toFloat(), 0f, null)
        outFile.outputStream().use { stream ->
            return canvasBitmap.compress(Bitmap.CompressFormat.PNG, 100, stream)
        }
    }

}