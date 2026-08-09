package com.seededge.runtime

import android.graphics.Bitmap
import android.graphics.Canvas
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.security.MessageDigest
import java.util.UUID
import java.util.concurrent.atomic.AtomicBoolean
import org.json.JSONObject

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
    const val BALANCED_THREADS = 5

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
            "\"temporal_frames\":5,\"target_frame_index\":0," +
            "\"encoder_threads\":$BALANCED_THREADS," +
            "\"dit_threads\":$BALANCED_THREADS," +
            "\"decoder_threads\":$BALANCED_THREADS" +
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

    /** Full-photo task result validated from native JSON plus task-local status. */
    data class FullPhotoResult(
        val success: Boolean,
        val state: String,
        val errorCode: String?,
        val errorMessage: String?,
        val taskId: String,
        val taskRoot: File,
        val correctedWidth: Int,
        val correctedHeight: Int,
        val tileCount: Int,
        val completedTiles: Int,
        val silentFallback: Boolean,
        val artifactsWritten: Boolean,
        val inputPng: File,
        val outputPng: File,
        val sideBySidePng: File,
        val statusJson: File
    )

    /** Reads width/height from the PNG IHDR chunk without decoding the image. */
    fun pngDimensions(file: File): Pair<Int, Int>? {
        if (!file.isFile) return null
        val header = ByteArray(24)
        file.inputStream().use { input ->
            var offset = 0
            while (offset < header.size) {
                val read = input.read(header, offset, header.size - offset)
                if (read < 0) break
                offset += read
            }
            if (offset < header.size) return null
        }
        val signature = byteArrayOf(0x89.toByte(), 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A)
        if (!header.copyOfRange(0, 8).contentEquals(signature)) return null
        if (header[12] != 0x49.toByte() || header[13] != 0x48.toByte() ||
            header[14] != 0x44.toByte() || header[15] != 0x52.toByte()) return null
        fun bigEndian(offset: Int): Int =
            ((header[offset].toInt() and 0xFF) shl 24) or
                ((header[offset + 1].toInt() and 0xFF) shl 16) or
                ((header[offset + 2].toInt() and 0xFF) shl 8) or
                (header[offset + 3].toInt() and 0xFF)
        return bigEndian(16) to bigEndian(20)
    }

    /**
     * Validates the native full-photo result JSON and the task-local status.json.
     * success is true only when the native bundle and task status both report a
     * completed, artifact-complete run with a correctly sized output.png.
     */
    fun validateFullPhotoResult(nativeResultText: String, taskRoot: File, width: Int, height: Int): FullPhotoResult {
        val inputPng = File(taskRoot, "input.png")
        val outputPng = File(taskRoot, "output.png")
        val sideBySidePng = File(taskRoot, "side_by_side.png")
        val statusJson = File(taskRoot, "status.json")
        var state = "failed"
        var artifactsWritten = false
        var silentFallback = false
        var errorCode: String? = "NATIVE_RESULT_INVALID"
        var errorMessage: String? = "native result JSON could not be parsed"
        try {
            val native = JSONObject(nativeResultText)
            state = native.optString("state", "failed")
            artifactsWritten = native.optBoolean("artifacts_written", false)
            silentFallback = native.optBoolean("silent_fallback", false)
            errorCode = native.optString("error_code", "").ifEmpty { null }
            errorMessage = native.optString("error_message", "").ifEmpty { null }
        } catch (_: Exception) {
            return FullPhotoResult(false, state, errorCode, errorMessage, "", taskRoot, width, height,
                0, 0, silentFallback, artifactsWritten, inputPng, outputPng, sideBySidePng, statusJson)
        }
        if (state != "completed" || !artifactsWritten || silentFallback) {
            return FullPhotoResult(false, state, errorCode, errorMessage, "", taskRoot, width, height,
                0, 0, silentFallback, artifactsWritten, inputPng, outputPng, sideBySidePng, statusJson)
        }
        if (!statusJson.isFile) {
            return FullPhotoResult(false, state, "STATUS_JSON_MISSING", "task status.json is missing",
                "", taskRoot, width, height, 0, 0, silentFallback, artifactsWritten,
                inputPng, outputPng, sideBySidePng, statusJson)
        }
        var taskId = taskRoot.name
        var tileCount = 0
        var completedTiles = 0
        var taskState = "unknown"
        var taskSilentFallback = false
        try {
            val status = JSONObject(statusJson.readText())
            taskId = status.optString("task_id", taskRoot.name)
            taskState = status.optString("state", "unknown")
            tileCount = status.optInt("tile_count", 0)
            completedTiles = status.optInt("completed_tiles", 0)
            taskSilentFallback = status.optBoolean("silent_fallback", false)
        } catch (_: Exception) {
            return FullPhotoResult(false, state, "STATUS_JSON_INVALID", "task status.json could not be parsed",
                taskId, taskRoot, width, height, 0, 0, silentFallback, artifactsWritten,
                inputPng, outputPng, sideBySidePng, statusJson)
        }
        val mergedSilentFallback = silentFallback || taskSilentFallback
        if (taskState != "completed" || mergedSilentFallback || tileCount <= 0 || completedTiles != tileCount) {
            return FullPhotoResult(false, state, "TASK_NOT_COMPLETED",
                "task status does not report a completed full-photo run", taskId, taskRoot, width, height,
                tileCount, completedTiles, mergedSilentFallback, artifactsWritten,
                inputPng, outputPng, sideBySidePng, statusJson)
        }
        val outputDims = pngDimensions(outputPng)
        val ok = inputPng.isFile && outputPng.isFile && sideBySidePng.isFile &&
            outputDims == (width to height)
        val code = when {
            !ok && !outputPng.isFile -> "OUTPUT_PNG_MISSING"
            !ok -> "OUTPUT_PNG_SIZE_MISMATCH"
            else -> null
        }
        return FullPhotoResult(ok, state, code,
            if (ok) null else "full-photo PNG artifacts are incomplete or wrong size",
            taskId, taskRoot, width, height, tileCount, completedTiles, mergedSilentFallback,
            artifactsWritten, inputPng, outputPng, sideBySidePng, statusJson)
    }

    /** MediaStore gate: only a completed, artifact-complete, non-cancelled task may be saved. */
    fun shouldSaveToGallery(result: FullPhotoResult, userCancelled: Boolean): Boolean =
        !userCancelled && result.success && !result.silentFallback &&
            result.tileCount > 0 && result.completedTiles == result.tileCount &&
            result.outputPng.isFile &&
            pngDimensions(result.outputPng) == (result.correctedWidth to result.correctedHeight)

    /** Runs the native full-photo pipeline and validates its task-local artifacts. */
    fun runFullPhoto(modelRoot: File, taskRoot: File, normalizedRgb: ByteArray,
                     width: Int, height: Int, requestJson: String): FullPhotoResult {
        val nativeText = NativeRuntime.nativeRunFullPhoto(
            modelRoot.path, taskRoot.path, normalizedRgb, width, height, requestJson)
        return validateFullPhotoResult(nativeText, taskRoot, width, height)
    }

    /**
     * Runs the native full-photo pipeline with cooperative progress and
     * cancellation. onProgress is invoked on the worker thread; returning
     * false from the bridge cancels at the next tile/fusion boundary.
     */
    fun runFullPhotoWithProgress(modelRoot: File, taskRoot: File, normalizedRgb: ByteArray,
                                 width: Int, height: Int, requestJson: String,
                                 cancelled: AtomicBoolean,
                                 onProgress: (stage: String, completed: Int, total: Int) -> Unit): FullPhotoResult {
        val bridge = FullPhotoProgressBridge(cancelled, onProgress)
        val nativeText = NativeRuntime.nativeRunFullPhotoWithProgress(
            modelRoot.path, taskRoot.path, normalizedRgb, width, height, requestJson, bridge)
        return validateFullPhotoResult(nativeText, taskRoot, width, height)
    }

    /** Stable full-photo UI stages shared by the progress bridge and tests. */
    object FullPhotoStages {
        const val PREPARING = "PREPARING"
        const val CHECKING_MODEL = "CHECKING_MODEL"
        const val RUNNING_TILE = "RUNNING_TILE"
        const val FUSING = "FUSING"
        const val ENCODING_PNG = "ENCODING_PNG"
        const val SAVING = "SAVING"
        const val COMPLETED = "COMPLETED"
        const val FAILED = "FAILED"
        const val CANCELLED = "CANCELLED"
    }

    /** Maps a native progress stage to a user-facing label. */
    fun fullPhotoStageLabel(stage: String, completed: Int, total: Int): String = when (stage) {
        FullPhotoStages.PREPARING -> "准备中"
        FullPhotoStages.CHECKING_MODEL -> "检查模型（1366 文件 SHA-256，仅一次）"
        FullPhotoStages.RUNNING_TILE ->
            if (total > 0) "正在增强：tile ${completed.coerceIn(0, total)}/$total"
            else "正在增强：tile"
        FullPhotoStages.FUSING -> "加权融合"
        FullPhotoStages.ENCODING_PNG -> "编码 PNG"
        FullPhotoStages.SAVING -> "保存到相册"
        FullPhotoStages.COMPLETED -> "完成"
        FullPhotoStages.FAILED -> "失败"
        FullPhotoStages.CANCELLED -> "已取消"
        else -> stage
    }

    /** Result object recorded after a successful MediaStore save. */
    data class GallerySaveRecord(
        val mediaStoreUri: String,
        val taskId: String,
        val correctedWidth: Int,
        val correctedHeight: Int,
        val tileCount: Int,
        val outputPngPath: String
    )

    fun gallerySaveRecord(result: FullPhotoResult, mediaStoreUri: String): GallerySaveRecord =
        GallerySaveRecord(mediaStoreUri, result.taskId, result.correctedWidth,
            result.correctedHeight, result.tileCount, result.outputPng.path)
}
