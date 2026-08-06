package com.seededge.runtime

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import androidx.test.platform.app.InstrumentationRegistry
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File

/**
 * Device-side Phase 18D gate.  These tests do not require the real model bundle:
 * the missing-model cases exercise the real JNI + task-local contract and the
 * PNG helpers are pure.  The full gallery photo gate runs after the models are
 * deployed and is driven from the MainActivity flow.
 */
class PhotoTaskDeviceTest {
    private val context get() = InstrumentationRegistry.getInstrumentation().targetContext

    @Test
    fun bitmapToRgb8ChannelOrder() {
        val bitmap = Bitmap.createBitmap(2, 1, Bitmap.Config.ARGB_8888)
        bitmap.setPixel(0, 0, 0xFF010203.toInt())
        bitmap.setPixel(1, 0, 0xFFA0B0C0.toInt())
        val rgb = PhotoTask.bitmapToRgb8(bitmap)
        assertEquals(6, rgb.size)
        assertEquals(1, rgb[0].toInt() and 0xFF)
        assertEquals(2, rgb[1].toInt() and 0xFF)
        assertEquals(3, rgb[2].toInt() and 0xFF)
        assertEquals(0xA0, rgb[3].toInt() and 0xFF)
        assertEquals(0xB0, rgb[4].toInt() and 0xFF)
        assertEquals(0xC0, rgb[5].toInt() and 0xFF)
    }

    @Test
    fun rgbFileEncodesPngWithExactDimensions() {
        val dir = File(context.cacheDir, "photo_task_test").apply { mkdirs() }
        val rgbFile = File(dir, "tiny.rgb")
        rgbFile.writeBytes(ByteArray(4 * 3 * 3) { it.toByte() })
        val pngFile = File(dir, "tiny.png")
        assertTrue(PhotoTask.writeRgbPng(rgbFile, pngFile, 4, 3))
        val decoded = BitmapFactory.decodeFile(pngFile.path)
        assertEquals(4, decoded.width)
        assertEquals(3, decoded.height)
    }

    @Test
    fun sideBySideCombinesInputAndOutputWidths() {
        val dir = File(context.cacheDir, "photo_task_test").apply { mkdirs() }
        val inputPng = File(dir, "left.png")
        val outputPng = File(dir, "right.png")
        assertTrue(PhotoTask.writeRgbPng(File(dir, "l.rgb").apply { writeBytes(ByteArray(2 * 2 * 3)) }, inputPng, 2, 2))
        assertTrue(PhotoTask.writeRgbPng(File(dir, "r.rgb").apply { writeBytes(ByteArray(3 * 2 * 3)) }, outputPng, 3, 2))
        val combined = File(dir, "combined.png")
        assertTrue(PhotoTask.writeSideBySide(inputPng, outputPng, combined))
        val decoded = BitmapFactory.decodeFile(combined.path)
        assertEquals(5, decoded.width)
        assertEquals(2, decoded.height)
    }

    @Test
    fun generatedTaskIdIsSafe() {
        assertTrue(PhotoTask.isSafeTaskId(PhotoTask.newTaskId()))
        assertTrue(Regex("[A-Za-z0-9_-]+").matches(PhotoTask.newTaskId()))
    }

    @Test
    fun malformedRequestRejectedBeforeModelAccess() {
        val rgb = ByteArray(64 * 64 * 3) { 7 }
        val result = JSONObject(NativeRuntime.nativeRunSinglePhotoTile(
            "missing-model", File(context.filesDir, "tasks/bad").path, rgb, 64, 64, "{"))
        assertEquals("INVALID_REQUEST_JSON", result.getString("error_code"))
        assertFalse(result.getBoolean("overall_pass"))
    }

    @Test
    fun taskLocalFlowWritesStatusOnMissingModel() {
        val taskId = "instrumented_missing_model_" + System.currentTimeMillis()
        val taskRoot = PhotoTask.taskRoot(context.filesDir, taskId)
        val rgb = ByteArray(64 * 64 * 3) { 9 }
        val request = PhotoTask.buildRequestJson(
            taskId, PhotoTask.MODEL_VERSION, "src", PhotoTask.sha256Hex(rgb), "manifest")
        val result = JSONObject(NativeRuntime.nativeRunSinglePhotoTile(
            File(context.cacheDir, "empty_model_root").apply { mkdirs() }.path,
            taskRoot.path, rgb, 64, 64, request))
        // On model failure the native entry returns the persisted status JSON,
        // not an error envelope, so the task id is the top-level identity.
        assertEquals(taskId, result.getString("task_id"))
        val tileRoot = PhotoTask.tileRoot(context.filesDir, taskId)
        assertTrue(File(tileRoot, "tile_input.rgb").isFile)
        assertTrue(File(tileRoot, "input_norm.f32").isFile)
        val statusFile = File(tileRoot, "status.json")
        assertTrue(statusFile.isFile)
        val status = JSONObject(statusFile.readText())
        assertEquals(taskId, status.getString("task_id"))
        assertEquals(PhotoTask.sha256Hex(rgb), status.getString("normalized_image_sha256"))
        assertTrue(status.isNull("checkpoint_root"))
        assertFalse(status.getBoolean("encoder_completed"))
        assertEquals(0, status.getInt("dit_block_completed"))
        assertFalse(status.getBoolean("decoder_completed"))
        val pipeline = status.getJSONObject("pipeline_result")
        assertFalse(pipeline.getBoolean("silent_cpu_fallback"))
        assertFalse(pipeline.getBoolean("model_set_verified"))
    }

    @Test
    fun nativeValidateModelManifestReportsMissingManifest() {
        val result = JSONObject(NativeRuntime.nativeValidateModelManifest(
            File(context.cacheDir, "empty_model_root").apply { mkdirs() }.path))
        assertFalse(result.getBoolean("valid"))
        assertFalse(result.getBoolean("silent_fallback"))
    }
}
