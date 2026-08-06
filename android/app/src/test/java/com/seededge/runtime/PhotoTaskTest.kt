package com.seededge.runtime

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.File

/** Red step for the Phase 18D Kotlin gallery flow: PhotoTask must exist. */
class PhotoTaskTest {
    @get:Rule
    val tmp = TemporaryFolder()

    @Test
    fun sha256HexMatchesKnownVector() {
        assertEquals(
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
            PhotoTask.sha256Hex("abc".toByteArray())
        )
    }

    @Test
    fun sha256HexOfFileMatchesBytes() {
        val file = tmp.newFile("source.jpg")
        file.writeBytes(byteArrayOf(1, 2, 3, 4, 5))
        assertEquals(PhotoTask.sha256Hex(byteArrayOf(1, 2, 3, 4, 5)), PhotoTask.sha256Hex(file))
    }

    @Test
    fun taskLayoutMatchesNativeContract() {
        val root = tmp.newFolder("files")
        val tile = PhotoTask.tileRoot(root, "photo_123_ab12")
        assertEquals("tile_0000", tile.name)
        assertEquals("tiles", tile.parentFile!!.name)
        assertEquals("photo_123_ab12", tile.parentFile!!.parentFile!!.name)
        assertEquals("tasks", tile.parentFile!!.parentFile!!.parentFile!!.name)
        // Regression: exactly files/tasks/<id>/tiles/tile_0000, never files/tasks/tasks/<id>.
        assertEquals("files", tile.parentFile!!.parentFile!!.parentFile!!.parentFile!!.name)
        assertFalse(tile.path.contains("tasks" + File.separator + "tasks"))
        assertTrue(tile.path.contains("tasks" + File.separator + "photo_123_ab12" + File.separator + "tiles"))
    }

    @Test
    fun requestJsonCarriesIdentityAndSchema() {
        val json = PhotoTask.buildRequestJson(
            taskId = "task_1",
            modelVersion = "phase18d-android-v1",
            sourceImageSha256 = "srcsha",
            normalizedImageSha256 = "normsha",
            manifestSha256 = "mfsha"
        )
        assertTrue(json.contains("\"schema_version\":1"))
        assertTrue(json.contains("\"task_id\":\"task_1\""))
        assertTrue(json.contains("\"model_version\":\"phase18d-android-v1\""))
        assertTrue(json.contains("\"source_image_sha256\":\"srcsha\""))
        assertTrue(json.contains("\"normalized_image_sha256\":\"normsha\""))
        assertTrue(json.contains("\"model_manifest_sha256\":\"mfsha\""))
        assertTrue(json.contains("\"tile_index\":0"))
    }

    @Test
    fun taskIdIsSafeForNativeRegex() {
        val generated = PhotoTask.newTaskId()
        assertTrue(PhotoTask.isSafeTaskId(generated))
        assertTrue(PhotoTask.isSafeTaskId("photo_123_ab12"))
        assertTrue(PhotoTask.isSafeTaskId("photo-123"))
        assertFalse(PhotoTask.isSafeTaskId("../evil"))
        assertFalse(PhotoTask.isSafeTaskId("a b"))
        assertFalse(PhotoTask.isSafeTaskId("a\u4e2db"))
        assertFalse(PhotoTask.isSafeTaskId(""))
    }
}