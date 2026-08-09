package com.seededge.runtime

import org.junit.Assert.assertNull
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
        assertTrue(json.contains("\"encoder_threads\":5"))
        assertTrue(json.contains("\"dit_threads\":5"))
        assertTrue(json.contains("\"decoder_threads\":5"))
        assertTrue(json.contains("\"runtime_mode\":\"auto\""))
        assertTrue(json.contains("\"tile_cohort_size\":4"))
        assertTrue(json.contains("\"checkpoint_mode\":\"production\""))
        assertTrue(json.contains("\"max_source_pixels\":16000000"))
        assertTrue(json.contains("\"max_tile_count\":2048"))
        assertTrue(json.contains("\"resume\":false"))
    }

    @Test
    fun resourceEstimateAndRuntimeLabelExposeRealMetadata() {
        val estimate = PhotoTask.estimateTaskResources(640, 427, "auto", 4)
        assertEquals(35, estimate.tileCount)
        assertEquals("hybrid", estimate.selectedMode)
        assertEquals(4, estimate.cohortSize)
        assertTrue(estimate.requiredFreeBytes > estimate.estimatedPeakBytes)
        assertFalse(estimate.deviceCalibrated)
        val label = PhotoTask.runtimeProgressLabel("""{"runtime_mode":"hybrid","tile_cohort_size":4,"checkpoint_generation":9,"stage_name":"dit","tile_index":3,"tile_count":35,"dit_block_index":7,"actual_task_disk_bytes":1234}""")
        assertTrue(label.contains("hybrid K=4"))
        assertTrue(label.contains("generation=9"))
        assertTrue(label.contains("tile=3/35"))
        assertTrue(label.contains("block=7"))
        assertTrue(label.contains("disk=1234"))
    }

    @Test
    fun tilePlanExplainsArbitrarySizeSplitAndCohorts() {
        val plan = PhotoTask.planTiles(640, 427, "auto", 4)
        assertEquals(640, plan.width)
        assertEquals(427, plan.height)
        assertEquals(128, plan.tileSize)
        assertEquals(32, plan.overlap)
        assertEquals(96, plan.stride)
        assertEquals(7, plan.columns)
        assertEquals(5, plan.rows)
        assertEquals(35, plan.tileCount)
        assertEquals("hybrid", plan.selectedMode)
        assertEquals(4, plan.cohortSize)
        assertEquals(9, plan.cohortCount)
        assertTrue(plan.summaryText().contains("grid=7x5"))
        assertTrue(plan.summaryText().contains("tiles=35"))
        assertTrue(plan.summaryText().contains("mode=hybrid K=4"))
    }

    @Test
    fun tileMapMarksCompletedCurrentAndPendingCells() {
        val plan = PhotoTask.planTiles(256, 256, "auto", 4)
        val map = PhotoTask.tileMapText(plan, completedTiles = 3, activeTileIndex = 4)
        assertEquals("###\n@>@\n@@@", map)
    }

    @Test
    fun overallProgressAdvancesAcrossWholePhotoNotJustCurrentTile() {
        val early = PhotoTask.overallProgressPercent(
            completedTiles = 5,
            totalTiles = 35,
            phase = "dit_streaming",
            ditBlockIndex = 0
        )
        val later = PhotoTask.overallProgressPercent(
            completedTiles = 5,
            totalTiles = 35,
            phase = "dit_streaming",
            ditBlockIndex = 31
        )
        val nextTile = PhotoTask.overallProgressPercent(
            completedTiles = 6,
            totalTiles = 35,
            phase = "vae_encoder"
        )
        assertTrue(early >= 14)
        assertTrue(later > early)
        assertTrue(nextTile > later)
        assertTrue(nextTile < 25)
    }

    @Test
    fun autoUsesMeasuredMediumModeAndFallsBackWhenDiskCannotFitIt() {
        val latency = PhotoTask.estimateTaskResources(512, 512, "auto", 4)
        assertEquals(25, latency.tileCount)
        assertEquals("stage_major", latency.selectedMode)
        assertEquals(0, latency.cohortSize)

        val bounded = PhotoTask.estimateTaskResources(512, 512, "auto", 4,
            availableDiskBytes = 2L * 1024 * 1024 * 1024)
        assertEquals("hybrid", bounded.selectedMode)
        assertEquals(4, bounded.cohortSize)
        assertTrue(bounded.requiredFreeBytes <= 2L * 1024 * 1024 * 1024)
    }

    @Test
    fun resumedRequestCarriesExplicitResumeFlag() {
        val json = PhotoTask.buildRequestJson("task_1", PhotoTask.MODEL_VERSION,
            "src", "norm", "manifest", resume = true)
        assertTrue(json.contains("\"runtime_mode\":\"auto\""))
        assertTrue(json.contains("\"tile_cohort_size\":4"))
        assertTrue(json.contains("\"resume\":true"))
    }

    @Test
    fun storagePreflightIncludesCheckpointAndAtomicReserve() {
        val estimate = PhotoTask.estimateRequiredStorageBytes(512, 512, 25)
        assertTrue(estimate > 25L * 96L * 1024L * 1024L)
        assertTrue(PhotoTask.hasStorageCapacity(estimate + 1, estimate))
        assertFalse(PhotoTask.hasStorageCapacity(estimate - 1, estimate))
    }


    @Test
    fun pngDimensionsParsesIhdr() {
        val png = tmp.newFile("dims.png")
        png.writeBytes(pngHeader(160, 96))
        assertEquals(160 to 96, PhotoTask.pngDimensions(png))
    }

    @Test
    fun pngDimensionsRejectsNonPng() {
        val file = tmp.newFile("not.png")
        file.writeBytes(ByteArray(32) { 0 })
        assertNull(PhotoTask.pngDimensions(file))
    }

    @Test
    fun validateFullPhotoResultAcceptsCompletedTask() {
        val root = tmp.newFolder("task")
        File(root, "input.png").writeBytes(pngHeader(160, 96))
        File(root, "output.png").writeBytes(pngHeader(160, 96))
        File(root, "side_by_side.png").writeBytes(pngHeader(320, 96))
        File(root, "status.json").writeText(
            """{"schema_version":1,"mode":"phase18d_full_photo","state":"completed","task_id":"t1","tile_count":2,"completed_tiles":2,"silent_fallback":false,"overall_pass":true}"""
        )
        val native =
            """{"schema_version":1,"mode":"phase18d_full_photo","state":"completed","artifacts_written":true,"silent_fallback":false}"""
        val result = PhotoTask.validateFullPhotoResult(native, root, 160, 96)
        assertTrue(result.success)
        assertEquals("completed", result.state)
        assertEquals(2, result.tileCount)
        assertEquals(2, result.completedTiles)
        assertEquals(File(root, "output.png"), result.outputPng)
        assertEquals(File(root, "status.json"), result.statusJson)
        assertEquals(File(root, "input.png"), result.inputPng)
        assertEquals(File(root, "side_by_side.png"), result.sideBySidePng)
    }

    @Test
    fun validateFullPhotoResultRejectsFailedNativeState() {
        val root = tmp.newFolder("task")
        val native =
            """{"schema_version":1,"mode":"phase18d_full_photo","state":"failed","artifacts_written":false,"silent_fallback":false,"error_code":"FUSION_FAILED"}"""
        val result = PhotoTask.validateFullPhotoResult(native, root, 160, 96)
        assertFalse(result.success)
        assertEquals("failed", result.state)
        assertEquals("FUSION_FAILED", result.errorCode)
    }

    @Test
    fun validateFullPhotoResultRejectsIncompleteTiles() {
        val root = tmp.newFolder("task")
        File(root, "status.json").writeText(
            """{"schema_version":1,"mode":"phase18d_full_photo","state":"completed","task_id":"t1","tile_count":2,"completed_tiles":1,"silent_fallback":false}"""
        )
        val native =
            """{"schema_version":1,"mode":"phase18d_full_photo","state":"completed","artifacts_written":true,"silent_fallback":false}"""
        assertFalse(PhotoTask.validateFullPhotoResult(native, root, 160, 96).success)
    }

    @Test
    fun validateFullPhotoResultRejectsMissingOrWrongSizedOutput() {
        val root = tmp.newFolder("task")
        File(root, "input.png").writeBytes(pngHeader(160, 96))
        File(root, "side_by_side.png").writeBytes(pngHeader(320, 96))
        File(root, "status.json").writeText(
            """{"schema_version":1,"mode":"phase18d_full_photo","state":"completed","task_id":"t1","tile_count":2,"completed_tiles":2,"silent_fallback":false}"""
        )
        val native =
            """{"schema_version":1,"mode":"phase18d_full_photo","state":"completed","artifacts_written":true,"silent_fallback":false}"""
        // output.png missing
        assertFalse(PhotoTask.validateFullPhotoResult(native, root, 160, 96).success)
        // output.png has the wrong dimensions
        File(root, "output.png").writeBytes(pngHeader(128, 128))
        assertFalse(PhotoTask.validateFullPhotoResult(native, root, 160, 96).success)
    }

    @Test
    fun validateFullPhotoResultRejectsSilentFallback() {
        val root = tmp.newFolder("task")
        File(root, "input.png").writeBytes(pngHeader(160, 96))
        File(root, "output.png").writeBytes(pngHeader(160, 96))
        File(root, "side_by_side.png").writeBytes(pngHeader(320, 96))
        File(root, "status.json").writeText(
            """{"schema_version":1,"mode":"phase18d_full_photo","state":"completed","task_id":"t1","tile_count":2,"completed_tiles":2,"silent_fallback":true}"""
        )
        val native =
            """{"schema_version":1,"mode":"phase18d_full_photo","state":"completed","artifacts_written":true,"silent_fallback":false}"""
        assertFalse(PhotoTask.validateFullPhotoResult(native, root, 160, 96).success)
    }

    @Test
    fun shouldSaveToGalleryRequiresSuccessAndNoCancel() {
        val root = tmp.newFolder("task")
        File(root, "input.png").writeBytes(pngHeader(160, 96))
        File(root, "output.png").writeBytes(pngHeader(160, 96))
        File(root, "side_by_side.png").writeBytes(pngHeader(320, 96))
        File(root, "status.json").writeText(
            """{"schema_version":1,"mode":"phase18d_full_photo","state":"completed","task_id":"t1","tile_count":2,"completed_tiles":2,"silent_fallback":false}"""
        )
        val native =
            """{"schema_version":1,"mode":"phase18d_full_photo","state":"completed","artifacts_written":true,"silent_fallback":false}"""
        val success = PhotoTask.validateFullPhotoResult(native, root, 160, 96)
        assertTrue(PhotoTask.shouldSaveToGallery(success, userCancelled = false))
        assertFalse(PhotoTask.shouldSaveToGallery(success, userCancelled = true))
        val failed = PhotoTask.validateFullPhotoResult(
            """{"schema_version":1,"mode":"phase18d_full_photo","state":"failed","artifacts_written":false,"silent_fallback":false}""",
            root, 160, 96
        )
        assertFalse(PhotoTask.shouldSaveToGallery(failed, userCancelled = false))
    }

    @Test
    fun fullPhotoStageLabelShowsTileProgress() {
        assertEquals("正在增强：tile 1/2", PhotoTask.fullPhotoStageLabel("RUNNING_TILE", 1, 2))
        assertEquals("检查模型（1366 文件 SHA-256，仅一次）", PhotoTask.fullPhotoStageLabel("CHECKING_MODEL", 0, 0))
        assertEquals("加权融合", PhotoTask.fullPhotoStageLabel("FUSING", 2, 2))
        assertEquals("编码 PNG", PhotoTask.fullPhotoStageLabel("ENCODING_PNG", 2, 2))
        assertEquals("已取消", PhotoTask.fullPhotoStageLabel("CANCELLED", 1, 2))
    }

    @Test
    fun gallerySaveRecordCarriesTaskIdentity() {
        val root = tmp.newFolder("task")
        File(root, "input.png").writeBytes(pngHeader(160, 96))
        File(root, "output.png").writeBytes(pngHeader(160, 96))
        File(root, "side_by_side.png").writeBytes(pngHeader(320, 96))
        File(root, "status.json").writeText(
            """{"schema_version":1,"mode":"phase18d_full_photo","state":"completed","task_id":"t1","tile_count":2,"completed_tiles":2,"silent_fallback":false}"""
        )
        val native =
            """{"schema_version":1,"mode":"phase18d_full_photo","state":"completed","artifacts_written":true,"silent_fallback":false}"""
        val result = PhotoTask.validateFullPhotoResult(native, root, 160, 96)
        assertTrue(result.success)
        val record = PhotoTask.gallerySaveRecord(result, "content://media/external/images/media/42")
        assertEquals("content://media/external/images/media/42", record.mediaStoreUri)
        assertEquals("t1", record.taskId)
        assertEquals(160, record.correctedWidth)
        assertEquals(96, record.correctedHeight)
        assertEquals(2, record.tileCount)
        assertEquals(File(root, "output.png").path, record.outputPngPath)
    }
    private fun pngHeader(width: Int, height: Int): ByteArray {
        val bytes = ByteArray(33)
        byteArrayOf(0x89.toByte(), 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A).copyInto(bytes, 0)
        bytes[8] = 0; bytes[9] = 0; bytes[10] = 0; bytes[11] = 13
        bytes[12] = 0x49; bytes[13] = 0x48; bytes[14] = 0x44; bytes[15] = 0x52
        fun bigEndian(offset: Int, value: Int) {
            bytes[offset] = (value ushr 24).toByte()
            bytes[offset + 1] = (value ushr 16).toByte()
            bytes[offset + 2] = (value ushr 8).toByte()
            bytes[offset + 3] = value.toByte()
        }
        bigEndian(16, width)
        bigEndian(20, height)
        return bytes
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
