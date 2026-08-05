package com.seededge.runtime

import android.util.Log
import androidx.test.platform.app.InstrumentationRegistry
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File

class RuntimeDeviceTest {
    private val context get() = InstrumentationRegistry.getInstrumentation().targetContext

    @Test fun fastConvolution3DMatchesBuiltinReference() {
        val result = JSONObject(NativeRuntime.nativeTestFastConvolution3D())
        assertTrue(result.toString(), result.getBoolean("finite"))
        assertTrue(result.toString(), result.getDouble("max_abs_error") <= 1e-4)
        assertTrue(result.toString(), result.getBoolean("overall_pass"))
    }

    @Test fun cpuTwentyRuns() {
        var last = ""
        repeat(20) { index ->
            last = NativeRuntime.nativeRunSyntheticCpuTest()
            val json = JSONObject(last)
            assertEquals(2, json.getInt("schema_version"))
            assertTrue(json.getBoolean("device_executed"))
            assertTrue(json.getBoolean("shape_pass"))
            assertTrue(json.getBoolean("finite"))
            assertTrue(json.getBoolean("expected_parity"))
            assertTrue(json.getBoolean("numeric_repeat_consistent"))
            assertFalse(json.getBoolean("silent_fallback"))
            assertTrue("CPU run ${index + 1}", json.getBoolean("overall_pass"))
        }
        context.openFileOutput("phase18b_cpu_last.json", 0).bufferedWriter().use { it.write(last) }
        Log.i("Phase18B", "CPU_20_PASS")
    }

    @Test fun vulkanTenRuns() {
        var last = ""
        repeat(10) { index ->
            last = NativeRuntime.nativeProbeVulkanSupport()
            val json = JSONObject(last)
            assertEquals(2, json.getInt("schema_version"))
            assertTrue(json.getBoolean("device_created"))
            assertTrue(json.getBoolean("blob_allocator_created"))
            assertTrue(json.getBoolean("staging_allocator_created"))
            assertTrue(json.getBoolean("gpu_upload_executed"))
            assertTrue(json.getBoolean("vulkan_command_executed"))
            assertTrue(json.getBoolean("gpu_download_executed"))
            assertTrue(json.getBoolean("synthetic_inference_executed"))
            assertTrue(json.getBoolean("finite"))
            assertTrue(json.getBoolean("cpu_gpu_parity"))
            assertFalse(json.getBoolean("silent_cpu_fallback"))
            assertTrue(json.getBoolean("cleanup_pass"))
            assertTrue("Vulkan run ${index + 1}", json.getBoolean("overall_pass"))
        }
        context.openFileOutput("phase18b_vulkan_last.json", 0).bufferedWriter().use { it.write(last) }
        Log.i("Phase18B", "VULKAN_10_PASS")
    }

    @Test fun vulkanOneDiagnosticRun() {
        val result = NativeRuntime.nativeProbeVulkanSupport()
        context.openFileOutput("phase18b_vulkan_diagnostic.json", 0).bufferedWriter().use { it.write(result) }
        val json = JSONObject(result)
        assertTrue(json.toString(), json.getBoolean("overall_pass"))
    }

    @Test fun phase18c3RealDecoderStageThreadBenchmark() {
        val probes = listOf(1, 2, 4, 5, 6, 8).map { requested ->
            val json = JSONObject(NativeRuntime.nativeRunOpenMpThreadProbe(requested))
            assertTrue(json.toString(), json.getBoolean("openmp_compiled"))
            assertTrue(json.toString(), json.getBoolean("overall_pass"))
            if (requested > 1) assertTrue(json.toString(), json.getInt("observed_worker_threads") > 1)
            json
        }
        val external = requireNotNull(context.getExternalFilesDir(null))
        val result = JSONObject(NativeRuntime.nativeRunDecoderStageBenchmark(
            File(external, "models/phase18c").absolutePath,
            File(external, "output/phase18c").absolutePath))
        File(external, "output/phase18c/phase18c3_openmp_thread_probe.json").bufferedWriter().use {
            it.write(probes.joinToString(prefix = "[", postfix = "]") { item -> item.toString() })
        }
        assertTrue(result.toString(), result.getBoolean("overall_pass"))
        assertEquals(6, result.getJSONArray("runs").length())
        Log.i("Phase18C3", "DECODER_STAGE_THREAD_BENCHMARK_PASS $result")
    }

    @Test fun phase18c3DecoderOnlyWritesPng() {
        val external = requireNotNull(context.getExternalFilesDir(null))
        val modelRoot = File(external, "models/phase18c")
        val outputRoot = File(external, "output/phase18c")
        val result = JSONObject(NativeRuntime.nativeRunDecoderOnly(
            modelRoot.absolutePath, outputRoot.absolutePath, outputRoot.absolutePath, "{\"mode\":\"decoder_only\"}"))
        assertTrue(result.toString(), result.getBoolean("overall_pass"))
        val png = File(outputRoot, "frame_000.png")
        assertTrue("RGB to PNG conversion failed", DecoderOutput.writeRgb24Png(File(outputRoot, "frame_000.rgb"), png))
        assertTrue("PNG is empty", png.length() > 0)
        Log.i("Phase18C3", "DECODER_ONLY_PNG_PASS ${png.length()}")
    }

    @Test fun phase18c4CreateDecUp0Checkpoint() {
        val external = requireNotNull(context.getExternalFilesDir(null))
        val modelRoot = File(external, "models/phase18c")
        val outputRoot = File(external, "output/phase18c")
        val result = JSONObject(NativeRuntime.nativeRunDecoderOnly(
            modelRoot.absolutePath, outputRoot.absolutePath, outputRoot.absolutePath,
            "{\"mode\":\"decoder_only\",\"stop_after\":\"dec_up0\"}"))
        assertTrue(result.toString(), result.getBoolean("overall_pass"))
        assertEquals("dec_up0", result.getString("checkpoint_stage"))
        assertEquals(64, result.getString("checkpoint_sha256").length)
        assertTrue(File(outputRoot, "decoder_after_dec_up0.f32").length() == 512L * 3L * 32L * 32L * 4L)
        Log.i("Phase18C4", "DEC_UP0_CHECKPOINT_PASS $result")
    }

    @Test fun phase18c4RunDecUp1Twice() {
        val external = requireNotNull(context.getExternalFilesDir(null))
        val modelRoot = File(external, "models/phase18c")
        val outputRoot = File(external, "output/phase18c")
        fun runOnce() = JSONObject(NativeRuntime.nativeRunDecoderOnly(
            modelRoot.absolutePath, outputRoot.absolutePath, outputRoot.absolutePath,
            "{\"mode\":\"decoder_only\",\"resume_from\":\"dec_up0\",\"stop_after\":\"dec_up1\"}"))
        val first = runOnce()
        assertTrue(first.toString(), first.getBoolean("overall_pass"))
        val second = runOnce()
        assertTrue(second.toString(), second.getBoolean("overall_pass"))
        assertEquals(first.getString("checkpoint_sha256"), second.getString("checkpoint_sha256"))
        assertEquals(64, second.getString("checkpoint_sha256").length)
        assertTrue(File(outputRoot, "decoder_after_dec_up1.f32").length() == 512L * 5L * 64L * 64L * 4L)
        File(outputRoot, "phase18c4_dec_up1_two_runs.json").bufferedWriter().use {
            it.write(JSONObject().put("run_1", first).put("run_2", second).put("bit_identical", true).toString())
        }
        Log.i("Phase18C4", "DEC_UP1_TWO_RUNS_PASS ${second.getString("checkpoint_sha256")}")
    }

    @Test fun phase18c4CompleteDecoderFromDecUp1() {
        val external = requireNotNull(context.getExternalFilesDir(null))
        val modelRoot = File(external, "models/phase18c")
        val outputRoot = File(external, "output/phase18c")
        val result = JSONObject(NativeRuntime.nativeRunDecoderOnly(
            modelRoot.absolutePath, outputRoot.absolutePath, outputRoot.absolutePath,
            "{\"mode\":\"decoder_only\",\"resume_from\":\"dec_up1\"}"))
        assertTrue(result.toString(), result.getBoolean("overall_pass"))
        val png = File(outputRoot, "frame_000.png")
        assertTrue("RGB to PNG conversion failed", DecoderOutput.writeRgb24Png(File(outputRoot, "frame_000.rgb"), png))
        assertTrue("PNG is empty", png.length() > 0)
        File(outputRoot, "phase18c4_decoder_completion.json").bufferedWriter().use { it.write(result.toString()) }
        Log.i("Phase18C4", "DECODER_FROM_DEC_UP1_PNG_PASS ${png.length()}")
    }

    @Test fun phase18c4ExportInputTensorForVisualComparison() {
        val external = requireNotNull(context.getExternalFilesDir(null))
        val source = File(context.filesDir, "phase18c_input/input_norm.f32")
        val destination = File(external, "output/phase18c/input_norm.f32")
        assertTrue("Internal input tensor missing", source.isFile)
        destination.parentFile?.mkdirs()
        source.copyTo(destination, overwrite = true)
        assertEquals(3L * 5L * 128L * 128L * 4L, destination.length())
        Log.i("Phase18C4", "INPUT_TENSOR_EXPORT_PASS ${destination.length()}")
    }

    @Test fun phase18c4RefreshCheckpointMetadataWithoutRecompute() {
        val external = requireNotNull(context.getExternalFilesDir(null))
        val modelRoot = File(external, "models/phase18c")
        val outputRoot = File(external, "output/phase18c")
        for (stage in listOf("dec_up0", "dec_up1")) {
            val result = JSONObject(NativeRuntime.nativeRunDecoderOnly(
                modelRoot.absolutePath, outputRoot.absolutePath, outputRoot.absolutePath,
                "{\"mode\":\"decoder_only\",\"resume_from\":\"$stage\",\"stop_after\":\"$stage\"}"))
            assertTrue(result.toString(), result.getBoolean("overall_pass"))
            assertEquals(0, result.getInt("recomputed_decoder_stages"))
            assertEquals(stage, result.getString("checkpoint_stage"))
        }
        Log.i("Phase18C4", "CHECKPOINT_METADATA_REFRESH_PASS")
    }

    @Test fun phase18cMinimalVideoDemo() {
        val result = MinimalVideoDemo.run(context)
        context.openFileOutput("phase18c_result.json", 0).bufferedWriter().use { it.write(result) }
        val json = JSONObject(result)
        assertTrue(json.toString(), json.getBoolean("model_set_verified"))
        assertTrue(json.toString(), json.getBoolean("vae_encoder_executed"))
        assertEquals(32, json.getInt("dit_blocks_executed"))
        assertEquals(1, json.getInt("dit_max_resident_blocks"))
        assertTrue(json.toString(), json.getBoolean("vae_decoder_executed"))
        assertEquals(4, json.getInt("output_frames_written"))
        assertTrue(json.toString(), json.getBoolean("video_encoded"))
        assertTrue(json.toString(), json.getBoolean("media_extractor_readable"))
        assertEquals(4, json.getInt("video_frame_count"))
        assertFalse(json.getBoolean("silent_cpu_fallback"))
        assertTrue(json.toString(), json.getBoolean("overall_pass"))
        Log.i("Phase18C", "MINIMAL_VIDEO_DEMO_PASS")
    }
}
