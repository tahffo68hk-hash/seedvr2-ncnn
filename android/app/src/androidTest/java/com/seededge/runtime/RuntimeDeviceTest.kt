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
