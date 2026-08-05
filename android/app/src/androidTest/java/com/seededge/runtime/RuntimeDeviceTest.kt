package com.seededge.runtime

import android.util.Log
import androidx.test.platform.app.InstrumentationRegistry
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

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
}
