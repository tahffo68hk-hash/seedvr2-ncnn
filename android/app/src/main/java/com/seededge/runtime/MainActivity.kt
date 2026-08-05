package com.seededge.runtime

import android.app.Activity
import android.os.Bundle
import android.view.ViewGroup
import android.widget.Button
import android.widget.LinearLayout
import android.widget.TextView
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

class MainActivity : Activity() {
    private lateinit var log: TextView
    private lateinit var cpuButton: Button
    private lateinit var vulkanButton: Button
    private var logText = "Log\n"
    private val executor = Executors.newSingleThreadExecutor()
    private val testRunning = AtomicBoolean(false)

    override fun onCreate(state: Bundle?) {
        super.onCreate(state)
        val root = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; setPadding(24, 24, 24, 24) }
        fun line(text: String) = TextView(this).also { it.setText(text, TextView.BufferType.NORMAL); root.addView(it) }
        line("SeedEdge Runtime — Phase 18B")
        line("Runtime version: ${NativeRuntime.nativeGetVersion()}")
        line("ABI / build: ${NativeRuntime.nativeGetBuildInfo()}")
        line("ncnn status: ${NativeRuntime.nativeGetNcnnInfo()}")
        line("Vulkan: compiled/prepared; device validation in progress")
        line("Model root: ${getExternalFilesDir(null)}/models/seedvr2")
        cpuButton = Button(this).apply { text = "Run synthetic CPU test"; setOnClickListener { runNative { NativeRuntime.nativeRunSyntheticCpuTest() } }; root.addView(this) }
        vulkanButton = Button(this).apply { text = "Run Vulkan synthetic test"; setOnClickListener { runNative { NativeRuntime.nativeProbeVulkanSupport() } }; root.addView(this) }
        log = TextView(this).apply {
            id = R.id.runtime_log
            isSaveEnabled = false
            setTextIsSelectable(false)
            setText(logText, TextView.BufferType.NORMAL)
            root.addView(this, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f))
        }
        setContentView(root)
    }

    private fun runNative(call: () -> String?) {
        if (!testRunning.compareAndSet(false, true)) return
        setButtonsEnabled(false)
        executor.execute {
            val result = try { call() } catch (t: Throwable) {
                "{\"schema_version\":2,\"overall_pass\":false,\"error_stage\":\"jni\",\"error_message\":\"${jsonEscape(t.message ?: t.javaClass.simpleName)}\"}"
            }
            log.post {
                appendResult(result)
                testRunning.set(false)
                setButtonsEnabled(true)
            }
        }
    }

    private fun setButtonsEnabled(enabled: Boolean) {
        cpuButton.isEnabled = enabled
        vulkanButton.isEnabled = enabled
    }

    private fun jsonEscape(value: String) = value.replace("\\", "\\\\").replace("\"", "\\\"").replace("\n", "\\n")

    override fun onDestroy() {
        executor.shutdown()
        super.onDestroy()
    }

    private fun appendResult(message: String?) {
        val next = SafeResultText.append(logText, message)
        log.post {
            logText = next
            log.setText(logText, TextView.BufferType.NORMAL)
        }
    }
}
