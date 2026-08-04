package com.seededge.runtime

import android.app.Activity
import android.os.Bundle
import android.view.ViewGroup
import android.widget.Button
import android.widget.LinearLayout
import android.widget.TextView

class MainActivity : Activity() {
    private lateinit var log: TextView
    private var logText = "Log\n"

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
        Button(this).apply { text = "Run synthetic CPU test"; setOnClickListener { appendResult(NativeRuntime.nativeRunSyntheticCpuTest()) }; root.addView(this) }
        Button(this).apply { text = "Probe Vulkan support"; setOnClickListener { appendResult(NativeRuntime.nativeProbeVulkanSupport()) }; root.addView(this) }
        log = TextView(this).apply {
            id = R.id.runtime_log
            isSaveEnabled = false
            setTextIsSelectable(false)
            setText(logText, TextView.BufferType.NORMAL)
            root.addView(this, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f))
        }
        setContentView(root)
    }

    private fun appendResult(message: String?) {
        val next = SafeResultText.append(logText, message)
        log.post {
            logText = next
            log.setText(logText, TextView.BufferType.NORMAL)
        }
    }
}
