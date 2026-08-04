package com.seededge.runtime

import android.app.Activity
import android.os.Bundle
import android.text.method.ScrollingMovementMethod
import android.view.ViewGroup
import android.widget.Button
import android.widget.LinearLayout
import android.widget.TextView

class MainActivity : Activity() {
    private lateinit var log: TextView
    override fun onCreate(state: Bundle?) {
        super.onCreate(state)
        val root = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; setPadding(24, 24, 24, 24) }
        fun line(text: String) = TextView(this).also { it.text = text; root.addView(it) }
        line("SeedEdge Runtime — Phase 18A")
        line("Runtime version: ${NativeRuntime.nativeGetVersion()}")
        line("ABI / build: ${NativeRuntime.nativeGetBuildInfo()}")
        line("ncnn status: ${NativeRuntime.nativeGetNcnnInfo()}")
        line("Vulkan: compiled/prepared; device not yet tested")
        line("Model root: ${getExternalFilesDir(null)}/models/seedvr2")
        Button(this).apply { text = "Run synthetic CPU test"; setOnClickListener { append(NativeRuntime.nativeRunSyntheticCpuTest()) }; root.addView(this) }
        Button(this).apply { text = "Probe Vulkan support"; setOnClickListener { append(NativeRuntime.nativeProbeVulkanSupport()) }; root.addView(this) }
        log = TextView(this).apply { movementMethod = ScrollingMovementMethod(); text = "Log\n"; root.addView(this, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f)) }
        setContentView(root)
    }
    private fun append(message: String) { log.append(message + "\n") }
}
