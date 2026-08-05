package com.seededge.runtime

import android.app.Activity
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.widget.*
import org.json.JSONObject
import java.io.File
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

class MainActivity : Activity() {
    private val executor = Executors.newSingleThreadExecutor()
    private val handler = Handler(Looper.getMainLooper())
    private val running = AtomicBoolean(false)
    private lateinit var status: TextView
    private lateinit var progress: ProgressBar
    private lateinit var log: TextView
    private lateinit var runButton: Button
    private lateinit var cancelButton: Button
    private lateinit var decoderRadio: RadioButton
    private var poll: Runnable? = null
    private val modelRoot get() = File(requireNotNull(getExternalFilesDir(null)), "models/phase18c")
    private val outputRoot get() = File(requireNotNull(getExternalFilesDir(null)), "output/phase18c")

    override fun onCreate(state: Bundle?) {
        super.onCreate(state)
        val root = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; setPadding(24,24,24,24) }
        fun label(value: String) = TextView(this).also { it.text = value; root.addView(it) }
        label("SeedEdge Runtime - Phase 18C.2")
        label("Device: ${android.os.Build.MANUFACTURER} ${android.os.Build.MODEL} / API ${android.os.Build.VERSION.SDK_INT} / ${android.os.Build.SUPPORTED_ABIS.firstOrNull() ?: "unknown"}")
        label("NCNN: ${NativeRuntime.nativeGetNcnnInfo()}")
        label("Model root: models/phase18c")
        status = TextView(this).also { it.text = "Model: NOT_CHECKED\nCheckpoint: NOT_CHECKED"; root.addView(it) }
        progress = ProgressBar(this, null, android.R.attr.progressBarStyleHorizontal).also { it.max = 1000; root.addView(it) }
        root.addView(Button(this).apply { text = "Inspect model"; setOnClickListener { inspectModel() } })
        root.addView(Button(this).apply { text = "Refresh checkpoint"; setOnClickListener { inspectCheckpoint() } })
        val modes = RadioGroup(this).apply {
            decoderRadio = RadioButton(context).apply { text = "Decoder Only (checkpoint)"; id = 1; isEnabled = false }
            addView(decoderRadio)
            addView(RadioButton(context).apply { text = "Full 4-frame demo"; id = 2; isChecked = true })
            addView(RadioButton(context).apply { text = "Synthetic test"; id = 3 })
        }
        root.addView(modes)
        val buttons = LinearLayout(this)
        runButton = Button(this).apply { text = "Run"; setOnClickListener { runSelected(modes.checkedRadioButtonId) } }
        cancelButton = Button(this).apply { text = "Stop task"; isEnabled = false; setOnClickListener { NativeRuntime.nativeRequestCancel(); append("cancel requested") } }
        buttons.addView(runButton); buttons.addView(cancelButton); root.addView(buttons)
        log = TextView(this).apply { setTextIsSelectable(true); root.addView(this, LinearLayout.LayoutParams(-1, 0, 1f)) }
        setContentView(root)
        restoreProgress()
    }

    private fun inspectModel() { runButton.isEnabled = false; executor.execute { val result = NativeRuntime.nativeInspectModel(modelRoot.absolutePath); handler.post { status.text = "Model: $result"; runButton.isEnabled = true } } }
    private fun inspectCheckpoint() { executor.execute { val result = NativeRuntime.nativeInspectCheckpoints(outputRoot.absolutePath); handler.post { status.append("\nCheckpoint: $result"); decoderRadio.isEnabled = JSONObject(result).optBoolean("checkpoint_verified", false) } } }
    private fun runSelected(mode: Int) {
        if (!running.compareAndSet(false, true)) return
        runButton.isEnabled = false; cancelButton.isEnabled = true; startPolling()
        executor.execute {
            val result = try {
                when (mode) {
                    3 -> NativeRuntime.nativeRunProgressSimulation()
                    1 -> NativeRuntime.nativeRunDecoderOnly(modelRoot.absolutePath, outputRoot.absolutePath, outputRoot.absolutePath, "{\"mode\":\"decoder_only\"}")
                    else -> MinimalVideoDemo.run(this)
                }
            } catch (t: Throwable) { "{\"overall_pass\":false,\"error_code\":\"UI_NATIVE\",\"error_message\":\"${t.message}\"}" }
            handler.post { append(result); running.set(false); runButton.isEnabled = true; cancelButton.isEnabled = false; stopPolling() }
        }
    }
    private fun startPolling() { poll = object : Runnable { override fun run() { try { val p = JSONObject(NativeRuntime.nativeGetProgressJson()); val block = p.optInt("dit_block_index", 0); progress.progress = (block * 1000 / 32).coerceIn(0, 1000); status.text = "${p.optString("phase")} / ${p.optString("stage_name")}\n${p.optString("message")}\nBlock $block/32  Frame ${p.optInt("output_frame_count", 0)}/4"; append("${p.optString("phase")} ${p.optString("stage_name")}") } catch (_: Throwable) {} ; if (running.get()) handler.postDelayed(this, 750) } }; handler.post(poll!!)
    }
    private fun stopPolling() { poll?.let(handler::removeCallbacks); poll = null }
    private fun restoreProgress() { executor.execute { val value = NativeRuntime.nativeGetProgressJson(); handler.post { append("last progress: $value") } } }
    private fun append(value: String) { log.setText((log.text.toString() + "\n" + value).takeLast(16000), TextView.BufferType.NORMAL) }
    override fun onDestroy() { stopPolling(); executor.shutdownNow(); super.onDestroy() }
}
