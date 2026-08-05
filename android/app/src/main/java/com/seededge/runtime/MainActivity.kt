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
    private var poll: Runnable? = null
    private val modelRoot get() = File(requireNotNull(getExternalFilesDir(null)), "models/phase18c")
    private val outputRoot get() = File(requireNotNull(getExternalFilesDir(null)), "output/phase18c")

    override fun onCreate(state: Bundle?) {
        super.onCreate(state)
        val root = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; setPadding(24,24,24,24) }
        fun label(s: String) = TextView(this).also { it.text = s; root.addView(it) }
        label("SeedEdge Runtime · Phase 18C.1")
        label("Device: ${android.os.Build.MANUFACTURER} ${android.os.Build.MODEL} · API ${android.os.Build.VERSION.SDK_INT} · ${android.os.Build.SUPPORTED_ABIS.firstOrNull() ?: "unknown"}")
        label("GPU/Vulkan: ${NativeRuntime.nativeGetNcnnInfo()}")
        label("Model root: models/phase18c")
        status = TextView(this).also { it.text = "Model: NOT_CHECKED\nCheckpoint: NOT_CHECKED"; root.addView(it) }
        progress = ProgressBar(this, null, android.R.attr.progressBarStyleHorizontal).also { it.max=1000; root.addView(it) }
        val inspect = Button(this).apply { text="检查模型"; setOnClickListener { inspectModel() } }; root.addView(inspect)
        val checkpoint = Button(this).apply { text="刷新 Checkpoint"; setOnClickListener { inspectCheckpoint() } }; root.addView(checkpoint)
        val modes = RadioGroup(this).apply {
            addView(RadioButton(context).apply { text="Decoder Only（推荐）"; id=1; isEnabled=false })
            addView(RadioButton(context).apply { text="完整 4 帧 Demo"; id=2; isChecked=true })
            addView(RadioButton(context).apply { text="Synthetic Test"; id=3 })
        }; root.addView(modes)
        val buttons = LinearLayout(this)
        runButton = Button(this).apply { text="运行"; setOnClickListener { runSelected(modes.checkedRadioButtonId) } }; buttons.addView(runButton)
        cancelButton = Button(this).apply { text="停止任务"; isEnabled=false; setOnClickListener { NativeRuntime.nativeRequestCancel(); append("cancel requested") } }; buttons.addView(cancelButton)
        root.addView(buttons)
        log = TextView(this).apply { setTextIsSelectable(true); root.addView(this, LinearLayout.LayoutParams(-1,0,1f)) }
        setContentView(root)
        restoreProgress()
    }

    private fun inspectModel() { runButton.isEnabled=false; executor.execute { val r=NativeRuntime.nativeInspectModel(modelRoot.absolutePath); handler.post { status.text="Model: "+r; runButton.isEnabled=true } } }
    private fun inspectCheckpoint() { executor.execute { val r=NativeRuntime.nativeInspectCheckpoints(outputRoot.absolutePath); handler.post { status.append("\nCheckpoint: "+r) } } }
    private fun runSelected(mode: Int) {
        if (!running.compareAndSet(false,true)) return
        runButton.isEnabled=false; cancelButton.isEnabled=true; startPolling()
        executor.execute {
            val result = try {
                if (mode==3) NativeRuntime.nativeRunSyntheticCpuTest()
                else MinimalVideoDemo.run(this)
            } catch (t: Throwable) { "{\"overall_pass\":false,\"error_code\":\"UI_NATIVE\",\"error_message\":\"${t.message}\"}" }
            handler.post { append(result); running.set(false); runButton.isEnabled=true; cancelButton.isEnabled=false; stopPolling() }
        }
    }
    private fun startPolling() { poll = object:Runnable { override fun run(){ try { val p=JSONObject(NativeRuntime.nativeGetProgressJson()); val done=p.optInt("output_frame_count",0); progress.progress=(p.optInt("dit_block_index",0)*1000/32).coerceIn(0,1000); status.text="${p.optString("phase")} · ${p.optString("stage_name")}\n${p.optString("message")}\nBlock ${p.optInt("dit_block_index")}/32 · Frame $done/4"; append("${p.optString("phase")} ${p.optString("stage_name")}") } catch(_:Throwable){}; if(running.get()) handler.postDelayed(this,750) } }; handler.post(poll!!)
    }
    private fun stopPolling(){ poll?.let(handler::removeCallbacks); poll=null }
    private fun restoreProgress(){ executor.execute { val p=NativeRuntime.nativeGetProgressJson(); handler.post { append("last progress: $p") } } }
    private fun append(s:String){ val text=(log.text.toString()+"\n"+s).takeLast(16000); log.setText(text, TextView.BufferType.NORMAL) }
    override fun onDestroy(){ stopPolling(); executor.shutdownNow(); super.onDestroy() }
}
