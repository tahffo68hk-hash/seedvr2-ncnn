package com.seededge.runtime

import android.app.Activity
import android.content.ContentValues
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.media.ExifInterface
import android.net.Uri
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.provider.MediaStore
import android.view.Gravity
import android.view.View
import android.widget.*
import java.io.File
import java.io.FileOutputStream
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

/** User-facing shell.  It never labels its local UI sample as a SeedVR2 result. */
class MainActivity : Activity() {
    private val worker = Executors.newSingleThreadExecutor()
    private val handler = Handler(Looper.getMainLooper())
    private val running = AtomicBoolean(false)
    private lateinit var page: LinearLayout
    private var selected: File? = null
    private var original: Bitmap? = null
    private var demo: Runnable? = null
    private var startedAt = 0L
    private val workDir get() = File(filesDir, "image_workflow").apply { mkdirs() }
    private val outputDir get() = File(workDir, "outputs").apply { mkdirs() }
    private val modelRoot get() = File(requireNotNull(getExternalFilesDir(null)), "models/phase18c")

    override fun onCreate(state: Bundle?) { super.onCreate(state); showHome() }
    private fun base(): LinearLayout {
        val scroll = ScrollView(this); page = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; setPadding(32,32,32,32) }
        scroll.addView(page); setContentView(scroll); return page
    }
    private fun text(value: String, large: Boolean = false) = TextView(this).apply { this.text = value; textSize = if (large) 26f else 16f; setPadding(0, 8, 0, 8); page.addView(this) }
    private fun button(value: String, action: () -> Unit) = Button(this).apply { text = value; isAllCaps = false; setOnClickListener { action() }; page.addView(this) }
    private fun image(bitmap: Bitmap) = ImageView(this).apply { setImageBitmap(bitmap); adjustViewBounds = true; page.addView(this, LinearLayout.LayoutParams(-1, 420)) }

    private fun showHome() {
        cancelDemo(); base(); text("SeedEdge", true); text("本地 AI 图片增强"); text("照片仅在设备本地处理")
        button("选择照片") { choosePhoto() }
        text("模型状态：${if (modelRoot.exists()) "已就绪（运行时仍部分完成）" else "缺失"}")
        text("运行引擎：ncnn CPU FP32\n性能模式：平衡（5线程）\n真实 SeedVR2 推理：开发中")
        button("高级诊断") { showDiagnostics() }
    }
    private fun choosePhoto() {
        startActivityForResult(Intent(Intent.ACTION_OPEN_DOCUMENT).apply {
            type = "image/*"; addCategory(Intent.CATEGORY_OPENABLE); addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION or Intent.FLAG_GRANT_PERSISTABLE_URI_PERMISSION)
        }, REQUEST_PHOTO)
    }
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode != REQUEST_PHOTO || resultCode != RESULT_OK) return
        val uri = data?.data ?: return
        try { contentResolver.takePersistableUriPermission(uri, Intent.FLAG_GRANT_READ_URI_PERMISSION) } catch (_: SecurityException) {}
        worker.execute {
            try {
                val name = "input_${System.currentTimeMillis()}.jpg"; val copy = File(workDir, name)
                contentResolver.openInputStream(uri)!!.use { input -> FileOutputStream(copy).use { input.copyTo(it) } }
                val decoded = BitmapFactory.decodeFile(copy.path) ?: error("不支持的图片格式")
                original = rotateForExif(copy, decoded); selected = copy
                handler.post { showEnhance() }
            } catch (t: Throwable) { handler.post { showError("格式不支持", t.message ?: "无法读取该照片") } }
        }
    }
    private fun rotateForExif(file: File, bitmap: Bitmap): Bitmap {
        val degree = when (ExifInterface(file.path).getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL)) {
            ExifInterface.ORIENTATION_ROTATE_90 -> 90f; ExifInterface.ORIENTATION_ROTATE_180 -> 180f; ExifInterface.ORIENTATION_ROTATE_270 -> 270f; else -> 0f
        }; return if (degree == 0f) bitmap else Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, android.graphics.Matrix().apply { postRotate(degree) }, true)
    }
    private fun showEnhance() {
        val source = selected ?: return showError("未选择照片", "请先选择照片")
        val bitmap = original ?: return showError("格式不支持", "无法解码")
        base(); text("准备增强", true); image(bitmap); text("${source.name}\n输入：${bitmap.width} × ${bitmap.height}\n文件大小：${source.length()} bytes")
        val modes = RadioGroup(this); listOf("平衡（5线程）", "低温（2线程）", "快速（8线程，实验性）").forEachIndexed { index, label -> modes.addView(RadioButton(this).apply { text = label; isChecked = index == 0 }) }; page.addView(modes)
        text("真实 SeedVR2 单图推理尚未就绪。开始后仅运行清晰标记的 UI DEMO，不会把测试输出称为增强结果。")
        button("开始增强（UI DEMO）") { startDemo((if (modes.checkedRadioButtonId == View.NO_ID) 5 else if (modes.indexOfChild(modes.findViewById(modes.checkedRadioButtonId)) == 1) 2 else 5)) }
        button("返回") { showHome() }
    }
    private fun startDemo(threads: Int) {
        running.set(true); startedAt = System.currentTimeMillis(); base(); text("正在增强 · UI DEMO", true); text("此页面展示界面流程，不执行 SeedVR2 推理。")
        val state = text("准备输入"); val meter = ProgressBar(this, null, android.R.attr.progressBarStyleHorizontal).apply { max = 40; page.addView(this) }
        val details = text("线程数：$threads\n内存：正在估算\n温度：正在估算")
        button("停止任务") { cancelDemo(); showError("任务取消", "UI DEMO 已安全停止；未启动真实模型。") }
        val stages = (listOf("准备输入", "模型检查", "VAE Encoder") + (1..32).map { "DiT Block $it / 32" } + (1..7).map { "VAE Decoder $it / 7" } + listOf("保存结果", "完成"))
        var index = 0
        demo = object : Runnable { override fun run() {
            if (!running.get()) return
            val label = stages[index]; state.text = "当前阶段：$label\nUI DEMO"
            meter.progress = index + 1; details.text = "已运行时间：${(System.currentTimeMillis()-startedAt)/1000}s\n线程数：$threads\nEncoder / DiT / Decoder 进度为界面演示"
            if (++index >= stages.size) { running.set(false); handler.postDelayed({ showResult() }, 250) } else handler.postDelayed(this, 180)
        } }; handler.post(demo!!)
    }
    private fun cancelDemo() { running.set(false); demo?.let(handler::removeCallbacks); demo = null }
    private fun showResult() {
        val bitmap = original ?: return showError("未选择照片", "没有可展示的测试输入")
        val test = File(outputDir, "SeedEdge_UI_TEST_${System.currentTimeMillis()}.png")
        FileOutputStream(test).use { bitmap.compress(Bitmap.CompressFormat.PNG, 100, it) }
        base(); text("测试结果", true); text("测试样例：输入照片的本地预览副本，非完整 SeedVR2 增强。")
        val row = LinearLayout(this); row.orientation = LinearLayout.HORIZONTAL
        val left = ImageView(this).apply { setImageBitmap(bitmap); adjustViewBounds = true }; val right = ImageView(this).apply { setImageBitmap(bitmap); adjustViewBounds = true }
        row.addView(left, LinearLayout.LayoutParams(0, 300, 1f)); row.addView(right, LinearLayout.LayoutParams(0, 300, 1f)); page.addView(row)
        text("原图                                      测试结果")
        button("保存测试结果到相册") { saveToGallery(test) }; button("重新处理") { showEnhance() }; button("选择其他照片") { choosePhoto() }
    }
    private fun saveToGallery(file: File) {
        try {
            val values = ContentValues().apply { put(MediaStore.Images.Media.DISPLAY_NAME, "SeedEdge_${System.currentTimeMillis()}.png"); put(MediaStore.Images.Media.MIME_TYPE, "image/png"); put(MediaStore.Images.Media.RELATIVE_PATH, "Pictures/SeedEdge"); put(MediaStore.Images.Media.IS_PENDING, 1) }
            val uri: Uri = contentResolver.insert(MediaStore.Images.Media.getContentUri(MediaStore.VOLUME_EXTERNAL_PRIMARY), values) ?: error("无法创建相册项目")
            contentResolver.openOutputStream(uri)!!.use { output -> file.inputStream().use { it.copyTo(output) } }; values.clear(); values.put(MediaStore.Images.Media.IS_PENDING, 0); contentResolver.update(uri, values, null, null)
            Toast.makeText(this, "已保存到 Pictures/SeedEdge", Toast.LENGTH_LONG).show()
        } catch (t: Throwable) { showError("保存失败", t.message ?: "无法写入系统相册") }
    }
    private fun showDiagnostics() { base(); text("高级诊断", true); text("App Interface = READY（界面能力）\nModel Runtime = PARTIAL\nImage Enhancement = NOT READY"); button("返回") { showHome() } }
    private fun showError(title: String, message: String) { cancelDemo(); base(); text(title, true); text(message); button("返回首页") { showHome() } }
    override fun onDestroy() { cancelDemo(); worker.shutdownNow(); super.onDestroy() }
    companion object { const val REQUEST_PHOTO = 1201 }
}
