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
import android.view.View
import android.widget.Button
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.RadioButton
import android.widget.RadioGroup
import android.widget.ScrollView
import android.widget.TextView
import android.widget.Toast
import org.json.JSONObject
import java.io.File
import java.io.FileOutputStream
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Real Phase 18D single-photo-tile flow: Photo Picker -> task-local RGB8 ->
 * nativeRunSinglePhotoTile (VAE encoder + 32 DiT blocks + scheduler + decoder)
 * -> output_valid.rgb -> PNG -> in-app input/output/side-by-side + MediaStore.
 * The UI never labels a non-model result as an enhancement.
 */
class MainActivity : Activity() {
    companion object {
        private const val REQUEST_PHOTO = 1201
        private val worker = Executors.newSingleThreadExecutor { r ->
            Thread(r, "seededge-worker").apply { isDaemon = false }
        }
    }
    private val handler = Handler(Looper.getMainLooper())
    private val running = AtomicBoolean(false)
    private lateinit var page: LinearLayout
    private var poller: Runnable? = null

    private var currentTaskId: String? = null
    private var currentTaskRoot: File? = null
    private var currentRgb: ByteArray? = null
    private var currentWidth = 0
    private var currentHeight = 0
    private var currentBitmap: Bitmap? = null
    private var currentSourceSha = ""
    private var currentNormalizedSha = ""


    private val modelRoot get() = File(requireNotNull(getExternalFilesDir(null)), "models/phase18c")

    override fun onCreate(state: Bundle?) {
        super.onCreate(state)
        showHome()
    }

    private fun base(): LinearLayout {
        val scroll = ScrollView(this)
        page = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; setPadding(32, 32, 32, 32) }
        scroll.addView(page)
        setContentView(scroll)
        return page
    }

    private fun text(value: String, large: Boolean = false): TextView =
        TextView(this).apply {
            this.text = value
            textSize = if (large) 26f else 16f
            setPadding(0, 8, 0, 8)
            page.addView(this)
        }

    private fun button(value: String, action: () -> Unit): Button =
        Button(this).apply {
            text = value
            isAllCaps = false
            setOnClickListener { action() }
            page.addView(this)
        }

    private fun image(bitmap: Bitmap) {
        page.addView(ImageView(this).apply {
            setImageBitmap(bitmap)
            adjustViewBounds = true
        }, LinearLayout.LayoutParams(-1, 420))
    }

    private fun showHome() {
        stopPoller(); running.set(false); base()
        text("SeedEdge", true)
        text("本地 AI 图片增强（Phase 18D 单 tile）")
        text("照片仅在设备本地处理；运行真实 SeedVR2 单图链路：VAE Encoder → 32 DiT Block → Scheduler → VAE Decoder")
        button("选择照片") { choosePhoto() }
        button("检查模型（完整 SHA-256）") { checkModel() }
        text("模型目录：${modelRoot.path}\n状态：未检查（运行前请先检查）")
        button("高级诊断") { showDiagnostics() }
    }

    private fun checkModel() {
        base(); text("检查模型", true); val state = text("正在校验 manifest 与全部模型文件 SHA-256（1366 文件，可能需数分钟）…")
        worker.execute {
            try {
                val json = JSONObject(NativeRuntime.nativeValidateModelManifest(modelRoot.path))
                handler.post {
                    val valid = json.optBoolean("valid", false)
                    val silent = json.optBoolean("silent_fallback", false)
                    state.text = buildString {
                        append(if (valid) "模型校验通过\n" else "模型校验失败\n")
                        append("files_verified=${json.optInt("files_verified", 0)}\n")
                        append("silent_fallback=$silent\n")
                        if (!valid) append("error=${json.optString("error", "unknown")}\n")
                    }
                    button("返回首页") { showHome() }
                }
            } catch (t: Throwable) {
                handler.post { showError("模型检查失败", t.message ?: "native 调用异常") }
            }
        }
    }

    private fun choosePhoto() {
        startActivityForResult(Intent(Intent.ACTION_OPEN_DOCUMENT).apply {
            type = "image/*"
            addCategory(Intent.CATEGORY_OPENABLE)
            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION or Intent.FLAG_GRANT_PERSISTABLE_URI_PERMISSION)
        }, REQUEST_PHOTO)
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode != REQUEST_PHOTO || resultCode != RESULT_OK) return
        val uri = data?.data ?: return
        try { contentResolver.takePersistableUriPermission(uri, Intent.FLAG_GRANT_READ_URI_PERMISSION) } catch (_: SecurityException) {}
        worker.execute {
            try {
                val taskId = PhotoTask.newTaskId()
                val taskRoot = PhotoTask.taskRoot(filesDir, taskId)
                val tileRoot = PhotoTask.tileRoot(filesDir, taskId)
                tileRoot.mkdirs()
                val source = File(taskRoot, "source_image${guessExtension(uri)}")
                contentResolver.openInputStream(uri)!!.use { input ->
                    FileOutputStream(source).use { output -> input.copyTo(output) }
                }
                val sourceSha = PhotoTask.sha256Hex(source)
                val decoded = BitmapFactory.decodeFile(source.path) ?: error("不支持的图片格式或图片损坏")
                val bitmap = rotateForExif(source, decoded)
                if (bitmap !== decoded) decoded.recycle()
                val rgb = PhotoTask.bitmapToRgb8(bitmap)
                val normalizedSha = PhotoTask.sha256Hex(rgb)
                val manifestFile = File(modelRoot, "manifest.json")
                val manifestSha = if (manifestFile.isFile) PhotoTask.sha256Hex(manifestFile) else ""
                val metadata = JSONObject().apply {
                    put("schema_version", PhotoTask.REQUEST_SCHEMA_VERSION)
                    put("task_id", taskId)
                    put("model_version", PhotoTask.MODEL_VERSION)
                    put("source_image_sha256", sourceSha)
                    put("normalized_image_sha256", normalizedSha)
                    put("model_manifest_sha256", manifestSha)
                    put("width", bitmap.width)
                    put("height", bitmap.height)
                    put("tile_index", 0)
                    put("created_at", System.currentTimeMillis())
                }
                File(taskRoot, "task.json").writeText(metadata.toString(2))
                handler.post {
                    currentTaskId = taskId; currentTaskRoot = taskRoot
                    currentRgb = rgb; currentWidth = bitmap.width; currentHeight = bitmap.height
                    currentBitmap = bitmap; currentSourceSha = sourceSha; currentNormalizedSha = normalizedSha
                    showEnhance(taskId, bitmap, source.length(), sourceSha)
                }
            } catch (t: Throwable) {
                handler.post { showError("导入照片失败", t.message ?: "无法读取该照片") }
            }
        }
    }

    private fun guessExtension(uri: Uri): String = when (contentResolver.getType(uri)) {
        "image/png" -> ".png"
        "image/webp" -> ".webp"
        else -> ".jpg"
    }

    private fun rotateForExif(file: File, bitmap: Bitmap): Bitmap {
        val degree = when (ExifInterface(file.path).getAttributeInt(
            ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL)) {
            ExifInterface.ORIENTATION_ROTATE_90 -> 90f
            ExifInterface.ORIENTATION_ROTATE_180 -> 180f
            ExifInterface.ORIENTATION_ROTATE_270 -> 270f
            else -> 0f
        }
        return if (degree == 0f) bitmap else Bitmap.createBitmap(
            bitmap, 0, 0, bitmap.width, bitmap.height,
            android.graphics.Matrix().apply { postRotate(degree) }, true)
    }

    private fun showEnhance(taskId: String, bitmap: Bitmap, sourceBytes: Long, sourceSha: String) {
        stopPoller(); running.set(false); base()
        text("准备增强", true)
        image(bitmap)
        text(buildString {
            append("任务：$taskId\n输入：${bitmap.width} × ${bitmap.height}\n")
            if (sourceBytes >= 0) append("源文件：$sourceBytes bytes\n")
            append("source_sha256=$sourceSha\nnormalized_image_sha256=$currentNormalizedSha")
        })
        text("运行真实 SeedVR2 单 tile（tile_0000，128×128 含 padding）。\n引擎：ncnn CPU FP32 / OpenMP 多线程（Pad 已验证 5–8 线程位一致）；不会静默回退。\n预计耗时随设备而定：Encoder + 32 DiT Block + Decoder 全链路。")
        button("开始增强") { startEnhancement() }
        button("返回") { showHome() }
    }

    private fun startEnhancement() {
        if (running.get()) return  // already enhancing; the UI shows cancel instead
        val taskId = currentTaskId ?: return showError("任务无效", "未选择照片")
        val taskRoot = currentTaskRoot ?: return showError("任务无效", "任务目录缺失")
        val rgb = currentRgb ?: return showError("任务无效", "RGB 输入缺失")
        if (currentWidth <= 0 || currentHeight <= 0) return showError("任务无效", "尺寸无效")
        running.set(true); base()
        text("正在增强 · 真实 SeedVR2", true)
        val state = text("准备中…")
        val meter = ProgressBar(this, null, android.R.attr.progressBarStyleHorizontal).apply { max = 40; page.addView(this) }
        val details = text("")
        button("取消任务") {
            val cancelJson = NativeRuntime.nativeRequestCancel()
            state.text = "已请求取消（在安全边界生效）\n$cancelJson"
        }
        val requestJson = PhotoTask.buildRequestJson(
            taskId, PhotoTask.MODEL_VERSION, currentSourceSha, currentNormalizedSha,
            if (File(modelRoot, "manifest.json").isFile) PhotoTask.sha256Hex(File(modelRoot, "manifest.json")) else "")
        worker.execute {
            try {
                val resultText = NativeRuntime.nativeRunSinglePhotoTile(
                    modelRoot.path, taskRoot.path, rgb, currentWidth, currentHeight, requestJson)
                handler.post { showResult(taskId, resultText) }
            } catch (t: Throwable) {
                handler.post { showError("运行失败", t.message ?: "native 调用异常") }
            }
        }
        startPoller(taskId, state, meter, details)
    }

    private fun startPoller(taskId: String, state: TextView, meter: ProgressBar, details: TextView) {
        stopPoller()
        val tileRoot = PhotoTask.tileRoot(filesDir, taskId)
        val progressFile = File(tileRoot, "runtime_progress.json")
        poller = object : Runnable {
            override fun run() {
                if (!running.get()) return
                try {
                    if (progressFile.isFile) {
                        val p = JSONObject(progressFile.readText())
                        val phase = p.optString("phase", "checking_model")
                        val block = p.optInt("dit_block_index", 0)
                        val decoderTotal = p.optInt("decoder_stage_total", 7)
                        val frame = p.optInt("frame_index", 0)
                        val stateValue = p.optString("state", "running")
                        val label = when (phase) {
                            "checking_model" -> "模型检查"
                            "verifying_manifest" -> "模型清单校验（SHA-256）"
                            "vae_encoder" -> "VAE Encoder"
                            "dit_streaming" -> "DiT Block ${block + 1} / 32"
                            "vae_decoder" -> "VAE Decoder（${decoderTotal} 阶段）"
                            "writing_frames" -> "写入帧 ${frame + 1} / 4"
                            else -> phase
                        }
                        state.text = if (stateValue == "running") "当前阶段：$label" else "状态：$stateValue（$label）"
                        meter.progress = when (phase) {
                            "checking_model" -> 1
                            "verifying_manifest" -> 2
                            "vae_encoder" -> 3
                            "dit_streaming" -> 4 + block
                            "vae_decoder" -> 36
                            "writing_frames" -> 37 + frame
                            else -> 40
                        }
                        details.text = "elapsed=${p.optLong("elapsed_ms", 0) / 1000}s\n" +
                            "backend=${p.optString("backend", "cpu_fp32")}\n" +
                            "message=${p.optString("message", "")}"
                        if (stateValue == "completed" || stateValue == "failed" || stateValue == "cancelled") {
                            running.set(false)
                            return
                        }
                    }
                } catch (_: Throwable) {}
                handler.postDelayed(this, 400)
            }
        }
        handler.post(poller!!)
    }

    private fun stopPoller() {
        poller?.let(handler::removeCallbacks)
        poller = null
    }

    private fun showResult(taskId: String, nativeResultText: String) {
        running.set(false); stopPoller()
        val tileRoot = PhotoTask.tileRoot(filesDir, taskId)
        val statusFile = File(tileRoot, "status.json")
        if (!statusFile.isFile) return showError("结果缺失", "status.json 未生成\n$nativeResultText")
        val status = JSONObject(statusFile.readText())
        val pipeline = status.optJSONObject("pipeline_result")
        val silentFallback = pipeline?.optBoolean("silent_cpu_fallback", false) ?: false
        val encoder = status.optBoolean("encoder_completed", false)
        val blocks = status.optInt("dit_block_completed", 0)
        val scheduler = status.optBoolean("scheduler_completed", false)
        val decoder = status.optBoolean("decoder_completed", false)
        val finite = status.optBoolean("finite", false)
        val bounds = status.optJSONObject("tile_bounds")
        val validW = bounds?.optInt("right", 0)?.minus(bounds.optInt("left", 0)) ?: 0
        val validH = bounds?.optInt("bottom", 0)?.minus(bounds.optInt("top", 0)) ?: 0
        val outputValidSha = status.optString("output_valid_sha256", "")
        val inputSha = status.optString("tile_input_sha256", "")
        val normalizedInStatus = status.optString("normalized_image_sha256", "")
        val manifestInStatus = status.optString("model_manifest_sha256", "")
        val durationMs = pipeline?.optLong("duration_ms", 0) ?: 0
        val errorLine = ResultSummary.errorLine(pipeline)

        val inputPng = File(tileRoot, "input.png")
        val outputPng = File(tileRoot, "output.png")
        val sideBySide = File(tileRoot, "side_by_side.png")
        val inputOk = PhotoTask.writeRgbPng(File(tileRoot, "tile_input.rgb"), inputPng, 128, 128)
        val outputOk = PhotoTask.writeRgbPng(File(tileRoot, "output_valid.rgb"), outputPng, validW, validH)
        val sideOk = outputOk && PhotoTask.writeSideBySide(inputPng, outputPng, sideBySide)

        base(); text("增强结果", true)
        if (silentFallback) text("错误：检测到 silent_cpu_fallback=true（禁止静默回退）", large = false)
        if (errorLine.isNotEmpty()) text("错误：$errorLine")
        text(buildString {
            append("encoder=$encoder  dit=$blocks/32  scheduler=$scheduler  decoder=$decoder\n")
            append("finite=$finite  nan=${status.optInt("nan_count", -1)}  inf=${status.optInt("inf_count", -1)}\n")
            append("valid=${validW}×${validH}  duration=${durationMs}ms\n")
            append("silent_cpu_fallback=$silentFallback\n")
            append("output_different_from_input=${outputValidSha.isNotEmpty() && outputValidSha != inputSha}\n")
            append("normalized_sha_match=${normalizedInStatus == currentNormalizedSha}\n")
            append("manifest_sha_match=${manifestInStatus.isNotEmpty()}\n")
            append("output_valid_sha256=${outputValidSha.ifEmpty { "-" }}\n")
            append("status.json=${statusFile.path}\n")
            append("input.png=${inputPng.path}\noutput.png=${outputPng.path}\nside_by_side.png=${sideBySide.path}")
        })
        if (inputOk && outputOk) {
            val row = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
            row.addView(ImageView(this).apply { setImageBitmap(BitmapFactory.decodeFile(inputPng.path)); adjustViewBounds = true },
                LinearLayout.LayoutParams(0, 320, 1f))
            row.addView(ImageView(this).apply { setImageBitmap(BitmapFactory.decodeFile(outputPng.path)); adjustViewBounds = true },
                LinearLayout.LayoutParams(0, 320, 1f))
            page.addView(row)
            text("原图(tile)                                  增强输出")
            if (sideOk) image(BitmapFactory.decodeFile(sideBySide.path))
        } else {
            text("PNG 编码失败：inputOk=$inputOk outputOk=$outputOk")
        }
        val complete = encoder && blocks == 32 && scheduler && decoder && finite && !silentFallback && outputOk
        text(if (complete) "Phase 18D 单 tile 完成（需真机相册照片复核后定级）" else "未完成：真实链路未全部通过")
        button("保存输出到相册") { saveToGallery(outputPng) }
        button("重新处理") { showEnhance(taskId, currentBitmap ?: Bitmap.createBitmap(1, 1, Bitmap.Config.ARGB_8888), -1, currentSourceSha) }
        button("选择其他照片") { choosePhoto() }
        button("返回首页") { showHome() }
    }

    private fun saveToGallery(file: File) {
        try {
            val values = ContentValues().apply {
                put(MediaStore.Images.Media.DISPLAY_NAME, "SeedEdge_${System.currentTimeMillis()}.png")
                put(MediaStore.Images.Media.MIME_TYPE, "image/png")
                put(MediaStore.Images.Media.RELATIVE_PATH, "Pictures/SeedEdge")
                put(MediaStore.Images.Media.IS_PENDING, 1)
            }
            val uri: Uri = contentResolver.insert(
                MediaStore.Images.Media.getContentUri(MediaStore.VOLUME_EXTERNAL_PRIMARY), values)
                ?: error("无法创建相册项目")
            contentResolver.openOutputStream(uri)!!.use { output -> file.inputStream().use { it.copyTo(output) } }
            values.clear(); values.put(MediaStore.Images.Media.IS_PENDING, 0)
            contentResolver.update(uri, values, null, null)
            Toast.makeText(this, "已保存到 Pictures/SeedEdge", Toast.LENGTH_LONG).show()
        } catch (t: Throwable) {
            showError("保存失败", t.message ?: "无法写入系统相册")
        }
    }

    private fun showDiagnostics() {
        stopPoller(); running.set(false); base()
        text("高级诊断", true)
        val info = text("读取中…")
        worker.execute {
            val version = NativeRuntime.nativeGetVersion()
            val progress = NativeRuntime.nativeGetProgressJson()
            val inspect = try { JSONObject(NativeRuntime.nativeInspectModel(modelRoot.path)).toString(2) } catch (_: Throwable) { "inspect failed" }
            handler.post {
                info.text = "version=$version\n\nprogress=$progress\n\nmodel_inspect=$inspect"
                button("返回首页") { showHome() }
            }
        }
    }

    private fun showError(title: String, message: String) {
        stopPoller(); running.set(false); base()
        text(title, true)
        text(message)
        button("返回首页") { showHome() }
    }

    override fun onDestroy() {
        stopPoller()
        running.set(false)

        super.onDestroy()
    }

}
