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
 * Real Phase 18D full-photo flow: Photo Picker -> task-local RGB8 ->
 * nativeRunFullPhoto (multi-tile VAE encoder + 32 DiT blocks + scheduler +
 * decoder + weighted fusion) -> task-level PNGs -> in-app preview + MediaStore.
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
        text("本地 AI 图片增强（Phase 18D full-photo）")
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
                File(taskRoot, "normalized.rgb").writeBytes(rgb)
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
        val plan = PhotoTask.planTiles(bitmap.width, bitmap.height, availableDiskBytes = filesDir.usableSpace)
        text(buildString {
            append("任务：$taskId\n输入：${bitmap.width} × ${bitmap.height}\n")
            if (sourceBytes >= 0) append("源文件：$sourceBytes bytes\n")
            append("source_sha256=$sourceSha\nnormalized_image_sha256=$currentNormalizedSha")
        })
        text(buildString {
            append("切分计划\n")
            append("tile=${plan.tileSize}×${plan.tileSize}，overlap=${plan.overlap}，stride=${plan.stride}\n")
            append("网格=${plan.columns}×${plan.rows}，总计=${plan.tileCount} tiles\n")
            append("Auto → ${plan.selectedMode}")
            if (plan.cohortSize > 0) append(" K=${plan.cohortSize}，cohorts=${plan.cohortCount}")
            append("\n预计耗时≈${PhotoTask.formatDuration(plan.estimate.estimatedRuntimeMs)}")
            append("，峰值空间≈${PhotoTask.formatBytes(plan.estimate.estimatedPeakBytes)}")
            append("，需空闲≈${PhotoTask.formatBytes(plan.estimate.requiredFreeBytes)}")
        })
        text("tile map：# 已完成，> 当前，@ 待处理\n${PhotoTask.tileMapText(plan, 0, 0)}")
        text("运行真实 SeedVR2 全图增强：按计划拆成多个 tile（每个 128×128 含 padding），逐 tile 完整推理后加权融合回原尺寸。\n引擎：ncnn CPU FP32 / OpenMP 多线程；不会静默回退。\n主进度条显示整张图总进度；详情行显示当前 tile 内部阶段。")
        button("开始增强") { startEnhancement() }
        button("返回") { showHome() }
    }

    private fun startEnhancement() {
        if (running.get()) return
        val taskId = currentTaskId ?: return showError("任务无效", "未选择照片")
        val taskRoot = currentTaskRoot ?: return showError("任务无效", "任务目录缺失")
        if (currentRgb == null || currentWidth <= 0 || currentHeight <= 0)
            return showError("任务无效", "RGB 输入或尺寸无效")
        val normalizedRgb = File(taskRoot, "normalized.rgb")
        val plan = PhotoTask.planTiles(currentWidth, currentHeight, availableDiskBytes = filesDir.usableSpace)
        val tileCount = plan.tileCount
        if (currentWidth.toLong() * currentHeight > PhotoTask.MAX_SOURCE_PIXELS ||
            tileCount > PhotoTask.MAX_TILE_COUNT)
            return showError("图片超过安全上限", "pixels=${currentWidth.toLong() * currentHeight}, tiles=$tileCount")
        val required = plan.estimate.requiredFreeBytes
        if (!normalizedRgb.isFile || !PhotoTask.hasStorageCapacity(filesDir.usableSpace, required))
            return showError("存储空间不足", "required=$required available=${filesDir.usableSpace}")

        running.set(true); base()
        text("正在增强 · 真实 SeedVR2", true)
        val state = text("准备中")
        val meter = ProgressBar(this, null, android.R.attr.progressBarStyleHorizontal).apply {
            max = 100; progress = 0; page.addView(this)
        }
        val details = text(buildString {
            append("总体 0% · tile 0/${plan.tileCount}")
            if (plan.cohortSize > 0) append(" · cohort 0/${plan.cohortCount}")
            append("\n${plan.summaryText()}")
            append("\n# 已完成，> 当前，@ 待处理\n${PhotoTask.tileMapText(plan, 0, 0)}")
        })
        button("暂停任务") {
            startService(PhotoEnhanceService.commandIntent(this, PhotoEnhanceService.ACTION_PAUSE))
            state.text = "已请求暂停（在下一个安全阶段边界生效）"
        }
        button("恢复任务") {
            startForegroundService(PhotoEnhanceService.commandIntent(this, PhotoEnhanceService.ACTION_RESUME))
            state.text = "正在从已验证 checkpoint 恢复"
        }
        button("取消任务") {
            startService(PhotoEnhanceService.commandIntent(this, PhotoEnhanceService.ACTION_CANCEL))
            state.text = "已请求取消（不可恢复）"
        }
        val manifest = File(modelRoot, "manifest.json")
        val requestJson = PhotoTask.buildRequestJson(taskId, PhotoTask.MODEL_VERSION,
            currentSourceSha, currentNormalizedSha,
            if (manifest.isFile) PhotoTask.sha256Hex(manifest) else "")
        startForegroundService(PhotoEnhanceService.startIntent(this, taskId, taskRoot, modelRoot,
            normalizedRgb, currentWidth, currentHeight, requestJson))
        startPoller(taskId, taskRoot, state, meter, details)
    }

    @Suppress("unused")
    private fun startEnhancementLegacy() {
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
        val cancelled = AtomicBoolean(false)
        button("取消任务") {
            cancelled.set(true)
            NativeRuntime.nativeRequestCancel()
            state.text = "已请求取消（在下一个安全边界生效）"
        }
        val requestJson = PhotoTask.buildRequestJson(
            taskId, PhotoTask.MODEL_VERSION, currentSourceSha, currentNormalizedSha,
            if (File(modelRoot, "manifest.json").isFile) PhotoTask.sha256Hex(File(modelRoot, "manifest.json")) else "")
        worker.execute {
            try {
                val result = PhotoTask.runFullPhotoWithProgress(
                    modelRoot, taskRoot, rgb, currentWidth, currentHeight, requestJson, cancelled) { stage, completed, total ->
                    handler.post {
                        state.text = "当前阶段：${PhotoTask.fullPhotoStageLabel(stage, completed, total)}"
                        details.text = "stage=$stage  tile=$completed/$total"
                        meter.progress = when (stage) {
                            PhotoTask.FullPhotoStages.PREPARING -> 1
                            PhotoTask.FullPhotoStages.CHECKING_MODEL -> 2
                            PhotoTask.FullPhotoStages.RUNNING_TILE ->
                                3 + (completed * 32 / total.coerceAtLeast(1)).coerceAtMost(32)
                            PhotoTask.FullPhotoStages.FUSING -> 38
                            PhotoTask.FullPhotoStages.ENCODING_PNG -> 39
                            PhotoTask.FullPhotoStages.SAVING -> 40
                            else -> meter.progress
                        }
                    }
                }
                handler.post { showResult(taskId, result) }
            } catch (t: Throwable) {
                handler.post { showError("运行失败", t.message ?: "native 调用异常") }
            }
        }
    }

    private fun startPoller(taskId: String, taskRoot: File, state: TextView, meter: ProgressBar, details: TextView) {
        stopPoller()
        poller = object : Runnable {
            override fun run() {
                if (!running.get()) return
                try {
                    val planned = if (currentWidth > 0 && currentHeight > 0)
                        PhotoTask.planTiles(currentWidth, currentHeight, availableDiskBytes = filesDir.usableSpace)
                    else null
                    var tileCount = planned?.tileCount ?: 0
                    var completed = 0
                    var taskState = "running"
                    val taskStatus = File(taskRoot, "status.json")
                    if (taskStatus.isFile) {
                        val s = JSONObject(taskStatus.readText())
                        taskState = s.optString("state", "running")
                        tileCount = s.optInt("tile_count", tileCount)
                        completed = s.optInt("completed_tiles", 0)
                        if (taskState == "completed") {
                            meter.progress = 100
                            state.text = "全图增强完成（$completed/$tileCount tile）"
                            running.set(false)
                            return
                        }
                        if (taskState == "cancelled" || taskState == "failed") {
                            state.text = if (taskState == "cancelled") "已取消"
                            else "增强失败：${s.optString("error_message", "")}"
                            running.set(false)
                            return
                        }
                    }
                    val tilesDir = File(taskRoot, "tiles")
                    val tileDirs = tilesDir.listFiles()
                        ?.filter { it.isDirectory && it.name.startsWith("tile_") }
                        ?.sortedBy { it.name }
                        ?: emptyList()
                    val progressFile = tileDirs.lastOrNull()?.let { File(it, "runtime_progress.json") }
                    if (progressFile != null && progressFile.isFile) {
                        val p = JSONObject(progressFile.readText())
                        val phase = p.optString("phase", "checking_model")
                        val block = p.optInt("dit_block_index", 0)
                        val decoderTotal = p.optInt("decoder_stage_total", 7)
                        val frame = p.optInt("frame_index", 0)
                        val activeTile = p.optInt("tile_index", completed).coerceAtLeast(completed)
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
                        val percent = PhotoTask.overallProgressPercent(completed, tileCount, phase, block, frame)
                        meter.progress = percent
                        val cohortSize = planned?.cohortSize ?: p.optInt("tile_cohort_size", 0)
                        val cohortCount = planned?.cohortCount ?: if (cohortSize > 0 && tileCount > 0)
                            (tileCount + cohortSize - 1) / cohortSize else 1
                        val cohortIndex = if (cohortSize > 0) activeTile / cohortSize + 1 else 1
                        if (stateValue == "running" || taskState == "running") {
                            state.text = "总体 $percent% · tile $completed/$tileCount · 当前：$label"
                        }
                        details.text = buildString {
                            append("elapsed=${p.optLong("elapsed_ms", 0) / 1000}s\n")
                            append("backend=${p.optString("backend", "cpu_fp32")}\n")
                            append("mode=${p.optString("runtime_mode", planned?.selectedMode ?: "auto")}")
                            if (cohortSize > 0) append(" K=$cohortSize · cohort $cohortIndex/$cohortCount")
                            append("\nblock=${if (phase == "dit_streaming") block + 1 else "-"} / 32")
                            append(" · message=${p.optString("message", "")}")
                            val mapPlan = planned
                            if (mapPlan != null) {
                                append("\n# 已完成，> 当前，@ 待处理\n")
                                append(PhotoTask.tileMapText(mapPlan, completed, activeTile))
                            }
                        }
                    } else if (tileCount > 0) {
                        val percent = PhotoTask.overallProgressPercent(completed, tileCount, "checking_model")
                        meter.progress = percent
                        state.text = "总体 $percent% · tile $completed/$tileCount · 等待 native 进度"
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

    private fun showResult(taskId: String, result: PhotoTask.FullPhotoResult) {
        running.set(false); stopPoller()
        val status = try { JSONObject(result.statusJson.readText()) } catch (_: Throwable) { null }
        base(); text("增强结果", true)
        if (result.silentFallback) text("错误：检测到 silent_cpu_fallback=true（禁止静默回退）", large = false)
        text(buildString {
            append("state=${result.state}\n")
            append("task_id=${result.taskId}\n")
            append("tiles=${result.completedTiles}/${result.tileCount}\n")
            append("尺寸=${result.correctedWidth}×${result.correctedHeight}\n")
            append("silent_fallback=${result.silentFallback}\n")
            append("output_full_sha256=${status?.optString("output_full_sha256", "-") ?: "-"}\n")
            append("status.json=${result.statusJson.path}\n")
            append("input.png=${result.inputPng.path}\noutput.png=${result.outputPng.path}\nside_by_side.png=${result.sideBySidePng.path}")
        })
        if (result.success) {
            val row = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
            row.addView(ImageView(this).apply { setImageBitmap(BitmapFactory.decodeFile(result.inputPng.path)); adjustViewBounds = true },
                LinearLayout.LayoutParams(0, 320, 1f))
            row.addView(ImageView(this).apply { setImageBitmap(BitmapFactory.decodeFile(result.outputPng.path)); adjustViewBounds = true },
                LinearLayout.LayoutParams(0, 320, 1f))
            page.addView(row)
            text("原图                                  增强输出")
            if (result.sideBySidePng.isFile) image(BitmapFactory.decodeFile(result.sideBySidePng.path))
            text("Phase 18D full-photo 完成（需真机相册照片复核后定级）")
            button("保存输出到相册") { saveToGallery(result.outputPng, result) }
        } else {
            text("未完成：${result.errorCode ?: "UNKNOWN"} ${result.errorMessage ?: ""}")
        }
        button("选择其他照片") { choosePhoto() }
        button("返回首页") { showHome() }
    }

    private fun saveToGallery(file: File, result: PhotoTask.FullPhotoResult) {
        try {
            val values = ContentValues().apply {
                put(MediaStore.Images.Media.DISPLAY_NAME, "SeedEdge_${System.currentTimeMillis()}.png")
                put(MediaStore.Images.Media.MIME_TYPE, "image/png")
                put(MediaStore.Images.Media.RELATIVE_PATH, "Pictures/SeedVR2")
                put(MediaStore.Images.Media.IS_PENDING, 1)
            }
            val uri: Uri = contentResolver.insert(
                MediaStore.Images.Media.getContentUri(MediaStore.VOLUME_EXTERNAL_PRIMARY), values)
                ?: error("无法创建相册项目")
            try {
                contentResolver.openOutputStream(uri)!!.use { output -> file.inputStream().use { it.copyTo(output) } }
                values.clear(); values.put(MediaStore.Images.Media.IS_PENDING, 0)
                contentResolver.update(uri, values, null, null)
            } catch (t: Throwable) {
                contentResolver.delete(uri, null, null)
                throw t
            }
            val record = PhotoTask.gallerySaveRecord(result, uri.toString())
            Toast.makeText(this, "已保存到 Pictures/SeedVR2", Toast.LENGTH_LONG).show()
            text("保存成功\nURI=${record.mediaStoreUri}\n任务=${record.taskId}\n" +
                "尺寸=${record.correctedWidth}×${record.correctedHeight}\n" +
                "tiles=${record.tileCount}\noutput.png=${record.outputPngPath}")
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
