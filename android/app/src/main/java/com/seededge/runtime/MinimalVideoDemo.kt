package com.seededge.runtime

import android.content.Context
import org.json.JSONObject
import java.io.File

object MinimalVideoDemo {
    fun run(context: Context): String {
        val external = requireNotNull(context.getExternalFilesDir(null))
        val modelRoot = File(external, "models/phase18c")
        val inputRoot = File(context.filesDir, "phase18c_input")
        val outputRoot = File(external, "output/phase18c")
        val frameRoot = File(outputRoot, "frames")
        DemoInput.prepare(context, inputRoot)
        frameRoot.mkdirs()
        val config = JSONObject().apply {
            put("frame_count", 4); put("fps", 4); put("backend", "mixed")
            put("threads", 1); put("use_fp16", false); put("run_id", "phase18c_demo")
        }.toString()
        val result = JSONObject(NativeRuntime.nativeRunMinimalVideoDemo(
            modelRoot.absolutePath, inputRoot.absolutePath, frameRoot.absolutePath, config))
        if (!result.optBoolean("overall_pass", false)) return result.toString()
        val video = DeviceVideoEncoder.encode(frameRoot, File(outputRoot, "seedvr2_android_demo.mp4"))
        result.put("video_encoded", true); result.put("output_video_path", video.path)
        result.put("output_video_size_bytes", video.sizeBytes); result.put("output_video_sha256", video.sha256)
        result.put("video_codec", "h264_avc"); result.put("pixel_format", "yuv420_i420")
        result.put("video_width", video.width); result.put("video_height", video.height)
        result.put("video_fps", video.fps); result.put("video_frame_count", video.frameCount)
        result.put("video_duration_us", video.durationUs); result.put("media_extractor_readable", video.readable)
        result.put("overall_pass", video.sizeBytes > 0 && video.readable && video.frameCount == 4)
        File(outputRoot, "phase18c_result.json").writeText(result.toString(2))
        return result.toString()
    }
}
