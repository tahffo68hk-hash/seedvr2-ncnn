package com.seededge.runtime

import android.media.MediaCodec
import android.media.MediaCodecInfo
import android.media.MediaExtractor
import android.media.MediaFormat
import android.media.MediaMuxer
import java.io.File
import java.security.MessageDigest

data class VideoEncodeResult(val path: String, val sizeBytes: Long, val sha256: String,
                             val width: Int, val height: Int, val fps: Int,
                             val frameCount: Int, val durationUs: Long, val readable: Boolean)

object DeviceVideoEncoder {
    fun encode(frameRoot: File, output: File, width: Int = 128, height: Int = 128,
               fps: Int = 4, frameCount: Int = 4): VideoEncodeResult {
        output.parentFile?.mkdirs(); if (output.exists()) output.delete()
        val format = MediaFormat.createVideoFormat(MediaFormat.MIMETYPE_VIDEO_AVC, width, height).apply {
            setInteger(MediaFormat.KEY_COLOR_FORMAT, MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420Flexible)
            setInteger(MediaFormat.KEY_BIT_RATE, 700_000)
            setInteger(MediaFormat.KEY_FRAME_RATE, fps)
            setInteger(MediaFormat.KEY_I_FRAME_INTERVAL, 1)
        }
        val codec = MediaCodec.createEncoderByType(MediaFormat.MIMETYPE_VIDEO_AVC)
        val muxer = MediaMuxer(output.absolutePath, MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4)
        var track = -1; var muxerStarted = false; var encodedFrames = 0
        val info = MediaCodec.BufferInfo()
        try {
            codec.configure(format, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE); codec.start()
            var inputIndex = 0; var eosQueued = false; var done = false
            while (!done) {
                if (!eosQueued) {
                    val slot = codec.dequeueInputBuffer(10_000)
                    if (slot >= 0) {
                        val buffer = codec.getInputBuffer(slot)!!; buffer.clear()
                        if (inputIndex < frameCount) {
                            val rgb = File(frameRoot, "frame_${inputIndex.toString().padStart(3, '0')}.rgb").readBytes()
                            val yuv = Yuv420.rgbToI420(rgb, width, height)
                            buffer.put(yuv)
                            codec.queueInputBuffer(slot, 0, yuv.size, inputIndex * 1_000_000L / fps, 0)
                            inputIndex++
                        } else {
                            codec.queueInputBuffer(slot, 0, 0, inputIndex * 1_000_000L / fps,
                                MediaCodec.BUFFER_FLAG_END_OF_STREAM); eosQueued = true
                        }
                    }
                }
                when (val out = codec.dequeueOutputBuffer(info, 10_000)) {
                    MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> {
                        check(!muxerStarted); track = muxer.addTrack(codec.outputFormat); muxer.start(); muxerStarted = true
                    }
                    MediaCodec.INFO_TRY_AGAIN_LATER -> Unit
                    else -> if (out >= 0) {
                        val buffer = codec.getOutputBuffer(out)!!
                        if (info.size > 0) {
                            check(muxerStarted); buffer.position(info.offset); buffer.limit(info.offset + info.size)
                            muxer.writeSampleData(track, buffer, info)
                            if (info.flags and MediaCodec.BUFFER_FLAG_CODEC_CONFIG == 0) encodedFrames++
                        }
                        done = info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0
                        codec.releaseOutputBuffer(out, false)
                    }
                }
            }
        } finally {
            codec.stop(); codec.release(); if (muxerStarted) muxer.stop(); muxer.release()
        }
        val extractor = MediaExtractor(); var readable = false; var duration = 0L
        try {
            extractor.setDataSource(output.absolutePath)
            readable = extractor.trackCount == 1
            if (readable) duration = extractor.getTrackFormat(0).getLong(MediaFormat.KEY_DURATION)
        } finally { extractor.release() }
        val digest = MessageDigest.getInstance("SHA-256").digest(output.readBytes()).joinToString("") { "%02x".format(it) }
        return VideoEncodeResult(output.absolutePath, output.length(), digest, width, height, fps, encodedFrames, duration, readable)
    }
}
