package com.seededge.runtime

import android.content.Context
import android.graphics.BitmapFactory
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder

object DemoInput {
    fun prepare(context: Context, root: File): File {
        root.mkdirs()
        val frames = (0 until 4).map { index ->
            context.assets.open("phase18c_demo/frame_${index.toString().padStart(3, '0')}.png").use {
                BitmapFactory.decodeStream(it).also { bitmap ->
                    require(bitmap.width == 128 && bitmap.height == 128)
                }
            }
        }
        val output = File(root, "input_norm.f32")
        FileOutputStream(output).channel.use { channel ->
            val buffer = ByteBuffer.allocateDirect(128 * 128 * 4).order(ByteOrder.LITTLE_ENDIAN)
            for (c in 0 until 3) for (t in 0 until 5) {
                val bitmap = frames[if (t < 4) t else 3]
                buffer.clear()
                for (y in 0 until 128) for (x in 0 until 128) {
                    val pixel = bitmap.getPixel(x, y)
                    val value = when (c) { 0 -> pixel shr 16; 1 -> pixel shr 8; else -> pixel } and 255
                    buffer.putFloat(value / 127.5f - 1f)
                }
                buffer.flip()
                while (buffer.hasRemaining()) channel.write(buffer)
            }
        }
        frames.forEach { it.recycle() }
        return output
    }
}
