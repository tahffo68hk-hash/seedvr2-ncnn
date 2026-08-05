package com.seededge.runtime

import android.graphics.Bitmap
import java.io.File
import java.nio.ByteBuffer

object DecoderOutput {
    /** Converts the native RGB24 decoder frame into a standard, shareable PNG. */
    fun writeRgb24Png(rgbFile: File, pngFile: File, width: Int = 128, height: Int = 128): Boolean {
        val rgb = rgbFile.readBytes()
        if (rgb.size != width * height * 3) return false
        val rgba = ByteArray(width * height * 4)
        for (pixel in 0 until width * height) {
            rgba[pixel * 4] = rgb[pixel * 3]
            rgba[pixel * 4 + 1] = rgb[pixel * 3 + 1]
            rgba[pixel * 4 + 2] = rgb[pixel * 3 + 2]
            rgba[pixel * 4 + 3] = 0xff.toByte()
        }
        val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        bitmap.copyPixelsFromBuffer(ByteBuffer.wrap(rgba))
        pngFile.outputStream().use { stream ->
            return bitmap.compress(Bitmap.CompressFormat.PNG, 100, stream)
        }
    }
}
