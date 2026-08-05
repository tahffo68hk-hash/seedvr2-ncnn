package com.seededge.runtime

object Yuv420 {
    fun rgbToI420(rgb: ByteArray, width: Int, height: Int): ByteArray {
        require(width > 0 && height > 0 && width % 2 == 0 && height % 2 == 0)
        require(rgb.size == width * height * 3)
        val frame = width * height
        val output = ByteArray(frame * 3 / 2)
        fun clamp(value: Int) = value.coerceIn(0, 255).toByte()
        for (y in 0 until height) for (x in 0 until width) {
            val p = (y * width + x) * 3
            val r = rgb[p].toInt() and 255
            val g = rgb[p + 1].toInt() and 255
            val b = rgb[p + 2].toInt() and 255
            output[y * width + x] = clamp(((66 * r + 129 * g + 25 * b + 128) shr 8) + 16)
        }
        for (y in 0 until height step 2) for (x in 0 until width step 2) {
            var r = 0; var g = 0; var b = 0
            for (dy in 0..1) for (dx in 0..1) {
                val p = ((y + dy) * width + x + dx) * 3
                r += rgb[p].toInt() and 255; g += rgb[p + 1].toInt() and 255; b += rgb[p + 2].toInt() and 255
            }
            r /= 4; g /= 4; b /= 4
            val chroma = (y / 2) * (width / 2) + x / 2
            output[frame + chroma] = clamp(((-38 * r - 74 * g + 112 * b + 128) shr 8) + 128)
            output[frame + frame / 4 + chroma] = clamp(((112 * r - 94 * g - 18 * b + 128) shr 8) + 128)
        }
        return output
    }
}
