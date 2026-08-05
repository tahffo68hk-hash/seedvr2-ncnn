package com.seededge.runtime

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class Yuv420Test {
    @Test fun convertsTwoByTwoRgbToI420() {
        val rgb = byteArrayOf(
            255.toByte(), 0, 0,  0, 255.toByte(), 0,
            0, 0, 255.toByte(),  255.toByte(), 255.toByte(), 255.toByte()
        )
        val yuv = Yuv420.rgbToI420(rgb, 2, 2)
        assertEquals(6, yuv.size)
        assertTrue((yuv[0].toInt() and 255) in 70..90)
        assertTrue((yuv[3].toInt() and 255) in 230..255)
    }

    @Test fun rejectsOddDimensions() {
        try {
            Yuv420.rgbToI420(ByteArray(3 * 3 * 3), 3, 3)
            throw AssertionError("odd dimensions accepted")
        } catch (_: IllegalArgumentException) { }
    }
}
