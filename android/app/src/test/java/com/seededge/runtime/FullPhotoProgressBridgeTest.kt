package com.seededge.runtime

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import java.util.concurrent.atomic.AtomicBoolean

class FullPhotoProgressBridgeTest {

    @Test
    fun onProgressForwardsStageCompletedTotal() {
        val cancelled = AtomicBoolean(false)
        var seen: Triple<String, Int, Int>? = null
        val bridge = FullPhotoProgressBridge(cancelled) { stage, completed, total ->
            seen = Triple(stage, completed, total)
        }
        assertTrue(bridge.onProgress("RUNNING_TILE", 1, 2))
        assertEquals(Triple("RUNNING_TILE", 1, 2), seen)
    }

    @Test
    fun onProgressReturnsFalseOnceCancelled() {
        val cancelled = AtomicBoolean(false)
        val bridge = FullPhotoProgressBridge(cancelled) { _, _, _ -> }
        assertTrue(bridge.onProgress("PREPARING", 0, 0))
        cancelled.set(true)
        assertFalse(bridge.onProgress("CHECKING_MODEL", 0, 0))
    }

    @Test
    fun onProgressExceptionCancelsAndReturnsFalse() {
        val cancelled = AtomicBoolean(false)
        val bridge = FullPhotoProgressBridge(cancelled) { _, _, _ -> throw IllegalStateException("boom") }
        assertFalse(bridge.onProgress("RUNNING_TILE", 0, 2))
        assertTrue(cancelled.get())
    }
}