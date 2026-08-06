package com.seededge.runtime

import java.util.concurrent.atomic.AtomicBoolean

/**
 * Cooperative full-photo progress bridge called from native on the worker
 * thread. onProgress returns false (which triggers cancellation at the next
 * safe boundary) as soon as the caller cancels. A throwing callback is never
 * silently swallowed: it cancels the task so native stops at the next tile
 * boundary.
 */
class FullPhotoProgressBridge(
    private val cancelled: AtomicBoolean,
    private val onStageProgress: (stage: String, completed: Int, total: Int) -> Unit
) {
    @Suppress("unused")
    fun onProgress(stage: String, completed: Int, total: Int): Boolean {
        try {
            onStageProgress(stage, completed, total)
        } catch (t: Throwable) {
            cancelled.set(true)
        }
        return !cancelled.get()
    }

    @Suppress("unused")
    fun isCancelled(): Boolean = cancelled.get()
}