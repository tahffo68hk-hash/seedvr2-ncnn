package com.seededge.runtime

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.os.Build
import android.os.IBinder
import android.os.PowerManager
import java.io.File
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicReference

/** Owns long-running real SeedVR2 work independently of the Activity lifecycle. */
class PhotoEnhanceService : Service() {
    companion object {
        const val ACTION_START = "com.seededge.runtime.photo.START"
        const val ACTION_PAUSE = "com.seededge.runtime.photo.PAUSE"
        const val ACTION_RESUME = "com.seededge.runtime.photo.RESUME"
        const val ACTION_CANCEL = "com.seededge.runtime.photo.CANCEL"
        private const val CHANNEL_ID = "seededge_photo_runtime"
        private const val NOTIFICATION_ID = 1804
        private const val PREFS = "photo_enhance_service"
        private val currentState = AtomicReference(PhotoEnhanceControl.State.IDLE)

        fun state(): PhotoEnhanceControl.State = currentState.get()
        fun startIntent(context: Context, taskId: String, taskRoot: File, modelRoot: File,
                        normalizedRgb: File, width: Int, height: Int, requestJson: String): Intent =
            Intent(context, PhotoEnhanceService::class.java).setAction(ACTION_START)
                .putExtra("task_id", taskId).putExtra("task_root", taskRoot.path)
                .putExtra("model_root", modelRoot.path).putExtra("normalized_rgb", normalizedRgb.path)
                .putExtra("width", width).putExtra("height", height).putExtra("request_json", requestJson)
        fun commandIntent(context: Context, action: String): Intent =
            Intent(context, PhotoEnhanceService::class.java).setAction(action)
    }

    private val executor = Executors.newSingleThreadExecutor { r -> Thread(r, "seededge-photo-service") }
    private val bridgeCancelled = AtomicBoolean(false)
    private var wakeLock: PowerManager.WakeLock? = null
    override fun onBind(intent: Intent?): IBinder? = null

    override fun onCreate() {
        super.onCreate()
        if (Build.VERSION.SDK_INT >= 26) getSystemService(NotificationManager::class.java)
            .createNotificationChannel(NotificationChannel(CHANNEL_ID,
                "SeedEdge photo enhancement", NotificationManager.IMPORTANCE_LOW))
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        when (intent?.action) {
            ACTION_START -> begin(intent, resume = false, cancelImmediately = false)
            ACTION_PAUSE -> requestPause()
            ACTION_RESUME -> begin(null, resume = true, cancelImmediately = false)
            ACTION_CANCEL -> requestCancel()
        }
        return START_NOT_STICKY
    }

    private fun begin(intent: Intent?, resume: Boolean, cancelImmediately: Boolean) {
        val before = currentState.get()
        val newTaskAllowed = before == PhotoEnhanceControl.State.IDLE ||
            before == PhotoEnhanceControl.State.CANCELLED || before == PhotoEnhanceControl.State.COMPLETED ||
            before == PhotoEnhanceControl.State.FAILED
        if ((!resume && !newTaskAllowed) ||
            (resume && before != PhotoEnhanceControl.State.PAUSED)) return
        if (intent != null) persist(intent)
        currentState.set(if (resume) PhotoEnhanceControl.transition(before, PhotoEnhanceControl.Command.RESUME)
            else PhotoEnhanceControl.transition(before, PhotoEnhanceControl.Command.START))
        bridgeCancelled.set(cancelImmediately)
        startForeground(NOTIFICATION_ID, notification("Preparing real SeedVR2 inference", true))
        acquireWakeLock()
        executor.execute { runPersisted(resume, cancelImmediately) }
    }

    private fun persist(intent: Intent) {
        val edit = getSharedPreferences(PREFS, MODE_PRIVATE).edit()
        for (key in arrayOf("task_id", "task_root", "model_root", "normalized_rgb", "request_json"))
            edit.putString(key, intent.getStringExtra(key) ?: "")
        edit.putInt("width", intent.getIntExtra("width", 0))
            .putInt("height", intent.getIntExtra("height", 0)).apply()
    }

    private fun runPersisted(resume: Boolean, cancelImmediately: Boolean) {
        val p = getSharedPreferences(PREFS, MODE_PRIVATE)
        try {
            val taskRoot = File(requireNotNull(p.getString("task_root", null)))
            val rgb = File(requireNotNull(p.getString("normalized_rgb", null))).readBytes()
            var request = requireNotNull(p.getString("request_json", null))
            if (resume) request = request.replace("\"resume\":false", "\"resume\":true")
            if (cancelImmediately) NativeRuntime.nativeRequestCancel()
            val result = PhotoTask.runFullPhotoWithProgress(
                File(requireNotNull(p.getString("model_root", null))), taskRoot, rgb,
                p.getInt("width", 0), p.getInt("height", 0), request, bridgeCancelled
            ) { stage, completed, total ->
                getSystemService(NotificationManager::class.java).notify(NOTIFICATION_ID,
                    notification(PhotoTask.fullPhotoStageLabel(stage, completed, total), true))
            }
            when (result.state) {
                "paused" -> {
                    currentState.set(PhotoEnhanceControl.transition(currentState.get(), PhotoEnhanceControl.Command.PAUSE_ACK))
                    releaseWakeLock()
                    getSystemService(NotificationManager::class.java).notify(NOTIFICATION_ID,
                        notification("Paused at a safe stage boundary", false))
                }
                "cancelled" -> finishTerminal(PhotoEnhanceControl.Command.CANCEL_ACK, "Cancelled")
                "completed" -> finishTerminal(PhotoEnhanceControl.Command.COMPLETE, "Completed")
                else -> finishTerminal(PhotoEnhanceControl.Command.FAIL, result.errorMessage ?: "Failed")
            }
        } catch (t: Throwable) {
            finishTerminal(PhotoEnhanceControl.Command.FAIL, t.message ?: "Service failure")
        }
    }

    private fun requestPause() {
        if (!currentState.compareAndSet(PhotoEnhanceControl.State.RUNNING,
                PhotoEnhanceControl.State.PAUSE_REQUESTED)) return
        bridgeCancelled.set(true)
        NativeRuntime.nativeRequestPause()
    }

    private fun requestCancel() {
        val before = currentState.get()
        if (before == PhotoEnhanceControl.State.PAUSED) {
            currentState.set(PhotoEnhanceControl.transition(before, PhotoEnhanceControl.Command.CANCEL))
            val root = getSharedPreferences(PREFS, MODE_PRIVATE).getString("task_root", null)
            val controlled = root != null && NativeRuntime.nativeControlStageMajor(root, "cancelled")
                .contains("\"ok\":true")
            if (controlled) finishTerminal(PhotoEnhanceControl.Command.CANCEL_ACK, "Cancelled")
            else finishTerminal(PhotoEnhanceControl.Command.FAIL, "Could not persist cancellation")
            return
        }
        if (before != PhotoEnhanceControl.State.RUNNING && before != PhotoEnhanceControl.State.PAUSE_REQUESTED) return
        currentState.set(PhotoEnhanceControl.transition(before, PhotoEnhanceControl.Command.CANCEL))
        bridgeCancelled.set(true)
        NativeRuntime.nativeRequestCancel()
    }

    private fun finishTerminal(command: PhotoEnhanceControl.Command, message: String) {
        currentState.set(PhotoEnhanceControl.transition(currentState.get(), command))
        releaseWakeLock()
        getSystemService(NotificationManager::class.java).notify(NOTIFICATION_ID, notification(message, false))
        stopForeground(false)
        stopSelf()
    }

    private fun acquireWakeLock() {
        if (wakeLock?.isHeld == true) return
        wakeLock = (getSystemService(POWER_SERVICE) as PowerManager).newWakeLock(
            PowerManager.PARTIAL_WAKE_LOCK, "SeedEdge:PhotoEnhance").apply { acquire() }
    }
    private fun releaseWakeLock() { wakeLock?.takeIf { it.isHeld }?.release(); wakeLock = null }

    private fun notification(message: String, running: Boolean): Notification {
        fun action(action: String, label: String, code: Int): Notification.Action {
            val pending = PendingIntent.getService(this, code, commandIntent(this, action),
                PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE)
            return Notification.Action.Builder(null, label, pending).build()
        }
        val builder = if (Build.VERSION.SDK_INT >= 26) Notification.Builder(this, CHANNEL_ID)
            else Notification.Builder(this)
        builder.setSmallIcon(android.R.drawable.stat_sys_download)
            .setContentTitle("SeedEdge full-photo enhancement").setContentText(message)
            .setOngoing(running).setOnlyAlertOnce(true)
        when (currentState.get()) {
            PhotoEnhanceControl.State.RUNNING -> builder.addAction(action(ACTION_PAUSE, "Pause", 1))
                .addAction(action(ACTION_CANCEL, "Cancel", 2))
            PhotoEnhanceControl.State.PAUSED -> builder.addAction(action(ACTION_RESUME, "Resume", 3))
                .addAction(action(ACTION_CANCEL, "Cancel", 2))
            else -> Unit
        }
        return builder.build()
    }

    override fun onDestroy() { releaseWakeLock(); executor.shutdown(); super.onDestroy() }
}
