package com.fedlearn.mobile

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Context
import android.content.Intent
import android.os.Build
import android.os.Handler
import android.os.IBinder
import android.os.Looper

// Foreground service for the lifetime of a training run (task 16 / E5). Keeps the process at
// foreground priority so the JS round loop survives Doze, and samples device state on a timer so
// per-round telemetry (thermal/battery) is fresh. Backgrounding the app no longer silently kills
// the run; stopping the service is the clean stop path.
class FlForegroundService : Service() {
  private val handler = Handler(Looper.getMainLooper())
  private val sampleIntervalMs = 5000L

  private val sampler = object : Runnable {
    override fun run() {
      DeviceState.sample(applicationContext)
      handler.postDelayed(this, sampleIntervalMs)
    }
  }

  override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
    if (intent?.action == ACTION_STOP) {
      stop()
      return START_NOT_STICKY
    }
    startForeground(NOTIFICATION_ID, buildNotification())
    handler.post(sampler)
    return START_STICKY
  }

  override fun onDestroy() {
    handler.removeCallbacks(sampler)
    super.onDestroy()
  }

  private fun stop() {
    handler.removeCallbacks(sampler)
    stopForeground(STOP_FOREGROUND_REMOVE)
    stopSelf()
  }

  private fun buildNotification(): Notification {
    val nm = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
      nm.createNotificationChannel(
        NotificationChannel(CHANNEL_ID, "Federated training", NotificationManager.IMPORTANCE_LOW),
      )
    }
    return Notification.Builder(this, CHANNEL_ID)
      .setContentTitle("FedLearn")
      .setContentText("Federated training in progress")
      .setSmallIcon(android.R.drawable.stat_sys_upload)
      .setOngoing(true)
      .build()
  }

  override fun onBind(intent: Intent?): IBinder? = null

  companion object {
    private const val CHANNEL_ID = "fedlearn.training"
    private const val NOTIFICATION_ID = 4711
    const val ACTION_STOP = "com.fedlearn.mobile.STOP_TRAINING"

    fun start(context: Context) {
      val intent = Intent(context, FlForegroundService::class.java)
      if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
        context.startForegroundService(intent)
      } else {
        context.startService(intent)
      }
    }

    fun stop(context: Context) {
      context.startService(Intent(context, FlForegroundService::class.java).setAction(ACTION_STOP))
    }
  }
}
