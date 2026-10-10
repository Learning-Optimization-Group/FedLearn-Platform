package com.fedlearn.mobile

import com.facebook.react.bridge.ReactApplicationContext
import com.facebook.react.bridge.ReactContextBaseJavaModule
import com.facebook.react.bridge.ReactMethod
import com.facebook.react.bridge.Promise

// RN module exposing start/stop of the training foreground service to JS (src/lib/foregroundService.ts).
// TrainingProvider starts it before the round loop and stops it in the finally block.
class FlServiceModule(private val ctx: ReactApplicationContext) : ReactContextBaseJavaModule(ctx) {
  private val delays = TrainingDelayScheduler()
  override fun getName(): String = "FlService"

  @ReactMethod
  fun start() {
    FlForegroundService.start(ctx)
  }

  @ReactMethod
  fun stop() {
    delays.cancelPending()
    FlForegroundService.stop(ctx)
  }

  @ReactMethod
  fun delay(milliseconds: Double, promise: Promise) {
    try {
      delays.wait(milliseconds) { promise.resolve(null) }
    } catch (error: IllegalArgumentException) {
      promise.reject("INVALID_TRAINING_DELAY", error)
    } catch (error: IllegalStateException) {
      promise.reject("TRAINING_DELAY_UNAVAILABLE", error)
    }
  }

  override fun invalidate() {
    delays.close()
    super.invalidate()
  }
}
