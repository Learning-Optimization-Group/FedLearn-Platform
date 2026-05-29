package com.fedlearn.mobile

import com.facebook.react.bridge.ReactApplicationContext
import com.facebook.react.bridge.ReactContextBaseJavaModule
import com.facebook.react.bridge.ReactMethod

// RN module exposing start/stop of the training foreground service to JS (src/lib/foregroundService.ts).
// TrainingScreen starts it before the round loop and stops it in the finally block (task 16).
class FlServiceModule(private val ctx: ReactApplicationContext) : ReactContextBaseJavaModule(ctx) {
  override fun getName(): String = "FlService"

  @ReactMethod
  fun start() {
    FlForegroundService.start(ctx)
  }

  @ReactMethod
  fun stop() {
    FlForegroundService.stop(ctx)
  }
}
