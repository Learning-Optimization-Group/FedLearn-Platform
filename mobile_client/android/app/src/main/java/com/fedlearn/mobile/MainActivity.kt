package com.fedlearn.mobile

import com.facebook.react.ReactActivity
import com.facebook.react.ReactActivityDelegate
import com.facebook.react.defaults.DefaultNewArchitectureEntryPoint.fabricEnabled
import com.facebook.react.defaults.DefaultReactActivityDelegate

class MainActivity : ReactActivity() {
  override fun getMainComponentName(): String = "FedLearn"

  // Sample thermal and battery whenever the app comes to the foreground, so the metrics a device reports before it
  // trains (the Home banner, the capability report sent with enrollment) are measured rather than defaults. The
  // training foreground service keeps sampling while a run trains. Sampling must never take the app down.
  override fun onResume() {
    super.onResume()
    try {
      DeviceState.sample(applicationContext)
    } catch (e: Throwable) {
      android.util.Log.w("FedLearn", "device state sampling failed", e)
    }
  }

  override fun createReactActivityDelegate(): ReactActivityDelegate =
    DefaultReactActivityDelegate(this, mainComponentName, fabricEnabled)
}
