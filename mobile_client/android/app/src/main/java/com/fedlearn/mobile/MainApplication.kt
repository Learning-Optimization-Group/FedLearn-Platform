package com.fedlearn.mobile

import android.app.Application
import com.facebook.react.PackageList
import com.facebook.react.ReactApplication
import com.facebook.react.ReactHost
import com.facebook.react.ReactNativeHost
import com.facebook.react.ReactPackage
import com.facebook.react.defaults.DefaultNewArchitectureEntryPoint.load
import com.facebook.react.defaults.DefaultReactHost.getDefaultReactHost
import com.facebook.react.defaults.DefaultReactNativeHost
import com.facebook.soloader.SoLoader

// VERIFY-BEFORE-USE: RN 0.80 New Architecture host. The app-defined C++ (CXX) TurboModule
// `NativeFedLearnCore` is surfaced by the cxxModuleProvider in bridge/android/jni/OnLoad.cpp; the
// New Arch TurboModule delegate must consult it (the wiring point is RN-version-specific).
class MainApplication : Application(), ReactApplication {

  override val reactNativeHost: ReactNativeHost =
    object : DefaultReactNativeHost(this) {
      override fun getPackages(): List<ReactPackage> =
        PackageList(this).packages.apply {
          add(FlServicePackage()) // exposes start/stop of the training foreground service to JS
        }

      override fun getJSMainModuleName(): String = "index"
      override fun getUseDeveloperSupport(): Boolean = BuildConfig.DEBUG
      override val isNewArchEnabled: Boolean = true
      override val isHermesEnabled: Boolean = true
    }

  override val reactHost: ReactHost
    get() = getDefaultReactHost(applicationContext, reactNativeHost)

  override fun onCreate() {
    super.onCreate()
    SoLoader.init(this, false)
    // Hand the native FL core its app-private dir (gRPC certs + on-device data) BEFORE any
    // TurboModule lookup constructs FedLearnCoreModule.
    FedLearnNative.setDataDir(filesDir.absolutePath)
    if (isNewArchEnabled) {
      load()
    }
  }
}
