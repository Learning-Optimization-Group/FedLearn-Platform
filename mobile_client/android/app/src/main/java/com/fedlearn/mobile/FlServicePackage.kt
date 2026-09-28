package com.fedlearn.mobile

import com.facebook.react.ReactPackage
import com.facebook.react.bridge.NativeModule
import com.facebook.react.bridge.ReactApplicationContext
import com.facebook.react.uimanager.ViewManager

class FlServicePackage : ReactPackage {
  override fun createNativeModules(ctx: ReactApplicationContext): List<NativeModule> =
    listOf(
      FlServiceModule(ctx),
      com.fedlearn.mobile.artifacts.ArtifactServiceModule(ctx),
      com.fedlearn.mobile.datasets.DatasetServiceModule(ctx),
    )

  override fun createViewManagers(ctx: ReactApplicationContext): List<ViewManager<*, *>> =
    emptyList()
}
