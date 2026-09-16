package com.fedlearn.mobile

// JNI bridge for one-time native init. nativeSetDataDir -> OnLoad.cpp
// (Java_com_fedlearn_mobile_FedLearnNative_nativeSetDataDir). RN 0.80 builds the app's native code into
// libappmodules.so (ReactNative-application.cmake); our OnLoad.cpp + FL core are grafted into it, so we
// load "appmodules" here to run JNI_OnLoad before the first TurboModule lookup.
object FedLearnNative {
  init {
    System.loadLibrary("appmodules")
  }

  @JvmStatic
  external fun nativeSetDataDir(dir: String)

  fun setDataDir(dir: String) = nativeSetDataDir(dir)
}
