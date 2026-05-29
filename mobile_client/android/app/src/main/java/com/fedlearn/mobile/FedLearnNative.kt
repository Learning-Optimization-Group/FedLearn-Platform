package com.fedlearn.mobile

// JNI bridge for one-time native init. nativeSetDataDir -> OnLoad.cpp
// (Java_com_fedlearn_mobile_FedLearnNative_nativeSetDataDir).
object FedLearnNative {
  init {
    System.loadLibrary("fedlearn_jni")
  }

  @JvmStatic
  external fun nativeSetDataDir(dir: String)

  fun setDataDir(dir: String) = nativeSetDataDir(dir)
}
