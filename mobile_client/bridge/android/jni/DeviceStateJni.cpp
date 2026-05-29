// DeviceStateJni.cpp — JNI shim so the Android layer (DeviceState.kt) can push thermal/battery
// into the shared C++ device-state holder (task 17). Kotlin calls nativeSetState(...) periodically
// from the foreground service; the C++ getDeviceMetrics reads the latest values.
//
// The Java class is com.fedlearn.mobile.DeviceState -> the JNI symbol below.
#include <jni.h>

#include <string>

#include "DeviceState.h"

extern "C" JNIEXPORT void JNICALL Java_com_fedlearn_mobile_DeviceState_nativeSetState(
    JNIEnv* env, jclass /*clazz*/, jstring thermalState, jdouble batteryLevel,
    jboolean batteryCharging) {
  const char* t = env->GetStringUTFChars(thermalState, nullptr);
  fedlearn::bridge::setPlatformDeviceState(
      {std::string(t ? t : "NOMINAL"), static_cast<double>(batteryLevel),
       batteryCharging == JNI_TRUE});
  if (t) env->ReleaseStringUTFChars(thermalState, t);
}
