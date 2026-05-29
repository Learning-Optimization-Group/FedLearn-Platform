package com.fedlearn.mobile

import android.content.Context
import android.os.BatteryManager
import android.os.Build
import android.os.PowerManager

// Reads platform thermal + battery and pushes them to the C++ device-state holder (task 17).
// nativeSetState -> DeviceStateJni.cpp (Java_com_fedlearn_mobile_DeviceState_nativeSetState).
object DeviceState {
  init {
    System.loadLibrary("fedlearn_jni")
  }

  @JvmStatic
  external fun nativeSetState(thermalState: String, batteryLevel: Double, charging: Boolean)

  fun sample(context: Context) {
    val thermal = thermalState(context)
    val bm = context.getSystemService(Context.BATTERY_SERVICE) as BatteryManager
    val level = bm.getIntProperty(BatteryManager.BATTERY_PROPERTY_CAPACITY) / 100.0
    nativeSetState(thermal, level, bm.isCharging)
  }

  private fun thermalState(context: Context): String {
    if (Build.VERSION.SDK_INT < Build.VERSION_CODES.Q) return "NOMINAL"
    val pm = context.getSystemService(Context.POWER_SERVICE) as PowerManager
    return when (pm.currentThermalStatus) {
      PowerManager.THERMAL_STATUS_NONE, PowerManager.THERMAL_STATUS_LIGHT -> "NOMINAL"
      PowerManager.THERMAL_STATUS_MODERATE -> "FAIR"
      PowerManager.THERMAL_STATUS_SEVERE -> "SERIOUS"
      else -> "CRITICAL"
    }
  }
}
