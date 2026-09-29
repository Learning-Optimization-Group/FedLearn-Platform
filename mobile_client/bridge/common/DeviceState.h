#pragma once
//
// DeviceState.h — a process-global, thread-safe holder for platform-only device telemetry
// (thermal state + battery) pushed in from the Android/iOS layer (task 17).
//
// The C++ core can read peak RSS portably (/proc on Android), but thermal/battery require platform
// APIs (Android PowerManager/BatteryManager, iOS ProcessInfo/UIDevice). Rather than a C++->Java
// upcall, the platform layer periodically PUSHES the latest values here and FedLearnCoreModule's
// getDeviceMetrics reads them. Decoupled, lock-guarded, no per-instance wiring.
//
#include <string>

namespace fedlearn::bridge {

struct PlatformDeviceState {
  // NOMINAL | FAIR | SERIOUS | CRITICAL, or UNKNOWN until the platform layer has sampled: a device that has not been
  // measured must not report a healthy thermal state it never read.
  std::string thermalState = "UNKNOWN";
  double batteryLevel = -1.0;            // 0..1; -1 if unknown
  bool batteryCharging = false;
};

// Set by the platform layer (Android JNI / iOS C entry point); read by getDeviceMetrics.
void setPlatformDeviceState(const PlatformDeviceState& state);
PlatformDeviceState getPlatformDeviceState();

}  // namespace fedlearn::bridge

// iOS (Swift) calls this C entry point; Android uses the JNI shim in DeviceStateJni.cpp.
extern "C" void FedLearnCoreSetDeviceState(const char* thermalState, double batteryLevel,
                                           bool batteryCharging);
