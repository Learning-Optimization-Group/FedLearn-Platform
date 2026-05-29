#include "DeviceState.h"

#include <mutex>

namespace fedlearn::bridge {
namespace {
std::mutex g_mutex;
PlatformDeviceState g_state;
}  // namespace

void setPlatformDeviceState(const PlatformDeviceState& state) {
  std::lock_guard<std::mutex> lk(g_mutex);
  g_state = state;
}

PlatformDeviceState getPlatformDeviceState() {
  std::lock_guard<std::mutex> lk(g_mutex);
  return g_state;
}

}  // namespace fedlearn::bridge

extern "C" void FedLearnCoreSetDeviceState(const char* thermalState, double batteryLevel,
                                           bool batteryCharging) {
  fedlearn::bridge::PlatformDeviceState s;
  s.thermalState = thermalState ? thermalState : "NOMINAL";
  s.batteryLevel = batteryLevel;
  s.batteryCharging = batteryCharging;
  fedlearn::bridge::setPlatformDeviceState(s);
}
