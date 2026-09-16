// Swift <-> C bridging header. Exposes the device-state C entry point (bridge/common/DeviceState.h)
// so DeviceState.swift can push iOS thermal/battery into the shared C++ holder (task 17).
#import "DeviceState.h"
