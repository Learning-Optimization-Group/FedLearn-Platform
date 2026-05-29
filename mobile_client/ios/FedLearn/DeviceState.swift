import Foundation
import UIKit

// Samples iOS thermal + battery and pushes them to the shared C++ device-state holder via the
// FedLearnCoreSetDeviceState C entry point (declared in the bridging header). Started while a run
// is active (foreground-only on iOS, E5). Task 17.
final class DeviceStateSampler {
  static let shared = DeviceStateSampler()
  private var timer: Timer?

  func start() {
    UIDevice.current.isBatteryMonitoringEnabled = true
    sample()
    timer = Timer.scheduledTimer(withTimeInterval: 5.0, repeats: true) { [weak self] _ in
      self?.sample()
    }
  }

  func stop() {
    timer?.invalidate()
    timer = nil
  }

  private func sample() {
    let thermal: String
    switch ProcessInfo.processInfo.thermalState {
    case .nominal: thermal = "NOMINAL"
    case .fair: thermal = "FAIR"
    case .serious: thermal = "SERIOUS"
    case .critical: thermal = "CRITICAL"
    @unknown default: thermal = "NOMINAL"
    }
    let level = Double(UIDevice.current.batteryLevel) // -1 if unknown
    let charging = UIDevice.current.batteryState == .charging || UIDevice.current.batteryState == .full
    thermal.withCString { FedLearnCoreSetDeviceState($0, level, charging) }
  }
}
