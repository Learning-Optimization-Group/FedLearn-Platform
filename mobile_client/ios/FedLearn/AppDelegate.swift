import UIKit
import React
import React_RCTAppDelegate

// VERIFY-BEFORE-USE: React Native 0.80 New Architecture iOS entry point. The app-defined C++ (CXX)
// TurboModule `NativeFedLearnCore` is provided by facebook::react::FedLearnCoreModuleProvider in
// bridge/ios/NativeFedLearnCore.mm; route the RCTTurboModuleManagerDelegate's
// getTurboModule(name:jsInvoker:) to it (the exact hook is RN-version-specific).
@main
class AppDelegate: RCTAppDelegate {
  override func application(
    _ application: UIApplication,
    didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]?
  ) -> Bool {
    self.moduleName = "FedLearn"
    self.initialProps = [:]
    return super.application(application, didFinishLaunchingWithOptions: launchOptions)
  }
}
