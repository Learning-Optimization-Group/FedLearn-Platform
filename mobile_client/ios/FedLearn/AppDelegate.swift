import UIKit
import React
import React_RCTAppDelegate
import ReactAppDependencyProvider

// When the native FL core pod is installed (FEDLEARN_NATIVE_IOS + the libtorch/gRPC xcframeworks),
// `FedLearnCore` is importable and we use its FedLearnFactoryDelegate, which registers the
// `NativeFedLearnCore` C++ TurboModule. Otherwise the app builds with the plain RN delegate and the
// TurboModule is simply absent (JS calls into it will throw — same as "native not built").
#if canImport(FedLearnCore)
import FedLearnCore
#endif

@main
class AppDelegate: UIResponder, UIApplicationDelegate {
  var window: UIWindow?

  var reactNativeDelegate: RCTDefaultReactNativeFactoryDelegate?
  var reactNativeFactory: RCTReactNativeFactory?

  func application(
    _ application: UIApplication,
    didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]? = nil
  ) -> Bool {
#if canImport(FedLearnCore)
    let delegate: RCTDefaultReactNativeFactoryDelegate = FedLearnFactoryDelegate()
#else
    let delegate: RCTDefaultReactNativeFactoryDelegate = ReactNativeDelegate()
#endif
    let factory = RCTReactNativeFactory(delegate: delegate)
    delegate.dependencyProvider = RCTAppDependencyProvider()

    reactNativeDelegate = delegate
    reactNativeFactory = factory

    window = UIWindow(frame: UIScreen.main.bounds)

    factory.startReactNative(
      withModuleName: "FedLearn",
      in: window,
      launchOptions: launchOptions
    )

    // Push iOS thermal/battery into the shared C++ device-state holder while the app runs (task 17;
    // no-op unless the native core is linked — the C entry point is guarded in DeviceState.swift).
    DeviceStateSampler.shared.start()

    return true
  }
}

class ReactNativeDelegate: RCTDefaultReactNativeFactoryDelegate {
  override func sourceURL(for bridge: RCTBridge) -> URL? {
    self.bundleURL()
  }

  override func bundleURL() -> URL? {
#if DEBUG
    RCTBundleURLProvider.sharedSettings().jsBundleURL(forBundleRoot: "index")
#else
    Bundle.main.url(forResource: "main", withExtension: "jsbundle")
#endif
  }
}
