// FedLearnFactoryDelegate.h — the iOS New-Architecture factory delegate that registers the shared
// C++ (CXX) TurboModule `NativeFedLearnCore` (bridge/common/FedLearnCoreModule). The Swift
// AppDelegate uses this in place of the template's default delegate so JS
// `TurboModuleRegistry.getEnforcing('NativeFedLearnCore')` resolves to the native FL core.
//
// This PUBLIC header is intentionally Swift-safe (no C++ in its surface) — the C++ `getTurboModule`
// override lives in the .mm. Exposed via the FedLearnCore pod module, so the Swift app does
// `import FedLearnCore` and `FedLearnFactoryDelegate()`.
#import <Foundation/Foundation.h>

// VERIFY-BEFORE-BUILD: the factory-delegate base class is RN-version-specific. In RN 0.80 it is
// RCTDefaultReactNativeFactoryDelegate from the React-RCTAppDelegate pod.
#import <React_RCTAppDelegate/RCTDefaultReactNativeFactoryDelegate.h>

NS_ASSUME_NONNULL_BEGIN

@interface FedLearnFactoryDelegate : RCTDefaultReactNativeFactoryDelegate
@end

NS_ASSUME_NONNULL_END
