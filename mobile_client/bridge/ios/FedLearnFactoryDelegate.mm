// FedLearnFactoryDelegate.mm — overrides the New-Architecture TurboModule lookup to hand React
// Native the shared C++ FL core, and supplies the JS bundle URL (the base delegate leaves it nil).
//
// VERIFY-BEFORE-BUILD (RN 0.80, New Architecture, version-specific):
//   - `-getTurboModule:jsInvoker:` is the RCTTurboModuleManagerDelegate hook the factory consults;
//     its exact name/signature can change between RN versions — reconcile against the pinned headers.
//   - FedLearnCoreModuleProvider is defined in bridge/ios/NativeFedLearnCore.mm (same pod target).
#import "FedLearnFactoryDelegate.h"

#import <React/RCTBundleURLProvider.h>
#import <ReactCommon/RCTTurboModule.h>

#include <memory>
#include <string>

namespace facebook::react {
// Defined in NativeFedLearnCore.mm — constructs FedLearnCoreModule for name "NativeFedLearnCore".
std::shared_ptr<TurboModule> FedLearnCoreModuleProvider(const std::string &name,
                                                        std::shared_ptr<CallInvoker> jsInvoker);
}  // namespace facebook::react

@implementation FedLearnFactoryDelegate

// Route the app-defined C++ TurboModule; fall back to the default (codegen/autolinked) modules.
- (std::shared_ptr<facebook::react::TurboModule>)
    getTurboModule:(const std::string &)name
         jsInvoker:(std::shared_ptr<facebook::react::CallInvoker>)jsInvoker {
  if (auto module = facebook::react::FedLearnCoreModuleProvider(name, jsInvoker)) {
    return module;
  }
  return [super getTurboModule:name jsInvoker:jsInvoker];
}

- (NSURL *)bundleURL {
#if DEBUG
  return [RCTBundleURLProvider.sharedSettings jsBundleURLForBundleRoot:@"index"];
#else
  return [NSBundle.mainBundle URLForResource:@"main" withExtension:@"jsbundle"];
#endif
}

- (NSURL *)sourceURLForBridge:(RCTBridge *)bridge {
  return [self bundleURL];
}

@end
