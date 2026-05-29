// NativeFedLearnCore.mm — iOS provider that hands React Native the shared C++ TurboModule
// (15-LLD-mobile.md §4 / §13 task 13). The actual implementation is the cross-platform
// bridge/common/FedLearnCoreModule.{h,cpp}; this ObjC++ shim only constructs and returns it.
//
// VERIFY-BEFORE-BUILD: the New Architecture provider hook is RN-version-specific. In recent RN,
// the app delegate's RCTTurboModuleManagerDelegate implements
// `- (std::shared_ptr<TurboModule>)getTurboModule:(const std::string&)name jsInvoker:(...)`.
// Call FedLearnCoreModuleProvider from there. `dataDir` MUST be the app's Documents directory
// (Data-Protection encrypted) holding the gRPC cert/key/CA + on-device data.
#import <Foundation/Foundation.h>

#import <ReactCommon/RCTTurboModule.h>

#include <memory>
#include <string>

#include "FedLearnCoreModule.h"

namespace facebook::react {

std::shared_ptr<TurboModule> FedLearnCoreModuleProvider(const std::string &name,
                                                        std::shared_ptr<CallInvoker> jsInvoker) {
  if (name == "NativeFedLearnCore") {
    NSString *docs = [NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES)
        firstObject];
    std::string dataDir = docs ? std::string([docs UTF8String]) : std::string();
    return std::make_shared<fedlearn::bridge::FedLearnCoreModule>(std::move(jsInvoker), dataDir);
  }
  return nullptr;
}

}  // namespace facebook::react
