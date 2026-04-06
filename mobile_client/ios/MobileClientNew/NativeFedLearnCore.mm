#import <React/RCTBridgeModule.h>
#import <ReactCommon/RCTTurboModule.h>
#import <React/RCTLog.h>

#include "NativeFedLearnCore.h"

// ObjC++ TurboModule wrapper for iOS
// Bridges React Native JavaScript calls to the C++ NativeFedLearnCoreImpl.
//
// After codegen, this should conform to the generated NativeFedLearnCoreSpec protocol.

@interface NativeFedLearnCoreModule : NSObject <RCTBridgeModule, RCTTurboModule>
@end

@implementation NativeFedLearnCoreModule {
  std::unique_ptr<fedlearn::NativeFedLearnCoreImpl> _impl;
}

RCT_EXPORT_MODULE(NativeFedLearnCore)

- (instancetype)init {
  if (self = [super init]) {
    _impl = std::make_unique<fedlearn::NativeFedLearnCoreImpl>();
  }
  return self;
}

+ (BOOL)requiresMainQueueSetup {
  return NO;
}

- (std::shared_ptr<facebook::react::TurboModule>)getTurboModule:
    (const facebook::react::ObjCTurboModule::InitParams &)params {
  // For CxxTurboModule integration, create a wrapper here.
  // The actual JSI binding is handled by the C++ side.
  return nullptr;
}

// --- Bridge methods (fallback for non-TurboModule setups) ---

RCT_EXPORT_METHOD(loadModel:(NSString *)modelPath
                  resolver:(RCTPromiseResolveBlock)resolve
                  rejecter:(RCTPromiseRejectBlock)reject) {
  NSString *expandedPath = [modelPath stringByExpandingTildeInPath];
  bool success = _impl->loadModel([expandedPath UTF8String]);
  resolve(@(success));
}

RCT_EXPORT_METHOD(getModelInfo:(RCTPromiseResolveBlock)resolve
                  rejecter:(RCTPromiseRejectBlock)reject) {
  std::string info = _impl->getModelInfo();
  resolve([NSString stringWithUTF8String:info.c_str()]);
}

RCT_EXPORT_METHOD(trainStep:(NSString *)inputPath
                  numEpochs:(double)numEpochs
                  lr:(double)lr
                  resolver:(RCTPromiseResolveBlock)resolve
                  rejecter:(RCTPromiseRejectBlock)reject) {
  std::string result = _impl->trainStep(
      [inputPath UTF8String],
      static_cast<int>(numEpochs),
      static_cast<float>(lr));
  resolve([NSString stringWithUTF8String:result.c_str()]);
}

RCT_EXPORT_METHOD(connect:(NSString *)serverAddress
                  clientId:(NSString *)clientId
                  resolver:(RCTPromiseResolveBlock)resolve
                  rejecter:(RCTPromiseRejectBlock)reject) {
  bool success = _impl->connect([serverAddress UTF8String],
                                 [clientId UTF8String]);
  resolve(@(success));
}

RCT_EXPORT_METHOD(disconnect) {
  _impl->disconnect();
}

RCT_EXPORT_METHOD(startFedAvgLoop:(NSString *)configJson) {
  _impl->startFedAvgLoop([configJson UTF8String]);
}

RCT_EXPORT_METHOD(startDeComFLLoop:(NSString *)configJson) {
  _impl->startDeComFLLoop([configJson UTF8String]);
}

RCT_EXPORT_METHOD(stopTraining) {
  _impl->stopTraining();
}

RCT_EXPORT_METHOD(getStatus:(RCTPromiseResolveBlock)resolve
                  rejecter:(RCTPromiseRejectBlock)reject) {
  std::string status = _impl->getStatus();
  resolve([NSString stringWithUTF8String:status.c_str()]);
}

RCT_EXPORT_METHOD(setZOConfig:(NSString *)configJson) {
  _impl->setZOConfig([configJson UTF8String]);
}

@end
