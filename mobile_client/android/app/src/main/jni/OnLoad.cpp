#include <DefaultComponentsRegistry.h>
#include <DefaultTurboModuleManagerDelegate.h>
#include <autolinking.h>
#include <fbjni/fbjni.h>
#include <react/renderer/componentregistry/ComponentDescriptorProviderRegistry.h>
#include <rncore.h>
#include <android/log.h>

#define FLLOG(...) __android_log_print(ANDROID_LOG_INFO, "FedLearnNative", __VA_ARGS__)

#ifdef REACT_NATIVE_APP_CODEGEN_HEADER
#include REACT_NATIVE_APP_CODEGEN_HEADER
#endif
#ifdef REACT_NATIVE_APP_COMPONENT_DESCRIPTORS_HEADER
#include REACT_NATIVE_APP_COMPONENT_DESCRIPTORS_HEADER
#endif

#include "NativeFedLearnCore.h"

namespace facebook::react {

class NativeFedLearnCoreTurboModule : public TurboModule {
 public:
  NativeFedLearnCoreTurboModule(std::shared_ptr<CallInvoker> jsInvoker)
      : TurboModule("NativeFedLearnCore", std::move(jsInvoker)) {

    methodMap_["loadModel"] = MethodMetadata{
        1, [](jsi::Runtime& rt, TurboModule& tm,
              const jsi::Value* args, size_t) -> jsi::Value {
          auto& self = static_cast<NativeFedLearnCoreTurboModule&>(tm);
          return jsi::Value(self.impl_.loadModel(args[0].asString(rt).utf8(rt)));
        }};

    methodMap_["getModelInfo"] = MethodMetadata{
        0, [](jsi::Runtime& rt, TurboModule& tm,
              const jsi::Value*, size_t) -> jsi::Value {
          auto& self = static_cast<NativeFedLearnCoreTurboModule&>(tm);
          return jsi::String::createFromUtf8(rt, self.impl_.getModelInfo());
        }};

    methodMap_["trainStep"] = MethodMetadata{
        3, [](jsi::Runtime& rt, TurboModule& tm,
              const jsi::Value* args, size_t) -> jsi::Value {
          auto& self = static_cast<NativeFedLearnCoreTurboModule&>(tm);
          return jsi::String::createFromUtf8(
              rt, self.impl_.trainStep(args[0].asString(rt).utf8(rt),
                                       static_cast<int>(args[1].asNumber()),
                                       static_cast<float>(args[2].asNumber())));
        }};

    methodMap_["connect"] = MethodMetadata{
        2, [](jsi::Runtime& rt, TurboModule& tm,
              const jsi::Value* args, size_t) -> jsi::Value {
          auto& self = static_cast<NativeFedLearnCoreTurboModule&>(tm);
          return jsi::Value(self.impl_.connect(args[0].asString(rt).utf8(rt),
                                               args[1].asString(rt).utf8(rt)));
        }};

    methodMap_["disconnect"] = MethodMetadata{
        0, [](jsi::Runtime&, TurboModule& tm,
              const jsi::Value*, size_t) -> jsi::Value {
          auto& self = static_cast<NativeFedLearnCoreTurboModule&>(tm);
          self.impl_.disconnect();
          return jsi::Value::undefined();
        }};

    methodMap_["startFedAvgLoop"] = MethodMetadata{
        1, [](jsi::Runtime& rt, TurboModule& tm,
              const jsi::Value* args, size_t) -> jsi::Value {
          auto& self = static_cast<NativeFedLearnCoreTurboModule&>(tm);
          self.impl_.startFedAvgLoop(args[0].asString(rt).utf8(rt));
          return jsi::Value::undefined();
        }};

    methodMap_["startDeComFLLoop"] = MethodMetadata{
        1, [](jsi::Runtime& rt, TurboModule& tm,
              const jsi::Value* args, size_t) -> jsi::Value {
          auto& self = static_cast<NativeFedLearnCoreTurboModule&>(tm);
          self.impl_.startDeComFLLoop(args[0].asString(rt).utf8(rt));
          return jsi::Value::undefined();
        }};

    methodMap_["stopTraining"] = MethodMetadata{
        0, [](jsi::Runtime&, TurboModule& tm,
              const jsi::Value*, size_t) -> jsi::Value {
          auto& self = static_cast<NativeFedLearnCoreTurboModule&>(tm);
          self.impl_.stopTraining();
          return jsi::Value::undefined();
        }};

    methodMap_["getStatus"] = MethodMetadata{
        0, [](jsi::Runtime& rt, TurboModule& tm,
              const jsi::Value*, size_t) -> jsi::Value {
          auto& self = static_cast<NativeFedLearnCoreTurboModule&>(tm);
          return jsi::String::createFromUtf8(rt, self.impl_.getStatus());
        }};

    methodMap_["setZOConfig"] = MethodMetadata{
        1, [](jsi::Runtime& rt, TurboModule& tm,
              const jsi::Value* args, size_t) -> jsi::Value {
          auto& self = static_cast<NativeFedLearnCoreTurboModule&>(tm);
          self.impl_.setZOConfig(args[0].asString(rt).utf8(rt));
          return jsi::Value::undefined();
        }};

    methodMap_["getRecentLogs"] = MethodMetadata{
        0, [](jsi::Runtime& rt, TurboModule& tm,
              const jsi::Value*, size_t) -> jsi::Value {
          auto& self = static_cast<NativeFedLearnCoreTurboModule&>(tm);
          return jsi::String::createFromUtf8(rt, self.impl_.getRecentLogs());
        }};
  }

  fedlearn::NativeFedLearnCoreImpl impl_;
};

// --------------- Standard RN callbacks ---------------

void registerComponents(
    std::shared_ptr<const ComponentDescriptorProviderRegistry> registry) {
#ifdef REACT_NATIVE_APP_COMPONENT_REGISTRATION
  REACT_NATIVE_APP_COMPONENT_REGISTRATION(registry);
#endif
  autolinking_registerProviders(registry);
}

std::shared_ptr<TurboModule> cxxModuleProvider(
    const std::string& name,
    const std::shared_ptr<CallInvoker>& jsInvoker) {
  FLLOG("cxxModuleProvider called with name: %s", name.c_str());
  if (name == "NativeFedLearnCore") {
    FLLOG("Creating NativeFedLearnCoreTurboModule");
    return std::make_shared<NativeFedLearnCoreTurboModule>(jsInvoker);
  }
  return autolinking_cxxModuleProvider(name, jsInvoker);
}

std::shared_ptr<TurboModule> javaModuleProvider(
    const std::string& name,
    const JavaTurboModule::InitParams& params) {
#ifdef REACT_NATIVE_APP_MODULE_PROVIDER
  auto module = REACT_NATIVE_APP_MODULE_PROVIDER(name, params);
  if (module != nullptr) {
    return module;
  }
#endif
  if (auto module = rncore_ModuleProvider(name, params)) {
    return module;
  }
  if (auto module = autolinking_ModuleProvider(name, params)) {
    return module;
  }
  return nullptr;
}

}  // namespace facebook::react

JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM* vm, void*) {
  FLLOG("JNI_OnLoad called for appmodules");
  return facebook::jni::initialize(vm, [] {
    FLLOG("Setting cxxModuleProvider");
    facebook::react::DefaultTurboModuleManagerDelegate::cxxModuleProvider =
        &facebook::react::cxxModuleProvider;
    facebook::react::DefaultTurboModuleManagerDelegate::javaModuleProvider =
        &facebook::react::javaModuleProvider;
    facebook::react::DefaultComponentsRegistry::
        registerComponentDescriptorsFromEntryPoint =
            &facebook::react::registerComponents;
    FLLOG("All providers set successfully");
  });
}
