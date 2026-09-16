// OnLoad.cpp — RN 0.80 New Architecture native entry point for libfedlearn_jni.so.
//
// Because this app uses a CUSTOM JNI CMakeLists it opts out of React Native's default entry point
// (ReactAndroid/cmake-utils/default-app-setup/OnLoad.cpp) and MUST replicate its full New-Architecture
// wiring here:
//   * cxxModuleProvider  — app C++ TurboModule (NativeFedLearnCore) + autolinked C++ TurboModules
//   * javaModuleProvider — core RN modules (rncore: PlatformConstants, ...) + autolinked Java modules
//   * registerComponents — autolinked Fabric component descriptors (screens/svg/safe-area-context/...)
// Missing any of these aborts at startup or falls back to the legacy paper interop (RNSScreen prop
// crash). `dataDir` is the Android filesDir (gRPC cert/key/CA + on-device data), pushed from Kotlin.
#include <jni.h>

#include <memory>
#include <string>

#include <fbjni/fbjni.h>  // facebook::jni::initialize — MUST run in JNI_OnLoad

#include <ReactCommon/CallInvoker.h>
#include <ReactCommon/JavaTurboModule.h>
#include <ReactCommon/TurboModule.h>

#include <DefaultComponentsRegistry.h>
#include <DefaultTurboModuleManagerDelegate.h>
#include <react/renderer/componentregistry/ComponentDescriptorProviderRegistry.h>
#include <rncore.h>        // rncore_ModuleProvider — core RN TurboModules
#include <autolinking.h>   // autolinking_{cxxModuleProvider,ModuleProvider,registerProviders} (generated)

#include "FedLearnCoreModule.h"

namespace facebook::react {

// App data dir, set once from Kotlin (FedLearnNative.setDataDir) before the first module lookup.
static std::string g_fedlearnDataDir;  // NOLINT(runtime/string)

void FedLearnCore_setDataDir(const std::string& dataDir) { g_fedlearnDataDir = dataDir; }

// Fabric component descriptors for autolinked view libraries (screens, svg, safe-area-context, ...).
void registerComponents(
    std::shared_ptr<const ComponentDescriptorProviderRegistry> registry) {
  autolinking_registerProviders(registry);
}

// App + autolinked C++ TurboModules.
std::shared_ptr<TurboModule> cxxModuleProvider(const std::string& name,
                                               const std::shared_ptr<CallInvoker>& jsInvoker) {
  if (name == "NativeFedLearnCore") {
    return std::make_shared<fedlearn::bridge::FedLearnCoreModule>(jsInvoker, g_fedlearnDataDir);
  }
  return autolinking_cxxModuleProvider(name, jsInvoker);
}

// Core RN modules (rncore: PlatformConstants, ...) then autolinked Java TurboModules.
std::shared_ptr<TurboModule> javaModuleProvider(const std::string& name,
                                                const JavaTurboModule::InitParams& params) {
  if (auto module = rncore_ModuleProvider(name, params)) {
    return module;
  }
  return autolinking_ModuleProvider(name, params);
}

}  // namespace facebook::react

// JNI_OnLoad — MUST initialize fbjni, then install all three New-Architecture entry points. Runs once
// when System.loadLibrary("fedlearn_jni") loads this .so (FedLearnNative.kt init block).
extern "C" JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM* vm, void* /*reserved*/) {
  return facebook::jni::initialize(vm, [] {
    facebook::react::DefaultTurboModuleManagerDelegate::cxxModuleProvider =
        &facebook::react::cxxModuleProvider;
    facebook::react::DefaultTurboModuleManagerDelegate::javaModuleProvider =
        &facebook::react::javaModuleProvider;
    facebook::react::DefaultComponentsRegistry::registerComponentDescriptorsFromEntryPoint =
        &facebook::react::registerComponents;
  });
}

// Kotlin (com.fedlearn.mobile.FedLearnNative) calls this at startup with filesDir, before the first
// TurboModule lookup, so the module is constructed with the app-private cert/data dir.
extern "C" JNIEXPORT void JNICALL Java_com_fedlearn_mobile_FedLearnNative_nativeSetDataDir(
    JNIEnv* env, jclass /*clazz*/, jstring dir) {
  const char* d = env->GetStringUTFChars(dir, nullptr);
  facebook::react::FedLearnCore_setDataDir(std::string(d ? d : ""));
  if (d) env->ReleaseStringUTFChars(dir, d);
}
