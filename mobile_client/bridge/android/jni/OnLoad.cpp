// OnLoad.cpp — registers the C++ (CXX) TurboModule on Android so that
// `TurboModuleRegistry.getEnforcing('NativeFedLearnCore')` in JS resolves to FedLearnCoreModule.
//
// VERIFY-BEFORE-BUILD: the exact registration hook is React Native New Architecture
// version-specific. In recent RN, app-defined C++ TurboModules are surfaced by a
// `cxxModuleProvider` that the app's TurboModuleManagerDelegate consults. Wire the provider below
// into the app (MainApplication's ReactNativeHost / the generated delegate) per the pinned RN
// version. `dataDir` MUST be the Android app filesDir (cert/key/CA + on-device data live there);
// the Java/Kotlin layer passes it at startup.
#include <jni.h>

#include <memory>
#include <string>

#include <ReactCommon/CallInvoker.h>
#include <ReactCommon/TurboModule.h>

#include "FedLearnCoreModule.h"

namespace facebook::react {

// App data dir, set once from Kotlin (e.g. via a JNI call) before the first module lookup.
static std::string g_fedlearnDataDir;  // NOLINT(runtime/string)

void FedLearnCore_setDataDir(const std::string& dataDir) { g_fedlearnDataDir = dataDir; }

std::shared_ptr<TurboModule> FedLearnCore_cxxModuleProvider(const std::string& name,
                                                            std::shared_ptr<CallInvoker> jsInvoker) {
  if (name == "NativeFedLearnCore") {
    return std::make_shared<fedlearn::bridge::FedLearnCoreModule>(std::move(jsInvoker),
                                                                  g_fedlearnDataDir);
  }
  return nullptr;
}

}  // namespace facebook::react

// Kotlin (com.fedlearn.mobile.FedLearnNative) calls this at startup with filesDir, before the
// first TurboModule lookup, so the module is constructed with the app-private cert/data dir.
extern "C" JNIEXPORT void JNICALL Java_com_fedlearn_mobile_FedLearnNative_nativeSetDataDir(
    JNIEnv* env, jclass /*clazz*/, jstring dir) {
  const char* d = env->GetStringUTFChars(dir, nullptr);
  facebook::react::FedLearnCore_setDataDir(std::string(d ? d : ""));
  if (d) env->ReleaseStringUTFChars(dir, d);
}

