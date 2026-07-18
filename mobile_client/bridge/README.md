# TurboModule bridge — React Native ↔ C++ FL core

This is the boundary that exposes the native C++ federated-learning core (`shared/`) to the
React Native app (15-LLD-mobile.md §5.1, §13 task 11). It is a **C++ (CXX) TurboModule** shared
by Android and iOS, so the FL logic is written once.

> **RN** = React Native · **JSI** = JavaScript Interface · **JNI** = Java Native Interface ·
> **CXX TurboModule** = a TurboModule whose implementation is pure C++ (not Java/ObjC) ·
> **codegen** = RN's TurboModule code generator.

## Files

```
bridge/
├── specs/NativeFedLearnCore.ts     # the TYPED codegen spec (source of truth for the interface)
├── common/
│   ├── BridgeTypes.h               # plain-C++ mirrors of the spec types (no JSI dependency)
│   ├── FedLearnCoreModule.h        # the CXX TurboModule declaration
│   └── FedLearnCoreModule.cpp      # impl: pure `do*` FL logic + the thin JSI/Promise layer
├── android/jni/
│   ├── OnLoad.cpp                  # provider: name -> FedLearnCoreModule (wire into the app delegate)
│   └── CMakeLists.txt              # arm64-v8a JNI lib; links fedlearn_core + fedlearn_grpc + RN
└── ios/NativeFedLearnCore.mm       # provider for the iOS RCTTurboModuleManagerDelegate
```

## Design — why it is shaped this way

- **One round per bridge call.** `runDeComFLRound` / `runFedAvgRound` each run exactly one round
  and return; the RN layer loops, checks the server deadline, samples device metrics, keeps the
  Android foreground service alive, and honors a user `stop()` between rounds (15-LLD §5.1 note;
  avoids multi-minute native calls being killed by Android Doze / iOS suspension).
- **Logic is isolated from the glue.** The private `do*` methods contain the real work and use
  only plain `bridge::*` structs + the verified core (`ModelManager`, `FedLearnClient`,
  `FederatedLoop`). The public methods are a thin JSI/Promise layer that marshals structs ↔
  `jsi::Object` field-by-field — **no hand-built JSON** (fixes A6 §L3, the v1 stringly-typed bridge).
- **Typed everywhere.** Every method returns a typed object the spec describes; `infer` returns a
  real softmax (`logits`/`probabilities`/`argmax`), killing the v1 fake `exp(-loss)` chart (C5 §3).

## Honest status — NOT buildable in this repo's CI-less environment

The JSI/TurboModule layer is **React Native New Architecture, version-specific**, and depends on
artifacts that are generated/cross-compiled on the build machine:

1. **codegen** produces the base class (`NativeFedLearnCoreCxxSpecJSI.h`) and the exact JSI method
   signatures from `specs/NativeFedLearnCore.ts`. The signatures in `FedLearnCoreModule.h` are
   written to the documented CXX-TurboModule pattern and **must be reconciled** against the
   generated header for the pinned RN version.
2. `react::createPromiseAsJSIValue` + `react::Promise` + the `CallInvoker` (used to resolve
   promises on the JS thread after the round runs on a worker) are RN runtime helpers.
3. The Android `ReactAndroid::reactnative` prefab target name and the iOS
   `RCTTurboModuleManagerDelegate` hook vary by RN version.

The `do*` logic and the `bridge::*` types are portable and reviewable on their own; the glue is
flagged in-file. None of this compiles without the RN toolchain + the gRPC/libtorch ARM64
artifacts, so it is gated on the build machine / `mobile.yml` CI.

### iOS is a scaffold — the JS app never crashes when the module is absent (MO-5)

`specs/NativeFedLearnCore.ts` resolves the native module with `TurboModuleRegistry.get()` (not
`getEnforcing()`), so **importing the spec never throws at load** even when the C++ core wasn't
compiled in. When the module is absent the default export is a typed fallback whose methods reject
with `native FL core unavailable on this platform` *only when invoked*, and `isNativeCoreAvailable()`
returns `false` so the RN app disables the training entry point (see `src/screens/HomeScreen.tsx`)
instead of launching into a crash. iOS is that absent case today: `ios/FedLearnCore.podspec` vendors a
**libtorch** xcframework that is incompatible with the shared **ExecuTorch** core, so
`FEDLEARN_NATIVE_IOS` stays strictly opt-in and the real iOS port is tracked in **MO-14**.

## Build & wire (on the build machine)

```bash
cd mobile_client && npm install
npx react-native codegen          # generates the CxxSpec from specs/NativeFedLearnCore.ts
cd ../proto && buf generate       # C++ gRPC stubs the gRPC layer needs
# Android: android/app/build.gradle externalNativeBuild -> bridge/android/jni/CMakeLists.txt
cd ../mobile_client/android && ./gradlew assembleRelease
```

Wire the provider into the app:
- **Android:** call `facebook::react::FedLearnCore_setDataDir(filesDir)` at startup, and return
  `FedLearnCore_cxxModuleProvider(name, jsInvoker)` from the TurboModuleManagerDelegate's
  `cxxModuleProvider`.
- **iOS:** return `facebook::react::FedLearnCoreModuleProvider(name, jsInvoker)` from the app
  delegate's `getTurboModule:jsInvoker:`.

## Open seam — on-device training data (task 14)

`RoundConfig` carries no data; raw features/labels live only on the device (FL invariant). Until
the RN app layer (task 14) wires the on-device dataset, call
`FedLearnCoreModule::setTrainingDataFromFiles(inputsF32Path, shape, targetsI64Path)` to supply a
validated batch. A round throws a clear error if no data is set.
