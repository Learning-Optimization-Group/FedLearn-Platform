# FedLearn Mobile Client (v2)

Native React Native + C++ (libtorch) federated-learning client. Built fresh per
[`docs/v2/build/15-LLD-mobile.md`](../docs/v2/build/15-LLD-mobile.md); see also the
build order in [`docs/v2/build/90-BUILD-SEQUENCE.md`](../docs/v2/build/90-BUILD-SEQUENCE.md).

> **RN** = React Native · **FL** = Federated Learning · **RNG** = Random Number Generator ·
> **ZO** = Zeroth-Order optimization · **gRPC** = Google Remote Procedure Call ·
> **JNI** = Java Native Interface · **NDK** = Native Development Kit · **CI** = Continuous Integration.

---

## Status — what is built so far

This is **increments 1–7: the full mobile unit in code** — the determinism contract, the C++ FL
core, the C++ gRPC layer, the TurboModule bridge, the React Native app layer + screens, the
native-dep prebuild scripts + mobile CI, and the Android/iOS app projects + foreground service +
device-metrics provider (15-LLD §13 tasks 2–19, plus the framework prerequisite). It is the load-bearing
foundation everything else depends on: the DeComFL protocol only works if the Python
server and this C++ client regenerate **identical** perturbation vectors from the same
seed, and produce matching gradient scalars.

| Built | File |
|---|---|
| Python source of truth | `../framework/src/fedlearn/estimators/perturbation.py` (`canonical_perturbation`) |
| Frozen golden vectors | `../framework/tests/fixtures/decomfl_golden/` (`*.f32` + `manifest.json`, torch 2.12.0) |
| Python parity test | `../framework/tests/test_perturbation.py` |
| C++ canonical RNG | `shared/src/Perturbation.cpp` + `include/fedlearn/Perturbation.h` |
| C++ dtype whitelist | `shared/src/DtypeMap.cpp` + `include/fedlearn/DtypeMap.h` |
| C++ parity gate (gtest) | `shared/tests/rng_parity_test.cpp` (**release blocker**, 15-LLD §13.4) |
| C++ dtype test (gtest) | `shared/tests/dtype_map_test.cpp` |
| C++ SHA-256 (verify-before-load) | `shared/src/Sha256.cpp` + `include/fedlearn/Sha256.h` (NIST KAT in `sha256_test.cpp`) |
| C++ ModelManager | `shared/src/ModelManager.cpp` (load + requires_grad-filtered flat params + symmetric serialize) |
| C++ ZerothOrderEstimator | `shared/src/ZerothOrderEstimator.cpp` (double-accumulated loss diff) |
| C++ DeComFLClient | `shared/src/DeComFLClient.cpp` (`fit()` snapshot-restore + `(eta/P)`; `rebuildModel()` Alg.2) |
| ZO golden reference | `../framework/tests/fixtures/decomfl_golden/` (`zo_model_tiny.pt`, batch, `zo_manifest.json`) + `generate_zo.py` |
| C++ FL-core tests (gtest) | `model_manager_test`, `flatparam_filter_test`, `g_scalar_parity_test`, `serialize_roundtrip_test`, `decomfl_equivalence_test` |
| Canonical gRPC proto | `../proto/fedlearn/v2/fedlearn.proto` (+ `buf.yaml`/`buf.gen.yaml`); mirror at `proto/fedlearn/v2/fedlearn.proto`, gated by `../scripts/check_proto_mirror.sh` |
| C++ gRPC client | `shared/src/FedLearnClient.cpp` (dual-channel TLS+mTLS, streaming + chunk framing/sha256, heartbeat-abort thread, proto↔core marshaling) — **opt-in** target |
| C++ federated loop | `shared/src/FederatedLoop.cpp` (DeComFL + FedAvg one-round bodies; torch-version gate; deadline/abort checks) — **opt-in** |
| C++ data loader | `shared/src/DataLoader.cpp` (validated on-device, client-private load) — **opt-in** |
| gRPC marshal test (gtest) | `shared/tests/grpc_marshal_test.cpp` (server-free; proto↔core + codec whitelist) |
| TurboModule spec (TS) | `bridge/specs/NativeFedLearnCore.ts` (typed codegen source of truth) |
| C++ CXX TurboModule | `bridge/common/FedLearnCoreModule.{h,cpp}` + `BridgeTypes.h` (pure `do*` logic + thin JSI layer; real-softmax `infer`) |
| Android / iOS registration | `bridge/android/jni/{OnLoad.cpp,CMakeLists.txt}`, `bridge/ios/NativeFedLearnCore.mm` |
| RN project scaffold | `package.json` (codegenConfig), `tsconfig.json`, `babel/metro/tailwind` config, `app.json`, `index.js` |
| RN app `lib/` | `src/lib/`: `nativeCore` (typed TurboModule wrapper), `clientId` (stable encrypted UUID), `deviceClass` (tier cap; 100M never on mobile), `restClient`, `runJoin` (the §6.1 join flow) |
| RN screens | `src/screens/`: `TrainingScreen` (join + DeComFL round loop + live metrics + device guard), `ModelLibraryScreen`, `ModelTestingScreen` (**real softmax**) — on NativeWind + OKLCH tokens, lucide icons |
| RN nav + root | `src/navigation/AppNavigator.tsx` (3-tab, lucide), `src/App.tsx`; `src/theme/` (local OKLCH token placeholder for `@fedlearn/tokens`) |
| Native-dep prebuild scripts | `scripts/`: `build_libtorch_arm64.sh` + `build_grpc_arm64.sh` (pinned cross-compile), `export_model.py` (1M/10M TorchScript — verified), `fetch_demo_data.sh` (MNIST, not committed) |
| Mobile CI | `../.github/workflows/mobile.yml` — proto-mirror + python-parity + cpp-parity gates (the **golden-vector test gates the build**); `android-so-size` budget job (gated by repo var) |
| Android app project | `android/`: Gradle (root + app, `externalNativeBuild`→JNI) + committed **Gradle wrapper** (`gradlew` + `gradle/wrapper/*`, Gradle 8.14.1), `AndroidManifest.xml`, `network_security_config.xml`, `MainApplication`/`MainActivity` (RN New-Arch host + data-dir init) |
| Foreground service (task 16) | `android/.../FlForegroundService.kt` + `FlServiceModule`/`FlServicePackage` + `src/lib/foregroundService.ts` (started around the round loop in `TrainingScreen`) |
| Device-metrics provider (task 17) | `android/.../DeviceState.kt` + `bridge/android/jni/DeviceStateJni.cpp`; iOS `DeviceState.swift`; shared `bridge/common/DeviceState.{h,cpp}` → `getDeviceMetrics` |
| iOS app project | `ios/`: committed **`FedLearn.xcodeproj`** (generated from the RN 0.80 template by `ios/generate_xcodeproj.sh`), `Podfile` (with `react_native_post_install`), `FedLearn/` (`AppDelegate.swift` RN-0.80 factory, `Info.plist` ATS-on + foreground-only, `LaunchScreen.storyboard`, `PrivacyInfo.xcprivacy`, `DeviceState.swift`, bridging header, app-icon assets), `.xcode.env` |
| Host build | `CMakeLists.txt`, `shared/CMakeLists.txt`, `shared/tests/CMakeLists.txt` |

**Buildability status.** The two template-scaffolding blockers are now resolved and committed: the
Android **Gradle wrapper** (`./gradlew` bootstraps Gradle 8.14.1 — verified) and the iOS
**`FedLearn.xcodeproj`** (generated from the pinned RN 0.80 template by `ios/generate_xcodeproj.sh`;
regenerate any time with that script). Per-machine bring-up:

```bash
npm install --legacy-peer-deps     # RN 0.80 deps (a react-navigation peer range needs this flag)
# Android (needs Android SDK/NDK + cross-compiled ARM64 libtorch/gRPC):
(cd android && ./gradlew assembleRelease -PLIBTORCH_DIR=… -PGENERATED_PROTO_DIR=…)
# iOS (needs full Xcode + CocoaPods):
(cd ios && pod install)            # or re-run ios/generate_xcodeproj.sh, which also pod-installs
```

**Native FL core — wired into both targets:**
- **Android** — `app/build.gradle` `externalNativeBuild` builds `libfedlearn_jni.so` from
  `bridge/android/jni/CMakeLists.txt` (core + gRPC + bridge), and `bridge/android/jni/OnLoad.cpp`
  now registers the `cxxModuleProvider` in `JNI_OnLoad` so JS
  `getEnforcing('NativeFedLearnCore')` resolves. Needs `-PLIBTORCH_DIR`/`-PGENERATED_PROTO_DIR`
  (cross-compiled ARM64 artifacts) to assemble.
- **iOS** — `ios/FedLearnCore.podspec` compiles `shared/` + `bridge/` + the gRPC layer into the app
  target; `bridge/ios/FedLearnFactoryDelegate.mm` is the New-Arch TurboModule hook (wired in
  `AppDelegate.swift` via `#if canImport(FedLearnCore)`); `ios/wire_native.rb` adds
  `DeviceState.swift` + the bridging header to the target. Enable with `FEDLEARN_NATIVE_IOS=1` +
  the libtorch/gRPC xcframework env vars (see the podspec); otherwise the JS shell builds without it.

**Still remaining** (build inputs + release): the cross-compiled **ARM64 libtorch + gRPC** artifacts
and buf-generated stubs for both platforms (`scripts/build_*_arm64.sh`, `buf generate`); a real
**release signing config** for both (Android app `build.gradle` and iOS currently use debug/none);
the shared `@fedlearn/tokens` package replacing the local `src/theme` placeholder; and on-device
training-data wiring (`FedLearnCoreModule::setTrainingDataFromFiles`). Once the Android project
assembles, set the repo variable `MOBILE_NATIVE_CI=true` to enable the `android-so-size` CI job.

The TurboModule bridge (tasks 11–13) is in `bridge/` — see `bridge/README.md` for its
build/codegen/wiring steps and the React Native version-specific caveats.

### Building the opt-in gRPC layer

`FedLearnClient` / `FederatedLoop` / `DataLoader` need the buf-generated C++ stubs and a
cross-compiled gRPC runtime, so they are **off by default** (the parity gate above builds
with only libtorch). To build them:

```bash
cd proto && buf generate          # emits C++ stubs into gen/cpp/fedlearn/v2/
cmake -S mobile_client -B mobile_client/build \
      -DLIBTORCH_DIR="$LIBTORCH_DIR" \
      -DFEDLEARN_BUILD_GRPC=ON \
      -DGENERATED_PROTO_DIR="$PWD/gen/cpp"
cmake --build mobile_client/build -j
ctest --test-dir mobile_client/build --output-on-failure   # incl. fedlearn_grpc_tests (marshal)
```

The end-to-end client↔server smoke (register → 1 DeComFL round → upload K·P scalars) runs
against a dev FL server per 15-LLD §11.6.

---

## The determinism contract (read this first)

`canonical_perturbation(seed, n)` generates `z ~ N(0, I_d)` **on the CPU** with a *local*
generator, then callers move it to their device. This is the only way to get the same `z`
on a CPU server and an Apple/Android device. The Python implementation is the **source of
truth**; the golden vectors in `framework/tests/fixtures/decomfl_golden/` are frozen from
it; this C++ core must reproduce them.

- **Re-freeze the fixture only on an intentional torch bump:**
  `cd framework && PYTHONPATH=src python tests/fixtures/decomfl_golden/generate.py`
  (then re-run the C++ parity test and review any change — never hand-edit the vectors).
- If `rng_parity_test` fails, the mobile build **must not ship** — a divergent `z`
  silently corrupts DeComFL aggregation between the server and this client.

---

## Build & run the C++ core tests in isolation (the parity gate)

Requires a host libtorch matching the pinned torch version (2.12.0). CMake ≥ 3.22, a
C++17 compiler, and network access for the gtest fetch.

```bash
# 1. Get libtorch (CPU build is fine for the host parity gate). Example:
#    download + unzip libtorch 2.12.0 from https://pytorch.org/get-started/locally/
export LIBTORCH_DIR=/path/to/libtorch

# 2. Configure + build
cmake -S mobile_client -B mobile_client/build -DLIBTORCH_DIR="$LIBTORCH_DIR"
cmake --build mobile_client/build -j

# 3. Run the parity + dtype tests
ctest --test-dir mobile_client/build --output-on-failure
```

Python side (same contract, bit-exact):

```bash
cd framework && PYTHONPATH=src pytest tests/test_perturbation.py -v
```

---

## Toolchain for the full app (later increments, 15-LLD §11.1)

Node 24 LTS, JDK 21, the Android SDK + NDK (for `arm64-v8a`), Xcode (iOS), CMake ≥ 3.22,
and the CI-prebuilt libtorch/gRPC ARM64 artifacts (`scripts/build_libtorch_arm64.sh`,
`scripts/build_grpc_arm64.sh` — not yet written). The Android app targets `arm64-v8a`
only; the C++ core is consumed through the JNI TurboModule bridge.

---

## Conventions

- One product name: **FedLearn** (the v1 "FedMob"/`com.mobileclientnew` names are retired).
- One canonical proto (`buf`-generated); no drifted copies (the v1 third proto is deleted).
- No committed dataset blobs; demo data is fetched (`scripts/fetch_demo_data.sh`, later).
- The C++ `SAFE_DTYPES` whitelist stays in lockstep with the Python serializer whitelist.
