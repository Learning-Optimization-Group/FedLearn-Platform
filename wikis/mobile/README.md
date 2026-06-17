# FedLearn Mobile — Wiki

> **Part of:** [FedLearn Platform Docs](../README.md)  
> **Stack:** React Native 0.80, TypeScript, native C++ (libtorch) via a TurboModule (JSI) bridge, Android + iOS  
> **Version:** `2.1.0` (adopted the **Ember** design system + brand fonts this cycle)

The mobile client (`mobile_client/`) is an **on-device** federated-learning participant for phones and tablets. The React Native / TypeScript layer owns the UI, authentication, and round orchestration; the heavy lifting — the **DeComFL zeroth-order (ZO) training path** — runs **natively in C++ on libtorch**, reached through a TurboModule bridge. Training data never leaves the device.

The single product name is **FedLearn** (the v1 `FedMob` / `com.mobileclientnew` names are retired). "Ember" is the design system/theme adopted in `2.1.0`, not a product rename.

---

## Architecture

```
React Native app (TS)         bridge/                 shared/ (C++ FL core)
┌──────────────────┐   JSI   ┌──────────────────┐   ┌──────────────────────────┐
│ screens / nav     │◄──────►│ NativeFedLearnCore │──►│ Perturbation (canonical RNG)│
│ runJoin (§6.1)    │ Turbo  │  (codegen spec +   │   │ ModelManager / ZerothOrder  │
│ nativeCore wrapper│ Module │   thin JSI layer)  │   │ DeComFLClient / FederatedLoop│
│ device guard      │        │ Android JNI / iOS  │   │ FedLearnClient (gRPC, opt-in)│
└──────────────────┘        │  .mm registration  │   └──────────────────────────┘
                             └──────────────────┘
```

- **JS/TS layer** (`src/`) — `TrainingScreen` (join + DeComFL round loop + live metrics + device-class guard), `ModelLibraryScreen`, `ModelTestingScreen` (real softmax). NativeWind + token theme, lucide icons, a 3-tab navigator.
- **TurboModule bridge** (`bridge/`) — `specs/NativeFedLearnCore.ts` is the typed codegen source of truth; `bridge/common/FedLearnCoreModule.{h,cpp}` is pure `do*` logic with a thin JSI layer. Android registers the `cxxModuleProvider` in `JNI_OnLoad` (`bridge/android/jni/OnLoad.cpp`); iOS wires it via `bridge/ios/` + the New-Arch factory delegate. The bridge separates JSI glue from the core logic so the C++ core stays platform-agnostic.
- **Native C++ core** (`shared/`) — the FL engine: canonical perturbation RNG, dtype whitelist, SHA-256 verify-before-load, `ModelManager`, `ZerothOrderEstimator`, `DeComFLClient`, and an **opt-in** gRPC layer (`FedLearnClient` / `FederatedLoop` / `DataLoader`).

---

## The determinism contract (the load-bearing invariant)

DeComFL only works if the Python server and this C++ client regenerate **identical** perturbation vectors from the same seed and produce matching gradient scalars. `canonical_perturbation(seed, n)` generates `z ~ N(0, I_d)` **on the CPU** with a *local* generator; callers then move it to their device.

- The **Python** implementation (`framework/src/fedlearn/estimators/perturbation.py`) is the **source of truth**.
- The **golden vectors** (`framework/tests/fixtures/decomfl_golden/`, frozen at torch 2.12.0) are derived from it.
- The **C++ parity gate** (`shared/tests/rng_parity_test.cpp`) is a **release blocker** — if `z` diverges, the build must not ship, because a divergent `z` silently corrupts aggregation.

---

## Project layout

```
mobile_client/
├── shared/             # C++ FL core (libtorch) + gtest parity/dtype/marshal tests
│   ├── src/  include/  tests/
│   └── CMakeLists.txt
├── bridge/             # TurboModule: TS spec, common C++ module, Android JNI + iOS .mm
├── src/                # React Native app: lib/, screens/, navigation/, theme/
├── android/            # Gradle project (+ committed wrapper), externalNativeBuild → JNI
├── ios/                # FedLearn.xcodeproj (generated), Podfile, Swift AppDelegate
├── proto/              # byte-mirror of canonical proto/fedlearn/v2 (mirror-checked)
├── scripts/            # ARM64 libtorch/gRPC cross-compile, model export, demo data
└── CMakeLists.txt      # host build (parity gate)
```

---

## Build status

Both app projects are now **buildable**: the two prior template-scaffolding blockers were resolved and committed — the Android **Gradle wrapper** (`./gradlew` bootstraps Gradle 8.14.1) and the iOS **`FedLearn.xcodeproj`** (regenerate with `ios/generate_xcodeproj.sh`). The native FL core is wired into both targets (Android `externalNativeBuild` → `libfedlearn_jni.so`; iOS `FedLearnCore.podspec`, enabled with `FEDLEARN_NATIVE_IOS=1`).

The **C++ core is fully implemented and tested** (parity, dtype, serialize round-trip, DeComFL equivalence, gRPC marshal). What still gates a shippable release: the cross-compiled **ARM64 libtorch + gRPC** artifacts and buf-generated stubs, real **signing configs** for both platforms, the shared `@fedlearn/tokens` package replacing the local theme placeholder, and on-device training-data wiring.

### Host parity gate (run the C++ core tests in isolation)

```bash
export LIBTORCH_DIR=/path/to/libtorch      # CPU build is fine; must match torch 2.12.0
cmake -S mobile_client -B mobile_client/build -DLIBTORCH_DIR="$LIBTORCH_DIR"
cmake --build mobile_client/build -j
ctest --test-dir mobile_client/build --output-on-failure
# Python side of the same contract:
cd framework && PYTHONPATH=src pytest tests/test_perturbation.py -v
```

CI (`.github/workflows/mobile.yml`) runs the proto-mirror, Python-parity, and C++-parity gates; the golden-vector test gates the build.

---

## Key cross-component interfaces

- Authenticates against the **Backend** `POST /api/auth/login` (same cookie/JWT contract as the web client).
- Connects to a **Framework** FL server over gRPC and runs the native DeComFL client path.
- Shares the **canonical** `proto/fedlearn/v2` contract (byte-mirrored into `mobile_client/proto/`, enforced by `scripts/check_proto_mirror.sh`). Note: the running Python framework still generates from the older `fedlearn.v1` proto — the v2 contract is ahead of the framework implementation.
- The C++ `SAFE_DTYPES` whitelist stays in lockstep with the Python serializer whitelist.

## Related documentation

- [Framework Wiki](../framework/README.md) — the FL engine and the DeComFL algorithm this client implements natively
- [Framework: DeComFL](../framework/06_decomfl.md) — zeroth-order estimation, seed/gradient protocol
- [Backend: Security & Authentication](../backend/02_security_and_auth.md) — the auth contract the client uses
