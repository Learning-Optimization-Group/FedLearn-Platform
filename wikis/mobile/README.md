# FedLearn Mobile — Wiki

> **Part of:** [FedLearn Platform Docs](../README.md)  
> **Stack:** React Native 0.80, TypeScript, native C++ (ExecuTorch on the shared core + Android; iOS libtorch xcframework build glue migration still pending) via a TurboModule (JSI) bridge, Android + iOS  
> **Version:** `2.1.0` (adopted the **Ember** design system + brand fonts this cycle)

The mobile client (`mobile_client/`) is an **on-device** federated-learning participant for phones and tablets. The React Native / TypeScript layer owns the UI, authentication, and round orchestration; the heavy lifting — the **DeComFL zeroth-order (ZO) training path** — runs **natively in C++ on the ExecuTorch runtime** (the shared C++ core and the Android build link ExecuTorch, not libtorch/ATen; the iOS libtorch xcframework glue is a pending, incompatible scaffold), reached through a TurboModule bridge. Training data never leaves the device.

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
├── shared/             # C++ FL core (ExecuTorch runtime) + gtest parity/dtype/marshal tests
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

The **C++ core is fully implemented and tested** (parity, dtype, serialize round-trip, DeComFL equivalence, gRPC marshal). What still gates a shippable release: the cross-compiled **ARM64 ExecuTorch + gRPC** artifacts and buf-generated stubs (the Android core links ExecuTorch statically; the iOS libtorch xcframework wiring is a separate, still-pending migration), real **signing configs** for both platforms, the shared `@fedlearn/tokens` package replacing the local theme placeholder, and on-device training-data wiring.

### Host parity gate (run the C++ core tests in isolation)

The C++ core links the **ExecuTorch runtime** (no libtorch/ATen), so build ExecuTorch v1.3.1
from source first, then point the build at it (must match the pinned torch 2.12.0):

```bash
# ET_SRC        — ExecuTorch v1.3.1 source tree (the dir MUST be named "executorch")
# ET_BUILD      — its host build output (cmake-out); the static libs are resolved out of it
# TORCH_INCLUDE — a venv torch include dir (supplies only the one missing headeronly macro header)
cmake -S mobile_client -B mobile_client/build -DFEDLEARN_BUILD_TESTS=ON \
  -DET_SRC=/tmp/executorch -DET_BUILD=/tmp/executorch/cmake-out \
  -DTORCH_INCLUDE=<venv>/lib/pythonX/site-packages/torch/include
cmake --build mobile_client/build -j
ctest --test-dir mobile_client/build --output-on-failure
# Python side of the same contract:
cd framework && PYTHONPATH=src pytest tests/test_perturbation.py -v
```

CI (`.github/workflows/mobile.yml`) runs the proto-mirror, Python-parity, and C++-parity gates; the golden-vector test gates the build.

---

## On-device model bundle — current state (honest disclosure)

The on-device DeComFL path does **not** yet load a per-recipe model. What it loads today is a
**fixed golden TinyNet fixture**: `Linear(4,5) → ReLU → Linear(5,3)` with `fc2` frozen —
**43 total parameters, 25 trainable** (`flat_dim = 25`). This is the committed fixture at
`framework/tests/fixtures/decomfl_golden/` (see `zo_manifest.json`), the same fixture that
backs the C++ parity gate.

In the end-to-end on-device run, that fixture is *staged as the device's local partition* — it
stands in for genuine per-device data, and its `.pte` loss/infer graphs stand in for a real
per-recipe model. This is a deliberate MVP shortcut, **not representative federation**:
per-recipe / per-device bundles and real on-device data are an explicit post-MVP step. The full
plumbing, phases, and manual acceptance runbook are documented in
[`mobile_client/ON_DEVICE_TRAINING_E2E.md`](../../mobile_client/ON_DEVICE_TRAINING_E2E.md)
(the loaded model reports exactly 25 trainable params, and a round's loss ≈ the fixture
`golden_loss` ~1.097).

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
