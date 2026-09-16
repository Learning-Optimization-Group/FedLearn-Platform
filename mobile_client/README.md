# FedLearn Mobile Client

An on-device federated-learning client: **React Native 0.80** on top of a **native C++ core** that
runs training on the phone through **ExecuTorch**, reached from JS over a **TurboModule (JSI)**
bridge. Raw features and labels are read from app-private files by the native core and never leave
the device.

Architecture deep-dive: [`../wikis/mobile/README.md`](../wikis/mobile/README.md).
Bridge build/codegen specifics: [`bridge/README.md`](bridge/README.md).

> **RN** = React Native · **FL** = Federated Learning · **ZO** = Zeroth-Order optimization ·
> **ET** = ExecuTorch · **JSI** = JavaScript Interface · **JNI** = Java Native Interface ·
> **NDK** = Native Development Kit · **CI** = Continuous Integration.

---

## Platform status — read this before you build

| Platform | Status |
|---|---|
| **Android** (`arm64-v8a`) | **The working path.** `app/build.gradle`'s `externalNativeBuild` points at `android/app/src/main/jni/CMakeLists.txt`, which runs React Native's own `ReactNative-application.cmake` and **grafts the FL core into `libappmodules.so`** (`add_subdirectory(shared)` + `target_link_libraries(appmodules fedlearn_core fedlearn_grpc)`; core + gRPC + training extension). RN globs `OnLoad.cpp` from that same directory, and it registers the `cxxModuleProvider` in `JNI_OnLoad` so `TurboModuleRegistry.get('NativeFedLearnCore')` resolves. Committed Gradle wrapper; `arm64-v8a` is the only shipped ABI. |
| **iOS** | **Native core not wired yet — the build glue migration is pending.** The JS app builds and runs; the native FL core does not. |

The iOS blocker is specific and worth stating plainly: `ios/FedLearnCore.podspec` still vendors a
cross-compiled **libtorch** xcframework, while `shared/` targets the **ExecuTorch** runtime. Those
are different, incompatible runtimes, so the iOS native wiring will not link as-is — it is a
scaffold (the podspec says so in its own header). Swapping the vendored framework for an ExecuTorch
iOS runtime and reconciling the bridge is tracked as **MO-14**. `FEDLEARN_NATIVE_IOS` is therefore
strictly opt-in on a dev machine and must not be enabled in CI. `scripts/build_libtorch_arm64.sh`
is retained *only* for that podspec; the Android and shared-core path is ExecuTorch.

When the native module is absent, nothing crashes: `bridge/specs/NativeFedLearnCore.ts` uses
`TurboModuleRegistry.get()` (not `getEnforcing()`), exposes `isNativeCoreAvailable()`, and falls
back to a typed no-op core whose methods **reject** with a clear message. The app disables its
on-device training entry point instead of taking down the JS bundle.

---

## What actually runs on the device

Two training paths, both native C++:

- **DeComFL (zeroth-order)** — the default. The device evaluates forward passes only; per round it
  uploads **perturbation seeds and K·P gradient scalars**, never weights. This is what makes the
  mobile client viable: no backward pass, and the uplink is a handful of numbers.
- **First-order FedAvg** — real backprop via ExecuTorch's training extension, uploading a weight
  blob that any gradient-aggregating server (FedAvg / FedProx / FedOpt / Robust) consumes over
  `SubmitModelUpdateStream`. It is **capability-gated**: it runs only when the backend provisioned a
  trainable `.pte` for the run (`manifest.firstOrderSupported`). Without one, `src/lib/training.ts`
  refuses **fail-closed** with `MobileFedAvgUnsupportedError` before touching the device, rather
  than uploading ZO scalars a FedAvg server cannot aggregate.

Models and data arrive per-run, not baked in: `src/lib/modelProvisioning.ts` fetches
`GET /api/runs/{runId}/model-bundle`, downloads each binary (the weights-free loss `.pte`, the
infer `.pte`, `inputs.f32`, `targets.i64`, and the optional `trainable.pte`), and stages each one
through `stageBundleFile`, which **sha256-verifies the decoded bytes before writing** and refuses
any file the bundle did not declare a hash for. A bundle that isn't staged yet 404s gracefully.

> Backend-side, `ScriptModelBundleStager` stages fixture-backed recipes (today: `TINYNET_GOLDEN`)
> with a stdlib-only copy needing no ExecuTorch toolchain, and takes a best-effort real export path
> (`scripts/export_model.py --recipe …`) for everything else. The TinyNet fixture
> (`Linear(4,5)→ReLU→Linear(5,3)`, `fc2` frozen: 43 params, 25 trainable) still stands in as the
> device's local partition on demo runs — genuine per-device data is a deliberate post-MVP step.
> See [`ON_DEVICE_TRAINING_E2E.md`](ON_DEVICE_TRAINING_E2E.md).

The app shell is a four-tab bottom bar (Home · Projects · Models · Settings) over a native stack
that pushes Model Testing, Playground and Project Detail; `src/state/TrainingContext.tsx` is the
single owner of the shared run (STOMP stream, heartbeat, stop semantics, log ring).

---

## The determinism contract (read this first)

DeComFL only works if the Python server and this C++ client regenerate **identical** perturbation
vectors from the same seed and produce matching gradient scalars. A divergent `z` silently corrupts
aggregation — no error, just a model that quietly stops learning.

- **Source of truth:** `../framework/src/fedlearn/estimators/perturbation.py`
  (`canonical_perturbation`). It generates `z ~ N(0, I_d)` **on the CPU** with a *local* generator;
  callers then move it to their device. That is the only way a CPU server and a phone agree.
- **Frozen golden vectors:** `../framework/tests/fixtures/decomfl_golden/` (`*.f32` + `manifest.json`,
  torch **2.12.0**).
- **The C++ reproduction:** `shared/include/fedlearn/RandnEngine.h` — an **ATen-free** reimplementation
  of `torch.randn(n, generator=Generator("cpu").manual_seed(seed))`, MT19937 plus PyTorch's own
  scalar/vectorised Box-Muller split at `n = 16`. No libtorch dependency, which is what let the core
  migrate from libtorch to ExecuTorch without touching the contract.
- **The gate:** `shared/tests/randn_parity_test.cpp`. If it fails, the mobile build **must not
  ship**.
- **Re-freeze the fixture only on an intentional torch bump:**
  `cd ../framework && PYTHONPATH=src python tests/fixtures/decomfl_golden/generate.py` — then re-run
  the C++ parity test and review the diff. Never hand-edit the vectors.

The multi-round trajectory is pinned too: `framework/tests/test_decomfl_multiround.py` freezes an
N-round DeComFL endpoint that `shared/tests/et_multiround_test.cpp` replays in C++.

---

## The wire is float32-only, and that is a design constraint

Model state travels as a **deterministic safetensors blob**, never `torch.save`/pickle. The C++ side
is a hand-written, **libtorch-free** decoder — `shared/src/Safetensors.cpp`, a small
recursive-descent parse of the compact header plus a bounds-checked `memcpy`. It supports **F32 and
nothing else** and throws on any other dtype.

That is the whole reason the wire is float32-only: a phone with no ATen has to be able to read it.
The framework holds the same line — `communication/safetensors_codec.py` emits `"dtype":"F32"`
unconditionally — and non-float32 buffers (a BatchNorm `num_batches_tracked` is `int64`) are
**excluded from the federated set** and kept local rather than raising, which is what unblocked
BatchNorm models on the `FULL` arm.

The decoder is also a hardening boundary, since it parses bytes from a possibly-plaintext gRPC
channel: the header length is compared **without** computing `8 + hlen` (an attacker-controlled
`uint64` would wrap), out-of-range `data_offsets` are rejected, and a byte span that isn't a whole
number of floats is refused before it can overflow the destination vector.

`shared/src/DtypeMap.cpp` keeps a separate `SAFE_DTYPES` whitelist in lockstep with `_SAFE_DTYPES`
in `framework/src/fedlearn/communication/serializer.py`; keep those two lists in sync.

---

## One canonical proto, byte-identical mirrors

The gradient-path contract lives at `../proto/fedlearn/v2/fedlearn.proto` (`package fedlearn.v2`),
under `buf` governance. This unit keeps a **byte-identical** in-tree copy at
`proto/fedlearn/v2/fedlearn.proto` for its CMake build. Never hand-edit the mirror — edit canonical,
then copy.

```bash
../scripts/check_proto_mirror.sh    # byte-compares all three mirrors; prints the exact cp fix
```

It is CI-gated in `proto.yml` **and** `mobile.yml` (not `ci.yml`). `proto.yml` additionally runs
`buf lint`, a breaking-change gate against `main`, and a regenerate-is-a-no-op check.

---

## Design system: Ledger

The mobile UI is styled from **Ledger**, the shared cross-platform design system (navy structural
ink on quiet paper surfaces, light-first). `design/tokens.json` at the repo root is the single
source of truth; `design/build-tokens.mjs` generates `src/theme/tokens.generated.ts` and
`src/theme/global.css`. **Both are generated — do not hand-edit them**, and don't hardcode
colours in a component.

- Raw inline / SVG values that must follow the active OS scheme go through `useThemeTokens()`.
- `src/theme/tokens.ts` is a retired shim that re-exports the generated module so old import paths
  keep resolving.
- `../scripts/check_design_tokens.sh` (repo root) runs unconditionally in CI (no path filter), so a hand-edit or a
  `tokens.json` change without regenerating fails the build instead of drifting silently.

---

## Build

### 1. JS app

```bash
npm install --legacy-peer-deps    # RN 0.80; a react-navigation peer range needs the flag
npm start                         # metro
npm run android                   # device/emulator
npm test                          # jest
npm run lint                      # ESLint 9 flat config
npx tsc --noEmit
```

Node **24+** (`engines` in `package.json`; the repo pins 24 in `.nvmrc` / `.tool-versions`).

### 2. C++ core tests in isolation — the parity gate

The core links the **ExecuTorch** runtime (v1.3.1), not libtorch/ATen. You need a host ExecuTorch
build, a CPU torch install matching the pinned golden version (**2.12.0**) — used *only* to supply
the `torch/headeronly/macros/cmake_macros.h` header, never linked — CMake ≥ 3.24, a C++17 compiler,
and network access for the gtest fetch. This mirrors the `cpp-parity` job in
[`../.github/workflows/mobile.yml`](../.github/workflows/mobile.yml).

```bash
# from the repo root
# 1. Build ExecuTorch v1.3.1 from source (the directory MUST be named "executorch");
#    use the configure flags from the mobile.yml cpp-parity job.
export ET_SRC=/tmp/executorch ET_BUILD=/tmp/executorch/cmake-out

# 2. Point TORCH_INCLUDE at a venv torch 2.12.0 include dir (headers only, no linking)
export TORCH_INCLUDE=$(python -c 'import torch, os; print(os.path.join(os.path.dirname(torch.__file__), "include"))')

# 3. Configure + build (add -DFEDLEARN_BUILD_TRAINING=ON for the first-order FedAvg gtests,
#    against an ET build configured with EXECUTORCH_BUILD_EXTENSION_TRAINING=ON)
cmake -S mobile_client -B mobile_client/build -DFEDLEARN_BUILD_TESTS=ON \
      -DET_SRC="$ET_SRC" -DET_BUILD="$ET_BUILD" -DTORCH_INCLUDE="$TORCH_INCLUDE"
cmake --build mobile_client/build -j

# 4. Run them
ctest --test-dir mobile_client/build --output-on-failure
```

Python side of the same contract:

```bash
cd framework && PYTHONPATH=src pytest tests/test_perturbation.py tests/test_decomfl_multiround.py -v --no-cov
```

> **`TORCH_INCLUDE` gotcha.** For any build that also links gRPC, `TORCH_INCLUDE` must be a
> **headeronly-scoped shim**, not the full `torch/include`: the full directory also carries torch's
> bundled `google/protobuf/*` headers, which shadow the gRPC-supplied protobuf and fail the compile
> with `unknown type name 'uint8'`. Copy just `torch/headeronly` into an otherwise-empty directory.
> The host parity gate above is gRPC-free, so the full directory is safe there.

### 3. Optional gRPC transport layer

`FedLearnClient` needs the buf-generated C++ stubs and a cross-compiled gRPC runtime, so it is
**off by default** (`FEDLEARN_BUILD_GRPC=OFF`; the parity gate builds against ExecuTorch alone).
`FederatedLoop` and `DataLoader` live in the core and stay gRPC-free via `IFedLearnClient`.

```bash
# from the repo root
(cd proto && buf generate)        # emits C++ stubs into proto/gen/cpp/fedlearn/v2/
cmake -S mobile_client -B mobile_client/build \
      -DET_SRC="$ET_SRC" -DET_BUILD="$ET_BUILD" -DTORCH_INCLUDE="$TORCH_INCLUDE" \
      -DFEDLEARN_BUILD_GRPC=ON \
      -DGENERATED_PROTO_DIR="$PWD/proto/gen/cpp"
cmake --build mobile_client/build -j
ctest --test-dir mobile_client/build --output-on-failure   # incl. fedlearn_grpc_tests (marshal)
```

`buf.gen.yaml` pins the C++ plugin to `protocolbuffers/cpp:v27.2` deliberately — it must match the
protobuf that gRPC 1.67.1 bundles, or the native compile fails with *"Protobuf C++ gencode is built
with an incompatible version"*.

### 4. Android APK with the native core

Needs the Android SDK/NDK (r27) plus the cross-compiled ARM64 artifacts, which the committed,
version-pinned scripts produce (the artifacts themselves are not committed):

```bash
# from mobile_client/ — the scripts write into ./.artifacts relative to the working directory
export ANDROID_NDK=/path/to/android-sdk/ndk/27.1.12297006   # both scripts require it

bash scripts/build_executorch_arm64.sh                      # ET_VERSION defaults to 1.3.1 -> .artifacts/
GRPC_CPP_VERSION=v1.67.1 bash scripts/build_grpc_arm64.sh   # no default — the script exits if unset
(cd ../proto && buf generate)

(cd android && ./gradlew assembleRelease \
   -PET_SRC=… -PET_BUILD=… -PTORCH_INCLUDE=… -PGRPC_DIR=… -PGENERATED_PROTO_DIR=…)
```

The APK build turns `FEDLEARN_BUILD_GRPC` and `FEDLEARN_BUILD_TRAINING` **on** and
`FEDLEARN_BUILD_TESTS` off (`android/app/src/main/jni/CMakeLists.txt`), and links the ExecuTorch
static libs into `libappmodules.so` — there is no `executorch-android` AAR, and the app never calls
ExecuTorch's Java API. (`bridge/android/jni/CMakeLists.txt` still builds a standalone
`libfedlearn_jni.so` target with the same flags, but Gradle no longer references it — the app build
moved into `android/app/src/main/jni/` in `1edf350`.)

**Still outstanding:** a real release signing config for both platforms (Android and iOS currently
use debug/none), and the iOS runtime migration above.

---

## Two measured on-device facts

Both come from the on-device benchmark campaign and are **not in conflict** — one is about
inference, the other about training. Don't carry either across.

**1. Inference: the shipped APK links reference kernels, and XNNPACK is ~83× faster.**
`shared/CMakeLists.txt:32` links `portable_kernels` — ExecuTorch's *reference* CPU kernels, not an
optimised backend. On the production frozen backbone (ResNet-18 @ 224×224, one mid-range handset,
Adreno 630) at batch 1: **4,093 ms with the shipped kernels vs 49.1 ms with XNNPACK**, and XNNPACK
is also 3.7× faster than the Vulkan GPU path (183 ms). All three backends match a PyTorch reference
to ~1e−7. Linking XNNPACK for the **feature-extraction / inference** path is a build-configuration
change with no model, protocol or API impact.

**2. Training: do NOT link XNNPACK.** At ResNet scale the XNNPACK-delegated *trainable* graph is
**18% faster per step and the model never learns** — `min_loss` stays at its initial value while the
portable build converges. It also costs +51% peak RSS and doubles the `.pte` (44.9 → 89.6 MB, a real
per-round wire cost). The first backward is *correct* (step-0 gradients agree — `fc.weight`
grad-L2 exactly, `conv1.weight` to ~1e−4); the divergence appears from step 1, consistent with the
delegate computing against stale cached weights.
Delegated training is documented as unsupported upstream, so this is an unsupported path behaving
badly rather than a filable defect — but a timing-only benchmark would have shipped a silently
broken trainer.

Caveat worth keeping attached to both numbers: **one handset, one SoC.** A flagship part could
plausibly change the ordering.

---

## CI

| Workflow | What it gates |
|---|---|
| `ci.yml` → `mobile-js` | `check_no_skipped_tests.sh mobile_client`, then `npm ci && npm run lint && npx tsc --noEmit && npm run test:coverage`. Path-filtered on `mobile_client/**`; the jest `coverageThreshold` in `package.json` is enforced. |
| `ci.yml` → `design-tokens` | `check_design_tokens.sh` — the generated mobile token files must match `design/tokens.json`. Runs unconditionally. |
| `mobile.yml` | `proto-mirror`, `python-parity` (the Python golden reproduces itself), `cpp-parity` (**the golden-vector test gates the build**). Plus `android-so-size`, a 60 MB native-lib budget that runs on the **nightly schedule** or on demand via the repo variable `MOBILE_NATIVE_CI=true` (TE-8) — too heavy for every PR, but it must not bit-rot. The nightly run uploads the APK. **Known stale:** that step greps the APK for `libfedlearn_jni.so`, which the current app build no longer emits (it produces `libappmodules.so`), so the size check needs repointing before the job can pass. |
| `proto.yml` | `buf lint` + breaking-change gate vs `main` + regenerate-is-a-no-op + the mirror check. |
| `release-mobile.yml` | Tag-triggered on **`mobile-v*`**. Cross-compiles ARM64 ExecuTorch + the gRPC C++ runtime from source, runs `buf generate`, and assembles a release APK whose native lib (`libappmodules.so`) embeds the FL core. **Android only** — iOS needs full Xcode, hand-built xcframeworks and signing, none of it CI-reproducible. |

There is no committed pre-commit config; these workflows are the gates.

---

## Conventions

- One product name: **FedLearn**. The v1 `FedMob` / `com.mobileclientnew` names are retired.
- One canonical proto, `buf`-generated, byte-identical mirrors. Never hand-edit a mirror.
- No committed dataset blobs — demo data is fetched (`scripts/fetch_demo_data.sh`).
- The C++ `SAFE_DTYPES` whitelist stays in lockstep with the Python serializer whitelist.
- Style from the generated Ledger tokens; never hardcode a colour or a spacing value.
- The 100M model tier is **never** offered on a phone (`src/lib/deviceClass.ts`) — its transient
  zeroth-order working set OOMs mid-tier Android.
