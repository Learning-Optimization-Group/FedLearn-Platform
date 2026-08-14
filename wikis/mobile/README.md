# FedLearn Mobile — Wiki

> **Part of:** [FedLearn Platform Docs](../README.md)  
> **Stack:** React Native 0.80, TypeScript, native C++ (ExecuTorch on the shared core + Android; the iOS native wiring is still a **libtorch** scaffold and its ExecuTorch migration is pending — MO-14) via a TurboModule (JSI) bridge, Android + iOS  
> **Version:** `2.1.0` (`mobile_client/package.json`) — the UI renders the **Ledger** design system

The mobile client (`mobile_client/`) is an **on-device** federated-learning participant for phones and tablets. The React Native / TypeScript layer owns the UI, authentication, join/enrol flow and round orchestration; the heavy lifting — the **DeComFL zeroth-order (ZO) training path** and, where the backend provisions a trainable graph, a **first-order (backprop) path** — runs **natively in C++ on the ExecuTorch runtime**, reached through a TurboModule bridge. Training data never leaves the device.

The single product name is **FedLearn** (the v1 `FedMob` / `com.mobileclientnew` names are retired).

> **Design system: Ledger, not Ember.** `src/theme/tokens.generated.ts` and `src/theme/global.css` are **generated** by `design/build-tokens.mjs` from `design/tokens.json` and both carry the Ledger palette (canvas `#F6F3EE`, surface `#FFFFFF`, ink `#191A1C`, muted `#6B6760`, navy accent `#1C314D` / hover `#14243A`; dark family `#0B1622` / `#4F8AC9`). Ledger landed in `2c50672` and rolled onto mobile in `3d54484`, superseding **Ember**, which had superseded *Instrument*. `src/theme/tokens.ts` is now a thin re-export kept only so older import paths resolve — do not hand-edit either generated file; CI's "Design tokens in sync with source of truth" step (`scripts/check_design_tokens.sh`, unconditional in `ci.yml`) fails on drift. The shipped brand fonts are **Hanken Grotesk** (sans *and* display) + **JetBrains Mono** (`src/assets/fonts/`); Bricolage Grotesque was Ember-era and is gone.

---

## Architecture

```
React Native app (TS)          bridge/                  shared/ (C++ FL core)
┌──────────────────────┐  JSI  ┌────────────────────┐  ┌────────────────────────────────┐
│ screens / navigation │◄─────►│ NativeFedLearnCore │─►│ RandnEngine  (canonical RNG)   │
│ TrainingContext      │ Turbo │  (codegen spec +   │  │ DtypeMap / Sha256 / Safetensors│
│ runJoin / training   │ Module│   thin JSI layer)  │  │ ModelManager / ExecutorchModel │
│ modelProvisioning    │       │ Android JNI /      │  │ EtZeroOrder / DeComFLClient    │
│ deviceClass guard    │       │  iOS .mm + podspec │  │ FederatedLoop / DataLoader     │
└──────────────────────┘       └────────────────────┘  │ FedLearnClient (gRPC, opt-in)  │
                                                       └────────────────────────────────┘
```

- **JS/TS layer** (`src/`) — a 4-tab bottom navigator (**Home · Projects · Models · Settings**, `navigation/AppNavigator.tsx`) inside an app stack (`ModelTesting`, `Playground`, `ProjectDetail`), wrapped by an auth root navigator (`navigation/RootNavigator.tsx` → `Login` / `Register`). Training is driven from `ProjectDetailScreen` through `state/TrainingContext.tsx` + `state/trainingReducer.ts`; the round loop itself lives in `lib/training.ts`. NativeWind + the generated token theme, lucide icons.
- **TurboModule bridge** (`bridge/`) — `specs/NativeFedLearnCore.ts` is the typed codegen source of truth; `bridge/common/FedLearnCoreModule.{h,cpp}` is pure `do*` logic behind a thin JSI layer, so the core stays platform-agnostic. Android registers the `cxxModuleProvider` in `JNI_OnLoad` (`bridge/android/jni/OnLoad.cpp`); iOS wires it through `bridge/ios/FedLearnFactoryDelegate.mm` + the New-Arch factory delegate.
- **Native C++ core** (`shared/`) — the FL engine: the ATen-free canonical RNG, the dtype whitelist, SHA-256 verify-before-load, a torch-free safetensors codec, `ModelManager`, the ExecuTorch forward/loss wrapper, the zeroth-order estimator, `DeComFLClient`, `FederatedLoop`, `DataLoader`, and an **opt-in** gRPC transport (`FedLearnClient`).

### What the native core is built from

`shared/CMakeLists.txt` compiles one static `fedlearn_core` from `DtypeMap · Sha256 · ModelManager · DeComFLClient · ExecutorchModel · Safetensors · DataLoader · FederatedLoop`, plus `TrainableExecutorchModel` when `FEDLEARN_BUILD_TRAINING=ON`. Several headers carry logic without a matching `.cpp` — the two on the FL hot path are `RandnEngine.h` (the canonical RNG) and `EtZeroOrder.h` (the ZO g-scalar); `AuthMetadata.h` and `EvalMetrics.h` are header-only too, each with its own gtest. Two CMake options gate the optional layers:

| Option | Default | Adds |
|---|---|---|
| `FEDLEARN_BUILD_TRAINING` | `OFF` (but **`ON` in the Android app build** and in the `cpp-parity` CI job) | `TrainableExecutorchModel` + `FederatedLoop::firstOrderRound`, the ExecuTorch `extension_training` / `extension_module` / `extension_named_data_map` libs, and the `FEDLEARN_HAS_TRAINING` define the bridge compiles against |
| `FEDLEARN_BUILD_GRPC` | `OFF` | `FedLearnClient` + the buf-generated C++ stubs; requires `-DGENERATED_PROTO_DIR=<proto/gen/cpp>` and a cross-compiled gRPC runtime |

---

## The determinism contract (the load-bearing invariant)

DeComFL only works if the Python server and this C++ client regenerate **identical** perturbation vectors from the same seed and produce matching gradient scalars. `canonical_perturbation(seed, n)` generates `z ~ N(0, I_d)` **on the CPU** with a *local* generator; callers then move it to their device.

- The **Python** implementation (`framework/src/fedlearn/estimators/perturbation.py`) is the **source of truth**.
- The **golden vectors** (`framework/tests/fixtures/decomfl_golden/`, frozen at torch 2.12.0) are derived from it.
- The **C++ side is ATen-free.** `shared/include/fedlearn/RandnEngine.h` reimplements `torch.randn(n, generator=Generator("cpu").manual_seed(seed), dtype=float32)` from first principles — PyTorch's MT19937 engine, the scalar Box-Muller path for `n < 16` and the vectorised `normal_fill` path for `n ≥ 16` — so the parity gate survived the migration off libtorch onto ExecuTorch. There is no `Perturbation.cpp`; the engine is a header.
- The **parity gate is `shared/tests/randn_parity_test.cpp`** (not `rng_parity_test.cpp`, which covered the retired `at::Tensor` wrapper). It is a **release blocker** — if `z` diverges, the build must not ship, because a divergent `z` silently corrupts aggregation.

Re-freeze the fixture only on a deliberate torch bump (`framework/tests/fixtures/decomfl_golden/generate.py`), then re-run both parity gates and review the diff. Never hand-edit the vectors.

---

## The wire: safetensors, float32-only, verified before load

`shared/include/fedlearn/Safetensors.h` + `src/Safetensors.cpp` are a **torch-free reimplementation of the Python codec** (`framework/src/fedlearn/communication/safetensors_codec.py`), byte-identical by construction:

```
u64_le(header_len) ++ compact_header_json_utf8 ++ raw_f32_data
```

Two details in the C++ encoder exist purely to preserve byte parity: a 0-d scalar is written with `"shape":[1]` (matching `np.ascontiguousarray`, which yields `ndim ≥ 1`), because emitting `[]` would change the header JSON, the `u64` length prefix, and every downstream sha256; and tensor entries stay in stored order so the blob is deterministic.

**This is why the FL wire is float32-only.** The decoder hard-rejects any tensor whose header dtype is not `"F32"` (`Safetensors.cpp`: `if (dtype != "F32") throw …`). A libtorch-free C++ client cannot carry torch's full dtype machinery, so the platform made the wire mono-dtype rather than have the phone silently coerce. Two consequences worth knowing:

- On the Python side, non-float32 buffers are **filtered out of the federated set** rather than raising (commit `3b13204`) — which is what lets BatchNorm models federate at all, since `num_batches_tracked` is `int64`.
- A malformed blob is rejected loudly, not tolerated: offsets outside the data region, a `data_offsets` span that is not a whole number of floats (a crafted 6-byte span would otherwise heap-overflow the `std::vector<float>` the memcpy writes into), and an unexpected tensor field all throw.

Separately, `DtypeMap.h/.cpp` mirror the Python `_SAFE_DTYPES` whitelist **exactly** — the same ten names (`float16, float32, float64, int8, int16, int32, int64, uint8, bool, bfloat16`); it is membership, not order, that must match (Python holds a `set`, C++ an `unordered_map`). An unknown dtype string is a hard error, never a silent default — this is the anti-injection guard for a malformed server payload, and it is broader than what the codec itself transports. Keep the two lists in lockstep.

Every staged artefact is **sha256-verified before it is used**: `stageBundleFile` hashes the *decoded* bytes and refuses to write on mismatch (MO-7, which covers `inputs.f32` / `targets.i64` that `loadModel` never re-checks), and `loadModel(path, expectedSha256)` verifies the `.pte` before handing it to ExecuTorch.

---

## The proto mirror

The gRPC contract has **one canonical home**, `proto/fedlearn/v2/fedlearn.proto` (package `fedlearn.v2`, buf-governed). `mobile_client/proto/fedlearn/v2/fedlearn.proto` is a **byte-identical mirror** kept in-tree so the native CMake build has the file locally; it is regenerated by copy, never hand-edited.

```bash
../scripts/check_proto_mirror.sh    # from mobile_client/ — byte-compares all three mirrors
```

The script checks **three** mirrors — this one, the framework's `fedlearn.proto`, and the framework's `fot.proto` — and on drift prints the literal `cp` that fixes it. It runs as a gating job in **`proto.yml`** and again as the `proto-mirror` job in **`mobile.yml`**; it is **not** in `ci.yml`. Nothing in the tree speaks `fedlearn.v1` any more.

---

## The TurboModule surface

`bridge/specs/NativeFedLearnCore.ts` is the codegen source of truth. Every method is async and returns a **typed object** — never a hand-built JSON string (that was the v1 stringly-typed bridge).

| Method | Purpose |
|---|---|
| `registerClient(serverAddress, runId, clientId, enrollmentToken, useTls)` | gRPC register; returns `assignedRound` (late-joiner) + the server protocol version |
| `getServerStatus(runId)` | live `serverState`, `currentRound`, participation counts, `roundDeadlineUnixMs` |
| `stop()` | sets the abort flag and joins the native threads |
| `setModelManifest(manifest)` | trainable param layout + total param count + the separate infer graph (+ optional trainable graph). **Must precede `loadModel`** — the ExecuTorch loss graph is weights-free, so the layout cannot come from the `.pte` |
| `loadModel(path, expectedSha256)` | integrity-checked load; returns `paramCount`, `trainableParamCount`, sha256, tier |
| `setTrainingDataFromFiles(inputsF32Path, inputShape, targetsI64Path)` | on-device data from app-private files; raw features/labels never enter a server table |
| `stageBundleFile(filename, base64Data, expectedSha256)` | sha256-verify then write into app-private storage; filename is basename-sanitised |
| `runDeComFLRound(runId, config)` / `runFedAvgRound(runId, config)` | **one round per call** — the RN layer owns the loop and the deadline check |
| `infer(inputJson)` | logits + **real softmax** probabilities + argmax (not `exp(-loss)`) |
| `getDeviceMetrics()` | native RSS sample, thermal state, battery level/charging |

The module is fetched with `TurboModuleRegistry.get()` (not `getEnforcing()`): on a build that did not compile the native core, `getEnforcing` throws *synchronously at import* and took the whole JS bundle down at launch. Instead, `isNativeCoreAvailable()` reports availability and a typed fallback core **rejects** every call with `NATIVE_CORE_UNAVAILABLE_MESSAGE`, so the UI disables the training entry point instead of crashing (MO-5). This is the mechanism that keeps the iOS shell usable while the native iOS port is unfinished.

---

## The JS/TS layer

**Auth is Bearer-token, not the web's cookie.** `lib/restClient.ts` sends `Authorization: Bearer <jwt>` plus `X-FedLearn-Client: fedlearn-mobile`. The backend's `JwtAuthenticationFilter` accepts a Bearer header **only** when that marker header is present (SE-9), and `AuthController` returns `accessToken` in the login body **only** to a marker-carrying native client — a browser login response carries identity only, and the JWT stays in the HttpOnly cookie. The token is persisted through `lib/authStore.ts` on `react-native-encrypted-storage` (Android Keystore / iOS Secure Enclave); `lib/credentialStore.ts` adds an explicit opt-in "save password" using the same store. The web JWT's SE-20 audience (`fedlearn-web`) still applies, so an FL connection token cannot be replayed here.

**Join → provision → train**, in order:

1. `lib/runJoin.ts` — `GET /api/client/projects/{id}` to find the active run, poll `GET /api/runs/{runId}/status` until it has a gRPC endpoint, then `POST /api/runs/{runId}/enroll` for the `partitionId`, `connectionToken`, CA fingerprint and the run **manifest** (`recipeKey`, `strategy`, `numRounds`, `seed`, `torchVersion`, and the `firstOrderSupported` capability flag).
2. `lib/modelProvisioning.ts` — `GET /api/runs/{runId}/model-bundle`, download each binary, `stageBundleFile` each one (sha256-verified), and return the local paths.
3. `lib/training.ts` — `setModelManifest` → `loadModel` → `setTrainingDataFromFiles` → the round loop.

**The MO-4 capability gate.** A run is trainable on-device with real backprop only when the backend provisioned a **trainable `.pte`** (the manifest's `firstOrderSupported`); then `firstOrderRound` uploads a weight blob via `SubmitModelUpdateStream`, which any gradient-aggregating server consumes. Without that bundle the only native path is the zeroth-order one, which uploads seeds + gradient scalars over `SubmitGradientScalars` — a wire a non-DeComFL server cannot aggregate. So the gate is `!firstOrderSupported && manifest.strategy !== 'DeComFL'`: rather than "train" into a void, `runTrainingLoop` throws `MobileFedAvgUnsupportedError` **before** any provisioning or native work, and the UI renders it as a precise "not provisioned for on-device training yet" message. A DeComFL run with no trainable bundle is *not* refused — that is the supported zeroth-order path.

**Resilience and telemetry.** `runResilientRoundLoop` (in `lib/training.ts`) is an injectable state machine: bounded per-round retries with exponential backoff, escalation to a full rejoin, a cap on total rejoins, and pacing between successful rounds so the server is not hot-polled (MO-8) — all unit-tested without the native module. `lib/statusHeartbeat.ts` polls server status on its **own** interval, because the round loop checks status only once per iteration and discards `currentRound`/`roundDeadlineUnixMs`; without the separate heartbeat the live view froze for the duration of a long round (MO-3). `lib/deviceMetricsPoll.ts` samples `getDeviceMetrics`.

**Guards and disclosures.** `lib/deviceClass.ts` maps total RAM to a maximum model tier (`≥ 6 GB → 10M`, else `1M`) and hard-refuses the `100M` tier on any phone — its ~2 GB transient ZO working set OOMs mid-tier Android and ~100 forward passes per round means minutes per round. `lib/evaluateEligibility.ts` is a copy of the desktop's canonical advisory self-gate (hard failures gate, soft warnings inform). `lib/privacyLabel.ts` is the static pre-join data-flow disclosure, written to client-side facts only. `lib/contributionLedger.ts` records one entry per **completed** round — deliberately labelled *submitted*, never *accepted*, because no backend API yet reports whether robust aggregation kept the update.

---

## On-device model bundle — current state (honest disclosure)

The bundle is now **per-run and served by the backend**, not a hard-coded client fixture: `GET /api/runs/{runId}/model-bundle` returns the two weight-free ExecuTorch graphs (loss `forward(flat, x, y)`, infer `forward(flat, x)`), the trainable param layout, an on-device data partition, and a sha256 for every file. Backend-side, `ScriptModelBundleStager` auto-stages it in the background when a run starts (`feature.model-bundle-autostage.enabled`, default **on**), and it is **recipe-aware**:

| Recipe class | Staging path | Requirements |
|---|---|---|
| Fixture-backed (`app.model-bundle.autostage.fixture-recipes`, default `TINYNET_GOLDEN`) | `scripts/stage_model_bundle.py` — copies the committed golden fixture | stdlib only, **no ExecuTorch toolchain** |
| Everything else | `scripts/export_model.py` — instantiates the run's recipe model and lowers both graphs | torch 2.12.0 + ExecuTorch 1.3.1 on the host |

Both paths write the identical served shape, staging is idempotent, and failure is best-effort: the phone treats a missing bundle as a graceful 404 rather than an error.

**What that means in practice today.** The demo path a phone actually completes is still the fixture one: `TINYNET_GOLDEN` — `Linear(4,5) → ReLU → Linear(5,3)` with `fc2` frozen, **43 total parameters, 25 trainable** (`flat_dim = 25`), the same fixture that backs the C++ parity gate (`framework/tests/fixtures/decomfl_golden/zo_manifest.json`, `golden_loss ≈ 1.0973`). The fixture's `zo_inputs.f32` / `zo_targets.i64` are staged *as the device's local partition*, so they stand in for genuine per-device data. That is a deliberate MVP shortcut and **not representative federation**; real per-device data remains an explicit post-MVP step. The fixture now also ships `tinynet_trainable.pte`, which is what lets the first-order path be exercised at all.

The real-export path is the documented seam for every other recipe, but **it does not currently run**: `scripts/export_model.py` resolves `recipes.py` from `backend/fl-platform-api/src/main/resources/scripts/`, a directory that no longer exists (the catalog lives in `fl-runtime/recipes.py`), so the import fails and the stager logs a non-zero exit. Until that path is fixed, a non-fixture recipe produces no bundle and the phone 404s.

The full plumbing, phases, and manual acceptance runbook are in [`mobile_client/ON_DEVICE_TRAINING_E2E.md`](../../mobile_client/ON_DEVICE_TRAINING_E2E.md).

---

## Project layout

```
mobile_client/
├── shared/             # C++ FL core (ExecuTorch runtime) + the gtest suite
│   ├── include/fedlearn/   # RandnEngine.h, EtZeroOrder.h, Safetensors.h, DtypeMap.h, …
│   ├── src/                # DtypeMap, Sha256, ModelManager, DeComFLClient, ExecutorchModel,
│   │                       # Safetensors, DataLoader, FederatedLoop, TrainableExecutorchModel,
│   │                       # FedLearnClient (gRPC, opt-in)
│   ├── tests/              # 17 gtest TUs, +2 under FEDLEARN_BUILD_TRAINING,
│   │                       # +grpc_marshal_test (its own target) under FEDLEARN_BUILD_GRPC
│   └── CMakeLists.txt
├── bridge/             # TurboModule: TS spec, common C++ module, Android JNI + iOS .mm
├── src/                # React Native app: lib/, screens/, navigation/, state/, theme/, components/
├── android/            # Gradle project (+ committed wrapper), externalNativeBuild → JNI
├── ios/                # FedLearn.xcodeproj (generated), Podfile, FedLearnCore.podspec, Swift AppDelegate
├── proto/              # byte-mirror of canonical proto/fedlearn/v2 (mirror-checked)
├── scripts/            # ARM64 ExecuTorch/gRPC cross-compile, PTE export, demo data
└── CMakeLists.txt      # host build (parity gate)
```

---

## Build status

Both app projects are **buildable**: the two prior template-scaffolding blockers were resolved and committed — the Android **Gradle wrapper** (`./gradlew` bootstraps Gradle 8.14.1) and the iOS **`FedLearn.xcodeproj`** (regenerate with `ios/generate_xcodeproj.sh`).

- **Android is the real native target.** `app/build.gradle`'s `externalNativeBuild` builds `libfedlearn_jni.so` from `bridge/android/jni/CMakeLists.txt`, which forces `FEDLEARN_BUILD_GRPC=ON` and `FEDLEARN_BUILD_TRAINING=ON` and links `fedlearn_core` + `fedlearn_grpc` + the React Native `jsi`/`reactnative` prefab. `arm64-v8a` is the only shipped ABI. It needs `-PET_SRC`, `-PET_BUILD`, `-PTORCH_INCLUDE`, `-PGRPC_DIR`, `-PGENERATED_PROTO_DIR` — the cross-compiled artefacts plus `buf generate` output.
- **iOS native is a scaffold, and the podspec says so.** `ios/FedLearnCore.podspec` vendors a cross-compiled **libtorch** xcframework while `shared/` targets the **ExecuTorch** runtime. Those are different, incompatible runtimes, so the iOS native wiring will not link/run as-is; the podspec's own build guard says do **not** enable `FEDLEARN_NATIVE_IOS` in CI or default builds. Swapping the vendored framework for an ExecuTorch iOS runtime and reconciling the bridge is tracked as MO-14. Until then iOS builds the JS shell, `canImport(FedLearnCore)` is false, and `isNativeCoreAvailable()` returns false so the training entry point is disabled rather than crashing. `scripts/build_libtorch_arm64.sh` is retained solely for that pending path.

What still gates a shippable release: the cross-compiled **ARM64 ExecuTorch + gRPC** artefacts and buf-generated stubs, the iOS ExecuTorch migration above, real **signing configs** for both platforms (the release APK is debug-signed today — the release workflow says so in the release notes), and real on-device training data.

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

> **`TORCH_INCLUDE` is a trap on any gRPC-linked target.** It exists only to supply the build-generated `torch/headeronly/macros/cmake_macros.h` that the ExecuTorch headers include, and it is ordered **last** on the include path. A *full* `torch/include` also carries torch's bundled `google/protobuf/*` headers, which — added as a normal include dir — shadow the gRPC-supplied protobuf (which arrives via `find_package` as a lower-priority `-isystem` include) and break the compile with `unknown type name 'uint8'`. The host parity gate is gRPC-free so the full dir is safe there; the Android job in `mobile.yml` copies just the `torch/headeronly` subtree into an otherwise-empty shim and asserts `google/` is absent.

---

## Measured on-device facts (kernels: inference vs training)

Two results from this repo's research record bear directly on what the APK ships. They point in opposite directions and are **not** in conflict — one is about inference, the other about training.

**(a) The shipped APK links the slowest available kernels.** `shared/CMakeLists.txt:32` lists `portable_kernels` and force-loads `portable_ops_lib`; nothing in `mobile_client/` links XNNPACK or Vulkan. (`scripts/build_executorch_arm64.sh` does pass `EXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON`, so the optimised kernel library is *built* — it is simply never linked.) Portable kernels are ExecuTorch's correctness-first **reference** implementations. Measured on a vivo 1805 (Snapdragon 845 / Adreno 630, Android 10, ExecuTorch 1.3.1) against a ResNet-18 feature extractor at 224×224 — the operation a frozen-backbone round spends essentially all its wall-clock in:

| Backend, batch 1 | Median latency | vs shipped |
|---|---|---|
| `portable_cpu` (**what ships**) | 4,093.45 ms | 1× |
| `vulkan_gpu` | 183.05 ms | 22.4× |
| `xnnpack_cpu` | 49.14 ms | **83×** |

All three agree with a PyTorch eager reference to ~1e-7 absolute, so the speed is not bought with a correctness trade. The GPU wins only at batch 1; at batch ≥ 8 the Vulkan partitioner fragments the graph and degrades to portable-CPU levels, while XNNPACK holds ~46 ms/image. Source: `research/results/benchmark/ondevice_vivo_backbone_cpu_vs_gpu.json`. **Caveat the record states plainly:** one 2018-era mid-range handset, one ExecuTorch version, inference only — a flagship Adreno 7xx/8xx could plausibly reverse the XNNPACK-vs-Vulkan ordering.

**(b) XNNPACK must NOT be linked for training.** At ResNet-18 scale (11.2M trainable params, 84 delegate partitions) the XNNPACK-lowered graph is ~18% faster per step but **never learns**: its minimum loss over 30 steps equals its initial loss (0.69308), while the portable build converges to 0.0187. There is no NaN, no error and no warning — lowering succeeds silently. Follow-ups pinned down the mechanism more precisely than the first write-up did: the forward pass is correct (max abs deviation ~1.07e-6 vs eager) and the **first** backward is correct (step-0 `fc.weight` gradient L2 matches exactly); subsequent steps compute against **stale, prepacked weights** — supported circumstantially (step-1 `fc.weight` gradients differ 2.3× from identical parameters), but the record marks it `NOT_proven`: the delegate's internal weight buffer was never read, and a per-parameter probe *refuted* the simpler "delegated params are frozen" story (62/62 receive gradients and move). It also costs +51% peak RSS and +99.6% model size — a real per-round wire cost in a federation. Upstream documents delegate+training integration as a work in progress, so the defensible complaint is narrow: the partitioner lowers a trainable graph silently instead of refusing. Sources: `research/results/benchmark/ondevice_xnnpack_training_produces_wrong_gradients.json` (incl. its `CORRECTIONS_2026_08_09` block) and `ondevice_vivo_optimized_kernels.json`.

**Reading:** linking XNNPACK is worth roughly 80× on the dominant on-device operation and is a pure build-configuration change — but it must be scoped to the **inference / feature-extraction** graphs, never the trainable one.

---

## CI and release

| Workflow | Job | What it gates |
|---|---|---|
| `ci.yml` | `mobile-js` (path-filtered on `mobile_client/**`) | `scripts/check_no_skipped_tests.sh mobile_client` (TE-10 — a `.skip`/`.only` fails the job), then `npm ci` → `npm run lint` (ESLint 9 flat config) → `npx tsc --noEmit` → `npm run test:coverage` (TE-11; thresholds live in `package.json`) |
| `ci.yml` | design tokens | `scripts/check_design_tokens.sh`, unconditional — a hand-edit of `src/theme/tokens.generated.ts` fails here |
| `mobile.yml` | `proto-mirror` · `python-parity` · `cpp-parity` | the mirror check; `test_perturbation.py` + `test_decomfl_multiround.py`; and the C++ gtests built against a from-source ExecuTorch with `FEDLEARN_BUILD_TRAINING=ON`. **The golden-vector test gates the build.** |
| `mobile.yml` | `android-so-size` | full arm64 APK + a `libfedlearn_jni.so` size budget (`SO_BUDGET_MB=60`). Too heavy for every PR, so TE-8 runs it on a **nightly schedule** (and uploads the APK) or on demand via the `MOBILE_NATIVE_CI` repo variable |
| `proto.yml` | buf lint · breaking-change gate vs `main` · regenerate-is-a-no-op · mirror check | the canonical contract |
| `release-mobile.yml` | `build-android` | **tag-triggered on `mobile-v*`** (so it never fires on `desktop-v*`), or `workflow_dispatch`. Cross-compiles **ARM64 ExecuTorch + gRPC from source**, runs `buf generate`, assembles the release APK, and publishes a GitHub Release. iOS is deliberately absent — it needs full Xcode plus manually cross-compiled xcframeworks and signing, none of it CI-reproducible |

Pinned in the two native workflows (`mobile.yml` + `release-mobile.yml`) and required to match: `TORCH_VERSION=2.12.0` (header only, not a link target), `ET_VERSION=1.3.1`, `GRPC_CPP_VERSION=v1.67.1`, NDK r27, JDK 21. `ci.yml`'s `mobile-js` is JS-only and pins none of them.

`mobile.yml`'s triggers include `framework/` paths (`estimators/**`, the golden fixtures, `test_perturbation.py`, `test_decomfl_multiround.py`) — a framework-side change to the RNG or the fixtures trips the mobile parity gate, which is the point.

---

## Key cross-component interfaces

- Authenticates against the **Backend** `POST /api/auth/login` as a **native client**: `Authorization: Bearer` + `X-FedLearn-Client: fedlearn-mobile` (SE-9). This is *not* the browser's cookie contract.
- Enrols through `GET /api/client/projects` → `GET /api/client/projects/{id}` → `GET /api/runs/{runId}/status` → `POST /api/runs/{runId}/enroll`, and provisions from `GET /api/runs/{runId}/model-bundle`.
- Connects to a **Framework** FL server over gRPC and runs the native DeComFL (or, when provisioned, first-order) client path.
- Shares the **canonical** `proto/fedlearn/v2` contract, byte-mirrored into `mobile_client/proto/` and enforced by `scripts/check_proto_mirror.sh` (three mirrors: this one plus the framework's `fedlearn.proto` and `fot.proto`). The framework's copy is byte-identical to canonical, so client and server share one `fedlearn.v2` contract with no version skew.
- The C++ `SAFE_DTYPES` whitelist stays in lockstep with the Python serializer whitelist, and the C++ safetensors codec stays byte-identical to `safetensors_codec.py`.
- The theme is generated from `design/tokens.json` — the same source the frontend and desktop consume.

## Related documentation

- [`mobile_client/README.md`](../../mobile_client/README.md) — the unit's own entry point: per-file build inventory and bring-up commands
- [`mobile_client/bridge/README.md`](../../mobile_client/bridge/README.md) — TurboModule codegen/wiring steps and RN-version caveats
- [`mobile_client/ON_DEVICE_TRAINING_E2E.md`](../../mobile_client/ON_DEVICE_TRAINING_E2E.md) — the six-phase on-device training plan and acceptance runbook
- [Framework Wiki](../framework/README.md) — the FL engine and the DeComFL algorithm this client implements natively
- [Framework: DeComFL](../framework/06_decomfl.md) — zeroth-order estimation, seed/gradient protocol
- [Backend: Security & Authentication](../backend/02_security_and_auth.md) — the auth contract, SE-9 and SE-20
