# A6 — Mobile FL Client Audit (`origin/fed-mobile:mobile_client/`)

**Audit date:** 2026-05-29
**Auditor unit:** A6 — Mobile
**Target:** `mobile_client/` on branch `origin/fed-mobile` (READ-ONLY)
**Status of branch:** forked from `dfde813 "Readme"` on **2026-02-17**, last commit `f0efead` on **2026-04-12**; **2 commits ahead** of its fork point, **157 commits behind `origin/main`** (verified: `git rev-list --left-right --count origin/main...origin/fed-mobile` → `157  2`). It is also **219 commits behind `main-clean`**.
**Builds on:** prior audit `docs/audit/2026-05-27/03-framework.md` (framework) and `README.md`. This report extends, does not duplicate, those findings.

---

## Executive summary

The mobile client is a **second, independent re-implementation of the entire FL client** — not a thin wrapper. It hand-ports `framework/src/fedlearn/{estimators,client}/*.py` into ~7 C++ files (`mobile_client/shared/src/`) running libtorch ARM64 + a C++ gRPC stub, bridged to React Native via JNI (Android) and an ObjC++ TurboModule (iOS). This is a large surface that must be kept bit-for-bit consistent with the Python framework's ZO math, RNG, and serialization — and today it is not, in several concrete ways.

The most damaging finding is structural, not a one-line bug: **the branch carries its own copies of the contract and the algorithm, and both have already drifted.** Two `fedlearn.proto` copies live in the tree, both renamed `SubmitModelUpdateRequest` → `SubmitModelUpdateReque` (a truncation typo), and the `src/federated/protos/` copy is additionally missing the entire DeComFL RPC/message block. The C++ compiles against the typo'd message and propagates it (`FedLearnClient.cpp` constructs `SubmitModelUpdateReque`). Because proto3 wire format is field-number-based, the typo does not break the wire — but it guarantees the generated symbols never match the canonical `fedlearn.v1` stubs, so the mobile C++ can never share generated code with the framework and any future contract change silently bypasses it.

Two findings cap the project's actual value: (1) the **DeComFL / ZO path is dead in the UI** — `TrainingScreen.jsx:31-33` marks both `zo_fl` and `decomfl` modes `disabled: true`, so the carefully hand-ported `ZerothOrderEstimator`/`DeComFLClient`/`deComFLLoop` is unreachable from the app; only FedAvg-with-backprop is exposed. (2) The headline architectural claim — that the C++ ZO matches the Python forward-difference estimator with seed-determinism — **cannot be relied upon**: the C++ casts loss to `float` before the division (`loss.item<float>()`), the perturbation RNG parity claim (`torch::Generator` == Python `torch.Generator` Mersenne-Twister stream) is asserted in a code comment with no test, and `setFlatParams` drops the Python `requires_grad` filter, so parameter flattening order/count can diverge for any model with frozen params.

Verdict for the subsystem: **rebuild the contract+algorithm sharing (proto codegen and the ZO core), salvage the bridge/UI scaffold.** The native-on-device DeComFL idea is genuinely valuable for a battery- and bandwidth-constrained mobile FL client (DeComFL's O(K·P) uplink is the right call for cellular), but the current implementation is a parallel fork that will rot. The v2 subtree lift must reconcile the contract first, generate C++ stubs from the canonical proto, and treat the ZO core as a tested port (golden-vector parity tests against the Python estimator) — not a maintained-by-eye duplicate.

---

## What was reviewed (evidence map)

| Area | Files read |
|---|---|
| Proto drift | `framework/src/fedlearn/communication/protos/fedlearn.proto`, `mobile_client/shared/proto/fedlearn.proto`, `mobile_client/src/federated/protos/fedlearn.proto` |
| ZO / DeComFL fidelity | `shared/src/ZerothOrderEstimator.{cpp,h}`, `shared/src/DeComFLClient.{cpp,h}`, `shared/src/Utils.h` vs `framework/.../estimators/zeroth_order.py`, `framework/.../client/decomfl_client.py` |
| FL loop & FedAvg | `shared/src/FederatedLoop.cpp`, `shared/src/ModelManager.cpp`, `shared/src/FedLearnClient.cpp`, `shared/src/DataLoader.cpp` |
| Build viability | `scripts/build_libtorch_{android,ios}.sh`, `scripts/build_grpc_android.sh`, `shared/CMakeLists.txt`, `android/.../jni/CMakeLists.txt` |
| Model export / size | `scripts/export_model_{1m,10m,100m}.py` |
| Bridge / UI | `ios/.../NativeFedLearnCore.mm`, `specs/NativeFedLearnCore.ts`, `src/screens/TrainingScreen.jsx`, `src/utils/nativeModelPath.js` |
| Security / dead code | `android/.../AndroidManifest.xml`, `src/utils/resourceMonitor.js`, `src/federated/DatasetLoader.js`, `package.json`, `patches/` |

---

## Findings

### CRITICAL

#### M-C1 — Proto drift: two copies, both with a renamed message, one missing all of DeComFL

**Evidence.** `diff` of the three files:

- `shared/proto/fedlearn.proto` vs canonical: **2 hunks**, both the same defect —
  - `rpc SubmitModelUpdate(SubmitModelUpdateReque) returns (...)` (canonical: `SubmitModelUpdateRequest`)
  - `message SubmitModelUpdateReque { ... }` (canonical: `SubmitModelUpdateRequest`)
  Otherwise byte-identical to canonical (it *does* carry the DeComFL block).
- `src/federated/protos/fedlearn.proto` vs canonical: the **same `SubmitModelUpdateReque` typo**, PLUS it is the **older pre-DeComFL revision** — it has no `GetDeComFLConfig`/`SubmitGradientScalars` RPCs and none of the `PerturbationSeeds`/`GradientScalars`/`RebuildHistory`/`*DeComFLConfig*` messages. It is a stale snapshot from before DeComFL was added to the contract.

**Why it matters.**
1. The C++ build (`shared/CMakeLists.txt` and `android/.../jni/CMakeLists.txt`) compiles `${PROTO_SRC_DIR}/fedlearn.proto` = `shared/proto/fedlearn.proto` (the typo'd-but-DeComFL-complete copy). So generated C++ symbol is `fedlearn::v1::SubmitModelUpdateReque`, and `FedLearnClient.cpp` is written to match (`::fedlearn::v1::SubmitModelUpdateReque request;`). It compiles and — because proto3 framing is field-number based and the server's `SubmitModelUpdate` RPC method name is unchanged — it will even **interoperate on the wire** with the canonical server. The typo is silent at runtime.
2. But it is **permanently un-sharable**: mobile can never `import` or link the framework's generated stubs, because the message type names differ. Every future contract change (new field, new RPC) must be hand-mirrored into the mobile copy by someone who remembers it exists.
3. The `src/federated/protos/` copy is a **third source of truth that is already a full version behind** (no DeComFL). It appears unused by the native build (the JS layer has no grpc-web client — see M-H4), so it is dead, but it is a live trap for anyone who edits "the proto" and picks the wrong file.

**Recommendation (rebuild).** Delete both copies. Make the canonical `framework/.../fedlearn.proto` the single source; generate C++ stubs from it in CI (the build scripts already invoke a host `protoc` + `grpc_cpp_plugin`, so point `--proto_path` at the canonical file via a symlink or a vendored copy stamped from CI with a checksum gate). Add a CI check that fails if any `mobile_client/**/fedlearn.proto` differs from canonical.

---

#### M-C2 — On-device DeComFL/ZO fidelity is unverified and has at least three concrete divergences from the Python reference

The header comment claims *"C++ torch::Generator uses the same Mersenne Twister as Python, producing identical outputs"* and *"bit-identical vectors"* (`ZerothOrderEstimator.cpp:9-11`, `.h:19-22`). There is **no test** anywhere in the tree asserting parity. Three divergences are visible by inspection:

1. **float32 truncation of the loss before division.** Python computes `g = (loss_x_perturbed - loss_x) / self.mu` on tensors and returns `g.item()` — the subtraction happens at the loss tensor's dtype (typically float32, but the division is done before extraction). C++ does `(loss_perturbed.item<float>() - loss_x.item<float>()) / mu_` — it extracts **two float32 scalars first, then subtracts**. For ZO the signal is `f(x+μz) − f(x)` with μ=0.001; this is a *catastrophic-cancellation* regime (two nearly-equal large losses subtracted to recover a tiny difference). Extracting to float32 *before* the subtraction discards exactly the low-order bits that carry the gradient signal. Python's path keeps the subtraction in-tensor. This is not cosmetic: it directly attacks the numerical stability the ZO estimator depends on. **Uncertain on magnitude** — needs measurement — but the structure is wrong and should be `double`-accumulated.

2. **`mu_` is `float`.** `ZerothOrderEstimator(float mu)`, member `float mu_`. Python's `smoothing_param` is a Python float (float64). Dividing a float32 difference by a float32 μ vs a float64 μ compounds (1).

3. **`setFlatParams` drops the `requires_grad` filter.** Python's `_get_flat_params`/`_set_flat_params` iterate **only `if p.requires_grad`** (`zeroth_order.py:_get_flat_params`/`_set_flat_params`). The C++ `getFlatParams`/`setFlatParams` (`Utils.h`) iterate **all** `model.parameters()` with no filter. For the current MNIST models every param is trainable, so they match *today*; for any model with frozen layers (LoRA, partial fine-tuning — exactly the LLM use-case DeComFL targets) the flattened vector length and ordering diverge, the seed→perturbation dimension (`x_current_.numel()`) no longer matches the server's notion of P-dimensional space, and the seed-replay reconstruction on the **server** (which uses the Python filtered count) will silently desync from the client. This breaks the core DeComFL invariant that client and server regenerate the *same* perturbation vector from a shared seed.

**Also note (parity-relevant):** the revert strategy differs. Python reverts by subtracting the accumulated `total_perturbation` in place (a deliberate "SECURE … reverse the exact perturbation" comment, `decomfl_client.py`). C++ reverts by `x_current_ = x_initial.clone()` captured at the top of `fit()`. These are mathematically equivalent at infinite precision but produce different rounding; combined with (1), the per-round model state can drift between a mobile client and the Python clients in the same federation.

**Recommendation (rebuild the ZO core as a tested port).** (a) Accumulate the loss difference in `double` and keep μ as `double`. (b) Add the `requires_grad` filter to `getFlatParams`/`setFlatParams` to match Python exactly. (c) Add a **golden-vector parity test**: in CI, generate `torch.randn(N, generator=manual_seed(s))` from Python for a fixed seed/N, serialize, and assert the C++ `generatePerturbation(s, N)` matches to bit/ULP tolerance — this validates the Mersenne-Twister claim instead of asserting it in a comment. (d) Add an end-to-end `g`-scalar parity test: same model weights, same batch, same seed → assert `|g_cpp − g_py|` within tolerance. Until these exist, the "matches the Python framework" claim is **unsubstantiated**.

> **Caveat / cannot verify here:** PyTorch's CPU `randn` with a `Generator` does use a Mersenne-Twister engine in both the Python and C++ frontends, so the claim is *plausible*. But (i) `torch.Generator(device='cpu').manual_seed` vs C++ `torch::Generator(); gen.set_current_seed(seed)` are different call shapes, and (ii) `randn` parity across `torch` versions is not guaranteed by API contract. I will not assert it works without the golden-vector test. Flagging as uncertain.

---

#### M-C3 — DeComFL is the project's reason to exist, and it is disabled in the UI

`TrainingScreen.jsx:31-33`:

```js
const TRAINING_MODES = [
  {key: 'fedavg',  label: 'FedAvg (SGD)'},
  {key: 'zo_fl',   label: 'ZO-FL (FedAvg + ZO)', disabled: true},
  {key: 'decomfl', label: 'DeComFL',              disabled: true},
];
```

Both ZO modes are `disabled: true`. The only reachable path is `fedavg`, which runs **first-order backprop** (`ModelManager::trainStep` does `loss.backward()` + manual SGD) and uploads **full model parameters** (`submitUpdate` → `tensorsToProto` → `SubmitModelUpdate`). So the entire hand-ported DeComFL/ZO C++ stack (`ZerothOrderEstimator`, `DeComFLClient`, `FederatedLoop::deComFLLoop`, the `GetDeComFLConfig`/`SubmitGradientScalars` RPCs) is **unreachable dead code in the shipped UI**. The branch's headline value — communication-light, dimension-independent on-device training — is not actually delivered to a user.

**Recommendation.** This is the decisive input to the verdict. Either (a) the v2 reconciliation re-enables and *tests* the DeComFL path (then the native core has a reason to exist), or (b) if mobile only ever needs FedAvg-with-backprop, the entire ZO C++ subsystem is **kill** candidate and the mobile client collapses to a far smaller surface. The current state — full ZO port present but disabled — is the worst of both: maximum maintenance surface, zero delivered capability.

---

#### M-C4 — On-device FedAvg uploads the full model in plaintext gRPC; trained model + private data sit unprotected on device

This extends framework audit **C4** (gRPC plaintext over WAN) to the mobile threat model, which is worse:

- `FedLearnClient.cpp` uses `grpc::InsecureChannelCredentials()` on **both** the training and heartbeat channels. A phone on a hostile cellular/Wi-Fi network ships its full trained weights (and, in the active FedAvg path, gradients that encode the on-device training data) in cleartext. The framework can at least be tunneled through Tailscale on a controlled LAN demo; a consumer phone in the field cannot.
- The model file is provisioned by **manual `adb push` + `run-as cp`** (`nativeModelPath.js:33-39`) into app-private storage, but on iOS it is read straight from `MainBundlePath` (shipped in the app bundle, world-readable to anyone who unzips the IPA). There is no model encryption, no signature/integrity check on the `.pt` file before `torch::jit::load` (`ModelManager::loadScriptModel`) — and **`torch.jit.load` executes a serialized module**; a tampered `.pt` is a code-execution vector on device. This is the mobile analogue of the desktop unsigned-binary finding (README P0 #4).
- Training data (`mnist_*.json`) is bundled or loaded from an arbitrary `inputPath` with no validation (`NativeFedLearnCore::trainStep`); `DataLoader::loadFromJson` parses attacker-controllable JSON. Low severity for MNIST demo data, but the pattern (unvalidated path → on-device training corpus) does not generalize safely to real user data.

**Recommendation.** v2 mobile must: require TLS on the gRPC channel (refuse insecure outside a dev flag, mirroring the framework C4 fix); verify a signature/hash on the model artifact before `jit::load`; store the model and any user training data in platform-encrypted storage (Android Keystore-wrapped / iOS Data Protection class), not the app bundle. Treat the model artifact as untrusted input.

---

### HIGH

#### M-H1 — libtorch ARM64 build is a from-source, training-enabled mobile build — high cost, large binary, fragile pinning

The build scripts compile **libtorch from PyTorch source** with `INTERN_BUILD_MOBILE=ON` **and** `BUILD_MOBILE_AUTOGRAD=ON` (`build_libtorch_android.sh`, `build_libtorch_ios.sh`). This is necessary because on-device training needs autograd, which the stock prebuilt PyTorch Mobile / `pytorch_android` AARs **do not ship** (they are inference-only). Implications:

- **No prebuilt artifact exists** for this configuration; every developer/CI must clone PyTorch and cross-compile (multi-hour build, ~tens of GB of build tree). There is no pinned PyTorch commit/tag in either script — `${PYTORCH_SRC}` is whatever the developer checked out. This is a reproducibility hole: the RNG-parity claim (M-C2) is meaningless without a pinned torch version on both sides.
- **Binary size:** the Android JNI link (`android/.../jni/CMakeLists.txt`) `--whole-archive`s **all** `${LIBTORCH_DIR}/lib/*.a` into the app's shared library, then links the full gRPC + protobuf static set. `--whole-archive` defeats dead-code elimination across the libtorch archives, so the ATen/c10 kernel set is pulled in wholesale. A static libtorch-mobile core is **~40–80 MB of native code per ABI** before gRPC/protobuf; with `--whole-archive` it trends to the high end. Shipping arm64-v8a alone, the `.so` will dominate the APK/IPA. **Uncertain on exact size** (no built artifact to measure here) but this is firmly in the "tens of MB native blob" range and needs a real measurement + a size budget gate.
- The Android build hardcodes `THIRD_PARTY_DIR=/tmp/fl-third-party` and notes a symlink workaround for spaces in paths — i.e. it depends on a hand-made symlink at a fixed absolute path. Not CI-friendly; not reproducible across machines.
- gRPC is pinned to `v1.62.0` (`build_grpc_android.sh`) — reasonable and recent; this one *is* pinned, unlike torch.

**Recommendation.** Pin the exact PyTorch tag in the build scripts and bake it into the parity tests. Produce the libtorch+gRPC ARM64 artifacts **once in CI**, cache them, and consume a pinned prebuilt rather than from-source per dev. Measure the per-ABI `.so` size and set a budget. Drop `--whole-archive` to whatever is actually needed (most of libtorch can be GC'd) — verify which kernels are referenced and only force-include those.

#### M-H2 — TorchScript export is inconsistent (trace vs script) and the 1M/10M/100M tiers create unrealistic battery/thermal/memory expectations

- **Export inconsistency:** `export_model_1m.py` and `export_model_10m.py` use `torch.jit.trace`; `export_model_100m.py` uses `torch.jit.script`. Tracing bakes in the example batch shape (`torch.randn(1,1,28,28)`) and erases control flow, so a traced 1M model may misbehave for batch sizes ≠ 1 unless the graph is shape-agnostic; scripting preserves control flow. Mixed strategies across tiers mean the three models are not interchangeable artifacts. The 1m script even comments that it traces specifically to avoid "scripted helper ops … not available in our Android libtorch build" — a sign the from-source libtorch is missing operators that full PyTorch has.
- **On-device cost of ZO at scale:** ZO does **2 forward passes per perturbation** (`computeGradientScalar`), and DeComFL runs K local steps × P perturbations → **2·K·P forward passes per round**. For the 100M-param TorchScript model, each forward is ~hundreds of MFLOPs–GFLOPs on a mobile CPU (libtorch is built `USE_VULKAN=OFF USE_METAL=OFF`, so this is **CPU-only** — no GPU/NPU). With K=… P=10 (the `TrainingConfig` default), that is 20+ full forward passes per round on the big model, single-threaded-ish (`USE_OPENMP=OFF`). Expect multi-second-to-minute rounds, sustained CPU load → **thermal throttling and battery drain** that make a 100M tier impractical for a phone in someone's pocket. The 100M model also needs **2× param memory transiently** during `flat_params + mu*perturbation` (a full-size temporary `perturbed` vector + the perturbation `z`, both `numel()`-length float32) on top of the loaded model — ~400 MB working set for 100M params just for the ZO temporaries, which will OOM mid-tier Android devices.
- The realistic on-device tier is **~1M (and maybe 10M with care)**; 100M is a benchmark artifact, not a deployable mobile config. This is consistent with the DeComFL thesis (its win is *communication*, not *compute* — on-device compute still scales with model size).

**Recommendation.** Standardize on `torch.jit.script` (or ExportedProgram via `torch.export` if the mobile libtorch supports it — **uncertain**, verify against the pinned build) for all tiers; or document the trace constraint (fixed batch). Cap the supported on-device model size by device-class detection. Treat 100M as a server/desktop tier, not mobile. Add per-round wall-clock, peak-RSS, and thermal-state telemetry (see observability below) before claiming any tier is viable.

#### M-H3 — No heartbeat-death detection (same class as framework H1), and the mobile loop has its own gaps

The mobile `heartbeatLoop` (`FedLearnClient.cpp`) catches `catch (...)` and continues — identical to framework **H1** (heartbeat thread death invisible to training). On mobile this is worse because the OS can suspend the app (backgrounding, Doze on Android, iOS background limits) and silently kill the channel; the training thread keeps running and the server times the client out and rejects the eventual upload as stale. There is also **no foreground-service / background-execution handling** in the AndroidManifest (only `INTERNET` permission) — a multi-minute FedAvg round will be killed by Android the moment the user backgrounds the app. iOS background execution is even more restrictive and is not addressed at all.

**Recommendation.** Adopt the framework H1 fix (shared `threading.Event`/atomic flag set on N consecutive heartbeat failures, checked between local steps). Add an Android foreground service for the training lifetime; on iOS, accept that long FL rounds only run in the foreground and surface that to the user. This is a genuine product constraint of mobile FL, not a bug to paper over.

#### M-H4 — Stale TF.js-era JavaScript still imports a dependency that is not installed → runtime crash if reached

`src/utils/resourceMonitor.js:6` and `src/federated/DatasetLoader.js:6` both `import * as tf from '@tensorflow/tfjs';`. **`@tensorflow/tfjs` is not in `package.json` dependencies** (verified — the dep list has no TF package). The `patches/` directory still carries `@tensorflow+tfjs-core+4.22.0.patch` and `@tensorflow+tfjs-react-native+1.0.0.patch`. This is residue from a **prior TF.js-based implementation** that was replaced by the native libtorch core but never deleted. Any code path that imports `resourceMonitor` or `DatasetLoader` will throw `Unable to resolve module @tensorflow/tfjs` at bundle/runtime. The `src/federated/protos/` stale proto (M-C1) and `src/federated/deactivate` (a stray shell snippet committed as a file) are part of the same dead layer.

**Recommendation.** Delete the TF.js-era layer wholesale: `src/federated/DatasetLoader.js`, `src/utils/resourceMonitor.js`, `src/federated/protos/`, `src/federated/deactivate`, the three `patches/@tensorflow*` files, and the empty-stub components (`ConnectionStatus.jsx`, `DeviceInfo.jsx`, etc. are 0-byte). Replace `resourceMonitor`'s (broken) memory telemetry with native metrics surfaced from the C++ core through the bridge.

---

### MEDIUM

#### M-M1 — `ModelManager::serializeStateDict()` builds and discards a dead dict; the live path is fragile

`serializeStateDict()` first constructs a `model_data` GenericDict, inserts an *empty* inner `"parameters"` dict and `"num_examples"`, then **throws it away** and builds a *second* `data_ivalue`/`gd` with the real params. The first ~8 lines are dead. The live path relies on `c10::IValue::toGenericDict()` returning a handle that shares storage with `data_ivalue` (so `gd.insert(...)` mutates `data_ivalue`) — this is true for libtorch IValue dicts (reference semantics) but is **load-bearing behavior asserted by nothing**. The serialized bytes must match Python's `torch.save({'parameters': state_dict, 'num_examples': N})` exactly or the framework's deserializer (framework C1's wrapper) rejects it. There is no roundtrip test. Note the C++ hardcodes `num_examples=0` here even though the FedAvg path passes a real `num_examples` to `submitUpdate` separately — the embedded-vs-RPC count are inconsistent.

**Recommendation.** Delete the dead block; add a Python↔C++ serialization roundtrip test (mobile serializes → Python `torch.load` succeeds and recovers identical tensors, and vice versa). This is the mobile mirror of framework C1 and must be tested on the same fixture.

#### M-M2 — `protoToTensors` dtype handling silently defaults unknown dtypes to float32

`FedLearnClient.cpp:protoToTensors` maps a handful of dtype strings and **falls through to `torch::kFloat32` for anything else**, including reinterpreting raw bytes via `from_blob`. If the server ever sends `bfloat16`/`uint8`/quantized tensors (plausible for LLM federations), the mobile client will silently misinterpret the bytes as float32 — garbage weights, no error. Mirrors framework H3 (fragile dtype handling) but the mobile side has *no* `_SAFE_DTYPES` whitelist at all.

**Recommendation.** Make unknown dtype a hard error, not a silent float32 fallback. Share the dtype enum semantics with the framework serializer.

#### M-M3 — gRPC `max_{send,receive}_message_length = 1 GB` on a phone

`makeChannelArgs()` sets 1 GiB send/receive limits — copied from the framework's LLM-chunking config. On a mobile device this is a memory-pressure footgun: a single unary `GetGlobalModel`/`SubmitModelUpdate` (the active FedAvg path uses **unary**, not the chunked stream) can buffer up to a gigabyte in RAM, instantly OOM-killing the app. The framework's parameter-chunking feature (>300 MB) exists precisely to avoid this, but the mobile FedAvg path calls the **unary** `SubmitModelUpdate`/`GetGlobalModel`, not the streaming variants — so chunking is bypassed on mobile. For a 100M model (~400 MB float32) the unary path will allocate the whole payload at once.

**Recommendation.** Mobile must use the streaming RPCs (`GetGlobalModelStream`/`SubmitModelUpdateStream`) for anything above a small threshold, exactly as the framework does, and cap message length to something phone-appropriate (tens of MB). Preserve the framework's chunking invariant on mobile.

#### M-M4 — `client_id = mobile_${Date.now()}` is not stable across restarts

`TrainingScreen.jsx` defaults `clientId` to `mobile_${Date.now()}`. DeComFL's missed-round rebuild (`rebuildModel`) assumes a stable client identity so the server can track which rounds a client missed. A timestamp-based ID changes every app launch, defeating rebuild/resume and inflating the server's client roster.

**Recommendation.** Persist a device-stable client ID (UUID in encrypted storage), surfaced read-only in the UI.

---

### LOW

- **L1.** `android/.../AndroidManifest.xml` sets `android:exported="true"` on `MainActivity` (default for the launcher; acceptable) but the app declares only `INTERNET` — no `usesCleartextTraffic` is set, yet the gRPC channel is cleartext (M-C4). On Android 9+ cleartext to arbitrary hosts may be blocked by default network-security-config; the demo "works" only because gRPC h2c may slip the policy. Document and pin a network-security-config.
- **L2.** `export_model_*.py` write to `mobile_client/assets/`, but `nativeModelPath.js` resolves `model_10m.pt` from `RNFS.DocumentDirectoryPath` (Android) / `MainBundlePath` (iOS) and only the 10M tier is wired (`MODEL_FILE_NAME = 'model_10m.pt'` hardcoded). The 1M/100M exports have no provisioning path. The whole model-provisioning story is "run a Python script, then `adb push` by hand" — not a release pipeline.
- **L3.** `NativeFedLearnCore` results are stringly-typed JSON hand-built with a local `jsonEscape` (`NativeFedLearnCore.cpp`) and parsed on the JS side; one missing escape (e.g. a Windows path with backslashes in an error message) breaks the bridge. The TurboModule spec (`specs/NativeFedLearnCore.ts`) declares typed return shapes but the impl returns strings — codegen and impl disagree.
- **L4.** `__tests__/App.test.tsx` is the only test in the entire mobile tree; there are zero tests for the C++ core, the bridge, or any FL logic. (Extends framework M6's "no ARM64 CI" to mobile: there is no mobile CI at all.)
- **L5.** MNIST raw IDX files are committed **twice** (`mobile_client/data/MNIST/raw/` and `mobile_client/android/data/MNIST/raw/`), ~11 MB of binary blobs in git. Repo bloat.

---

## Decision table (verdicts)

| Module / subsystem | Verdict | One-line rationale |
|---|---|---|
| Proto copies (`shared/proto`, `src/federated/protos`) | **rebuild** | Two drifted copies with a renamed message and one missing all DeComFL; generate C++ stubs from canonical `fedlearn.v1`, delete both. |
| ZO / DeComFL C++ core (`ZerothOrderEstimator`, `DeComFLClient`, `Utils.h` flatten) | **rebuild** | float32 truncation, unfiltered `requires_grad`, unverified RNG-parity claim — port must be re-derived against golden parity tests, not maintained by eye. |
| DeComFL UI path | **rebuild** (re-enable + test) or **kill** | Disabled in `TrainingScreen` today; decide whether mobile delivers DeComFL at all — if not, the entire ZO core is kill. |
| FedAvg loop + `ModelManager` (active path) | **refactor** | Works for MNIST demo but uploads full model over insecure unary gRPC; needs TLS, streaming, dead-code removal in `serializeStateDict`. |
| `FedLearnClient` gRPC layer | **refactor** | Insecure channel + 1 GB message limits + silent dtype fallback + duplicated heartbeat-death bug; align to framework fixes. |
| libtorch/gRPC ARM64 build (`scripts/`, JNI CMake) | **refactor** | From-source training build is necessary but unpinned, unreproducible (`/tmp` symlink), `--whole-archive` bloats binary; pin + CI-prebuild + size budget. |
| Model export scripts (`export_model_*.py`) | **refactor** | Inconsistent trace/script; no provisioning pipeline; 100M tier non-viable on-device. |
| RN bridge (JNI / ObjC++ TurboModule / spec) | **salvage** | Architecturally sound (CxxTurboModule + JSI); stringly-typed returns and spec/impl mismatch are minor. |
| RN screens / navigation scaffold | **salvage** | Standard RN; reusable once dead modules are pruned. |
| TF.js-era JS (`resourceMonitor`, `DatasetLoader`, `src/federated/*`, `patches/@tensorflow*`) | **kill** | Imports an uninstalled dep; dead residue of a replaced implementation. |
| Committed MNIST blobs + duplicate `data/MNIST` | **kill** | ~11 MB binary in git, duplicated; move to LFS or a fetch script. |

---

## What the v2 subtree lift must reconcile (deferred step)

The branch forked **2026-02-17** and froze **2026-04-12**; it is **157 commits behind `origin/main`** and **219 behind `main-clean`**. Bringing it current is not a merge — it is a contract re-baseline. Concretely, before the subtree lift:

1. **Contract.** Re-baseline against the *current* `fedlearn.v1` proto and the framework fixes that landed after Feb 17 — in particular framework audit **C1** (the chunked-upload `{'parameters', 'num_examples'}` wrapper) and **C3** (the proposed `compressed` flag baked into `ModelChunk`). If C1/C3 ship, the mobile `ModelManager::serializeStateDict`/`loadStateDict` and `FedLearnClient` chunk handling must match the new wrapper/flag or mobile uploads break exactly as the Python LLM clients did. The mobile copies of the proto must be deleted and regenerated from canonical (M-C1).
2. **Auth.** The platform is cookie-only HttpOnly JWT (no Bearer). The mobile gRPC client has **no auth at all** — it registers with a bare `client_id`. v2 must define how a mobile client authenticates to the FL server (the gRPC plane is currently unauthenticated for *all* clients; this is a platform-wide gap, but mobile makes it a public-internet gap). Reconcile with whatever the backend's FL-server-spawn flow expects post-V5 identity.
3. **DeComFL server side.** Confirm the server's seed-generation, P-dimension, and rebuild-history semantics still match what `DeComFLClient`/`FedLearnClient` parse. If the server changed K/P encoding or added fields, the mobile parsers (`getDeComFLConfig`) silently drop them.
4. **Build provenance.** Pin the PyTorch tag used for the ARM64 libtorch build and assert the same tag is used for the Python parity fixtures (M-C2/M-H1). RNG/serialization parity is only meaningful against a pinned torch on both sides.
5. **Observability parity.** The prior audit's Theme 3 (observability absent) and the per-run FL telemetry goal apply doubly to mobile: a phone client must report per-round wall-clock, peak memory, thermal state, battery delta, and dropped-heartbeat counts so a federation operator can see *why* a mobile client is slow or churning. None of this exists today; the only telemetry (`resourceMonitor.js`) is broken (M-H4). Wire native metrics through the bridge and onto the same `/topic/results/{projectId}` / metrics path the rest of the platform is moving toward.

---

## Prioritized recommendations

**P0 (before any v2 lift):**
1. Delete both mobile proto copies; generate C++ stubs from canonical `fedlearn.v1` in CI with a checksum gate (M-C1).
2. Decide DeComFL-on-mobile go/no-go (M-C3). If go, fix ZO numerics (double accumulation, `requires_grad` filter) and add golden-vector + g-scalar parity tests (M-C2). If no-go, delete the ZO C++ subsystem.
3. Require TLS on the mobile gRPC channel and integrity-check the `.pt` before `jit::load` (M-C4).

**P1 (hygiene + viability):**
4. Delete the TF.js-era dead layer and the empty stub components (M-H4).
5. Pin the PyTorch tag; CI-prebuild libtorch+gRPC ARM64; measure and budget the per-ABI `.so` size; drop `--whole-archive` to the referenced kernel set (M-H1).
6. Standardize TorchScript export (script, document trace constraints); cap supported on-device model size by device class; demote 100M to non-mobile (M-H2).
7. Switch mobile to streaming RPCs with phone-appropriate message limits; preserve the chunking invariant (M-M3).
8. Adopt framework H1 heartbeat-death detection; add an Android foreground service for the training lifetime (M-H3).

**P2 (correctness + product):**
9. Add Python↔C++ serialization roundtrip test; delete dead block in `serializeStateDict`; fix the embedded `num_examples=0` inconsistency (M-M1).
10. Hard-error on unknown dtypes in `protoToTensors` (M-M2).
11. Stable persisted client ID (M-M4).
12. Native per-round telemetry (wall-clock, peak RSS, thermal, battery, heartbeat drops) surfaced through the bridge and onto the platform metrics path.
13. Move committed MNIST blobs out of git; first mobile CI job (build the C++ core + run parity tests on an ARM runner) (L4, L5).
