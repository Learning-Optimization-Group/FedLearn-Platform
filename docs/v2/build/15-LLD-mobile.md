# 15 — LOW-LEVEL DESIGN (LLD): MOBILE FL (Federated Learning) CLIENT

**Document type:** Production build specification — Low-Level Design (LLD) for one deployable unit.
**Unit:** the **mobile FL (Federated Learning) client** — React Native (RN) TypeScript app + a native C++ libtorch on-device FL core, bridged by a TurboModule.
**Audience:** a mid-sized local Large Language Model (LLM, ~30 billion parameters) implementing the unit. Every contract, version, path, type, and command below is **pre-decided**. Do not choose alternatives. Implement exactly what is written; fill in method bodies where a signature is given.
**Status:** build-authoritative for the mobile unit. This unit is **lifted/built in milestone M10** (`90-BUILD-SEQUENCE.md:477`), the last client milestone, after the proto (M2), the corrected DeComFL framework + golden vectors (M3), the control plane (M4), the substrate (M5), and observability (M6). It is specified fully **now** so M10 is a transcription task, not a design task.
**Date authored:** 2026-05-29.

> **CONFORMANCE.** This document does not redefine any contract. It conforms to, and references by exact name:
> - `02-TECH-STACK.md §17` (mobile pinned versions), `§3` (proto + buf), `§4.2` (safetensors), `§25` (hard invariants).
> - `03-DATA-MODEL.md §5.2` (`round_results`, `model_artifacts`, `fl_runs`, `determinism_manifests`).
> - `04-API-CONTRACTS.md §10` (gRPC `fedlearn.v2`), `§5` (internal callbacks), `§13` (per-run token), `§14` (traceparent).
> - The DeComFL correctness spec `docs/v2/specs/2026-05-29-decomfl-correctness-design.md` (the **determinism contract** and golden-vector fixtures), and the FL-framework LLD `11-LLD-fl-framework.md` (now on disk), which together pin the Python-side determinism slice this unit's C++ port must reproduce. This LLD references those authoritative sources and `04 §10`; it does **not** redefine `11-LLD`'s contracts.

---

## 0. Abbreviation key (first-use full forms; thereafter the short form is used)

| Short form | Full form |
|---|---|
| LLD | Low-Level Design |
| FL | Federated Learning |
| RN | React Native |
| LLM | Large Language Model |
| API | Application Programming Interface |
| gRPC | Google Remote Procedure Call |
| RPC | Remote Procedure Call |
| TLS | Transport Layer Security |
| mTLS | mutual Transport Layer Security |
| RNG | Random Number Generator |
| ZO | Zeroth-Order (optimization) |
| DeComFL | Dimension-Free Communication Federated Learning (zeroth-order; the v1 wiki's "Decomposed" expansion is wrong per the paper, `B1-paper-alignment.md:33`) |
| FedAvg | Federated Averaging |
| SPSA | Simultaneous Perturbation Stochastic Approximation |
| JSI | JavaScript Interface (React Native's C++ bridge layer) |
| JNI | Java Native Interface (Android native bridge) |
| NDK | Native Development Kit (Android C/C++ toolchain) |
| ABI | Application Binary Interface |
| ObjC++ | Objective-C++ (Objective-C with C++) |
| CMake | Cross-platform Make (the C++ build generator) |
| AAR | Android Archive (Android library package) |
| APK | Android Package (Android app package) |
| IPA | iOS App Store Package |
| CI | Continuous Integration |
| UUID | Universally Unique Identifier |
| sha256 | Secure Hash Algorithm 256-bit |
| ULP | Unit in the Last Place (floating-point precision unit) |
| FP | Floating-Point |
| BN | Batch Normalization |
| MNIST | Modified National Institute of Standards and Technology (handwritten-digit dataset) |
| CXX / C++17 | the C++ language, ISO standard 2017 |
| OS | Operating System |
| RSS | Resident Set Size (peak physical memory) |
| OKLCH | a perceptually-uniform color space (Lightness, Chroma, Hue) |
| WS | WebSocket |
| STOMP | Simple Text Oriented Messaging Protocol |
| W3C | World Wide Web Consortium |
| DLG | Deep Leakage from Gradients |
| DP | Differential Privacy |
| EKS | (AWS) Elastic Kubernetes Service |
| ARN | (AWS) Amazon Resource Name |
| QR | Quick Response (matrix barcode) |
| VBU | "verify-before-use" — confirm the latest patch/build exists before pinning (a `02-TECH-STACK.md` convention) |
| TDD | Test-Driven Development |
| gtest | GoogleTest (the C++ unit-test framework) |
| MFLOPs / GFLOPs | Mega / Giga Floating-point Operations |
| ANE / NPU | Apple Neural Engine / Neural Processing Unit |

---

## 1. Purpose & single responsibility

**Single responsibility.** The mobile FL client is a **phone-resident FL participant** that:
1. Joins a specific FL **run** (`fl_runs.id`) on the long-running v2 FL server over gRPC `fedlearn.v2`.
2. Runs **on-device training** — DeComFL (the primary, communication-light path) or FedAvg (the fallback, full-model path) — against **client-private** local data that never leaves the device (the FL invariant, `03-DATA-MODEL.md §1`).
3. Uploads **only** what the strategy requires: for DeComFL, gradient **scalars + seeds** (the O(K·P) wedge, never the perturbation vector `z`); for FedAvg, chunked `safetensors` model parameters.
4. Reports per-round telemetry (loss, accuracy, compute time, plus mobile-specific battery/thermal/memory) so a federation operator can see why a phone client is slow or churning.

**What it is NOT (scope boundaries, locked):**
- It is **not** a second re-implementation of the framework's algorithm maintained "by eye." The v1 mobile branch was exactly that and it drifted (`A6-mobile.md` Executive summary). The C++ ZO core in this unit is a **tested port** whose correctness is defined by the Python golden-vector fixtures (`docs/v2/specs/2026-05-29-decomfl-correctness-design.md §6`), not by inspection.
- It does **not** own the proto. There is **one** buf-generated `fedlearn.v2` source; the mobile unit consumes generated C++ stubs (`02-TECH-STACK.md §3.3`). The v1 two-drifted-copies state (`A6-mobile.md` M-C1) is structurally killed.
- It does **not** serve as the desktop training engine. Desktop uses the PyInstaller Python sidecar (`02-TECH-STACK.md §16` note); the C++ core is **mobile-only** for v2 (`02-TECH-STACK.md §17.3` rejected alternative).
- It does **not** call the control-plane REST API for run lifecycle directly from C++. Run **lifecycle** (create/start/stop) is a REST concern handled by the RN TypeScript layer talking to `04 §4`; the C++ core talks **only** gRPC to the FL server.

**Federated-learning context (stated per the project rule):**
- **Aggregation strategy assumed:** DeComFL (primary) with the corrected `1/P` averaging and CPU-canonical RNG (`docs/v2/specs/2026-05-29-decomfl-correctness-design.md §3`, Bugs 1 & 2); FedAvg (secondary) with `safetensors` chunking.
- **Client heterogeneity:** the mobile client is **CPU-only** (libtorch built `USE_VULKAN=OFF USE_METAL=OFF`, `A6-mobile.md` M-H2). It coexists in one federation with desktop (CPU/MPS/CUDA) and Jetson clients. The CPU-canonical perturbation contract is what makes a CPU phone and a CUDA server agree on `z` (`B1-paper-alignment.md` B1-C2 / `C3-reproducibility.md §5.1`).
- **Communication-round-bounded:** yes. Every round has a server-enforced deadline (`round_deadline_unix_ms`, `04 §10.2 GetServerStatusResponse`); the client never blocks forever.

---

## 2. Position in the system

### 2.1 Dependency graph

```
                         ┌─────────────────────────────────────────────┐
                         │  v2 FL SERVER  (long-running, keyed run_id)  │
                         │  M3 framework + M5 substrate                 │
                         └───────────────▲─────────────────────────────┘
                                         │ gRPC fedlearn.v2 (TLS+mTLS)
                                         │ (consumes §2.2 EXPOSED-by-server RPCs)
   ┌─────────────────────────────────────┴───────────────────────────────────┐
   │  MOBILE FL CLIENT (this unit)                                            │
   │                                                                          │
   │  React Native TS layer  ──TurboModule (JSI)──►  C++ FL core (libtorch)   │
   │  (screens, navigation,                          (DeComFLClient,          │
   │   nativewind tokens)                             ZerothOrderEstimator,   │
   │        │                                         FederatedLoop,          │
   │        │ REST (cookie/token via RN)              ModelManager,           │
   │        ▼                                         FedLearnClient/gRPC)    │
   │  CONTROL PLANE  /api/runs/* /api/projects/*  ◄───┘                       │
   │  (04 §3, §4)  — run lifecycle, dataset/version pin                       │
   └──────────────────────────────────────────────────────────────────────────┘
            │ consumes                                   produces ▼
   proto/ fedlearn.v2 (M2, buf)            round_results / model_artifacts (via SERVER, 03 §5.2)
   @fedlearn/tokens (OKLCH design system)  client metrics (ReportClientMetrics RPC, 04 §10.2)
   framework golden fixtures (M3)          determinism_manifests (read-only, via REST 04 §4.4)
```

### 2.2 Interfaces CONSUMED (referenced by exact name)

| Interface | Source contract | What the mobile unit uses it for |
|---|---|---|
| gRPC service `FederatedLearningService` (`fedlearn.v2`) | `04-API-CONTRACTS.md §10.1` | The C++ core's gRPC stub. RPCs consumed: `RegisterClient`, `GetServerStatus`, `Heartbeat`, `GetDeComFLConfig`, `SubmitGradientScalars`, `GetGlobalModel`/`GetGlobalModelStream`, `SubmitModelUpdate`/`SubmitModelUpdateStream`, `ReportClientMetrics`. |
| gRPC framing rules | `04-API-CONTRACTS.md §10.3` | TLS+mTLS default, cert-CN identity, `codec` whitelist `{safetensors, lz4+safetensors}`, chunk symmetry, `max_payload_bytes`, never-transmit-`z`, round deadline + quorum, status-code mapping. |
| REST `POST /api/projects/{projectId}/runs`, `GET /api/runs/{runId}`, `GET /api/runs/{runId}/status`, `GET /api/runs/{runId}/manifest`, `POST /api/runs/{runId}/stop` | `04-API-CONTRACTS.md §4` | RN layer launches/monitors/stops a run; reads the `DeterminismManifestDto` (`§4.4`) to display lineage and to assert the client's `torch_version`/`goldenVectorSha256` match. |
| REST auth (cookie `jwtToken`, `withCredentials`) | `04-API-CONTRACTS.md §1`, `§2` | RN layer authenticates the **user** for REST. (gRPC client identity is separate: cert CN + `enrollment_token`, `04 §10.3`.) |
| `enrollment_token` (minted by the backend at launch) | `04-API-CONTRACTS.md §10.1 RegisterClientRequest.enrollment_token` | Anti-Sybil binding for `RegisterClient`. The RN layer obtains it from the run-join REST flow and passes it to the C++ core. |
| Golden-vector fixtures `framework/tests/fixtures/decomfl_golden/` | DeComFL correctness spec `§6` | The C++ `ZerothOrderEstimator` must reproduce these vectors (the parity gate, `§10`). |
| `@fedlearn/tokens` (OKLCH token package, NativeWind theme) | `02-TECH-STACK.md §17.2`, `C5-design-ux.md §7.1` | The RN screens' design tokens. One brand across web/desktop/mobile. |
| `proto/fedlearn/v2/fedlearn.proto` (buf, M2) | `02-TECH-STACK.md §3.3` | The single proto source; C++ stubs are buf-generated from it in CI. |

### 2.3 Interfaces EXPOSED

| Interface | Consumer | Shape |
|---|---|---|
| TurboModule `NativeFedLearnCore` (JSI spec) | the RN TypeScript layer | Typed async methods (§5.1) — `registerClient`, `getServerStatus`, `runDeComFLRound`, `runFedAvgRound`, `loadModel`, `getDeviceMetrics`, `stop`. Returns **typed objects**, not stringly-typed JSON (fixes `A6-mobile.md` L3). |
| `ReportClientMetricsRequest` (outbound gRPC) | the FL server (M6 observability) | Per-round mobile telemetry: loss, accuracy, `compute_ms`, `client_type="mobile"` (`04 §10.2`). |
| (no REST exposed) | — | The mobile unit is a pure consumer of REST; it exposes no REST endpoints. |

> **Reasoning — the C++ core never touches REST.** Splitting "run lifecycle" (REST, RN layer) from "FL data plane" (gRPC, C++ core) mirrors the platform-wide split (`04 §0`) and keeps the cookie-JWT auth (`04 §1`) entirely in the JS layer where `withCredentials` and the OS keychain already live (`A6-mobile.md` "What the v2 subtree lift must reconcile" item 2). The C++ core's only credential is the gRPC cert + `enrollment_token`.

---

## 3. Tech stack for this unit (pinned from `02-TECH-STACK.md §17`; one-line reasoning each)

| Layer | Technology | Pinned version | Reasoning (one line) |
|---|---|---|---|
| App shell / bridge | React Native (TurboModule / JSI) | `0.8x` latest stable, VBU via `npm view react-native version` (`02 §17.1`, `§24.7`) | RN bridge wiring is sound; salvage RN, rebuild styling (`02 §17.1`). |
| Styling | NativeWind | `4.x` VBU (`02 §17.2`) | Tailwind-for-RN consuming the shared OKLCH tokens; kills the v1 inline Bootstrap hex (`C5 §1.1`). |
| Components | react-native-reusables | latest, VBU from npm (`02 §17.2`) | shadcn-for-RN; same component vocabulary as web/desktop (`C5 §7.1`). |
| Design tokens | `@fedlearn/tokens` | monorepo-internal (`02 §11`) | One OKLCH source of truth across all three surfaces (`C5 §7.1`). |
| Language (UI) | TypeScript | `5.x` exact, unified across surfaces (`02 §11`, `§24.7`) | One TS version removes drift (`B7-standards-dx.md:189`). |
| Native core language | C++17 | (libtorch requires C++17) (`02 §17.3`) | Matches libtorch's ABI and the existing C++ core. |
| On-device tensor math | libtorch (PyTorch C++) | **match server torch `2.12.0`** as closely as the ARM64 build allows; pin in `mobile_client/CMakeLists.txt` `LIBTORCH_DIR`, VBU the ARM64 build exists (`02 §17.3`, `§24.7`) | RNG parity is only meaningful against a pinned torch on both sides (`C3 §5.1`, `B1-H1`). |
| gRPC C++ runtime | gRPC C++ | current release, VBU; replaces v1 `v1.62.0` in the cross-compile script (`02 §3.2`, `§24.7`) | buf generates the **stubs** but NOT the linked C++ runtime; the cross-compile script stays (`02 §3.2`). |
| Wire serialization | Protocol Buffers (`fedlearn.v2`) + safetensors blobs | protobuf runtime from buf stubs (VBU); safetensors `0.4.x` (VBU) (`02 §3.1`, `§4.2`) | Typed, no `torch.save`/pickle on the wire (`04 §10.3` codec whitelist). |
| Proto toolchain | buf CLI | `1.70.0` (`02 §3.3`) | Single source + breaking-change gate kills `SubmitModelUpdateReque` drift (`02 §3.3`). |
| Android native build | CMake + NDK (via Gradle `externalNativeBuild`) | NDK VBU (pin in `build.gradle`); CMake `3.22+` (RN's default) | The JNI bridge + C++ core compile per-ABI; the existing JNI CMake is salvageable (`A6 §decision table` "RN bridge → salvage"). |
| iOS native build | Xcode + ObjC++ TurboModule | Xcode VBU (matched to the pinned RN) | The ObjC++ TurboModule is the iOS bridge; CxxTurboModule + JSI is architecturally sound (`A6 §decision table`). |
| C++ tests | GoogleTest (gtest) | VBU (pin in CMake `FetchContent`) | The golden-vector + roundtrip parity tests run in CI on an ARM runner (`A6 §L4`, DeComFL spec `§7`). |
| Secure storage | Android Keystore / iOS Data Protection (via RN secure-storage) | RN library VBU | Model + client-id + any user data stored encrypted, not in the app bundle (`A6 §M-C4`). |

> **ARM64 note (locked, `02 §17.3`):** libtorch is ~267 MB (CPU). The Android JNI link must **not** `--whole-archive` all of libtorch (v1 did, defeating dead-code elimination — `A6 §M-H1`); force-include only the referenced kernels (§11.5). Set a per-ABI `.so` size budget and gate it in CI.

---

## 4. Module / file structure

Exact tree under `mobile_client/` (after the M10 lift; the v1 dead layers in `A6 §M-H4`/`§L5` are deleted). One-line responsibility per file.

```
mobile_client/
├── CMakeLists.txt                      # top-level C++ build; pins LIBTORCH_DIR, gRPC runtime, gtest
├── package.json                        # RN 0.8x, nativewind 4.x, react-native-reusables, @fedlearn/tokens
├── tsconfig.json                       # TS 5.x, strict
├── babel.config.js                     # nativewind babel preset
├── tailwind.config.js                  # consumes @fedlearn/tokens OKLCH theme (no inline hex)
├── metro.config.js                     # RN bundler config
├── app.json                            # one product name: "FedLearn" (retire "FedMob", C5 §8)
│
├── proto/                              # NOT a copy — a CI-stamped symlink/vendored mirror of the
│   └── fedlearn/v2/fedlearn.proto      #   canonical proto; checksum-gated against proto/ root (§9 E6)
│
├── shared/                             # the cross-platform C++ FL core (the "tested port")
│   ├── CMakeLists.txt                  # builds libfedlearn_core.a from src/; links libtorch + grpc + proto stubs
│   ├── include/fedlearn/
│   │   ├── DeComFLClient.h             # DeComFL on-device client: fit() + rebuildModel() interface
│   │   ├── ZerothOrderEstimator.h      # ZO forward/central gradient-scalar estimator interface
│   │   ├── Perturbation.h              # canonical_perturbation(seed,n,dtype) — CPU-canonical RNG (Bug-2 contract)
│   │   ├── FederatedLoop.h             # round loop: deComFLLoop() + fedAvgLoop() interfaces
│   │   ├── ModelManager.h              # TorchScript load + flat-param get/set + safetensors serialize
│   │   ├── FedLearnClient.h            # gRPC client: dual channel (training + heartbeat), chunking
│   │   ├── DataLoader.h                # on-device data loading (client-private; validated input)
│   │   ├── Types.h                     # Seeds2D, GradientScalars2D, RoundConfig, RoundResult, DeviceMetrics
│   │   └── DtypeMap.h                  # SAFE_DTYPES whitelist; unknown dtype = hard error (M-M2)
│   ├── src/
│   │   ├── DeComFLClient.cpp           # impl: snapshot-restore revert, (eta/P)*delta update, rebuild
│   │   ├── ZerothOrderEstimator.cpp    # impl: double-accumulated loss diff, requires_grad filter
│   │   ├── Perturbation.cpp            # impl: torch::Generator(cpu).manual_seed -> randn (matches Python)
│   │   ├── FederatedLoop.cpp           # impl: K local steps x P perturbations, deadline checks
│   │   ├── ModelManager.cpp            # impl: jit::load + integrity check, flat params w/ requires_grad
│   │   ├── FedLearnClient.cpp          # impl: TLS+mTLS channel, streaming RPCs, dtype-safe protoToTensors
│   │   ├── DataLoader.cpp              # impl: validated JSON/tensor load
│   │   └── DtypeMap.cpp                # impl: string<->torch::Dtype, hard-error fallthrough
│   └── tests/                          # gtest — the parity & roundtrip contract
│       ├── CMakeLists.txt
│       ├── rng_parity_test.cpp         # T2: canonical_perturbation == committed golden fixture
│       ├── g_scalar_parity_test.cpp    # end-to-end g parity vs Python reference (tolerance)
│       ├── serialize_roundtrip_test.cpp# T3 mirror: safetensors save<->load symmetry
│       └── flatparam_filter_test.cpp   # requires_grad filter: frozen-layer model count matches Python
│
├── bridge/                             # TurboModule (the RN <-> C++ boundary)
│   ├── specs/
│   │   └── NativeFedLearnCore.ts       # codegen spec — TYPED return shapes (fixes L3 stringly-typed)
│   ├── common/
│   │   ├── FedLearnCoreModule.h        # CxxTurboModule (shared C++ TurboModule impl)
│   │   └── FedLearnCoreModule.cpp      # marshals JSI <-> C++ core; no hand-built JSON
│   ├── android/
│   │   └── jni/
│   │       ├── CMakeLists.txt          # JNI lib: links libfedlearn_core.a; NO --whole-archive (M-H1)
│   │       └── OnLoad.cpp              # JNI_OnLoad — registers the CxxTurboModule
│   └── ios/
│       └── NativeFedLearnCore.mm       # ObjC++ TurboModule shim -> common CxxTurboModule
│
├── android/                            # Android app project
│   ├── app/build.gradle                # externalNativeBuild(cmake), abiFilters arm64-v8a, NDK pin
│   ├── app/src/main/AndroidManifest.xml# INTERNET + FOREGROUND_SERVICE; network-security-config (L1)
│   └── app/src/main/res/xml/network_security_config.xml  # pin TLS; no cleartext outside dev
│
├── ios/                                # iOS app project
│   ├── Podfile                         # RN pods + libtorch/grpc xcframework refs
│   └── FedLearn/Info.plist             # ATS (App Transport Security) on; background-mode limits noted
│
├── src/                                # RN TypeScript app layer
│   ├── App.tsx                         # root; nativewind provider + navigation
│   ├── navigation/AppNavigator.tsx     # 3-tab bottom bar (Training / Library / Testing) — lucide icons
│   ├── screens/
│   │   ├── TrainingScreen.tsx          # join-run + DeComFL/FedAvg launch + live round progress
│   │   ├── ModelLibraryScreen.tsx      # on-device saved models list (encrypted storage)
│   │   └── ModelTestingScreen.tsx      # on-device inference; REAL softmax (kills fake confidence, C5 §3)
│   ├── components/                     # shared RN-reusables wrappers (StatusBadge, MetricTile, ...)
│   ├── lib/
│   │   ├── nativeCore.ts               # typed wrapper over NativeFedLearnCore TurboModule
│   │   ├── restClient.ts               # axios withCredentials -> /api/runs, /api/projects
│   │   ├── runJoin.ts                  # join-run flow: REST -> enrollment_token -> nativeCore.registerClient
│   │   ├── clientId.ts                 # stable persisted UUID client id (fixes M-M4)
│   │   └── deviceClass.ts              # device-class detection -> max supported model tier (M-H2)
│   └── theme/                          # re-exports @fedlearn/tokens for nativewind
│
├── scripts/
│   ├── build_libtorch_arm64.sh         # CI-prebuild libtorch ARM64; PINNED torch tag (fixes M-H1)
│   ├── build_grpc_arm64.sh             # CI-prebuild gRPC C++ ARM64; pinned release
│   ├── export_model.py                 # ONE export script, torch.jit.script for all tiers (M-H2)
│   └── fetch_demo_data.sh              # fetch MNIST instead of committing blobs (fixes L5)
│
└── assets/                             # bundled demo artifacts only (no committed MNIST raw, L5)
```

**Deleted vs v1 (do not recreate):** `src/federated/protos/` (third drifted proto, `A6 §M-C1`), `src/utils/resourceMonitor.js` + `src/federated/DatasetLoader.js` (import uninstalled `@tensorflow/tfjs`, `A6 §M-H4`), `patches/@tensorflow*`, `src/federated/deactivate`, the 0-byte stub components, and the duplicate committed `data/MNIST/` + `android/data/MNIST/` blobs (`A6 §L5`).

---

## 5. Key interfaces & type signatures

### 5.1 TurboModule spec — `bridge/specs/NativeFedLearnCore.ts` (TYPED; fixes `A6 §L3`)

```ts
// React Native TurboModule spec (codegen source of truth).
// Every method is async (Promise) and returns a TYPED object — never a hand-built JSON string.
import type { TurboModule } from 'react-native';
import { TurboModuleRegistry } from 'react-native';

export type Strategy = 'DeComFL' | 'FedAvg';
export type GradEstimateMethod = 'forward' | 'central';

export interface RegisterResult {
  accepted: boolean;
  message: string;
  assignedRound: number;     // RegisterClientResponse.assigned_round (late-joiner round)
  serverProtocolVersion: number;
}

export interface ServerStatus {
  serverState: string;       // GetServerStatusResponse.ServerState name
  currentRound: number;
  requiredClientsForRound: number;
  receivedUpdatesThisRound: number;
  activeClients: number;
  roundDeadlineUnixMs: number;
}

export interface RoundConfig {
  strategy: Strategy;
  learningRate: number;      // eta
  mu: number;                // DeComFL ZO smoothing radius (double on the C++ side)
  numPerturbations: number;  // P  (1..256, 04 §4.2)
  numLocalSteps: number;     // K  (1..1000)
  gradEstimateMethod: GradEstimateMethod;  // default 'forward' (B1-H2)
  seed: number;              // optimizer seed (distinct from data seed)
  torchVersion: string;      // must match server's GetDeComFLConfigResponse.torch_version
}

export interface RoundResult {
  round: number;
  loss: number;
  accuracy: number;          // -1.0 if not evaluated this round
  scalarsTransmitted: number;// K*P (DeComFL) ; 0 for FedAvg
  uplinkBytes: number;
  downlinkBytes: number;
  computeMs: number;
  reverted: boolean;         // model restored to pre-round snapshot (DeComFL invariant)
}

export interface DeviceMetrics {
  peakRssBytes: number;      // native RSS sample (replaces broken resourceMonitor.js, M-H4)
  thermalState: string;      // 'NOMINAL'|'FAIR'|'SERIOUS'|'CRITICAL' (platform thermal API)
  batteryLevel: number;      // 0.0..1.0
  batteryCharging: boolean;
}

export interface ModelInfo {
  paramCount: number;        // model dimension d
  trainableParamCount: number; // requires_grad-filtered count (must match server's P-dim, M-C2)
  sha256: string;            // integrity hash of the .pt (verified before jit::load, M-C4)
  tier: '1M' | '10M' | '100M';
}

export interface NativeFedLearnCoreSpec extends TurboModule {
  // ---- gRPC lifecycle ----
  registerClient(
    serverAddress: string, runId: string, clientId: string,
    enrollmentToken: string, useTls: boolean): Promise<RegisterResult>;
  getServerStatus(runId: string): Promise<ServerStatus>;
  stop(): Promise<void>;                       // sets the abort flag; joins threads

  // ---- model ----
  loadModel(modelPath: string, expectedSha256: string): Promise<ModelInfo>;  // integrity-checked

  // ---- one round (the bridge runs ONE round per call; the RN layer loops + checks deadline) ----
  runDeComFLRound(runId: string, config: RoundConfig): Promise<RoundResult>;
  runFedAvgRound(runId: string, config: RoundConfig): Promise<RoundResult>;

  // ---- inference (Model Testing screen) — REAL softmax, not exp(-loss) (C5 §3) ----
  infer(inputJson: string): Promise<{ logits: number[]; probabilities: number[]; argmax: number }>;

  // ---- telemetry ----
  getDeviceMetrics(): Promise<DeviceMetrics>;
}

export default TurboModuleRegistry.getEnforcing<NativeFedLearnCoreSpec>('NativeFedLearnCore');
```

> **Reasoning — one round per bridge call, the RN layer loops.** Long-running multi-minute native calls across the bridge are the worst case for Android Doze / iOS background suspension (`A6 §M-H3`). Returning after each round lets the RN layer (a) check the server deadline, (b) sample device metrics, (c) keep the foreground service heartbeat alive, and (d) honor a user `stop()` between rounds. The heartbeat gRPC stub still runs in parallel inside the C++ core during the blocking round (§6.4).

### 5.2 C++ core — key class signatures

```cpp
// include/fedlearn/Perturbation.h  — THE determinism contract (Bug-2 fix, DeComFL spec §3)
namespace fedlearn {
// Device-independent N(0, I_d). Generated on CPU for bit-stable output, regardless of
// where the forward pass runs. MUST mirror Python canonical_perturbation EXACTLY:
//   torch.Generator(device="cpu").manual_seed(seed); torch.randn(n, generator=g, dtype, device="cpu")
// The seed widening int32(wire) -> int64(set_current_seed) is safe in [0, 2^31) (B1 Low, C3 §2.2).
torch::Tensor canonical_perturbation(int64_t seed, int64_t num_params,
                                     torch::Dtype dtype = torch::kFloat32);
}
```

```cpp
// include/fedlearn/ZerothOrderEstimator.h
namespace fedlearn {
class ZerothOrderEstimator {
 public:
  // mu is DOUBLE (Python smoothing_param is float64) — fixes A6 M-C2 #2.
  ZerothOrderEstimator(double mu, GradEstimateMethod method /*forward|central*/);

  // Returns the scalar g for one perturbation seed.
  //   forward: g = (f(x+mu*z) - f(x)) / mu
  //   central: g = (f(x+mu*z) - f(x-mu*z)) / (2*mu)
  // CRITICAL (M-C2 #1): the loss DIFFERENCE is accumulated in DOUBLE before dividing;
  //   do NOT extract two float32 scalars and subtract. Keep the subtraction in-tensor / double.
  double computeGradientScalar(
      torch::jit::Module& model,
      const torch::Tensor& flatParams,    // requires_grad-filtered (see ModelManager)
      int64_t perturbationSeed,
      const torch::Tensor& batchInputs,
      const torch::Tensor& batchTargets);

 private:
  double mu_;
  GradEstimateMethod method_;
};
}
```

```cpp
// include/fedlearn/ModelManager.h
namespace fedlearn {
class ModelManager {
 public:
  // Verifies sha256(file) == expected BEFORE jit::load (untrusted-input rule, M-C4), then loads.
  ModelInfo loadScriptModel(const std::string& path, const std::string& expectedSha256);

  // Flatten ONLY parameters with requires_grad == true, in module-iteration order
  // (matches Python _get_flat_params; fixes A6 M-C2 #3 frozen-layer divergence).
  torch::Tensor getFlatParams(const torch::jit::Module& model) const;
  void          setFlatParams(torch::jit::Module& model, const torch::Tensor& flat) const;

  // safetensors blob symmetric with the framework serializer (codec="safetensors", 04 §10.3).
  // Wraps {"parameters": state_dict, "num_examples": n} — symmetric save/load (fixes Bug 3 / M-M1).
  std::string serializeStateDict(const torch::jit::Module& model, int64_t numExamples) const;
  void        loadStateDict(torch::jit::Module& model, const std::string& blob) const;

  int64_t trainableParamCount(const torch::jit::Module& model) const;
};
}
```

```cpp
// include/fedlearn/DeComFLClient.h
namespace fedlearn {
class DeComFLClient {
 public:
  DeComFLClient(ModelManager& mm, ZerothOrderEstimator& zo, double eta, int P, int K);

  // One DeComFL round. Returns the per-(k,p) gradient scalars to upload (NEVER z).
  // Invariant: snapshot x_initial = getFlatParams().clone() at the top; restore EXACTLY at the end
  //   (snapshot-restore revert — the paper-faithful pattern, B1-M1; do NOT subtract a running sum).
  // Local update inside the round uses (eta / P) * delta (the 1/P factor — matches server after Bug-1 fix).
  GradientScalars2D fit(torch::jit::Module& model,
                        const Seeds2D& seeds,            // [K][P] from GetDeComFLConfigResponse
                        const DataBatch& batch);

  // Replay missed rounds r' = t_i .. r-1: regenerate z from seed, apply x -= (eta/P)*g*z (Alg.2 rebuild).
  void rebuildModel(torch::jit::Module& model, const RebuildHistory& history);

 private:
  ModelManager& mm_; ZerothOrderEstimator& zo_;
  double eta_; int P_; int K_;
};
}
```

```cpp
// include/fedlearn/FedLearnClient.h  — the gRPC layer
namespace fedlearn {
class FedLearnClient {
 public:
  // useTls=false ONLY when the RN layer passes a dev flag; default secure (TLS+mTLS), 04 §10.3.
  FedLearnClient(const std::string& serverAddress, bool useTls,
                 const std::string& clientCertPath, const std::string& clientKeyPath,
                 const std::string& caCertPath);

  RegisterClientResponse registerClient(const std::string& runId, const std::string& clientId,
                                        const std::string& enrollmentToken, int protocolVersion);
  GetServerStatusResponse getServerStatus(const std::string& runId);
  GetDeComFLConfigResponse getDeComFLConfig(const std::string& runId, const std::string& clientId);

  // DeComFL upload: scalars + num_examples (the O(K*P) wedge). Returns bytes_received.
  SubmitGradientScalarsResponse submitGradientScalars(const std::string& runId,
      const std::string& clientId, int trainedOnRound,
      const GradientScalars2D& gradients, int64_t numExamples);

  // FedAvg: STREAMING for anything above a small threshold (fixes M-M3 unary 1 GB footgun).
  ModelParameters getGlobalModelStream(const std::string& runId, const std::string& clientId);
  SubmitModelUpdateResponse submitModelUpdateStream(const std::string& runId,
      const std::string& clientId, int trainedOnRound,
      const std::string& safetensorsBlob, int64_t numExamples);

  void reportClientMetrics(const ReportClientMetricsRequest& m);  // telemetry, best-effort

  // Heartbeat runs on its OWN thread (dual-channel). Sets abortFlag_ on N consecutive failures (M-H3).
  void startHeartbeat(const std::string& runId, const std::string& clientId, int currentRound);
  void stopHeartbeat();
  bool shouldStop() const;   // reads HeartbeatResponse.should_stop OR abortFlag_

 private:
  std::shared_ptr<grpc::Channel> trainingChannel_;
  std::shared_ptr<grpc::Channel> heartbeatChannel_;   // SEPARATE channel (dual-heartbeat, 04 §10.1.6)
  std::atomic<bool> abortFlag_{false};
  std::thread heartbeatThread_;
};
}
```

### 5.3 Plain-data types — `include/fedlearn/Types.h`

```cpp
namespace fedlearn {
using Seeds2D          = std::vector<std::vector<int64_t>>;   // [K][P] perturbation seeds
using GradientScalars2D= std::vector<std::vector<double>>;    // [K][P] g scalars (double — B1 wire is double)

enum class GradEstimateMethod { Forward, Central };

struct RoundConfig {       // mirrors the TurboModule RoundConfig
  std::string strategy; double learningRate; double mu;
  int numPerturbations; int numLocalSteps; GradEstimateMethod method;
  int64_t seed; std::string torchVersion;
};

struct DeviceMetrics { int64_t peakRssBytes; std::string thermalState;
                       double batteryLevel; bool batteryCharging; };

struct ModelInfo { int64_t paramCount; int64_t trainableParamCount;
                   std::string sha256; std::string tier; };
}
```

### 5.4 Data row types the unit READS/WRITES (via the server, not directly) — see §7

```ts
// REST DTOs the RN layer consumes (04 §4.4) — typed at the wire with Zod (04 §1 validation row)
interface DeterminismManifestDto {     // GET /api/runs/{runId}/manifest
  runId: string; seed: number; strategy: 'DeComFL' | 'FedAvg';
  hyperparameters: Record<string, unknown>;
  torchVersion: string;                // mobile MUST match this for RNG parity (else warn/refuse)
  numpyVersion: string; frameworkGitSha: string;
  datasetVersionId: string; datasetSha256: string; partitionRecipeId: string;
  modelInitSha256: string; goldenVectorSha256: string;  // the fixture hash the C++ port validated
  createdAt: string;
}
```

---

## 6. Core algorithms & flows

### 6.1 Join-run flow (RN layer → REST → C++ gRPC)

```
USER taps "Join run" on TrainingScreen
  │
  ▼
runJoin.ts: POST /api/projects/{projectId}/runs (or join an existing run)   [04 §4]
  │   body: StartRunRequest { strategy, numRounds, minClients, launcher, datasetVersionId, ... }
  │   <- 202 RunDto { id=runId, grpcEndpoint (null until RUNNING), ... }
  ▼
runJoin.ts: poll GET /api/runs/{runId}/status until status==RUNNING && grpcEndpoint != null
  │   (also GET /api/runs/{runId}/manifest -> assert manifest.torchVersion == libtorch build version)
  ▼
runJoin.ts: obtain enrollment_token (minted by backend at launch, surfaced via the join response)
  ▼
clientId.ts: load/create a STABLE persisted UUID (encrypted storage) — fixes M-M4
  ▼
nativeCore.registerClient(grpcEndpoint, runId, clientId, enrollmentToken, useTls=true)
  │   -> C++ FedLearnClient.registerClient(...) -> RegisterClient RPC
  │      server checks cert CN + enrollment_token + protocol_version (04 §10.3); REJECTED on mismatch
  ▼
loadModel(modelPath, manifest.modelInitSha256)  -> integrity-check then jit::load
  ▼
RN layer enters the round loop (§6.2 or §6.3)
```

> **Reasoning — the manifest is checked before any round.** `04 §4.4`'s `DeterminismManifestDto.torchVersion` and `goldenVectorSha256` are the run's reproducibility contract. If the phone's libtorch build version disagrees, RNG parity is not guaranteed (`C3 §6.3` "federation version-compatibility gate"); the client warns (or refuses outside dev). This is the mobile half of the gate `C3` puts on every node.

### 6.2 DeComFL round (the primary, communication-light path)

This is the heart of the unit. It must produce gradient scalars that the server can combine with **the same `z`** the server regenerates — which only holds because both sides use `canonical_perturbation` on CPU (Bug-2 contract).

```
RN loop, per round r (until currentRound == numRounds OR shouldStop):
  ┌─────────────────────────────────────────────────────────────────────────┐
  │ 1. cfg = GetDeComFLConfig(runId, clientId)   [04 §10.2 GetDeComFLConfigResponse]│
  │    -> current_seeds [K][P], rebuild_history (rounds this client missed),       │
  │       config{lr,mu,P,K}, torch_version, grad_estimate_method, golden_vector_sha256│
  │ 2. IF cfg.torch_version != localTorchVersion -> WARN (refuse outside dev)       │
  │ 3. IF rebuild_history non-empty -> DeComFLClient.rebuildModel(model, history)   │
  │      (replay missed rounds with (eta/P)*g*z; regenerate z via canonical_perturbation)│
  │ 4. scalars = DeComFLClient.fit(model, current_seeds, batch)                     │
  │ 5. SubmitGradientScalars(runId, clientId, r, scalars, numExamples)              │
  │    -> bytes_received == K*P*8 (the O(K*P) comm-cost number, 04 §10.2 / B3 §6.2) │
  │ 6. ReportClientMetrics(loss, accuracy, compute_ms, client_type="mobile")        │
  │ 7. getDeviceMetrics() -> surface battery/thermal/RSS to the UI + telemetry      │
  └─────────────────────────────────────────────────────────────────────────┘
```

`DeComFLClient::fit` pseudocode (the paper-faithful body; `B1` C1/M1, `A6` M-C2):

```cpp
GradientScalars2D DeComFLClient::fit(Module& model, const Seeds2D& seeds, const DataBatch& batch) {
  // --- snapshot for exact revert (B1-M1: snapshot-restore, NOT running-sum subtract) ---
  torch::Tensor x_initial = mm_.getFlatParams(model).clone();   // requires_grad-filtered
  torch::Tensor x_current = x_initial.clone();

  GradientScalars2D out(K_, std::vector<double>(P_, 0.0));
  for (int k = 0; k < K_; ++k) {
    torch::Tensor delta = torch::zeros_like(x_current);          // accumulate over P
    for (int p = 0; p < P_; ++p) {
      int64_t s = seeds[k][p];
      // z on CPU (canonical), then move to compute device if forward runs elsewhere.
      torch::Tensor z = fedlearn::canonical_perturbation(s, x_current.numel(), torch::kFloat32);
      // g uses DOUBLE-accumulated loss difference inside the estimator (M-C2 #1).
      double g = zo_.computeGradientScalar(model, x_current, s, batch.inputs, batch.targets);
      out[k][p] = g;
      delta += g * z;                                            // delta += g_p * z_p
    }
    // local step with the 1/P averaging factor (matches the server AFTER the Bug-1 fix).
    x_current = x_current - (eta_ / P_) * delta;
    mm_.setFlatParams(model, x_current);
  }
  // --- exact revert: client reverts to pre-round state; server owns the true global trajectory ---
  mm_.setFlatParams(model, x_initial);
  return out;     // upload ONLY scalars + seeds-by-reference; z NEVER leaves the device
}
```

`ZerothOrderEstimator::computeGradientScalar` (the catastrophic-cancellation fix, `A6 §M-C2 #1`):

```cpp
double ZerothOrderEstimator::computeGradientScalar(
    Module& model, const Tensor& flatParams, int64_t seed,
    const Tensor& inputs, const Tensor& targets) {
  Tensor z = fedlearn::canonical_perturbation(seed, flatParams.numel(), torch::kFloat32);

  // f(x + mu*z)  — keep the loss as a tensor; extract to DOUBLE late.
  mm_.setFlatParams(model, flatParams + mu_ * z);
  double loss_plus = lossTensor(model, inputs, targets).item<double>();   // <-- double, not float

  double loss_ref;
  if (method_ == GradEstimateMethod::Central) {
    mm_.setFlatParams(model, flatParams - mu_ * z);
    loss_ref = lossTensor(model, inputs, targets).item<double>();
    mm_.setFlatParams(model, flatParams);                                  // restore
    return (loss_plus - loss_ref) / (2.0 * mu_);                           // central, O(mu^2) bias
  } else {
    mm_.setFlatParams(model, flatParams);                                  // restore for f(x)
    loss_ref = lossTensor(model, inputs, targets).item<double>();
    return (loss_plus - loss_ref) / mu_;                                   // forward, O(mu) bias
  }
}
```

> **Why double, not float (the load-bearing numeric fix):** `g = (f(x+μz) − f(x))/μ` with μ=0.001 is a catastrophic-cancellation regime — two nearly-equal losses subtracted to recover a tiny signal. v1's C++ extracted two `float32` scalars *then* subtracted, discarding exactly the low-order bits that carry the gradient (`A6 §M-C2 #1`). Extracting each loss to `double` (and keeping `mu_` `double`) preserves them and matches Python's float64 division.

### 6.3 FedAvg round (the fallback, full-model path)

```
RN loop, per round r:
  1. params = GetGlobalModelStream(runId, clientId)   // STREAMING, not unary (fixes M-M3)
  2. ModelManager.setFlatParams(model, params)        // (or load_state_dict for full dict)
  3. local SGD: K steps of loss.backward() + manual SGD on the model
  4. blob = ModelManager.serializeStateDict(model, numExamples)   // safetensors, wrapped {parameters,num_examples}
  5. SubmitModelUpdateStream(runId, clientId, r, blob, numExamples)// chunked if > threshold; codec="safetensors"
```

Chunk symmetry is enforced exactly as the framework (`04 §10.3`): the reassembled blob is a `safetensors` dict; sender wraps `{parameters, num_examples}`, receiver unwraps; `sha256` must match. The mobile cap on `max_{send,receive}_message_length` is **tens of MB** (not 1 GB — fixes `A6 §M-M3`), forcing the streaming path for anything large.

### 6.4 Dual-heartbeat (parallel stub) sequence

```
 main/training thread                          heartbeat thread (started before the round loop)
 ──────────────────                            ────────────────────────────────────────────────
 registerClient()
 startHeartbeat() ──────────────────────────►  loop every Hb interval:
 for round in 0..N:                               Heartbeat(runId, clientId, status, step, round)
   fit()  [BLOCKS this thread for the round]      resp = HeartbeatResponse
   if shouldStop(): break  ◄───────────────────── if resp.should_stop OR N consecutive failures:
   submit...                                          set abortFlag_  (M-H3: death is now VISIBLE)
 stopHeartbeat() / join()
```

> **Reasoning — preserve the two stubs (locked invariant).** The training stub blocks during `fit()`; the heartbeat stub runs on a parallel thread so the server does not time the client out during long rounds (`02 §4.1`, `04 §10.1.6`). v1 mobile caught `catch(...)` in the heartbeat loop and continued, so a dead heartbeat was invisible and the server rejected the eventual upload as stale (`A6 §M-H3`). v2 sets an `std::atomic<bool> abortFlag_` on N consecutive failures, checked **between** local steps — the framework H1 fix ported to mobile. On Android the round loop runs under a **foreground service** for its lifetime (§9 E5); on iOS long rounds run only in the foreground and the UI says so.

### 6.5 Inference flow (Model Testing screen) — real softmax (kills `C5 §3` fake chart)

```
USER draws/loads a 28x28 input on ModelTestingScreen
  -> nativeCore.infer(inputJson)
     -> C++: logits = model.forward(input);  probs = torch::softmax(logits, dim=-1)
     -> returns { logits, probabilities, argmax }   // REAL probabilities
  -> UI renders the actual softmax bar chart (NOT exp(-loss) — that fake proxy is deleted, C5 §3/§9)
```

The v1 28×28 grid of 784 absolutely-positioned `<View>`s (`A6 §M-H2`/`C5 §3.1` perf liability) is replaced by a single canvas/`Skia` draw surface in the RN layer.

---

## 7. Data it owns

**The mobile unit owns NO control-plane table.** Per the FL invariant (`03-DATA-MODEL.md §1`), raw training features/labels live **only on the device** and never enter any table. The mobile client **contributes rows indirectly** — it sends gRPC telemetry/updates to the FL server, which is the writer of these tables (`03 §5.2`). The columns the mobile path populates (via the server) are:

| Table (`03 §5.2`) | Columns the mobile client's data flows into | Via |
|---|---|---|
| `round_results` | `round_idx`, `loss`, `accuracy`, `uplink_bytes`, `downlink_bytes`, `scalars_transmitted` (= K·P for DeComFL), `num_clients_reported`, `round_started_at`, `round_ended_at` | FL server writes after `SubmitGradientScalars`/`SubmitModelUpdate` + `ReportClientMetrics`; server POSTs `RoundResultDto` to `/api/internal/runs/{runId}/results` (`04 §5.1`). |
| `model_artifacts` | `sha256`, `storage_uri`, `size_bytes`, `kind`, `fl_run_id`, `round_idx` | Only the **server** aggregates and writes the final/checkpoint artifact (the phone uploads its update; it does not author an artifact row). |
| `fl_runs` | (read-only to mobile) `status`, `round_idx`, `grpc_endpoint`, `config` | The RN layer **reads** these via `GET /api/runs/{runId}` / `/status`. |
| `determinism_manifests` | (read-only) `torch_version`, `seed`, `golden_vector` hash, `manifest_json` | The RN layer **reads** via `GET /api/runs/{runId}/manifest` to run the version gate (§6.1). |

> **Exact column names are authoritative in `03-DATA-MODEL.md §5.2`.** `round_results.round_idx` is the `serverRound` field on the wire (`04 §5.1` reasoning); `scalars_transmitted` is the DeComFL bandwidth wedge column.

**On-device (in-memory / encrypted local) structures the mobile unit DOES own:**

| Structure | Type | Lifetime | Storage |
|---|---|---|---|
| Stable client id | `UUID` string | persisted across restarts (fixes `A6 §M-M4`) | encrypted (Android Keystore / iOS Data Protection) |
| Loaded model | `torch::jit::Module` | per-run | RAM; the `.pt` file in app-private encrypted storage (`A6 §M-C4`) |
| `x_initial` snapshot | `torch::Tensor` (`numel` = trainable params) | per-round | RAM (transient; see §10 memory bounds) |
| Perturbation `z` | `torch::Tensor` (`numel` = trainable params) | per-(k,p) | RAM (transient; the 2× working-set driver, §10) |
| Seeds `[K][P]` / scalars `[K][P]` | `Seeds2D` / `GradientScalars2D` | per-round | RAM (tiny: K·P int64/double) |
| gRPC cert + key + CA | files | app lifetime | app-private encrypted storage |
| `enrollment_token` | string | per-run | RAM only (never persisted) |

---

## 8. Configuration & environment variables

The mobile app is configured by **RN config + native build args**, not server env vars. (The `04 §13` `FEDLEARN_*` env vars are injected into the **FL server** executor, not the phone — the phone receives the equivalent values over REST/gRPC.)

| Name | Layer | Type | Default | Where |
|---|---|---|---|---|
| `FEDLEARN_API_URL` | RN (`.env` / app config) | string (URL) | — (required) | `restClient.ts`; the control-plane base for `/api/*`. |
| `FEDLEARN_ALLOW_INSECURE_GRPC` | RN config | boolean | `false` | dev-only flag → `useTls=false` to `registerClient`. **Default secure** (`04 §10.3`). Refused outside a dev build. |
| `FEDLEARN_MAX_MODEL_TIER` | RN config / `deviceClass.ts` | enum `1M\|10M\|100M` | device-class detected (§10) | caps the supported on-device model (`A6 §M-H2`). |
| `FEDLEARN_HEARTBEAT_INTERVAL_MS` | RN config | int | `5000` | passed to `startHeartbeat`. |
| `FEDLEARN_HEARTBEAT_FAILURE_LIMIT` | RN config | int | `3` | N consecutive failures → `abortFlag_` (`A6 §M-H3`). |
| `FEDLEARN_GRPC_MAX_MESSAGE_BYTES` | RN config → C++ channel args | int | `33554432` (32 MB) | phone-appropriate cap, not 1 GB (`A6 §M-M3`). |
| `LIBTORCH_DIR` | CMake (`mobile_client/CMakeLists.txt`) | path | (CI-prebuilt artifact path) | pins the ARM64 libtorch matched to torch `2.12.0` (`02 §17.3`). |
| `PYTORCH_TAG` | `scripts/build_libtorch_arm64.sh` | git tag | matches torch `2.12.0` | the PINNED tag (fixes `A6 §M-H1` unpinned `${PYTORCH_SRC}`). |
| `GRPC_CPP_VERSION` | `scripts/build_grpc_arm64.sh` | version | current release (VBU) | replaces v1 hardcoded `v1.62.0` (`02 §3.2`). |
| `ANDROID_ABI` | Android `build.gradle` | string | `arm64-v8a` | the only shipped ABI (size budget, `A6 §M-H1`). |

**Build "profiles" (mirror Spring/Vite profiles 1:1):** a `dev` RN build allows `FEDLEARN_ALLOW_INSECURE_GRPC=true` and a localhost API; a `release` build forbids it (TLS+mTLS mandatory) and points at the production control plane.

---

## 9. Error handling & edge cases

Enumerate the real failure modes and the exact handling. Codes map to `04 §12.1` (REST) or gRPC status (`04 §10.3`).

| # | Failure mode | Where | Exact handling |
|---|---|---|---|
| E1 | Protocol-version mismatch at registration | `RegisterClient` | Server returns `RegisterClientResponse.status = REJECTED`; C++ surfaces `RegisterResult.accepted=false`; RN shows "client out of date" and aborts join. (v1 had no `protocol_version` — `04 §10.1.2`.) |
| E2 | `torch_version` mismatch (RNG-parity risk) | `GetDeComFLConfig` step 2 (§6.2) | WARN in dev; **refuse to train** in release (the federation version gate, `C3 §6.3`). RNG parity is undefined across torch versions (`B1-H1`). |
| E3 | Golden-vector parity failure (C++ `z` ≠ Python `z`) | CI `rng_parity_test.cpp` | **Release blocker** (`B1-H1`, `C3 §9 risk 2`). Never ship a build whose perturbation diverges from the Python fixture — silent non-learning otherwise (`B1-C2`). |
| E4 | Heartbeat thread death (channel dropped, app backgrounded) | heartbeat thread | After `FEDLEARN_HEARTBEAT_FAILURE_LIMIT` consecutive failures, set `abortFlag_`; the training loop checks it between local steps and aborts cleanly (`A6 §M-H3`). |
| E5 | OS suspends the app mid-round (Android Doze / iOS background) | RN layer | Android: run the round loop under a **foreground service** for its lifetime. iOS: long rounds run only in foreground; the UI states this. Backgrounding triggers a clean `stop()` + server-visible heartbeat gap (E4). |
| E6 | Proto drift (mobile copy ≠ canonical) | CI `proto.yml` | Checksum gate fails the build if `mobile_client/proto/**` differs from canonical `proto/fedlearn/v2/fedlearn.proto` (`A6 §M-C1`, `02 §3.3`). |
| E7 | Unknown tensor dtype on the wire | `FedLearnClient::protoToTensors` | **Hard error** (gRPC `INVALID_ARGUMENT`), never a silent float32 fallback (fixes `A6 §M-M2`); use `DtypeMap` SAFE_DTYPES whitelist (`04 §10` Tensor.dtype whitelist). |
| E8 | Tampered / corrupt `.pt` model file | `ModelManager::loadScriptModel` | Verify `sha256(file) == expectedSha256` BEFORE `jit::load`; on mismatch throw and refuse to load (a tampered `.pt` is a code-execution vector — `A6 §M-C4`). |
| E9 | Frozen-layer model (LoRA / partial fine-tune) | `getFlatParams`/`setFlatParams` | `requires_grad` filter ensures the flattened length/order matches the server's P-dimension (fixes `A6 §M-C2 #3`); `flatparam_filter_test.cpp` asserts the count equals Python's. |
| E10 | Round deadline passes before the phone submits | round loop | The server proceeds with ≥ min-quorum and may mark the client late; the client respects `round_deadline_unix_ms` and does not block forever (`04 §10.3` round-deadline rule). |
| E11 | Payload exceeds `max_payload_bytes` | streaming upload | gRPC `RESOURCE_EXHAUSTED`; the client must chunk (it already does); never buffer a 1 GB unary payload on a phone (`A6 §M-M3`). |
| E12 | OOM on a too-large model tier | `loadModel` / round | `deviceClass.ts` caps `FEDLEARN_MAX_MODEL_TIER`; 100M is rejected on mid-tier devices (the 2× ZO working set OOMs them — §10, `A6 §M-H2`). |
| E13 | gRPC channel insecure outside dev | `FedLearnClient` ctor | Refuse to construct an insecure channel unless the dev flag is set; default TLS+mTLS (`04 §10.3`, `A6 §M-C4`). |
| E14 | Run already terminal when the client calls back | server | gRPC `FAILED_PRECONDITION` / REST `409 RUN_TERMINAL`; the client stops the loop and reports the terminal status to the UI. |
| E15 | `enrollment_token` invalid/expired | `RegisterClient` | gRPC `UNAUTHENTICATED`; RN re-runs the join flow to mint a fresh token. |

---

## 10. Battery / thermal / memory bounds for 1M / 10M / 100M-parameter models

This is a first-class deliverable: the mobile compute story (`A6 §M-H2`). libtorch on mobile is **CPU-only** (`USE_VULKAN=OFF USE_METAL=OFF`, no GPU/NPU), and DeComFL does **2 forward passes per perturbation** (forward-diff) or **3** (central), so a round is **2·K·P** (forward) or **3·K·P** (central) full forward passes.

**Round forward-pass count (defaults K and P=10, `04 §4.2`):**
- forward, K=5, P=10 → 2·5·10 = **100 forward passes/round**.
- central, K=5, P=10 → 3·5·10 = **150 forward passes/round**.

**Working-set memory model (the OOM driver).** During `fit()` the transient peak above the loaded model is roughly:
- `x_initial` (trainable-param vector, float32) +
- `x_current` (same size) +
- `z` (same size, per perturbation) +
- `delta` (same size accumulator).

So the ZO temporaries add **~3–4× the trainable-param vector** on top of the loaded model. For float32, the trainable-param vector alone is `4 · d` bytes.

| Tier | params `d` | model RAM (fp32, ~`4d`) | ZO temporaries (~`4·4d`) | total working set (rough) | fwd-pass cost (CPU) | Verdict |
|---|---|---|---|---|---|---|
| **1M** | 1×10⁶ | ~4 MB | ~16 MB | **~20–40 MB** | ~ms–tens of ms | **Deployable.** Rounds in seconds; battery/thermal acceptable. The realistic on-device tier. |
| **10M** | 1×10⁷ | ~40 MB | ~160 MB | **~200–300 MB** | ~tens–hundreds of ms | **Deployable with care.** 100 fwd passes → tens of seconds/round; watch thermal throttling on sustained rounds. Cap K·P. |
| **100M** | 1×10⁸ | ~400 MB | ~1.6 GB | **~2 GB transient** | hundreds of ms–seconds each | **NOT mobile.** 100 fwd passes → minutes/round; ~2 GB working set OOMs mid-tier Android (`A6 §M-H2`). Demote to server/desktop tier. |

**Bounds enforcement (locked):**
1. `deviceClass.ts` detects total RAM and CPU class → sets `FEDLEARN_MAX_MODEL_TIER`. 100M is **never** offered on a phone (`A6 §M-H2` recommendation; `C5 §3`/`A6` "100M is a benchmark artifact, not a deployable mobile config").
2. Per-round telemetry (`getDeviceMetrics` → `ReportClientMetrics`) reports peak RSS, thermal state, and battery delta so the operator sees a throttling/draining phone (`A6 §M-H2`/`§Observability parity`).
3. The RN layer pauses the loop and warns if `thermalState ∈ {SERIOUS, CRITICAL}` or battery is low and not charging.
4. **DeComFL is the right mobile strategy** precisely because its **uplink** is O(K·P) scalars (~K·P·8 bytes, e.g. 50·8 = 400 bytes/round) independent of `d` — the bandwidth win on cellular (`A6 §M-H2`, `B3 §6.2`). On-device **compute** still scales with `d`, which is why the tier cap is needed; the win is communication, not compute.

> **Numbers are order-of-magnitude (flagged uncertain).** Exact per-ABI `.so` size and per-tier round wall-clock require a measurement on a real built artifact, which does not exist yet (`A6 §M-H1`/`§M-H2` both flag "uncertain on exact size/magnitude — needs measurement"). M10's done-condition (§13) includes measuring these and setting a CI size budget. The memory model and the 100M-is-not-mobile verdict are firm; the wall-clock seconds are estimates.

---

## 11. Build & run (verify the unit in isolation)

### 11.1 One-time toolchain
```bash
# Node + RN toolchain (pinned TS 5.x, RN 0.8x).
cd mobile_client && npm install

# buf-generated C++ stubs (single source; M2 produces these; mobile consumes).
buf generate           # from the repo proto/ root; emits C++ stubs into the build tree
```

### 11.2 CI-prebuild the native dependencies (once, cached — fixes `A6 §M-H1`)
```bash
# Pinned libtorch ARM64 (PYTORCH_TAG matches torch 2.12.0).
PYTORCH_TAG=v2.12.0 bash scripts/build_libtorch_arm64.sh   # outputs LIBTORCH_DIR artifact

# Pinned gRPC C++ ARM64 (replaces v1 hardcoded v1.62.0).
GRPC_CPP_VERSION=<current-release> bash scripts/build_grpc_arm64.sh
```

### 11.3 Build & run the C++ core tests in isolation (the parity gate)
```bash
cmake -S mobile_client/shared -B build/shared \
      -DLIBTORCH_DIR=/path/to/libtorch-arm64 -DCMAKE_BUILD_TYPE=Release
cmake --build build/shared --target fedlearn_core_tests
ctest --test-dir build/shared --output-on-failure
# MUST pass: rng_parity_test, g_scalar_parity_test, serialize_roundtrip_test, flatparam_filter_test
```

### 11.4 Android build
```bash
cd mobile_client/android
./gradlew assembleRelease       # externalNativeBuild(cmake) compiles the JNI lib (arm64-v8a only)
# Verify the .so size budget (CI gate):
unzip -l app/build/outputs/apk/release/app-release.apk | grep libfedlearn   # check size vs budget
```

### 11.5 iOS build
```bash
cd mobile_client/ios && pod install
xcodebuild -workspace FedLearn.xcworkspace -scheme FedLearn -configuration Release
```

### 11.6 Verify against a dev FL server (end-to-end smoke)
```bash
# Start a dev FL server (M3/M5, LOCAL_PROCESS launcher) and a dev control plane.
# Run the RN app in dev (insecure gRPC allowed) pointed at it; join a DeComFL run.
cd mobile_client && npm run android   # or: npm run ios
# Expected: registers, runs >=1 DeComFL round, uploads K*P scalars (NOT z), telemetry visible.
```

**Drop `--whole-archive` (`A6 §M-H1`):** `bridge/android/jni/CMakeLists.txt` must force-include only the referenced libtorch kernels, not all of `lib/*.a`. Verify which symbols are referenced and gate the per-ABI `.so` size.

---

## 12. Reasoning & alternatives

| Decision | Why this | Rejected alternative & why | Audit driver |
|---|---|---|---|
| **One buf-generated proto; C++ stubs in CI; checksum gate** | Kills the v1 two-drifted-copies + `SubmitModelUpdateReque` typo permanently; mobile can finally share generated symbols. | Keep vendored per-unit proto copies (v1's model) — produced the drift; un-sharable forever. | `A6 §M-C1`, `02 §3.3` |
| **C++ ZO core as a tested port (golden-vector gate)** | Correctness is defined by the Python fixtures, not "maintained by eye"; the parity test converts "bit-identical" from a comment to a CI fact. | Maintain the hand-port and trust the header comment — v1 did, and float32 truncation + unfiltered `requires_grad` made the claim false. | `A6 §M-C2`, `B1-H1`, `C3 §5.1` |
| **CPU-canonical perturbation on every node** | A CPU phone and a CUDA server regenerate the **same** `z` from a shared seed — the whole DeComFL premise; without it the federation silently does not learn. | Generate `z` on the compute device — breaks across CPU/CUDA/MPS (`torch.randn` not bit-identical). | `B1-C2`, `C3-1` (headline) |
| **double-accumulated loss diff + double μ** | Preserves the low-order bits the ZO signal lives in (catastrophic cancellation at μ=0.001). | Extract two float32 losses then subtract (v1) — discards the gradient signal. | `A6 §M-C2 #1/#2` |
| **`requires_grad` filter in flatten** | Flattened length/order matches the server's P-dimension for frozen-layer models (the LLM/LoRA case DeComFL targets). | Flatten all `model.parameters()` (v1) — diverges for any frozen layer; breaks seed→`z` dimension. | `A6 §M-C2 #3` |
| **TLS+mTLS default; integrity-check `.pt`; encrypted storage** | A phone on hostile cellular must not ship weights/gradients in cleartext; `jit::load` of a tampered `.pt` is code execution. | Insecure channel + bundle-shipped `.pt` (v1) — exfiltration + code-exec vector. | `A6 §M-C4`, `04 §10.3` |
| **Streaming RPCs + 32 MB cap** | Avoids buffering a 1 GB unary payload on a phone; preserves the chunking invariant. | Unary `SubmitModelUpdate` with 1 GB limits (v1) — instant OOM on a 100M model. | `A6 §M-M3` |
| **Dual-heartbeat with `abortFlag_` + foreground service** | Heartbeat death becomes visible; the OS cannot silently kill a multi-minute round. | `catch(...) continue` (v1) — dead heartbeat invisible; server times out the client. | `A6 §M-H3` |
| **One round per bridge call, RN loops** | Lets the RN layer check the deadline, sample metrics, and honor `stop()` between rounds; survives backgrounding boundaries. | One giant native call for the whole run — worst case for Doze/background suspension. | `A6 §M-H3` |
| **DeComFL re-enabled + tested (not disabled)** | The native core's reason to exist is delivered; otherwise it is max maintenance surface, zero capability. | Ship FedAvg-only with DeComFL `disabled:true` (v1) — the worst of both. | `A6 §M-C3` |
| **Tier cap; 100M is not mobile** | The 2× ZO working set + 100 CPU forward passes/round are impractical on a phone; the DeComFL win is communication, not compute. | Offer 100M on mobile — OOM + thermal throttle. | `A6 §M-H2` |
| **NativeWind + react-native-reusables + `@fedlearn/tokens`; one product name "FedLearn"** | One brand/token source across web/desktop/mobile; kills the v1 Bootstrap-hex "FedMob" island. | Keep inline hex + emoji tabs + "FedMob" (v1) — three products from three companies. | `C5 §1.1/§7/§8` |
| **Real softmax on Model Testing; delete fake confidence chart** | The chart must show true probabilities, not `exp(-loss)`. | Keep the `exp(-loss)` "Per-Class Score" proxy (v1) — visualizes a fake. | `C5 §3/§9` |
| **Typed TurboModule returns (not JSON strings)** | Spec and impl agree; one missing escape can't break the bridge. | Hand-built `jsonEscape` strings (v1) — spec/impl disagreement (`A6 §L3`). | `A6 §L3` |
| **Stable persisted UUID client id** | DeComFL missed-round rebuild needs a stable identity; a timestamp id defeats rebuild and inflates the roster. | `mobile_${Date.now()}` (v1). | `A6 §M-M4` |
| **Pinned `PYTORCH_TAG` + CI-prebuilt libtorch; no `--whole-archive`** | Reproducible builds; RNG parity is meaningful only against a pinned torch; smaller `.so`. | Unpinned `${PYTORCH_SRC}` + `/tmp` symlink + `--whole-archive` (v1) — unreproducible, bloated. | `A6 §M-H1` |

---

## 13. Build task checklist for the ~30B local model (ORDERED, dependency-respecting)

Each task is ~one file/feature with a concrete done-condition. Build in this order — later tasks depend on earlier ones. This unit is **M10** and assumes M0 (buf scaffold), M2 (proto), M3 (corrected DeComFL + golden fixtures), M4 (control plane), M5 (substrate), M6 (observability) are done (`90-BUILD-SEQUENCE.md §M10`).

1. **Lift + clean the subtree.** Bring `mobile_client/` onto the trunk; **delete** the v1 dead layers (`src/federated/protos/`, `resourceMonitor.js`, `DatasetLoader.js`, `patches/@tensorflow*`, `deactivate`, 0-byte stubs, committed `data/MNIST/` blobs). **Done:** the tree matches §4; `git grep "@tensorflow/tfjs"` returns nothing; no committed MNIST raw blobs.
2. **Single proto + checksum gate.** Make `mobile_client/proto/fedlearn/v2/fedlearn.proto` a CI-stamped mirror of canonical; wire `proto.yml` to fail on divergence. **Done:** `buf generate` emits C++ stubs; CI fails if the mirror differs (E6).
3. **`Perturbation.{h,cpp}` — `canonical_perturbation`.** CPU `torch::Generator`, `manual_seed`, `randn(..., dtype, cpu)`. **Done:** compiles; `rng_parity_test.cpp` (next task) will pin it.
4. **`rng_parity_test.cpp` (gtest).** Load the committed Python golden fixture (`framework/tests/fixtures/decomfl_golden/`); assert `canonical_perturbation(seed,n,fp32)` matches bit/ULP-tolerance. **Done:** test passes; it is a **release blocker** if it fails (E3).
5. **`DtypeMap.{h,cpp}` — SAFE_DTYPES.** string↔`torch::Dtype`; unknown = hard error. **Done:** unit test asserts unknown dtype throws (E7).
6. **`ModelManager.{h,cpp}`.** `loadScriptModel` with sha256 verify-before-`jit::load`; `getFlatParams`/`setFlatParams` with the **`requires_grad` filter**; `serializeStateDict`/`loadStateDict` wrapping `{parameters,num_examples}` in safetensors. **Done:** `flatparam_filter_test.cpp` (frozen-layer count == Python) and `serialize_roundtrip_test.cpp` pass; loading a tampered `.pt` throws (E8).
7. **`ZerothOrderEstimator.{h,cpp}`.** `double mu_`; double-accumulated loss diff; forward + central. **Done:** `g_scalar_parity_test.cpp` matches the Python reference within tolerance for the same model/batch/seed.
8. **`DeComFLClient.{h,cpp}`.** `fit()` (snapshot-restore revert, `(eta/P)*delta`, scalars out); `rebuildModel()` (Alg.2 replay). **Done:** a local test reproduces the framework's participate-vs-rebuild equivalence (mirror of DeComFL spec T1) within tolerance.
9. **`FedLearnClient.{h,cpp}` — gRPC layer.** Dual channel (training + heartbeat); TLS+mTLS default; `registerClient`/`getServerStatus`/`getDeComFLConfig`/`submitGradientScalars`; **streaming** `getGlobalModelStream`/`submitModelUpdateStream`; dtype-safe `protoToTensors`; `reportClientMetrics`; `startHeartbeat` with `abortFlag_`. **Done:** registers against a dev FL server; uploads scalars (verified `bytes_received == K*P*8`); heartbeat death sets `abortFlag_` (E4/E13).
10. **`FederatedLoop.{h,cpp}` + `DataLoader.{h,cpp}`.** `deComFLLoop`/`fedAvgLoop` one-round bodies; deadline checks; validated data load. **Done:** one DeComFL round and one FedAvg round complete end-to-end against the dev server.
11. **TurboModule spec `NativeFedLearnCore.ts` + `FedLearnCoreModule.{h,cpp}`.** Typed returns (§5.1); JSI↔C++ marshaling; **no** hand-built JSON. **Done:** RN codegen produces the typed module; `nativeCore.ts` calls compile in TS strict.
12. **Android JNI bridge.** `bridge/android/jni/CMakeLists.txt` (NO `--whole-archive`, arm64-v8a only); `OnLoad.cpp`; `AndroidManifest.xml` (`INTERNET` + `FOREGROUND_SERVICE` + `network_security_config.xml`). **Done:** `./gradlew assembleRelease` builds; `.so` size within the CI budget (E12 measurement, §10).
13. **iOS ObjC++ bridge.** `NativeFedLearnCore.mm` → common CxxTurboModule; `Info.plist` ATS on. **Done:** `xcodebuild ... Release` builds and links libtorch/grpc xcframeworks.
14. **RN app layer.** `clientId.ts` (stable UUID), `restClient.ts` (`withCredentials`), `runJoin.ts` (REST→`enrollment_token`→`registerClient`, manifest version gate), `deviceClass.ts` (tier cap). **Done:** join flow reaches `RUNNING` and registers the C++ core; 100M is never offered (E2/E12).
15. **Screens on the shared design system.** `TrainingScreen.tsx` (DeComFL **enabled** + live round progress + battery/thermal banner), `ModelLibraryScreen.tsx` (encrypted-storage model list), `ModelTestingScreen.tsx` (**real softmax**, canvas grid). NativeWind + react-native-reusables + `@fedlearn/tokens`; lucide icons; product name "FedLearn". **Done:** no inline hex (`git grep "#0" src/screens` empty); no emoji tab icons; the fake `exp(-loss)` chart is gone (C5 §3/§9).
16. **Foreground service + heartbeat lifecycle.** Android foreground service for the round-loop lifetime; iOS foreground-only notice. **Done:** backgrounding mid-round triggers clean `stop()` + visible heartbeat gap, not a silent stale upload (E4/E5).
17. **Telemetry wiring.** `getDeviceMetrics` (native RSS/thermal/battery) → `ReportClientMetrics` each round → server → `round_results`/Grafana (§7). **Done:** per-round mobile metrics appear on the run-observability surface with `client_type="mobile"`.
18. **`scripts/build_libtorch_arm64.sh` + `build_grpc_arm64.sh` (pinned) + `export_model.py` (`torch.jit.script`, all tiers) + `fetch_demo_data.sh`.** **Done:** CI prebuilds and caches the native deps from a pinned `PYTORCH_TAG`; one export script produces 1M/10M models; demo data is fetched, not committed (E? / `A6 §M-H1/§M-H2/§L5`).
19. **Mobile CI job `mobile.yml`.** Build the C++ core + run the four gtest parity/roundtrip tests on an ARM runner; enforce the `.so` size budget; checksum-gate the proto. **Done:** `mobile.yml` is green; the golden-vector test gates the build (the M10 CHECKPOINT condition, `90-BUILD-SEQUENCE.md:499`).

---

*End of 15-LLD-mobile.md. Every contract traces to a foundation doc (`02-TECH-STACK.md §17/§3/§4`, `03-DATA-MODEL.md §5.2`, `04-API-CONTRACTS.md §4/§5/§10/§13/§14`) or the DeComFL correctness spec; every design decision cites the audit finding it closes (`A6-mobile.md`, `B1-paper-alignment.md`, `C3-reproducibility.md`, `C5-design-ux.md`). The FL-framework LLD `11-LLD-fl-framework.md` is on disk; this unit's C++ determinism slice reproduces the Python contract pinned there and in the DeComFL correctness spec.*
