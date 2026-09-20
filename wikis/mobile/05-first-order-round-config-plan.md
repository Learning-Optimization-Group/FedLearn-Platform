# Android First-Order Round Settings Plan

**Goal:** Consume the first-order settings already sent with each global-model download, so FedOpt uses its server-selected learning rate and local epochs; reject unsupported proximal training instead of treating it as FedAvg.

**Architecture:** The FL server puts `learning_rate`, `local_epochs`, and `proximal_mu` into the first `ModelChunk.config` when a strategy supplies them. The Android gRPC client currently drops the map. Carry it through the existing transport seam to `FederatedLoop::firstOrderRound`, validate numeric values before training, and use them in place of JavaScript defaults. FedAvg/Robust retain their current defaults only when the server sends no values; execution contract v1 will replace that fallback.

**Spec:** [Execution Contract v1](03-android-execution-contract-v1-design.md). This is an incremental wire-use step, not contract publication or generic optimizer parity.

## Task 1: Preserve round settings across the download boundary

**Files:** `mobile_client/shared/include/fedlearn/IFedLearnClient.h`, `mobile_client/shared/include/fedlearn/FedLearnClient.h`, `mobile_client/shared/src/FedLearnClient.cpp`, two C++ test mocks.

- [x] Add failing native round tests for non-default `learning_rate` and `local_epochs`, missing settings, and malformed settings. The endpoint test compares against the committed five-step SGD golden and checks the uploaded weight blob.
- [x] Extend `getGlobalModelStream` with an optional output parameter for the first chunk's string config; keep the verified safetensors blob and round number unchanged. Update test mocks to supply controlled maps.
- [x] Reject an empty stream; take config only from the first chunk, as the server sends it. Reject incomplete settings before training.
- [x] Build and run the host `FEDLEARN_BUILD_TRAINING=ON` C++ parity suite (62 passing).

## Task 2: Use validated values and keep strategy routing honest

**Files:** `mobile_client/shared/src/FederatedLoop.cpp`, `mobile_client/shared/include/fedlearn/FederatedLoop.h`, `mobile_client/bridge/common/FedLearnCoreModule.cpp`, `mobile_client/src/lib/training.ts`.

- [x] Parse full numeric strings, reject NaN/infinity, nonpositive rate/steps, and nonzero `proximal_mu` before a model update. Require server values for FedOpt; allow the legacy TinyNet FedAvg/Robust fallback until execution contract v1 is published.
- [x] Pass the actual strategy from TypeScript through `RoundConfig`; FedOpt requires the server map, while DeComFL always takes the scalar path and FedProx remains refused before provisioning.
- [x] Run targeted C++ and TypeScript tests, then the relevant complete suites (62 C++ and 242 mobile tests). TypeScript checking and targeted lint pass; the Android debug APK builds.
- [x] Review the diff and commit locally; do not push.

## Task 3: Keep the desktop TinyNet step aligned

**Files:** `fl-runtime/client.py`, `fl-runtime/tests/test_client_tinynet_golden_fedavg_wire.py`.

- [x] Add a failing endpoint test proving a FedOpt-style `learning_rate=0.02` reaches `ZOSLClient.fit`'s TinyNet SGD step.
- [x] Validate and apply the server learning rate for TinyNet; leave other recipes' dataset-specific rates unchanged until their execution contracts are specified.
- [x] Run the relevant runtime suite (297 passed, 29 deselected; one Hugging Face download test excluded for unavailable DNS).

## Remaining parity gates

The next work must replace FedAvg/Robust fallback settings with validated execution-contract values, implement FedProx's proximal update, add non-TinyNet artifacts and local file-picked data, and qualify CPU/GPU backends. A passing TinyNet/FedOpt test does not establish full Android FL support.
