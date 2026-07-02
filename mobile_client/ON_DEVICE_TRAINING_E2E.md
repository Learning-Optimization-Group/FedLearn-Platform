# On-device training — end-to-end spec

**Goal:** make the mobile client run a **real on-device DeComFL round against a real server**, with training data staying on the device (only perturbation seeds + gradient scalars leave).

**Status today:** the mobile client is fully wired client-side (`registerClient → setModelManifest →
loadModel → setTrainingDataFromFiles → runDeComFLRound` loop; `provisionTrainingBundle()` is a
throwing placeholder). It cannot run end-to-end because of three cross-stack gaps below. None are
mobile-side.

## The three gaps

1. **Proto / server.** The mobile core speaks `fedlearn.v2`; the running framework server speaks
   `fedlearn.v1`. They cannot communicate: package mismatch, missing `run_id` /
   `protocol_version` / `enrollment_token` on `RegisterClient`, `int32`-vs-`int64` seed truncation,
   absent `perturbation_seeds` on `SubmitGradientScalarsRequest`, no `codec`/`sha256` chunk framing,
   no `ReportClientMetrics` RPC, and colliding `ServerState` enums (v1 `AGGREGATING=3` ↔ v2
   `TRAINING=3`).
2. **Export / bundle.** A loadable `.pte` bundle (loss graph + infer graph + a manifest carrying
   `paramLayout` + sha256s + fixture data) must be produced and staged per run.
3. **Backend / data.** `GET /api/runs/{runId}/model-bundle` + binary file serving + an
   `enrollment_token` do not exist yet.

## Key leverage (why this is plumbing, not a rebuild)

- The v1 servicer **already implements all five DeComFL RPCs** and the seed/gradient proto helpers
  (`framework/src/fedlearn/server/grpc_servicer.py`) — this is a proto upgrade + field plumbing.
- `framework/tests/fixtures/decomfl_golden/` **already ships a complete TinyNet bundle**
  (`Linear(4,5)→ReLU→Linear(5,3)`, fc2 frozen, 25 trainable params): both weight-free `.pte` graphs,
  `zo_inputs.f32`, `zo_targets.i64`, safetensors state, and a manifest with `golden_g`/`golden_loss`
  and every sha256 — the perfect MVP payload, no new export needed.
- **The DeComFL path never touches `GetGlobalModelStream`** (that is the FedAvg weight path and the
  source of the `validateCodec` crash), so a pure-DeComFL MVP ships without any weight transfer and
  without that crash on the critical path.

## Recommended MVP

Single-client, single-round, **pure DeComFL** (avoids the FedAvg weight/codec path entirely):

1. **P0** — regenerate framework stubs from `fedlearn.v2` so the server speaks the same package/fields.
2. **P1** — plumb only the five DeComFL RPCs in `grpc_servicer.py`: `RegisterClient` (accept
   `run_id`/`protocol_version`/`enrollment_token` permissively, return `assigned_round`),
   `GetServerStatus` (correct v2 enum + `round_deadline_unix_ms` + `active_clients`), `Heartbeat`
   (read `run_id`), `GetDeComFLConfig` (`torch_version` + `grad_estimate_method` +
   `golden_vector_sha256`), `SubmitGradientScalars` (read `int64` `perturbation_seeds` → coordinator),
   plus a minimal `ReportClientMetrics`.
3. **P3** — stage the committed golden TinyNet fixture into `/var/models/{runId}/` (no export needed).
4. **P2** — serve it via `GET /api/runs/{runId}/model-bundle` + a whitelisted file endpoint behind the
   existing JWT gate; mint an `enrollment_token` at enroll.
5. **P4** — implement `provisionTrainingBundle` to download + stage into app-private storage and return
   local paths.

**Result:** the device enrolls, stages the tiny bundle, `loadModel` verifies sha256 and reports 25
trainable params, `setTrainingDataFromFiles` reads the local f32/i64, and one `runDeComFLRound` uploads
only seeds + scalars — a full on-device DeComFL round against a real v2 server with data staying local.
(The fixture stands in as the device's local partition; genuine per-device data is a deliberate
post-MVP step.)

## Phases (critical-path order)

| # | Phase | Component | Effort | Goal |
|---|---|---|---|---|
| P0 | Proto v2 convergence + stub regen | proto | M | One authoritative `fedlearn.v2`; regenerate py/java/ts/cpp stubs from one `buf` run |
| P1 | v2 servicer: DeComFL 5-RPC plumbing + telemetry | framework-server | L | v2 client completes a DeComFL round with correct seed/scalar semantics |
| P3 | Bundle staging / export pipeline | export-pipeline | M | Per-run dir with `manifest.json` + loss/infer `.pte` + `inputs.f32` + `targets.i64`, sha256-consistent |
| P2 | Backend model-bundle endpoint + file serving + enrollment token | backend-api | M | Authenticated REST fetch of bundle + binaries; mint `enrollment_token` |
| P4 | Mobile file-staging + `provisionTrainingBundle` | mobile | L | Real fetch→verify→stage→return-paths; thread `enrollment_token` into `RegisterClient` |
| P5 | E2E one-round integration + data-locality guardrails | cross | M | One real on-device DeComFL round; assert no raw feature/label bytes on the wire |

**Critical path:** P0 → P1 → P3 → P2 → P4 → P5.

### Verification per phase
- **P0:** regenerated pb2 descriptor reports `fedlearn.v2` and exposes `perturbation_seeds`,
  `int64` seeds, `codec/sha256`, `ReportClientMetrics`; `scripts/check_proto_mirror.sh` exits 0.
- **P1:** per-RPC pytest + an integration harness asserting `int64` seeds survive (no int32
  truncation), `perturbation_seeds` reach the coordinator, and reconstructed `golden_g` matches
  `zo_manifest.json` within tolerance.
- **P3:** recomputed sha256 of each staged file equals the manifest; ExecuTorch loads both graphs;
  `paramLayout` order matches `named_parameters()` requires_grad order (`test_pte_export.py`).
- **P2:** authed request returns the bundle JSON; each binary downloads with body sha256 == manifest;
  404/401/403 for unknown/unauth/non-member.
- **P4:** on device, the loop passes provisioning; staged files exist in app-private storage;
  airplane-mode read of inputs/targets confirms device-local; `setTrainingDataFromFiles` succeeds.
- **P5:** server round counter advances to `TRAINING_COMPLETE`; gRPC capture contains no raw
  input/target payloads; `RoundResult` loss ≈ fixture `golden_loss` (~1.097).

## Open questions (decide before P0)

- **Package strategy:** fully migrate framework + backend to `fedlearn.v2` (recommended, clean, but
  breaks any deployed v1 server), or keep the v1 wire package and only add fields for a transitional
  dual-stack? (Plan assumes full v2.)
- **`enrollment_token`:** reuse `ConnectionTokenService` (HMAC over `app.jwt.secret`, 120s TTL) or a
  separate anti-Sybil token; log-only vs hard-reject for the MVP.
- **Constraint 7:** does the coordinator actually *reconstruct* the ZO-SGD update from
  `perturbation_seeds`, or only store scalars? If reconstruction is missing, P1 grows.
- **Real per-device data (post-MVP):** pre-generated fixtures vs on-demand `seed+partitionId` slicing
  (mirror `run_local_test.py:79-99`) vs synthetic. The MVP bundles server-origin fixture data — a demo
  shortcut, **not representative federation**.
- **torch/ExecuTorch version parity:** the golden fixture is torch 2.12.0 — the native arm64
  libtorch/ExecuTorch build must match or `golden_vector_sha256` / RNG parity breaks on device.
- **RN file staging:** `react-native-fs` (verify RN 0.80 + New Architecture + arm64 support) vs the
  native `GetGlobalModelStream` path.
- **`protocol_version` policy:** exact server constant; mismatch hard-reject vs warn.
- **Quorum/deadline for a single-client demo:** `min_clients=1`, deadline long enough for one device.

## Risks to watch
- `buf` remote-plugin version drift silently changes wire codegen — pin exact versions; verify all four
  stub sets (py/java/ts/cpp) come from the same `buf` run.
- Switching the framework to `fedlearn.v2` is breaking for any deployed v1 server (no dual-stack).
- Manifest field-name mismatch (fixture snake_case vs mobile `ModelManifest` camelCase) silently fails
  `loadModel` — add an adapter/validator.
- Path traversal via the file endpoint `{filename}` — strict whitelist.
- Whole-file (non-streaming) download can OOM on real bundles — fine for the tiny fixture; revisit.

## Status (2026-07-02)

All six phases landed on `vinf`. Each was verified by an automated test that runs without a device or a
live backend, so the whole critical path is CI-checkable:

| # | Commit | Verified by |
|---|---|---|
| P0 | `f8ee131` | Regenerated `fedlearn_pb2` reports package `fedlearn.v2`; `perturbation_seeds`, `int64` `LocalStepSeeds.seeds`, `run_id`/`protocol_version`/`enrollment_token`, `ReportClientMetrics` all present. |
| P1 | `e03ed46` | Per-RPC in-process tests: register accepts/rejects on `protocol_version`, status exposes `active_clients` + `round_deadline_unix_ms`, config carries `torch_version`/`grad_estimate_method`, `perturbation_seeds` reach the coordinator, metrics recorded. |
| P3 | `a530ea3` | `stage_model_bundle.py` stages 5 files + `manifest.json`; recomputed sha256 == fixture manifest; data byte-sizes match declared shapes. |
| P2 | `d991578` | `RunServiceModelBundleTest` (5 Mockito tests): DTO mapping, per-file sha, whitelist + path-traversal 404, missing-bundle 404, feature-flag-disabled 404. |
| P4 | `5b41aab` | `modelProvisioning.test.ts` (fetch→base64→`stageBundleFile`→mapping, 404 path) + native `stageBundleFile` compiles into the arm64 `.so`. |
| P5 | *(this)* | `framework/tests/test_v2_grpc_e2e.py`: a full DeComFL round over a **real gRPC socket** (register→heartbeat→status→config→submit→aggregate→advance), protocol-version rejection, int64-seed wire integrity. |

**What P5 proves and what it does not.** The gRPC e2e test stands up the real `grpc.Server` + servicer +
`FLCoordinator` + `DeComFL` strategy and drives the exact RPC sequence a mobile client performs, over a
real client channel — so P0 (wire schema) and P1 (servicer semantics) are validated over the socket, not
just via direct method calls, and the coordinator aggregates + advances the round. Because DeComFL is
**server-authoritative on seeds**, the scalars-only submission is sufficient: the server reconstructs the
update from its own `seed_history`, so a device computing scalars for those same seeds produces an
equivalent result. This is the automatable half.

The **remaining, inherently hands-on half** is the device-in-the-loop: a physical phone completing one
round against a live backend + v2 `fl_server`, with a gRPC capture confirming no raw feature/label bytes
leave the device. That needs a reachable backend the phone can hit and a person driving the app; it can't
run in CI. Runbook:

1. **Stage a bundle:** `python scripts/stage_model_bundle.py --run-id <uuid> --out /var/models` (uses the
   golden TinyNet fixture). Confirm `/var/models/<uuid>/manifest.json` + the 5 binaries exist.
2. **Backend:** `SPRING_PROFILES_ACTIVE=dev APP_MODEL_BUNDLE_DIR=/var/models ./gradlew bootRun`. Confirm
   `GET /api/runs/<uuid>/model-bundle` (authed) returns the DTO and each `/files/*` binary downloads.
3. **v2 fl_server:** launch the framework server on the DeComFL strategy with `min_clients=1` and a long
   round deadline, bound to a LAN/Tailscale address the phone can reach (gRPC is plaintext — audit #37).
4. **Phone:** install the arm64 APK, log in, pick the run; the app calls `provisionTrainingBundle`, stages
   the files, then registers with `protocol_version=2` + `enrollment_token` and runs the DeComFL loop.
5. **Assert data-locality:** capture the phone↔server gRPC (Tailscale tap / mitm / server-side frame log)
   and confirm only seeds + scalars + metrics appear — **no `inputs.f32`/`targets.i64` bytes**. Confirm the
   server round counter advances and `RoundResult` loss ≈ fixture `golden_loss` (~1.097).

Steps 1–2 are already covered by automated tests; 3–5 are the manual acceptance run.
