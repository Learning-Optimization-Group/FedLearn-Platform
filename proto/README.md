# FedLearn Protocol Buffers — the canonical gRPC contract

This directory is the **single source of truth** for the FedLearn client ⇄ FL-server wire protocol.
Codegen for all four targets (Python, Java, TypeScript, C++) is configured here and driven by
[`buf`](https://buf.build). Editing a copy instead of the original is what produced the v1 drift the
audit found — a malformed `SubmitModelUpdate` RPC that diverged across the mobile copies (see
`buf.yaml`'s header) — so every in-tree copy is now byte-checked in CI.

> **gRPC** = Google Remote Procedure Call · **RPC** = Remote Procedure Call ·
> **CI** = Continuous Integration · **TS** = TypeScript · **FoT** = Federation over Text.

## Layout

```
proto/
├── buf.yaml                       # module + lint (STANDARD, 4 documented excepts) + breaking (FILE)
├── buf.gen.yaml                   # one config -> Python / Java / TS / C++ stubs
├── fedlearn/v2/fedlearn.proto     # THE gradient contract (package fedlearn.v2)
├── fedlearn/fot/v1/fot.proto      # Federation over Text (package fedlearn.fot.v1)
└── gen/{python,java,ts,cpp}/      # buf output — GITIGNORED (.gitignore:179), regenerate locally
```

## Two contracts, two packages

**There is no `fedlearn.v1` anywhere in the tree.** The v1 package is retired, not co-hosted. FoT is
deliberately a **separate** service and package so the text-federation server cannot perturb the
gradient wire, and it is additive to — never a replacement for — the gradient path.

### `fedlearn/v2/fedlearn.proto` — `package fedlearn.v2`

`option java_package = "com.fedlearn.v2"`, `java_multiple_files = true`.
Service **`FederatedLearningService`**, **10 RPCs**:

| Group | RPC |
|---|---|
| lifecycle / control | `RegisterClient`, `GetServerStatus`, `Heartbeat` |
| model transfer (FedAvg path) | `GetGlobalModel`, `GetGlobalModelStream` (server-stream `ModelChunk`), `SubmitModelUpdate`, `SubmitModelUpdateStream` (client-stream `ModelUpdateChunk`) |
| DeComFL path | `GetDeComFLConfig`, `SubmitGradientScalars` — seeds and scalars only, **no weights on the wire** |
| telemetry | `ReportClientMetrics` |

`Heartbeat` is bidirectional in effect, not just liveness: `HeartbeatResponse.should_stop` is how the
server aborts a client mid-round. `ModelChunk` / `ModelUpdateChunk` **declare** explicit framing —
`codec`, `compressed`, `total_bytes`, `sha256` — so nothing need be inferred from the environment.
⚠️ Only the download direction (`ModelChunk`) actually populates them today: the Python client's
upload generator leaves all four unset on `ModelUpdateChunk` and the server never reads them
(`framework/src/fedlearn/client/grpc_client.py:248-261`,
`framework/src/fedlearn/server/grpc_servicer.py:295-360`). The schema is ahead of the
implementation here — an upload-side integrity gap, not a symmetric contract.

### `fedlearn/fot/v1/fot.proto` — `package fedlearn.fot.v1`

`option java_package = "com.fedlearn.fot.v1"`.
Service **`FoTService`**, **2 RPCs**:

| RPC | Purpose |
|---|---|
| `SubmitReasoningTrace` | client uploads one abstracted reasoning trace (never the raw problem) |
| `GetInsightLibrary` | client fetches the current insight library; `known_version` enables an `unchanged` short-circuit |

Payloads are plain JSON strings (`ReasoningTrace.to_json` / `InsightLibrary.to_json`) — no vector
store, no tensors.

## What the schema does *not* carry

Framing rules proto cannot express are enforced in code, not by the contract. Do not assume a
generated stub gives you any of them:

- **Transport security** is opt-in: plaintext is the default; `FEDLEARN_GRPC_USE_TLS=1` turns TLS on,
  `FEDLEARN_GRPC_REQUIRE_CLIENT_AUTH=1` demands a client certificate, and `FEDLEARN_REQUIRE_TLS=1`
  makes the server fail closed rather than serve plaintext.
- **Client authentication** is separate from TLS: an `x-connection-token` metadata header, verified
  server-side and enforced only when `FEDLEARN_REQUIRE_CLIENT_AUTH=1` — note this is a *different*
  switch from the mTLS `FEDLEARN_GRPC_REQUIRE_CLIENT_AUTH` above, despite the near-identical name.
- **The `codec` value** must be `safetensors` (`lz4+safetensors` when compressed). The two decode
  paths are deliberately asymmetric: a legacy `torch.save`/pickle blob is **rejected** on the
  server-side upload path but still **accepted** on the client-side global-model download, so a new
  client keeps working against an older server during a staged rollout.
- **Ingress caps** on the streaming upload: `FEDLEARN_MAX_UPLOAD_BYTES` (default 2 GiB),
  `FEDLEARN_MAX_UPLOAD_CHUNKS` (100,000), `FEDLEARN_MAX_UPLOAD_SECONDS` (600) — a client that lies
  about `total_bytes` or never sends `is_final_chunk` is cut off.
- **The round deadline and quorum** (`FEDLEARN_ROUND_TIMEOUT_S`, default 120 s) and the gRPC
  status-code mapping.

⚠️ `ModelChunk`'s comment says chunking is "used for models > 300 MB". The implementation gates the
*streaming path* at `STREAMING_THRESHOLD_MB = 100` (or any transformer, unconditionally) and then
chunks at a hardcoded **50 MB** in both directions (`grpc_client.py:249`, `grpc_servicer.py:184`).
Believe the code; the comment is stale.

## Generate the stubs

```bash
cd proto
buf lint                                   # STANDARD, minus the 4 excepts documented in buf.yaml
buf breaking --against '.git#branch=main'  # fail on a breaking change to the contract
buf generate                               # writes gen/python, gen/java, gen/ts, gen/cpp
```

Every plugin in `buf.gen.yaml` is pinned to a concrete version. **The C++ pin is load-bearing:**
`protocolbuffers/cpp` is held at `v27.2` to match the protobuf runtime that `grpc/cpp:v1.67.1`
bundles — a newer plugin emits a version guard that fails the mobile native compile against the
cross-compiled runtime. Do not bump it in isolation. (The file's own `VERIFY-BEFORE-USE` header still
calls the version suffixes "placeholders"; that header predates the pins below it.)

Managed mode is intentionally off for Java: the `.proto` declares `java_package` itself. The gRPC C++
*runtime* for mobile is not produced by buf — it is cross-compiled separately by
`mobile_client/scripts/build_grpc_arm64.sh`; buf emits only the stubs that compile against it.

## Generated code: what is committed and what is not

**`proto/gen/` is gitignored** (`.gitignore:179`) — committing buf output invites exactly the drift
this gate exists to prevent, so it is produced locally or in CI and never checked in. Consumers:

| Consumer | Where its stubs come from |
|---|---|
| Python framework | **committed** stubs at `framework/src/fedlearn/communication/generated/` (`fedlearn_pb2*`, `fot_pb2*`), generated with `grpc_tools.protoc` from the framework's own mirror |
| Mobile C++ core | `gen/cpp`, passed to CMake as `-DGENERATED_PROTO_DIR=…` — nothing is committed under `mobile_client/` |
| Java / TS | configured and generated on demand, but **no consumer links them today**: the Spring backend declares no gRPC/protobuf dependency (it shells out to the Python FL server instead), and no JS/TS unit depends on `@bufbuild/protobuf` (protobuf-es;
`buf.gen.yaml` pins its `bufbuild/es` plugin). The targets are kept so the contract stays generatable for those surfaces |

## The three in-tree mirrors

Three units keep a **byte-identical copy** of a canonical file in-tree, so a unit ships with the
contract it speaks instead of depending on a `buf` fetch at build time:

| Mirror | Path | What reads it |
|---|---|---|
| **framework** | `framework/src/fedlearn/communication/protos/fedlearn.proto` | the source the framework's committed `fedlearn_pb2*` stubs are generated from |
| **framework (FoT)** | `framework/src/fedlearn/communication/protos/fot.proto` | same, for `fot_pb2*` and the FoT servicer |
| **mobile** | `mobile_client/proto/fedlearn/v2/fedlearn.proto` | the mobile unit's in-tree copy of the contract. Note: **no build file under `mobile_client/` currently reads it** — the CMake build links the C++ stubs from `proto/gen/cpp` (see above), so this copy's job today is to keep the mobile unit self-describing and byte-checked |

All three are **mirrors, not independent copies**, and **must never be hand-edited**. Edit only under
`proto/`, then `cp` out — from the repo root:

```bash
cp proto/fedlearn/v2/fedlearn.proto framework/src/fedlearn/communication/protos/fedlearn.proto
cp proto/fedlearn/v2/fedlearn.proto mobile_client/proto/fedlearn/v2/fedlearn.proto
cp proto/fedlearn/fot/v1/fot.proto  framework/src/fedlearn/communication/protos/fot.proto
```

You do not have to remember those paths: `scripts/check_proto_mirror.sh` byte-compares all three and,
on any difference, prints the diff **and the exact `cp` command that fixes it**. On success it also
prints the canonical sha256 of each contract, so "same bytes" is checkable at a glance:

```
$ ./scripts/check_proto_mirror.sh
OK: mobile proto mirror matches canonical.
OK: framework proto mirror matches canonical.
OK: framework-fot proto mirror matches canonical.
  sha256 (fedlearn.v2): 07a173f9d3dd35770f60766acf1a383664b886b27d81b6cdb65fbd4016162ec5
  sha256 (fot.v1):      40a2eec3c6e8ebdde7f7037319ff19e9cfcaf597b29e1657f4a38cf66ef149be
```

## What CI gates

**`.github/workflows/proto.yml`** — triggers on `proto/**` and `framework/**/protos/**`:

| Step | What it enforces |
|---|---|
| `buf lint` | STANDARD ruleset, minus the four excepts `buf.yaml` documents (shared request/response types across the unary + streaming variants, and short enum value names) |
| `buf breaking --against '.git#branch=main'` | a wire-breaking change fails the PR. Needs `fetch-depth: 0`, which the workflow sets |
| `buf generate` freshness | regeneration must leave the tracked tree unchanged. Since `gen/` is gitignored, what this actually proves today is that **generation succeeds** and writes nothing outside `gen/`; there are no committed buf stubs left for it to diff |
| `scripts/check_proto_mirror.sh` | all three mirrors byte-identical to canonical |

**`.github/workflows/mobile.yml`** runs `check_proto_mirror.sh` as its own gating `proto-mirror` job
and triggers on `mobile_client/**` and `proto/**`. That matters: `proto.yml`'s path filter does *not*
include `mobile_client/proto/**`, so the mobile mirror is covered by `mobile.yml`, not by `proto.yml`.
Between the two workflows, every mirror path is gated.

## Changing the contract

1. Edit the canonical file under `proto/` only (`fedlearn/v2/fedlearn.proto`, or
   `fedlearn/fot/v1/fot.proto` for FoT). Never a mirror.
2. Run `buf lint` and `buf breaking` (a breaking change requires a deliberate package bump).
3. Re-run `buf generate`, and regenerate the framework's committed Python stubs if the gradient or
   FoT messages changed.
4. `cp` **all three** mirrors, then run `scripts/check_proto_mirror.sh` before pushing.
