# FedLearn Protocol Buffers — the canonical gRPC contract

This directory is the **single source of truth** for the FedLearn client ⇄ FL-server wire
protocol. Every language (Python, Java, TypeScript, C++) generates its stubs from here via
[`buf`](https://buf.build); nothing is hand-written or copied. This governance is what kills
the v1 drift the audit found — a malformed `SubmitModelUpdate` RPC and a message-type typo
that diverged across the mobile copies.

> **gRPC** = Google Remote Procedure Call · **RPC** = Remote Procedure Call ·
> **CI** = Continuous Integration · **TS** = TypeScript · **RNG** = Random Number Generator.

## Layout

```
proto/
├── buf.yaml                       # module + lint (STANDARD) + breaking (FILE) config
├── buf.gen.yaml                   # one config -> Python / Java / TS / C++ stubs
├── fedlearn/v2/fedlearn.proto     # THE gradient contract (package fedlearn.v2)
└── fedlearn/fot/v1/fot.proto      # Federation over Text (package fedlearn.fot.v1)
```

The gradient package is `fedlearn.v2` (the v1 package `fedlearn.v1` is retired, not co-hosted —
nothing in the tree speaks it any more). FoT is deliberately a **separate** service/package
(`fedlearn.fot.v1`) so the text-federation server cannot perturb the gradient wire. Framing rules
that proto cannot express (the transport TLS/mTLS policy, the `codec` whitelist, chunk `sha256`
symmetry, `max_payload_bytes`, the round deadline/quorum, and the gRPC status-code mapping) are
not carried by the schema — they are enforced in code. Transport security is **opt-in**:
plaintext is the default (`FEDLEARN_GRPC_USE_TLS=1` turns TLS on; `FEDLEARN_REQUIRE_TLS=1` makes
the server fail-closed rather than serve plaintext).

## Generate the stubs

```bash
cd proto
buf lint                                   # STANDARD lint (see buf.yaml for the documented excepts)
buf breaking --against '.git#branch=main'  # fail the build on a breaking change to the contract
buf generate                               # writes gen/python, gen/java, gen/ts, gen/cpp
```

Pin the exact remote-plugin and runtime versions before first use (the `:vX.Y.Z` suffixes in
`buf.gen.yaml` are placeholders to resolve against https://buf.build/plugins). The C++/gRPC
runtime for mobile is **not** produced by buf — it is cross-compiled separately by
`mobile_client/scripts/build_grpc_arm64.sh` (pinned); buf emits only the C++ stubs that compile
against it.

## The in-tree mirrors (framework + mobile)

Two units keep a **byte-identical copy** of these files in-tree because their build systems need
the proto locally rather than pulling from `buf`:

| Copy | Path | Why it exists |
|---|---|---|
| **framework** | `framework/src/fedlearn/communication/protos/fedlearn.proto` | the running Python framework generates its `fedlearn_pb2` stubs from here (package `fedlearn.v2`, same bytes as canonical) |
| **mobile** | `mobile_client/proto/fedlearn/v2/fedlearn.proto` | the native mobile core's CMake build |
| **framework (FoT)** | `framework/src/fedlearn/communication/protos/fot.proto` | the Python FoT servicer's stubs, mirrored from `fedlearn/fot/v1/fot.proto` |

All three are **mirrors, not independent copies**: they are regenerated/synced from the canonical
files and **must never be hand-edited**. Edit only under `proto/`, then `cp` out:

```bash
cp proto/fedlearn/v2/fedlearn.proto framework/src/fedlearn/communication/protos/fedlearn.proto
cp proto/fedlearn/v2/fedlearn.proto mobile_client/proto/fedlearn/v2/fedlearn.proto
cp proto/fedlearn/fot/v1/fot.proto  framework/src/fedlearn/communication/protos/fot.proto
```

Two gates enforce this so a mirror can never silently drift:

- **`scripts/check_proto_mirror.sh`** — byte-compares *all three* mirrors against canonical and exits
  non-zero (with the exact `cp` fix) on any difference. Run it locally before pushing.
- **`.github/workflows/proto.yml`** — the CI proto gate. Runs `buf lint`, `buf breaking`
  (against `main`), a `buf generate` freshness check, and `check_proto_mirror.sh`. It triggers on
  any change under `proto/**` or `framework/**/protos/**`, so a stray edit to a mirror fails the
  PR.

## Changing the contract

1. Edit the canonical file under `proto/` only (`fedlearn/v2/fedlearn.proto`, or
   `fedlearn/fot/v1/fot.proto` for FoT).
2. Run `buf lint` and `buf breaking` (a breaking change requires a deliberate package bump).
3. Re-run `buf generate` in every consumer; commit the regenerated stubs (or generate in CI).
4. `cp` **all three** mirrors (framework, mobile, framework-FoT) and run `scripts/check_proto_mirror.sh`.
