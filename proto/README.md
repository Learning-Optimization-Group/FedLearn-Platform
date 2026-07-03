# FedLearn Protocol Buffers — the canonical gRPC contract

This directory is the **single source of truth** for the FedLearn client ⇄ FL-server wire
protocol. Every language (Python, Java, TypeScript, C++) generates its stubs from here via
[`buf`](https://buf.build); nothing is hand-written or copied. This governance is what kills
the v1 drift the audit found — a malformed `SubmitModelUpdate` RPC and a message-type typo
that diverged across the mobile copies (`docs/audit/2026-05-29/A3-framework.md` §5).

> **gRPC** = Google Remote Procedure Call · **RPC** = Remote Procedure Call ·
> **CI** = Continuous Integration · **TS** = TypeScript · **RNG** = Random Number Generator.

## Layout

```
proto/
├── buf.yaml                       # module + lint (STANDARD) + breaking (FILE) config
├── buf.gen.yaml                   # one config -> Python / Java / TS / C++ stubs
└── fedlearn/v2/fedlearn.proto     # THE contract (package fedlearn.v2)
```

The package is `fedlearn.v2` (the v1 package `fedlearn.v1` is retired, not co-hosted). The
authoritative definition lives in `docs/v2/build/04-API-CONTRACTS.md §10.2`; this file is its
checked-in form. Framing rules that proto cannot express (TLS+mTLS by default, the `codec`
whitelist, chunk `sha256` symmetry, `max_payload_bytes`, the round deadline/quorum, and the
gRPC status-code mapping) are specified in `04-API-CONTRACTS.md §10.3` and enforced in code.

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

Two units keep a **byte-identical copy** of this file in-tree because their build systems need
the proto locally rather than pulling from `buf`:

| Copy | Path | Why it exists |
|---|---|---|
| **framework** | `framework/src/fedlearn/communication/protos/fedlearn.proto` | the running Python framework generates its `fedlearn_pb2` stubs from here |
| **mobile** | `mobile_client/proto/fedlearn/v2/fedlearn.proto` | the native mobile core's CMake build |

Both are **mirrors, not independent copies**: they are regenerated/synced from this canonical
file and **must never be hand-edited**. Edit only `fedlearn/v2/fedlearn.proto`, then `cp` it out:

```bash
cp proto/fedlearn/v2/fedlearn.proto framework/src/fedlearn/communication/protos/fedlearn.proto
cp proto/fedlearn/v2/fedlearn.proto mobile_client/proto/fedlearn/v2/fedlearn.proto
```

Two gates enforce this so a mirror can never silently drift:

- **`scripts/check_proto_mirror.sh`** — byte-compares *both* mirrors against canonical and exits
  non-zero (with the exact `cp` fix) on any difference. Run it locally before pushing.
- **`.github/workflows/proto.yml`** — the CI proto gate. Runs `buf lint`, `buf breaking`
  (against `main`), a `buf generate` freshness check, and `check_proto_mirror.sh`. It triggers on
  any change under `proto/**` or `framework/**/protos/**`, so a stray edit to a mirror fails the
  PR.

## Changing the contract

1. Edit `fedlearn/v2/fedlearn.proto` only.
2. Run `buf lint` and `buf breaking` (a breaking change requires a deliberate package bump).
3. Re-run `buf generate` in every consumer; commit the regenerated stubs (or generate in CI).
4. `cp` **both** mirrors (framework + mobile) and run `scripts/check_proto_mirror.sh`.
