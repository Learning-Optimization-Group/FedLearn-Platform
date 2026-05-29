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

## The mobile mirror

The native mobile core needs the proto in-tree for its CMake build, so a byte-identical mirror
lives at `mobile_client/proto/fedlearn/v2/fedlearn.proto`. It is **not** an independent copy —
`scripts/check_proto_mirror.sh` (wired into CI as `proto.yml`) fails the build if it diverges
from this canonical file. To update the mirror after changing the contract:

```bash
cp proto/fedlearn/v2/fedlearn.proto mobile_client/proto/fedlearn/v2/fedlearn.proto
```

## Changing the contract

1. Edit `fedlearn/v2/fedlearn.proto` only.
2. Run `buf lint` and `buf breaking` (a breaking change requires a deliberate package bump).
3. Re-run `buf generate` in every consumer; commit the regenerated stubs (or generate in CI).
4. `cp` the mirror and run `scripts/check_proto_mirror.sh`.
