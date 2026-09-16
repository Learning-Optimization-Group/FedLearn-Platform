# FedLearn — Distributed Federated Learning Framework

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B%20(CI%3A%203.12)-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.12.0](https://img.shields.io/badge/PyTorch-2.12.0%20(pinned)-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](../LICENSE)

## Overview

FedLearn is the installable **library** at the heart of the platform (`pip install -e .` →
`import fedlearn`). It is **built from scratch**: no Flower / `flwr` dependency and none of its
server/client/strategy semantics. Clients and servers talk direct gRPC over a custom protobuf
contract (`fedlearn.v2`), and model weights cross the wire as deterministic **safetensors** —
never `torch.save`/pickle.

> **`flwr` is gone repo-wide.** It used to be pulled in elsewhere for dataset partitioning; that
> dependency was removed (commit `65048b6`), which also cleared the `cryptography<45` and
> `protobuf<5` caps it imposed. If a doc still calls `flwr` a current dependency, it is stale.

> **Library vs. runtime.** This package is the library. The *executable* FL layer the Spring Boot
> backend actually shells out to lives in [`../fl-runtime/`](../fl-runtime/) (`client.py`,
> `fl_server.py`, `recipes.py`, …), and it imports this package. If you are looking for the
> model-recipe catalog or the training-arm CLI, look there.

**Key features:**
- 🌐 **Server–client architecture** — gRPC, parallel heartbeat, server-driven stop
- 🔄 **Six registered strategies** — FedAvg, FedProx, FedOpt, FedLoRA, DeComFL, RobustAggregator
- 🔐 **Deterministic safetensors wire** — float32, byte-identical across Python and the mobile C++ core
- 🧪 **In-process simulator** — thousands of clients in one process, no gRPC and no port pool
- 🛡️ **Byzantine-robust aggregation** — coordinate-wise median / trimmed mean + norm clipping
- 🔏 **Central DP** — a from-scratch RDP accountant (no Opacus/TF-Privacy at runtime)
- 🤖 **Multi-model** — CNNs, Transformers, LoRA-adapted LLMs
- 📊 **Non-IID partitioning** — Dirichlet, pathological, shard, IID

## Quick Start

### Installation

```bash
git clone https://github.com/Learning-Optimization-Group/FedLearn-Platform.git
cd FedLearn-Platform/framework

# 1. Install torch FIRST, from the PyTorch index (this is exactly what CI does).
#    The version is PINNED to 2.12.0 — see "Why torch is pinned" below.
pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.12.0"

# 2. Then the rest of the dependencies, and the framework itself.
pip install -r requirements.txt
pip install -e .
```

**Requirements:** Python 3.10+ (`setup.py` floor; CI tests on **3.12**). A CUDA GPU is optional and
only pays off for LLM training.

**Why `torch` is pinned to `2.12.0`** (`requirements.txt:9`): the DeComFL golden fixtures and the
`executorch==1.3.1` native extension were built against it, and `test_torch_version_matches_manifest`
fails the suite on a mismatch. "PyTorch 2.0+" would be misleading. `torchvision`/`torchaudio` are
deliberately **not** listed — the framework never imports them, and installing them from PyPI against
a PyTorch-index `torch` build produces an ABI mismatch. `setup.py` filters `torch*` out of
`install_requires`, so `pip install -e .` alone will *not* give you torch.

### 5-minute example

**Server:**
```python
import fedlearn as fl

model = YourModel()

strategy = fl.FedAvg(
    initial_parameters=model.state_dict(),
    min_fit_clients=2,       # minimum reporting clients before a round aggregates
    clients_per_round=5,     # cohort size (defaults to min_fit_clients)
)

fl.server.start_server(
    server_address="0.0.0.0:50051",
    config=fl.server.ServerConfig(num_rounds=10),
    strategy=strategy,
)
```

**Client:**
```python
import fedlearn as fl

class MyClient(fl.Client):
    def get_parameters(self):
        return self.model.state_dict()

    def fit(self, parameters, config):
        # local training …
        return updated_parameters, num_samples

fl.client.start_client(
    server_address="localhost:50051",
    client=MyClient(),
    client_id="client_0",
)
```

A round aggregates as soon as the cohort reports; an incomplete cohort is resolved after
`FEDLEARN_ROUND_TIMEOUT_S` (default **120 s**, `server/coordinator.py`).

### 1,000 clients in one process — no ports, no subprocesses

The deployed path reserves one TCP port per FL server from `50000-50010`, which caps the platform at
11 concurrent federations. `SimulatedFederation` sidesteps that by driving the **production**
`FLCoordinator` and the **production** `Strategy` objects by direct method call:

```python
import torch.nn as nn
from fedlearn.server.strategy import FedAvg
from fedlearn.simulation.federation import SimulatedFederation

fed = SimulatedFederation(
    strategy=FedAvg(initial_parameters=nn.Linear(8, 4).state_dict()),
    client_factory=make_client,   # (client_id, ClientRng) -> object with .fit(params, config)
    num_clients=1000,
    clients_per_round=10,
    seed=0,
    wire_in_the_loop=0.25,        # route 25% of updates through the real safetensors codec
)
result = fed.run(num_rounds=5)    # -> SimulationResult; .to_json() has {"meta", "per_round"}
```

Three properties it is responsible for: **determinism** (every draw derives from
`(seed, client_id, round)`, so adding a client cannot perturb its peers), **no wall-clock
dependence** (a dropout round resolves inline instead of sleeping out the deadline), and **the wire
stays testable** (`wire_in_the_loop` routes a fraction of updates through the real encode/decode; the
suite asserts on and off agree bit-for-bit). Clients are built and released per participation, so
peak memory scales with `clients_per_round`, not `num_clients` — which is what makes a 5,000-client
sweep runnable on a laptop-class machine.

Partitioners live alongside it: `dirichlet_partition`, `pathological_partition`, `shard_partition`,
`iid_partition`, `partition_report` (`fedlearn.simulation`).

## Strategies

Six strategies are registered in `server/strategy_factory.py`. Names match case-insensitively and
ignore hyphens/underscores (`"Fed-Prox"` == `"fedprox"`):

| Name | Class | What it does |
|---|---|---|
| `fedavg` | `FedAvg` | num-examples-weighted mean of client models |
| `fedprox` | `FedProx` | FedAvg aggregation + a client-side proximal penalty `(mu/2)·‖w − w_global‖²`; rejects a `(mu, lr)` pair past the stability limit, where the penalty *amplifies* drift |
| `fedopt` | `FedOpt` | server-side adaptive optimization — FedAdam / FedYogi ([Reddi et al.](https://arxiv.org/abs/2003.00295)) over the pseudo-gradient, with moment state persisted across rounds |
| `fedlora` | `FedLoRA` | adapter-only federation; under `FFA_LORA` the frozen `A` is re-attached bit-identically so `avg(B)@A == avg(B@A)` stays exact. Carries the optional central-DP path |
| `decomfl` | `DeComFL` | dimension-free zeroth-order path — **seeds and scalars only, no weights on the wire** |
| `robust` | `RobustAggregator` | coordinate-wise median / β-trimmed mean, non-finite rejection, L2 update clipping, and a breakdown-point guard that refuses the round rather than aggregating past it |

```python
from fedlearn.server.strategy_factory import create_strategy
strategy = create_strategy("fedprox", initial_parameters=params, proximal_mu=0.1)
```

`FedAvg`, `FedProx`, `FedOpt` and `DeComFL` are re-exported at the top level (`fl.FedAvg`);
`FedLoRA` and `RobustAggregator` live under `fl.server`. Details and paper mappings:
[`wikis/framework/05_strategies.md`](../wikis/framework/05_strategies.md).

## The wire

**Deterministic safetensors, not pickle.** `communication/safetensors_codec.py` encodes
`u64_le(header_len) ++ compact_json_header ++ raw_data`, in insertion order, little-endian **float32
only** — which is what lets the libtorch-free mobile C++ core produce byte-identical output and a
golden fixture pin the contract. A malformed or hostile blob fails loudly (header/offset/shape
validation, non-finite rejection) instead of being mis-read.

**Non-float32 tensors are excluded from the federated set, not coerced.** Every BatchNorm module
carries an int64 `num_batches_tracked`, which used to kill any FULL-arm run on a BatchNorm model.
`estimators/params.py` now supplies `federable_state()` / `non_federable_names()` — **one** filter
used by *both* sides, so client and server agree by construction, and what was withheld is logged
rather than silently dropped. `running_mean`/`running_var` are float32 and are still federated;
dropping those would be FedBN, a different algorithm. The encoder itself remains fail-loud: a
non-float32 tensor that reaches `state_dict_to_safetensors` raises.

**The two decode paths are asymmetric — know which one you are in:**

| Path | Behaviour on a legacy `torch.save`/pickle blob |
|---|---|
| Server-side upload/receive (`serializer.py`) | **rejects loudly** — sniffs the zip (`PK\x03\x04`) / pickle (`0x80`) magic after a positive safetensors check |
| Client-side global-model download (`client/grpc_client.py`) | **still accepts** it, falling back to `torch.load(..., weights_only=True)`, so a new client keeps working against an older server during a staged rollout |

On the **download** path the `codec` field is the primary signal and magic-byte sniffing the backstop,
and the payload's sha256 is verified format-agnostically *before* either. **The upload path carries
no framing and no integrity check** — a real gap, not a design: `_generate_model_chunks`
(`client/grpc_client.py:248-261`) leaves `codec`, `compressed`, `total_bytes` and `sha256` unset on
every `ModelUpdateChunk`, and the server's `SubmitModelUpdateStream` handler never reads them (it
hardcodes `compressed=False` and relies solely on the magic-byte sniff inside
`chunks_to_parameters`). The proto declares the fields in both directions; only the download
direction populates them.

**Chunking is size-gated at the call site, then unconditional inside the streaming path.**
`GrpcClient.submit_update()` picks the streaming upload only when the model looks like a transformer
(`ALWAYS_STREAM_TRANSFORMERS = True`, detected from parameter names) **or** exceeds
`STREAMING_THRESHOLD_MB = 100`; otherwise it sends a unary upload. Once streaming is chosen the blob
is chunked at a **hardcoded 50 MB** in both directions (`client/grpc_client.py:249`,
`server/grpc_servicer.py:184`) — a small blob simply emits one chunk. Note that
`FEDLEARN_CHUNK_SIZE_MB` (default 4 MB) sets `serializer.CHUNK_SIZE`, which is only the *default*
argument to `parameters_to_chunks`; the streaming call site overrides it, so that env var does
**not** govern the gRPC wire today.

**Heartbeat is bidirectional and runs on its own stub.** A client holds **two** gRPC stubs: the
training stub blocks during `fit()`, while the heartbeat stub keeps the server from timing the client
out mid-round. A `HeartbeatResponse` with `should_stop=True` latches `_stop_training`, which the fit
loop polls to abort the round. Keep both stubs *and* the stop signal.

The contract itself is governed by `buf` and lives at [`../proto/`](../proto/) — see
[`proto/README.md`](../proto/README.md).

## Security

**gRPC TLS exists and is opt-in — it is not absent, and it is not on by default.**

| Switch | Effect |
|---|---|
| `FEDLEARN_GRPC_USE_TLS=1` | client builds `ssl_channel_credentials` + `secure_channel`; server builds `ssl_server_credentials`. Default (unset) is **plaintext** |
| `FEDLEARN_GRPC_SERVER_CERT` / `_KEY` | required when TLS is on — the server refuses to start without them |
| `FEDLEARN_GRPC_ROOT_CERT`, `_CLIENT_CERT`, `_CLIENT_KEY` | client-side trust roots and client certificate |
| `FEDLEARN_GRPC_REQUIRE_CLIENT_AUTH=1` | mTLS: the server demands a client certificate |
| `FEDLEARN_REQUIRE_TLS=1` | **policy gate** (`security/tls.py`): fail closed rather than serve the FL boundary in plaintext. Deployed profiles set it |

**Client authentication is a separate mechanism** from transport security — and note the two
similarly named switches are *different*: `FEDLEARN_GRPC_REQUIRE_CLIENT_AUTH` is mTLS (above),
`FEDLEARN_REQUIRE_CLIENT_AUTH` is the connection-token gate (below). The backend mints a
short-lived HMAC-JWT per enrollment; the client receives it as `FEDLEARN_CONNECTION_TOKEN` and a
client interceptor attaches it as `x-connection-token` metadata on every call. Server-side,
`ConnectionTokenInterceptor` verifies it with PyJWT (HMAC family only — `none` and all asymmetric algs
are rejected) and aborts `UNAUTHENTICATED` otherwise. Enforcement is gated on
`FEDLEARN_REQUIRE_CLIENT_AUTH=1`, so local runs fail *open* while a misconfigured deployment (enforce
on, no secret) fails *closed*. `security/identity.py` additionally pins one token's server-assigned
`partitionId` to one wire `client_id`, so a single valid token cannot be replayed as a whole cohort.

## Privacy

`privacy/dp_accountant.py` is a **from-scratch, pure-Python + numpy** Rényi-DP accountant for the
Sampled Gaussian Mechanism (Mironov et al., [arXiv:1908.10530](https://arxiv.org/abs/1908.10530)) —
no Opacus or TF-Privacy at runtime. Per-order RDP matches Opacus to ~1e-9; the RDP→(ε, δ) conversion
uses the classic Mironov (2017) bound rather than the tighter Balle bound, so the reported ε is
deliberately **conservative** (never under-reports loss).

`privacy/dp_mechanism.py` is the central-DP mechanism on the FedLoRA adapter delta — clip each
client's delta jointly to L2 norm `S`, **uniform**-average (never num-examples-weighted: an attacker
controls its own reported example count, and weighting would void the ε claim), then add
`N(0, (z·S/N)²)` on the aggregatable keys only. Enable it through `FedLoRA(dp_enabled=True, …)` with
either an explicit `dp_noise_multiplier` **or** a `dp_target_epsilon` the accountant solves `z` from.

Runnable, seeded ε-vs-accuracy and breakdown-point benchmarks live in
[`benchmarks/`](benchmarks/) — start with [`benchmarks/README.md`](benchmarks/README.md).

## Specialization: frozen backbones, trainable subsets, adapter bundles

- **`estimators/params.py`** — the canonical trainable-parameter manifest: `param_layout`,
  `flat_params`, `trainable_state`, `frozen_state`, plus the wire filter `federable_state`. Anything
  that flattens, counts or federates parameters delegates here so client and server agree on layout;
  a mismatch would silently misalign DeComFL's shared-seed perturbation.
- **`server/subset_federation.py`** — federate only a model's `requires_grad` subset under FedAvg,
  with a fail-loud per-client guard on keys *and* shapes. The guard must run **before** aggregation:
  `FedAvgAggregator` derives its key set from the first client, so validating the aggregate can never
  catch a later client's bad payload.
- **`backbone/distribution.py`** — serialize a frozen backbone to deterministic, content-addressed
  bytes; fetch through an injected `Callable[[], bytes]` seam, verify sha256, cache by content
  address, reconstruct fail-loud.
- **`bundle/manifest.py`** — the versioned adapter-bundle manifest: identity by `artifact_sha256`,
  frozen base, LoRA config, license, eval-card reference and a per-file sha256 list, validated
  against a committed JSON schema. It is the cross-language sha256 provenance contract between the
  registry, the serving path and on-device training.

## Examples

`examples/` holds historical, hand-run federations. **Check the status column before you copy a
command** — several were committed with an empty `data.py` and do not import.

| Example | Status | Notes |
|---|---|---|
| `simple_federation/` (MNIST + CNN) | ⚠️ server runs, client **broken** | `run_client.py` raises `SyntaxError: name 'logger' is assigned to before global declaration` |
| `llm_federation/` (OPT-125M, SuperGLUE CB) | ✅ runs | `run_client.py --help` itself crashes on an unescaped `%` in a help string; passing real flags is fine |
| `ecg_federation/` (Transformer) | ❌ broken | `data.py` is committed empty → `ImportError: cannot import name 'get_test_loader'` |
| `ecg_decomfl_central/`, `ecg_decomfl_multiclient/` | ❌ broken | same empty `data.py` |
| `ecg_decomfl_framework_integration/` | ⚠️ runs, no CLI | `run_server.py` takes **no** arguments and starts a federation immediately; run it from its own directory |
| `fot_text_federation/` | ✅ runs | offline Federation-over-Text demo, no network and no LLM (see the FoT note below) |

Working invocation for the LLM example (flags verified against its `argparse`):

```bash
cd examples/llm_federation

# Terminal 1 — server
python run_server.py --dataset cb --num_rounds 5 --clients_per_round 3 --min_fit_clients 3

# Terminals 2-4 — clients
python run_client.py --id 0 --dataset cb --server_address localhost:50051
```

**On the accuracy numbers.** Earlier revisions of this README quoted "~83% on CB after 5 rounds" and
"~93.8% ECG after 3 rounds". The result CSVs committed next to those examples say otherwise: on CB
the 3-client run is at **55.4%** at round 5 and only reaches 80.4% by round 64 (the 2-client run
reaches 85.7% at round 50), and the ECG run is at **58.4%** at round 3 and reaches 93.8% around round
100. The headline figures were end-of-run, not early-round. No committed result exists for the MNIST
example, so no number is quoted for it.

For a federation you can actually run today, prefer the in-process simulator above, or this
directory's own end-to-end drivers — none of them take arguments:

| Script | What it needs |
|---|---|
| `python run_local_test.py` | nothing — spawns its own server and clients via `multiprocessing` |
| `python run_full_test_suite.py` | nothing — heavier multi-scenario run (CNN/MNIST, ECG transformer, DeComFL) on ports `50051-50053` |
| `python run_platform_e2e_test.py` | **the Spring Boot backend running on `:8081`** (override with `API_BASE_URL`); it drives the REST API and asserts round results persist |

## Testing

```bash
cd framework
PYTHONPATH=src python -m pytest -q          # exactly how CI runs it
PYTHONPATH=src python -m pytest -q --no-cov tests/test_federable_state.py   # a subset
```

`pytest.ini` is not a bare pytest config — it adds two things that will bite you if you forget them:

- **Coverage is enforced.** `--cov=fedlearn --cov-report=term-missing --cov-fail-under=73`. Running a
  hand-picked subset reports low coverage and fails the floor *by design* — pass `--no-cov`.
- **`-m "not slow"` is the default deselection.** Tests that download models or run full training are
  marked `slow` and skipped by default; deselection is not a skip and never trips the guard below.

CI (`.github/workflows/ci.yml`, path-filtered to `framework/**` — and also to `backend/**`, so a
backend change that breaks Java↔Python token compatibility trips this gate) installs CPU
`torch==2.12.0` plus `executorch==1.3.1`, then runs the command above with
`FEDLEARN_FAIL_ON_UNEXPECTED_SKIP=1`: a test **skipped** for a non-allowlisted reason fails the job,
so an import-guard skip cannot quietly turn the suite green. 105 test modules live under `tests/`.

`ruff` and `mypy` run locally but are **not** CI gates here and ship no committed config.

## Framework structure

```
framework/
├── src/fedlearn/            # the package (57 modules)
│   ├── backbone/            # frozen-backbone serialization + content-addressed cache
│   ├── bundle/              # adapter-bundle manifest + JSON schema
│   ├── client/              # Client ABC, gRPC client, DeComFL client, LocalTrainer
│   ├── communication/       # safetensors codec, serializer, protos + generated stubs
│   ├── estimators/          # parameter manifest, perturbation, zeroth-order estimator
│   ├── fot/                 # Federation over Text (additive, torch-free — see below)
│   ├── privacy/             # RDP accountant + central-DP mechanism
│   ├── security/            # TLS policy, token interceptors, identity binding
│   ├── server/              # server, coordinator, strategies, robust aggregation, subset federation
│   └── simulation/          # in-process federation, partitioners, seeded RNG streams
├── benchmarks/              # seeded, re-runnable benchmark harnesses
├── examples/                # historical example federations (see the status table)
├── tests/                   # 105 test modules
└── docs/                    # API reference and guides
```

## Documentation

📚 **Start here:**

- **[Installation Guide](docs/installation.md)** · **[Quick Start](docs/quickstart.md)**
- **[API Reference](docs/api-reference/)** — [Server](docs/api-reference/server.md) ·
  [Client](docs/api-reference/client.md) · [Strategies](docs/api-reference/strategies.md) ·
  [Core Modules](docs/api-reference/core-modules.md)
- **[Advanced](docs/advanced/)** — [Custom Strategies](docs/advanced/custom-strategies.md) ·
  [Extending the Framework](docs/advanced/extending-framework.md)
- **Deep dives live in the wiki:** [`wikis/framework/`](../wikis/framework/) — architecture, gRPC,
  server/client internals, strategies, DeComFL, partitioning, developer guide.

## Research

- **DeComFL** — *Achieving Dimension-Free Communication in Federated Learning via Zeroth-Order
  Optimization* (ICLR 2025, [arXiv:2405.15861](https://arxiv.org/abs/2405.15861)); reference
  implementation [ZidongLiu/DeComFL](https://github.com/ZidongLiu/DeComFL) (Apache-2.0).
  **Implemented and exercised** — the paper's server protocol (Algorithm 3) is
  `server/decomfl_strategy.py`, the client protocol (Algorithm 4) is `client/decomfl_client.py`, and
  the perturbation/estimator math is `estimators/`. Golden fixtures under
  `tests/fixtures/decomfl_golden/` pin it across Python and the mobile C++ core; line-by-line
  paper-to-code mapping in [`wikis/framework/06_decomfl.md`](../wikis/framework/06_decomfl.md).
- **Federation over Text (FoT)** — [arXiv:2604.16778](https://arxiv.org/abs/2604.16778). What ships
  in [`src/fedlearn/fot/`](src/fedlearn/fot/) is a complete, torch-free **vertical slice of the
  scaffolding**, orthogonal to the gradient path — *not* a validated implementation of the method.
  Be aware before citing it: **no LLM has ever run through it** (`backend.get_backend()` wires only
  `DeterministicStubBackend`; `local-http`, `vllm` and `ollama` all raise "not implemented in this
  build"); "distillation" is deduplication plus argmax-by-length over verbatim client strings, not
  synthesis; the default `quorum=2` counts distinct client IDs and drops sub-quorum insights, which
  inverts the paper's transfer goal; `Insight.tags` is declared, serialized and never populated; and
  `max_insights = 37` is an unexplained cap. What *is* solid: provenance and quorum counting are
  computed from real traces and never LLM-trusted, and canonicalization is tie-broken so insight IDs
  never depend on LLM ordering.
- Developed at Rochester Institute of Technology under Professor Haibo Yang.

## Citation

```bibtex
@inproceedings{li2025decomfl,
  title={Achieving Dimension-Free Communication in Federated Learning via Zeroth-Order Optimization},
  author={Li, Zhe and Ying, Bicheng and Liu, Zidong and Dong, Chaosheng and Yang, Haibo},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2025}
}
```

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for how to extend the framework, add a
strategy, and the code-style and testing expectations.

## License

Apache License 2.0 — see [LICENSE](../LICENSE).

## Support

- 📖 **Documentation**: [docs/](docs/) · [wikis/framework/](../wikis/framework/)
- 🐛 **Issues**: [GitHub Issues](https://github.com/Learning-Optimization-Group/FedLearn-Platform/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/Learning-Optimization-Group/FedLearn-Platform/discussions)

## Acknowledgments

Developed by the Learning Optimization Group at Rochester Institute of Technology.

**Principal Investigator:** Professor Haibo Yang

---

**New to federated learning?** Read the [Quick Start Guide](docs/quickstart.md), then run the
in-process simulator above — it needs no ports, no second terminal and no dataset.
