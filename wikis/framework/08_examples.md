# 08 — Examples & Benchmarks

## Table of Contents
- [Examples vs. Benchmarks](#examples-vs-benchmarks)
- [Overview](#overview)
- [Example 1: Simple Federation (MNIST + CNN)](#example-1-simple-federation-mnist--cnn)
  - [What It Demonstrates](#what-it-demonstrates)
  - [Architecture](#architecture)
  - [Running the Example](#running-the-example)
  - [Step-by-Step Trace](#step-by-step-trace)
  - [Expected Results](#expected-results)
- [Example 2: LLM Federation (OPT-125M)](#example-2-llm-federation-opt-125m)
  - [What It Demonstrates](#what-it-demonstrates-1)
  - [Architecture](#architecture-1)
  - [Running the Example](#running-the-example-1)
  - [Memory and Streaming Notes](#memory-and-streaming-notes)
  - [Expected Results](#expected-results-1)
- [Example 3: ECG Classification (Transformer)](#example-3-ecg-classification-transformer)
  - [What It Demonstrates](#what-it-demonstrates-2)
  - [Architecture](#architecture-2)
  - [Running the Example](#running-the-example-2)
  - [Expected Results](#expected-results-2)
- [Example 4: ECG + DeComFL (Multi-Client)](#example-4-ecg--decomfl-multi-client)
  - [What It Demonstrates](#what-it-demonstrates-3)
  - [Communication Comparison](#communication-comparison)
  - [Running the Example](#running-the-example-3)
- [Running Multiple Examples at Once](#running-multiple-examples-at-once)
- [The Committed Benchmark Harnesses](#the-committed-benchmark-harnesses)
- [End-to-End Test Suites](#end-to-end-test-suites)

---

## Examples vs. Benchmarks

Two different directories with two different jobs. Do not cite one where you mean the other.

| | `framework/examples/` | `framework/benchmarks/` |
|---|---|---|
| Purpose | show the **API** end to end | **measure** something and record it |
| Seeded / reproducible | not systematically | yes, by construction |
| Produces a machine-readable record | no | yes — `results/<name>.{json,md}` |
| Numbers quotable in a report | **no** | yes, with their caveats |
| Tracked in git | yes | the harnesses **yes**, `benchmarks/results/` **no** |

> **The "Expected Results" blocks in the example walkthroughs below are illustrative shapes, not
> measurements.** They show what a healthy run's output *looks like* — decreasing loss, rising
> accuracy — so you can tell a working run from a broken one. They are not reproduced from any
> committed record and must not be quoted as results. Every quotable number in this repo comes from
> a benchmark harness and lands in a JSON file next to it.

---

## Overview

The `framework/examples/` directory contains seven end-to-end runnable experiments:

| Example | Model | Strategy | Clients per round (**as shipped**) | Dataset |
|---------|-------|----------|---------|---------|
| `simple_federation` | CNN (`SimpleCNN`) | FedAvg | 2 — `min_fit_clients=2`, `clients_per_round` unset so it falls back | MNIST |
| `llm_federation` | OPT-125M | FedAvg | 2 (`--clients_per_round` default; `--min_fit_clients` defaults to 8) | SuperGLUE CB |
| `ecg_federation` | Transformer | FedAvg | 1 (`--num_clients` / `--min_fit_clients` / `--clients_per_round` all default to 1) | ECG (binary) |
| `ecg_decomfl_central` | **MLP** (`ECGModel`, 140→64→64→2) | DeComFL | 1 (`config.NUM_CLIENTS`) | ECG |
| `ecg_decomfl_multiclient` | **MLP** (`ECGModel`) | DeComFL | 2 of 5 (`NUM_CLIENTS=5 × CLIENT_FRACTION=0.4`) | ECG |
| `ecg_decomfl_framework_integration` | **MLP** (`ECGModel`) | DeComFL | 2 of 5 (`CLIENTS_PER_ROUND=2`) | ECG (via full framework) |
| `fot_text_federation` | — (stub backend) | FoT | in-process | synthetic text |

> The counts above are the **committed defaults**, not a recommendation — override them with the
> flags each script actually exposes (listed per example below) or by editing that example's
> `config.py`.

The six gradient examples are each self-contained with their own `run_server.py` and `run_client.py`
(`simple_federation` also has a `run_server_debug.py`); most carry a local `config.py`, `data.py` and
`model.py`, and several ship committed CSV/PNG artefacts from past runs.

**`fot_text_federation` is the exception** — it is a single in-process `run_fot.py` demo of the
Federation over Text mode: a separate, offline, local-LLM-only research path that is *additive and
orthogonal* to the gradient path rather than a replacement for it. See
`framework/examples/fot_text_federation/README.md`.

> **The FoT example runs against a deterministic stub, and no LLM has ever run through that path.**
> `fot.backend.get_backend()` wires only `DeterministicStubBackend`; asking for `local-http`, `vllm`
> or `ollama` raises `BackendError("not implemented in this build")`. The example demonstrates the
> plumbing, and nothing about FoT should be cited as a validated capability of this platform.

---

## Example 1: Simple Federation (MNIST + CNN)

### What It Demonstrates
- Basic FedAvg workflow end-to-end
- Custom `Client` subclass implementation
- Server evaluation with test accuracy metrics
- A crude contiguous-slice data split

> **It does *not* demonstrate Dirichlet partitioning.** `data.get_mnist_loader` hands client `i` the
> contiguous index range `[i·N/num_clients, (i+1)·N/num_clients)` of MNIST, and `run_client.py:36`
> calls it with a hardcoded `num_clients=10` regardless of how many clients you launch — so each
> client trains on 6,000 samples and the "non-IID" character is whatever MNIST's storage order
> happens to give. For a real, seeded, contract-tested split use
> `fedlearn.simulation.partition` — see [07 — Data Partitioning](07_data_partitioning.md).

### Architecture

```
┌─────────────────────────────────────────────────────┐
│                    Server                            │
│                                                      │
│  FedAvg Strategy                                     │
│    initial_parameters = SimpleCNN().state_dict()     │
│    evaluate_fn = server_side_evaluate()              │
│    min_fit_clients = 2                               │
│    clients_per_round → defaults to min_fit_clients=2 │
│                                                      │
│  FLCoordinator (--num_rounds, default 5)            │
└──────────────────────┬──────────────────────────────┘
                       │ gRPC :50051
             ┌─────────┴─────────┐
             ▼                   ▼
         Client 0            Client 1
         6,000               6,000
         samples             samples
    (contiguous MNIST slices; get_mnist_loader is called
     with num_clients=10 hardcoded in run_client.py:36)
```

> `run_server.py` never passes `clients_per_round`, so it falls back to `min_fit_clients` (2) — the
> round aggregates as soon as **two** clients report, and a third client's update for that round is
> then rejected as stale. Launch two clients, or pass `clients_per_round` in the source.

### Running the Example

```bash
cd examples/simple_federation

# Terminal 1: Start server — the ONLY flags are --port and --num_rounds
python run_server.py --port 50051 --num_rounds 10

# Terminal 2: Client 0 — the ONLY flags are --id (required) and --server_address
python run_client.py --id 0 --server_address localhost:50051

# Terminal 3: Client 1
python run_client.py --id 1 --server_address localhost:50051
```

### Step-by-Step Trace

**Server startup log:**
```json
{"timestamp": "2024-01-15T10:00:00Z", "level": "INFO", "message": "Starting FedLearn server on 0.0.0.0:50051"}
{"timestamp": "2024-01-15T10:00:00Z", "level": "WARNING", "message": "gRPC server running without TLS."}
{"timestamp": "2024-01-15T10:00:00Z", "level": "INFO", "message": "gRPC server started and listening on 0.0.0.0:50051"}
```

**Client 0 registration:**
```json
{"level": "INFO", "message": "RegisterClient: client_0 registered"}
{"level": "INFO", "message": "[client_0] Registered with server; starting heartbeat"}
```

**Round 1 begins:**
```
Server: coordinator.start_round()           ← clears updates buffer
Client: GetGlobalModelStream                ← downloads SimpleCNN: 85,822 params ≈ 0.33 MB
Client: fit(parameters, config)             ← 1 local epoch, 6000/32 ≈ 188 batches
  → heartbeats: status="training", step=1..188/188
Client: SubmitModelUpdate                   ← 0.33 MB is under STREAMING_THRESHOLD_MB=100
                                              and not a transformer → the UNARY upload path
Server: submit_client_update()
Server: [waiting for 1 more client...]
```

**Aggregation trigger (when the second client submits — `clients_per_round` is 2 here):**
```json
{"level": "INFO", "message": "All 2 clients reported for round 1; aggregating"}
{"level": "INFO", "message": "FedAvg eval round=1 loss=0.4312 metrics={'accuracy': 0.8734}"}
{"level": "INFO", "message": "[Server] Round 1 complete. Metrics: {'loss': 0.4312, 'accuracy': 0.8734}"}
```

**Client poll after round completion:**
```
Client: GetGlobalModelStream     ← now returns updated global model (round=2)
Client: server_round (2) > last_completed_round (1) → start round 2
```

### Expected Results

*Illustrative shape only — not a measurement (see [Examples vs. Benchmarks](#examples-vs-benchmarks)).*

```
Round  1: loss=0.43, accuracy=87.3%
Round  3: loss=0.31, accuracy=91.2%
Round  5: loss=0.24, accuracy=93.1%
Round  7: loss=0.19, accuracy=94.4%
Round 10: loss=0.15, accuracy=95.2%
```

What to actually check: loss decreasing monotonically-ish and accuracy rising. A flat curve means the
clients are not receiving the aggregated model; a curve that rises then explodes means a learning-rate
problem (and, on the DeComFL path, is the signature the LR stability envelope exists to catch —
see [06](06_decomfl.md#the-learning-rate-stability-envelope)).

---

## Example 2: LLM Federation (OPT-125M)

### What It Demonstrates
- Federated fine-tuning of a 125M parameter language model
- Automatic streaming upload (model is 500 MB — exceeds threshold)
- HuggingFace Transformers integration
- SuperGLUE CommitmentBank (3-class textual entailment)

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│                       Server                             │
│  FedAvg with evaluate_fn                                 │
│  Initial model: facebook/opt-125m (500 MB)               │
│  Task: 3-class classification (entailment/contradiction/ │
│        neutral)                                          │
└────────────────────────┬────────────────────────────────┘
                         │ gRPC :50051
                         │ streaming (50 MB chunks)
           ┌─────────────┼─────────────┐
           ▼             ▼             ▼
       Client 0      Client 1      Client 2
       ~83 CB        ~83 CB        ~83 CB
       examples      examples      examples
```

### Running the Example

```bash
cd examples/llm_federation

# Requires ~2 GB RAM per client (model + optimizer + activations).
# There is no --device flag on either script — the device is picked in config.py.

# Terminal 1: Server. Flags: --port --dataset --num_rounds --data_fraction
#                            --min_fit_clients --clients_per_round
python run_server.py \
    --port 50051 \
    --dataset cb \
    --num_rounds 5 \
    --min_fit_clients 3 \
    --clients_per_round 3

# Terminals 2-4: Clients. Flags: --server_address --id --dataset --num_clients --data_fraction
# NOTE the client's --server_address DEFAULT is a hardcoded AWS IP, not localhost — pass it.
python run_client.py \
    --id 0 \
    --dataset cb \
    --num_clients 3 \
    --server_address localhost:50051
```

### Memory and Streaming Notes

OPT-125M has 125M parameters → 500 MB as float32.

The `GrpcClient.submit_update()` detects transformer architecture:
```python
# Auto-detection triggers streaming
is_transformer = any(
    keyword in name.lower()
    for name in params.keys()
    for keyword in ['transformer', 'bert', 'gpt', 'opt', 'attention', 'encoder', 'decoder']
)
# OPT parameter names contain 'decoder' and 'attention' → streaming selected
```

Upload sequence *(shape only — `submit_update` logs MiB, i.e. `params * 4 / 1024²`, so ~125M params
prints ≈ 478 MB, not the 500 decimal-MB figure above)*:
```
[client_0] Model: 477.75 MB (125,237,760 params)
[client_0] Streaming upload selected (transformer)
[client_0] Zero-copy streaming upload
[client_0] Streaming upload complete in 12.3s
```

Memory footprint during streaming upload:
```
state_dict → safetensors blob → sliced into 50 MB chunks
Peak extra memory: ~500 MB (just the serialized buffer)
Total: ~1.5 GB per client (model + gradients + buffer)
```

> The upload wire is the deterministic **safetensors** codec, not `torch.save`/pickle — see
> [02 — gRPC Communication](02_grpc_communication.md#chunked-streaming-for-large-models). It is
> float32-only and fails loud on any non-float32 tensor.

### Expected Results

*Illustrative shape only — not a measurement.*

```
Round 1: loss=1.12, accuracy=72.3%
Round 2: loss=0.89, accuracy=78.1%
Round 3: loss=0.73, accuracy=81.4%
Round 4: loss=0.64, accuracy=82.7%
Round 5: loss=0.58, accuracy=83.1%
```

The CB dataset has only 250 training examples and 56 test examples — run-to-run variance is large
enough that a single run's accuracy is not a result at any scale.

---

## Example 3: ECG Classification (Transformer)

### What It Demonstrates
- Federated learning for medical time-series data
- Custom Transformer model for 1D signal classification
- Larger client count (5 clients) with non-IID ECG data
- Binary classification: Normal (0) vs. Abnormal (1)

### Architecture

```
ECG Signal (140 time steps) → Transformer (multi-head attention) → FC → Binary label

Transformer config (ECGTransformer, defined inline in run_server.py / run_client.py —
this example has no model.py):
  input_dim          = 140      ← config.ECG_CONFIG; ecg.csv is 141 columns wide
  d_model            = 64
  nhead              = 4
  num_layers         = 2
  dim_feedforward    = 256      ← hardcoded in the encoder layer, not in config.py

Parameters: ~111K (small — the upload takes the UNARY path)
```

### Running the Example

```bash
cd examples/ecg_federation

# Download ECG dataset first (Kaggle PTB-ECG or MIT-BIH)
# Place at ecg_data/ecg.csv — the committed file is 141 columns: cols 0-139 signal, col 140 label

# Server binds via --port (there is no --server_address on the server side)
python run_server.py \
    --data_path ecg_data/ecg.csv \
    --num_clients 5 \
    --min_fit_clients 5 \
    --clients_per_round 5 \
    --num_rounds 3 \
    --port 50051

# 5 separate client terminals
for i in 0 1 2 3 4; do
    python run_client.py \
        --id $i \
        --num_clients 5 \
        --data_path ecg_data/ecg.csv \
        --server_address localhost:50051 &
done
```

### Expected Results

*Illustrative shape only — not a measurement.*

```
Round 1: loss=0.68, accuracy=81.2%
Round 2: loss=0.42, accuracy=89.7%
Round 3: loss=0.31, accuracy=93.8%
```

The ECG Transformer converges quickly because binary classification is simple and the signal features
are highly structured — which also makes it a poor discriminator between aggregation strategies. Use
`benchmarks/algo_comparison.py` if you want to compare algorithms, not this example.

---

## Example 4: ECG + DeComFL (Multi-Client)

### What It Demonstrates
- DeComFL communication efficiency on the ECG **MLP** (`ECGModel`, 140→64→64→2 — *not* the
  Transformer that `ecg_federation` uses; all three DeComFL examples ship the MLP)
- Gradient scalar protocol (O(K×P) scalars per round)
- Model rebuild for late-joining clients
- Comparison to standard FedAvg on same task

### Communication Comparison

Arithmetic on the shipped `ECGModel` (13,314 trainable parameters), at a hypothetical `K=5, P=10`
(the committed `config.py` ships `NUM_LOCAL_STEPS = 1`, `NUM_PERTURBATIONS = 10`):

| | FedAvg (ECG MLP) | DeComFL (K=5, P=10) |
|--|--------------------------|---------------------|
| Upload per client | ~53 KB (13,314 params × 4 B) | 400 bytes (50 scalars × 8 B) |
| Download per client | ~53 KB | ~600 bytes (seeds + history) |
| Ratio | 1× | ~130× smaller |

> **This model is far too small for DeComFL to pay off dramatically** — the O(K×P) win scales with
> `d`, and `d = 13,314` here. The million-fold figures in
> [06 — Communication Comparison](06_decomfl.md#communication-comparison) are for LLM-scale `d`.
> Do not quote either number as a measurement; the committed measurement is
> `benchmarks/zeroth_vs_first_order.py`.

### Running the Example

> **Two ECG DeComFL examples, only one of them federated.** `ecg_decomfl_multiclient` (and
> `ecg_decomfl_central`) is a *single-process simulation* with its own in-file server/client classes:
> no gRPC, **no argparse at all** — `config.py` is the only knob — and its `run_client.py` has no
> `__main__`, so it is imported rather than executed. The example that actually runs the framework
> over gRPC (`fl.server.start_server` + `start_decomfl_client`) is
> `ecg_decomfl_framework_integration`.

```bash
# The federated one — this is the example to copy
cd examples/ecg_decomfl_framework_integration

# Server: no CLI flags. Edit config.py for NUM_ROUNDS / NUM_LOCAL_STEPS / NUM_PERTURBATIONS /
# LEARNING_RATE / SMOOTHING_PARAM / SERVER_ADDRESS (default localhost:50051).
python run_server.py

# Clients: the only two flags are --client-id (required) and --server
python run_client.py --client-id client_0
python run_client.py --client-id client_1 --server localhost:50051
python run_client.py --client-id client_2 --server localhost:50051

# The single-process simulation, for contrast — no flags, config.py is the only knob
cd ../ecg_decomfl_multiclient && python run_server.py
```

**Server-side round log for DeComFL** *(shape only; `model_dim` is the shipped `ECGModel`'s 13,314
trainable params, and the committed config is `K=1`)*:
```json
{"level": "INFO", "message": "DeComFL initialised: K=1, P=10, eta=0.001, mu=0.001, model_dim=13314"}
{"level": "INFO", "message": "DeComFL config request from client_0 for round 1"}
{"level": "INFO", "message": "Sending 1 local steps, 0 missed rounds"}
{"level": "INFO", "message": "Receiving gradient scalars from client_0 for round 1"}
{"level": "INFO", "message": "Received 1 local steps, 10 perturbations per step"}
{"level": "INFO", "message": "All 2 DeComFL updates received for round 1; aggregating"}
{"level": "INFO", "message": "Round 1 complete (loss=0.71, metrics={'accuracy': 0.782})"}
```

**Late-joining client rebuild example:**
```json
{"level": "DEBUG", "message": "Sending 1 local steps, 3 missed rounds"}
{"level": "DEBUG", "message": "Rebuilding model from 3 missed rounds"}
{"level": "DEBUG", "message": "Model rebuild complete"}
```

---

## Running Multiple Examples at Once

Use the `launch_all.sh` script at the repository root to start all components. It is **macOS-only** — it
drives AppleScript (`osascript`) to open a Terminal window per service, so on Linux you must start each
component by hand.

```bash
# From FedLearn-Platform root (macOS)
./launch_all.sh

# Or for just the framework (manual multi-terminal setup):
# 1. Start Spring Boot backend (provides orchestration)
# 2. Backend spawns Python FL server automatically
# 3. Backend spawns Python FL clients automatically
```

For pure Python testing (no Spring Boot):

```bash
#!/bin/bash
# Quick multi-client test script

PORT=50051
SERVER_ADDR="localhost:${PORT}"
# simple_federation aggregates at clients_per_round, which defaults to min_fit_clients=2.
NUM_CLIENTS=2
NUM_ROUNDS=5

# Start server in background (--port and --num_rounds are its only flags)
python examples/simple_federation/run_server.py \
    --num_rounds $NUM_ROUNDS \
    --port $PORT &

SERVER_PID=$!
sleep 2   # wait for server to be ready

# Start clients (--id and --server_address are its only flags)
for i in $(seq 0 $((NUM_CLIENTS - 1))); do
    python examples/simple_federation/run_client.py \
        --id $i \
        --server_address $SERVER_ADDR &
done

# Wait for all clients to finish
wait
kill $SERVER_PID 2>/dev/null
echo "Training complete"
```

---

## The Committed Benchmark Harnesses

`framework/benchmarks/` holds **17 seeded, re-runnable measurement harnesses** plus one shared
accounting module (`wire_bytes.py`) and a `README.md` — 19 files, all tracked in git.
**`benchmarks/results/` is not tracked** — the JSON/Markdown records are generated, so regenerate
rather than expecting them in a fresh clone.

Every harness runs against the **real** strategy and client code, not a mock, and writes
`results/<name>.json` — the machine-readable record. Most also write `results/<name>.md`, a
rendering of it; `decomfl_vs_fedavg_dim.py`, `frozen_vs_finetune_xray.py` and
`strategy_device_sweep.py` currently emit JSON only.

```bash
cd framework
PYTHONPATH=src python benchmarks/<name>.py [--flags]
```

`wire_bytes.py` is the exception to that command: it has **no `__main__`** and writes nothing — it is
the byte-accounting library `algo_comparison` imports, not a harness you run.

### Algorithms and communication

| Harness | What it measures |
|---|---|
| `algo_comparison.py` | Apples-to-apples: one task, one fixed non-IID partition, one seed, through each algorithm's real `aggregate_fit`. Per round: test accuracy, loss, wall-clock, and **truthful cumulative wire bytes** |
| `wire_bytes.py` | The per-round wire-byte accounting `algo_comparison` uses. Exists because the platform measured accuracy but never bytes — the proto's `bytes_received` field is unwired — and a fair FedAvg-vs-DeComFL comparison lives or dies on the communication axis |
| `comms_regimes.py` | Three-regime per-round comms: full-model FedAvg vs head-only (frozen-backbone) FedAvg vs DeComFL, unified into one contrast |
| `decomfl_vs_fedavg_dim.py` | Zeroth- vs first-order **as a function of model dimension** |
| `zeroth_vs_first_order.py` | Both families' convergence **and** bytes side by side — the measured convergence↔communication trade-off |
| `strategy_device_sweep.py` | Every aggregation strategy on CPU versus GPU at the model scale this platform federates |

### Byzantine robustness (FR-12)

| Harness | What it measures |
|---|---|
| `robust_aggregation_attack.py` | Whether the real `RobustAggregator` retains held-out accuracy under a family of Byzantine attacks where the real `FedAvg` collapses, on a non-IID split |
| `robust_breakdown_point.py` | The **measured breakdown point** — `robust_aggregation_attack` runs at one fixed Byzantine fraction, which shows the estimators defend *a* level; this sweeps the fraction to find where they stop |

### Central DP (FR-13)

| Harness | What it measures |
|---|---|
| `dp_epsilon_accuracy.py` | The ε-vs-accuracy trade-off on the real federated-LoRA loop: no-DP baseline plus several target ε, everything else fixed and seeded |
| `dp_snr_crossing.py` | Whether pushing the utility-SNR toward 1 (by cohort size) recovers FedLoRA utility |
| `dp_on_head.py` | Whether DP on a **small head** (low `d`) escapes the high-dimension collapse — on a seeded synthetic task |
| `dp_on_head_xray.py` | The same question on **real chest X-ray** data over a frozen backbone, sweeping ε |
| `dp_on_head_xray_cohort.py` | Its cohort-size (`N`) axis |
| `dp_on_head_xray_domain.py` | Whether a **domain** backbone beats frozen ImageNet for DP-on-head |
| `dp_head_cohort_sweep.py` | The cohort-size axis of the DP utility-SNR on a small head |
| `dp_subsampling_amplification.py` | Poisson client subsampling (`q < 1`) amplifying privacy — exercising the accountant's `q` parameter |

### Frozen backbone / transfer (DA-11, DA-14)

| Harness | What it measures |
|---|---|
| `frozen_backbone_fl.py` | Head-only federated learning over a shared frozen backbone: communication **and** utility |
| `frozen_vs_finetune_xray.py` | The non-DP 2×2 that separates "pretraining helped" from "freezing helped" |

### How to read a benchmark result honestly

The `benchmarks/README.md` is worth reading in full, because it models the standard these records are
held to. The FR-13 entry is the clearest example: the DP mechanism and accountant are **validated
end-to-end** (DP solves `z` from each target ε, the accountant certifies the accounted ε back to the
requested budget exactly), and utility nonetheless **collapses at every tested ε** at laptop scale.
The README quantifies *why* — the utility SNR `N/(z·√d)` is ≪ 1, and it is **independent of the clip
`S`**, so no tuning helps — rather than reporting a hand-tuned curve. That is a negative result,
recorded as one.

Several harnesses have paired smoke tests in `framework/tests/` (`test_algo_comparison_smoke.py`,
`test_dp_on_head_smoke.py`, `test_robust_aggregation_breakdown.py`, …) so the harnesses themselves
cannot silently rot between full runs.

---

## End-to-End Test Suites

These are **separate from the pytest suite**, which is what CI actually gates on:

```bash
cd framework
PYTHONPATH=src python -m pytest -q          # ~100 test modules; how CI runs it
```

`pytest.ini` sets `-m "not slow"` (deselecting tests that download models or run full training) and
enforces coverage: `--cov=fedlearn --cov-fail-under=73`. That floor is a **regression guard, not a
target** — measured line coverage is ~77%. Consequence worth knowing: **running a hand-picked subset
of tests will report low coverage and trip the floor by design** — pass `--no-cov` for a subset run.

The three scripts below are heavier, hand-run end-to-end checks. **None of them take CLI
arguments** — each is configured in the source (or, for the platform test, by environment variable).
Run them from `framework/`.

### run_local_test.py — Quick Sanity Test

```bash
cd framework/
python run_local_test.py
```

Spins up **1 gRPC server and 3 clients** locally using the `SimpleCNN` model from
`examples/simple_federation`. Each client trains on a partition of MNIST with real SGD, and the server
evaluates accuracy after each round.

Exercises:
- Server startup and client registration
- FedAvg across three real local-training clients
- Model serialisation/deserialisation round-trip
- Heartbeat flow

### run_full_test_suite.py — Comprehensive E2E

```bash
cd framework/
python run_full_test_suite.py
```

Runs all three major example configurations in sequence, spinning up a gRPC server + clients for each,
running federated training, and verifying accuracy/convergence. Logs are stored per-test.

- Test 1: `SimpleCNN` + MNIST + FedAvg (10 rounds)
- Test 2: `ECGTransformer` + ECG data + FedAvg (10 rounds)
- Test 3: ECG MLP + ECG data + DeComFL (10 rounds)

### run_platform_e2e_test.py — Integration with Spring Backend

```bash
cd framework/
python run_platform_e2e_test.py

# The backend API base is the ONLY knob, via env var (default shown):
API_BASE_URL=http://localhost:8081/api python run_platform_e2e_test.py
```

Exercises:
- Registering/logging in a test user via the Spring Boot backend (default `:8081`)
- Creating a training project through the REST API
- Backend-triggered FL server spawn (`FlServerManager`) + locally spawned simulated Python clients
- Polling the backend API for `RoundResults` to verify persistence — to **PostgreSQL**, which backs
  every Spring profile (H2 has been retired)
- Automatic cleanup

> **Prerequisite:** the backend needs a local Postgres. From `backend/fl-platform-api/`, run
> `docker compose up -d` (postgres:16.6-alpine; db/user/password all `federance` on `:5432`) before
> starting Spring Boot.
