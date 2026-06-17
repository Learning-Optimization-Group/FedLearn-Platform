# 08 — Examples Walkthrough

## Table of Contents
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
- [End-to-End Test Suites](#end-to-end-test-suites)

---

## Overview

The `framework/examples/` directory contains six end-to-end runnable experiments:

| Example | Model | Strategy | Clients | Dataset |
|---------|-------|----------|---------|---------|
| `simple_federation` | CNN | FedAvg | 3 | MNIST |
| `llm_federation` | OPT-125M | FedAvg | 3 | SuperGLUE CB |
| `ecg_federation` | Transformer | FedAvg | 5 | ECG (binary) |
| `ecg_decomfl_central` | Transformer | DeComFL | 1 | ECG |
| `ecg_decomfl_multiclient` | Transformer | DeComFL | 3 | ECG |
| `ecg_decomfl_framework_integration` | Transformer | DeComFL | 3 | ECG (via full framework) |

Each example is self-contained with its own `run_server.py` and `run_client.py`.

---

## Example 1: Simple Federation (MNIST + CNN)

### What It Demonstrates
- Basic FedAvg workflow end-to-end
- Custom `Client` subclass implementation
- Server evaluation with test accuracy metrics
- Non-IID data partitioning with Dirichlet α=0.5

### Architecture

```
┌─────────────────────────────────────────────────────┐
│                    Server                            │
│                                                      │
│  FedAvg Strategy                                     │
│    initial_parameters = CNN().state_dict()           │
│    evaluate_fn = evaluate_on_test_set()              │
│    min_fit_clients = 2                               │
│    clients_per_round = 3                             │
│                                                      │
│  FLCoordinator (10 rounds)                          │
└──────────────────────┬──────────────────────────────┘
                       │ gRPC :50051
          ┌────────────┼────────────┐
          ▼            ▼            ▼
    Client 0      Client 1      Client 2
    ~20,000       ~20,000       ~20,000
    samples       samples       samples
    (biased       (different    (different
    classes)      bias)         bias)
```

### Running the Example

```bash
cd examples/simple_federation

# Terminal 1: Start server
python run_server.py \
    --num_rounds 10 \
    --num_clients 3 \
    --server_address 0.0.0.0:50051

# Terminal 2: Client 0
python run_client.py \
    --id 0 \
    --num_clients 3 \
    --alpha 0.5 \
    --server_address localhost:50051

# Terminal 3: Client 1
python run_client.py --id 1 --num_clients 3 --alpha 0.5 --server_address localhost:50051

# Terminal 4: Client 2
python run_client.py --id 2 --num_clients 3 --alpha 0.5 --server_address localhost:50051
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
Client: GetGlobalModelStream                ← downloads 4.8 MB CNN
Client: fit(parameters, config)             ← 1 local epoch, ~400 batches
  → heartbeats: status="training", step=1..400/400
Client: SubmitModelUpdateStream             ← uploads 4.8 MB
Server: submit_client_update()
Server: [waiting for 2 more clients...]
```

**Aggregation trigger (when client 2 submits):**
```json
{"level": "INFO", "message": "All 3 clients reported for round 1; aggregating"}
{"level": "INFO", "message": "FedAvg eval round=1 loss=0.4312 metrics={'accuracy': 0.8734}"}
{"level": "INFO", "message": "[Server] Round 1 complete. Metrics: {'loss': 0.4312, 'accuracy': 0.8734}"}
```

**Client poll after round completion:**
```
Client: GetGlobalModelStream     ← now returns updated global model (round=2)
Client: server_round (2) > last_completed_round (1) → start round 2
```

### Expected Results

```
Round  1: loss=0.43, accuracy=87.3%
Round  3: loss=0.31, accuracy=91.2%
Round  5: loss=0.24, accuracy=93.1%
Round  7: loss=0.19, accuracy=94.4%
Round 10: loss=0.15, accuracy=95.2%
```

Training time: ~5 minutes on CPU for 3 clients × 10 rounds with a small CNN.

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
                         │ streaming (500 MB chunks)
           ┌─────────────┼─────────────┐
           ▼             ▼             ▼
       Client 0      Client 1      Client 2
       ~83 CB        ~83 CB        ~83 CB
       examples      examples      examples
```

### Running the Example

```bash
cd examples/llm_federation

# Requires ~2 GB RAM per client (model + optimizer + activations)
# Optional: --device mps for Apple Silicon, --device cuda for NVIDIA GPU

# Terminal 1: Server
python run_server.py \
    --dataset cb \
    --num_rounds 5 \
    --num_clients 3 \
    --server_address 0.0.0.0:50051

# Terminals 2-4: Clients
python run_client.py \
    --id 0 \
    --dataset cb \
    --device cpu \  # or mps/cuda
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

Download sequence:
```
[client_0] Model: 500.24 MB (125,237,760 params)
[client_0] Streaming upload selected (transformer)
[client_0] Zero-copy streaming upload
[client_0] Streaming upload complete in 12.3s
```

Memory footprint during streaming upload:
```
torch.save() → BytesIO buffer → memoryview (zero-copy)
Peak extra memory: ~500 MB (just the serialized buffer)
Total: ~1.5 GB per client (model + gradients + buffer)
```

### Expected Results

```
Round 1: loss=1.12, accuracy=72.3%
Round 2: loss=0.89, accuracy=78.1%
Round 3: loss=0.73, accuracy=81.4%
Round 4: loss=0.64, accuracy=82.7%
Round 5: loss=0.58, accuracy=83.1%
```

The CB dataset has only 250 training examples and 56 test examples — accuracy variance between runs is expected.

---

## Example 3: ECG Classification (Transformer)

### What It Demonstrates
- Federated learning for medical time-series data
- Custom Transformer model for 1D signal classification
- Larger client count (5 clients) with non-IID ECG data
- Binary classification: Normal (0) vs. Abnormal (1)

### Architecture

```
ECG Signal (187 time steps) → Transformer (multi-head attention) → FC → Binary label

Transformer config:
  d_model = 64
  nhead = 4
  num_encoder_layers = 2
  dim_feedforward = 256

Parameters: ~500K (small — unary upload used)
```

### Running the Example

```bash
cd examples/ecg_federation

# Download ECG dataset first (Kaggle PTB-ECG or MIT-BIH)
# Place at ecg_data/ecg.csv (col 0-186: signal, col 187: label)

python run_server.py \
    --data_path ecg_data/ecg.csv \
    --num_clients 5 \
    --num_rounds 3 \
    --server_address 0.0.0.0:50051

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

```
Round 1: loss=0.68, accuracy=81.2%
Round 2: loss=0.42, accuracy=89.7%
Round 3: loss=0.31, accuracy=93.8%
```

The ECG Transformer converges quickly because:
1. Binary classification is simple
2. The signal features are highly structured
3. All 5 clients contribute gradient information from different patient profiles

---

## Example 4: ECG + DeComFL (Multi-Client)

### What It Demonstrates
- DeComFL communication efficiency for the ECG Transformer
- Gradient scalar protocol (O(K×P) scalars per round)
- Model rebuild for late-joining clients
- Comparison to standard FedAvg on same task

### Communication Comparison

| | FedAvg (ECG Transformer) | DeComFL (K=5, P=10) |
|--|--------------------------|---------------------|
| Upload per client | ~2 MB (500K params × 4B) | 400 bytes (50 scalars × 8B) |
| Download per client | ~2 MB | ~600 bytes (seeds + history) |
| Ratio | 1× | ~5,000× smaller |

### Running the Example

```bash
cd examples/ecg_decomfl_multiclient

# Server
python run_server.py \
    --num_clients 3 \
    --num_rounds 10 \
    --num_local_steps 5 \
    --num_perturbations 10 \
    --learning_rate 0.001 \
    --smoothing_param 0.001 \
    --server_address 0.0.0.0:50051

# Clients (note: DeComFL clients use start_decomfl_client)
python run_client.py --id 0 --data_path ecg_data/ecg.csv --server_address localhost:50051
python run_client.py --id 1 --data_path ecg_data/ecg.csv --server_address localhost:50051
python run_client.py --id 2 --data_path ecg_data/ecg.csv --server_address localhost:50051
```

**Server-side round log for DeComFL:**
```json
{"level": "INFO", "message": "DeComFL initialised: K=5, P=10, eta=0.001, mu=0.001, model_dim=512000"}
{"level": "INFO", "message": "DeComFL config request from client_0 for round 1"}
{"level": "INFO", "message": "Sending 5 local steps, 0 missed rounds"}
{"level": "INFO", "message": "Receiving gradient scalars from client_0 for round 1"}
{"level": "INFO", "message": "Received 5 local steps, 10 perturbations per step"}
{"level": "INFO", "message": "All 3 DeComFL updates received for round 1; aggregating"}
{"level": "INFO", "message": "Round 1 complete (loss=0.71, metrics={'accuracy': 0.782})"}
```

**Late-joining client rebuild example:**
```json
{"level": "DEBUG", "message": "Sending 5 local steps, 3 missed rounds"}
{"level": "DEBUG", "message": "Rebuilding model from 3 missed rounds"}
{"level": "DEBUG", "message": "Model rebuild complete"}
```

---

## Running Multiple Examples at Once

Use the `launch_all.sh` script at the repository root to start all components:

```bash
# From FedLearn-Platform root
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

SERVER_ADDR="localhost:50051"
NUM_CLIENTS=3
NUM_ROUNDS=5

# Start server in background
python examples/simple_federation/run_server.py \
    --num_rounds $NUM_ROUNDS \
    --num_clients $NUM_CLIENTS \
    --server_address "0.0.0.0:${SERVER_ADDR##*:}" &

SERVER_PID=$!
sleep 2   # wait for server to be ready

# Start clients
for i in $(seq 0 $((NUM_CLIENTS - 1))); do
    python examples/simple_federation/run_client.py \
        --id $i \
        --num_clients $NUM_CLIENTS \
        --server_address $SERVER_ADDR &
done

# Wait for all clients to finish
wait
kill $SERVER_PID 2>/dev/null
echo "Training complete"
```

---

## End-to-End Test Suites

The framework ships with three test scripts:

### run_local_test.py — Quick Sanity Test

```bash
# Tests single-client FL on a tiny subset of MNIST
# Completes in ~30 seconds
python run_local_test.py
```

Exercises:
- Server startup and client registration
- One round of FedAvg (single client)
- Model serialisation/deserialisation round-trip
- Heartbeat flow

### run_full_test_suite.py — Comprehensive E2E

```bash
# Runs all examples in sequence (takes ~20 minutes)
python run_full_test_suite.py \
    --test simple \      # or llm, ecg, decomfl
    --num_rounds 3 \
    --num_clients 3 \
    --timeout 300        # seconds per test
```

Exercises:
- All strategy types (FedAvg, DeComFL)
- Multiple client counts
- Streaming upload/download paths
- Heartbeat and stale update rejection
- Graceful shutdown sequence

### run_platform_e2e_test.py — Integration with Spring Backend

```bash
# Requires running Spring Boot backend
python run_platform_e2e_test.py \
    --backend_url http://localhost:8080 \
    --api_key your_api_key \
    --project_id test_project_123
```

Exercises:
- Backend-triggered FL server spawn
- WebSocket log streaming from Python server to frontend
- Project result persistence to the backend database (H2)
- Full round-trip through the platform API
