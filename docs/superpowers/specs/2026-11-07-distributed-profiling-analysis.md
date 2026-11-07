# Distributed Execution Profiling & Heterogeneous Analysis

## 1. Overview & Objective

This document formalizes the per-round timing breakdown profiling methodology and empirical distribution findings across heterogeneous federated client cohorts (Area 5 of the MSc Capstone reliability program).

Historically, the platform recorded total round duration without granular visibility into client vs. network vs. server execution phases. To diagnose compute bottlenecks, network serialization overheads, and straggler dynamics, the protocol and coordinator were extended to ingest a seven-component telemetry vector across each round:
1. `time_data_loading_s` (Client): Time spent loading, batching, and transforming data batches from disk/memory into tensors.
2. `time_forward_backward_s` (Client): Compute duration for local forward pass, loss calculation, and backpropagation.
3. `time_optimizer_step_s` (Client): Local parameter update duration (SGD / Adam step) and zeroing gradients.
4. `time_wire_encode_s` (Client): Wire serialization duration (safetensors format encoding).
5. `time_wire_decode_s` (Server): Server-side wire deserialization and tensor validation.
6. `time_aggregation_s` (Server): Server-side federated aggregation (e.g., FedAvg weighted averaging, coordinate-wise trimmed mean).
7. `time_eval_s` (Server): Global model evaluation against held-out validation data.

---

## 2. Experimental Setup

The benchmark is implemented in `framework/benchmarks/timing_profiling_benchmark.py` and evaluates a 3-round federated training run across a heterogeneous cohort of simulated edge nodes:
- **Desktop Node**: High compute capacity, 400 training samples, batch size 64, 2 local epochs (`compute_multiplier = 1.0`).
- **Edge Laptop**: Moderate compute capacity, 200 training samples, batch size 32, 2 local epochs (`compute_multiplier = 1.5`).
- **Mobile Device**: Low-power constrained device, 100 training samples, batch size 16, 2 local epochs (`compute_multiplier = 2.5`).

The training task uses a two-layer multi-layer perceptron (`BenchmarkClassifier`: 64-dim input, 128 hidden units, 2 classes) trained via federated averaging (`FedAvg`) with held-out validation evaluation.

---

## 3. Empirical Results

The empirical measurements across 3 federated rounds are summarized below:

| Telemetry Component | Mean Duration (s) | Std Dev (s) | Share of Round Budget (%) | Phase Domain |
|---|---|---|---|---|
| `time_data_loading_s` | 0.000915 | 0.000098 | 4.50% | Client Data Ingestion |
| `time_forward_backward_s` | 0.017824 | 0.001574 | 87.58% | Client Local Compute |
| `time_optimizer_step_s` | 0.000405 | 0.000071 | 1.99% | Client Local Compute |
| `time_wire_encode_s` | 0.000038 | 0.000005 | 0.19% | Client Wire Serialization |
| `time_wire_decode_s` | 0.000084 | 0.000015 | 0.41% | Server Ingestion |
| `time_aggregation_s` | 0.000052 | 0.000022 | 0.26% | Server Reduction |
| `time_eval_s` | 0.001033 | 0.000624 | 5.08% | Server Validation |
| **Total Mean Round** | **0.020351** | — | **100.00%** | — |

### High-Level Subsystem Breakdown:
- **Client-Side Compute & Ingestion**: **94.07%** of total execution budget (`time_forward_backward_s` + `time_data_loading_s` + `time_optimizer_step_s`).
- **Wire Overhead (Encode + Decode)**: **0.60%** of total execution budget (`time_wire_encode_s` + `time_wire_decode_s`).
- **Server Aggregation & Evaluation**: **5.34%** of total execution budget (`time_aggregation_s` + `time_eval_s`).

---

## 4. Bottleneck Identification & Architectural Implications

1. **Dominant Bottleneck**:
   The primary execution bottleneck across rounds is `time_forward_backward_s`, representing **87.58%** of the per-round budget. On-device backpropagation through deep architectures is the primary rate-limiting step, exacerbated by client heterogeneity (slower mobile/edge CPU clock and smaller batch sizes requiring more gradient evaluation steps).

2. **Safetensors Wire Efficiency**:
   Serialization via `safetensors` remains negligible (< 1% total duration). The memory-mapped, zero-copy wire format avoids Python pickle overhead and C++ serialization friction, confirming that communication latency on LAN/Wi-Fi is bounded by transport rather than payload serialization.

3. **Straggler Mitigation Strategy**:
   Because client compute dominates over 94% of the per-round time, synchronous federated learning is fundamentally constrained by the slowest participating node (the "straggler" effect). In heterogeneous deployments with physical mobile clients, deadline-based partial aggregation (`round_timeout_s`) and frozen-backbone training (`FROZEN_HEAD`) are critical:
   - Freezing backbone feature extractors and training only linear classification heads reduces the gradient parameter space and backpropagation depth by > 90%, shrinking `time_forward_backward_s` by an order of magnitude on mobile devices.
