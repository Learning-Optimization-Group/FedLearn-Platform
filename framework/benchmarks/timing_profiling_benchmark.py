"""Fine-grained per-round timing breakdown benchmark across simulated heterogeneous clients.

Area 5 requirement:
- Profile distributed execution across simulated heterogeneous clients (desktop, edge laptop, mobile).
- Record the 7 canonical timing breakdown components:
    * time_data_loading_s (client)
    * time_forward_backward_s (client)
    * time_optimizer_step_s (client)
    * time_wire_encode_s (client)
    * time_wire_decode_s (server)
    * time_aggregation_s (server)
    * time_eval_s (server)
- Report distribution statistics (mean, std, percentage share) and identify the dominant bottleneck.
"""
from __future__ import annotations

import argparse
import copy
from collections import OrderedDict
import json
import math
import os
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, "..", "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from fedlearn.communication.serializer import chunks_to_parameters, state_dict_to_safetensors
from fedlearn.server.coordinator import FLCoordinator
from fedlearn.server.strategy import FedAvg


class BenchmarkClassifier(nn.Module):
    """Two-layer MLP classifier for profiling federated execution."""

    def __init__(self, in_features: int = 64, hidden_dim: int = 128, num_classes: int = 2):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.relu(self.fc1(x)))


def _generate_synthetic_dataset(num_samples: int, in_features: int = 64, num_classes: int = 2, seed: int = 42):
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(num_samples, in_features, generator=generator, dtype=torch.float32)
    y = torch.randint(0, num_classes, (num_samples,), generator=generator, dtype=torch.long)
    return TensorDataset(x, y)


def simulate_client_training(
    client_id: str,
    global_params: OrderedDict[str, torch.Tensor],
    dataloader: DataLoader,
    epochs: int,
    compute_multiplier: float = 1.0,
) -> tuple[OrderedDict[str, torch.Tensor], dict[str, float], bytes]:
    """Simulate on-device training and measure client-side timing breakdown."""
    model = BenchmarkClassifier()
    model.load_state_dict(global_params)
    model.train()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    total_data_loading_s = 0.0
    total_forward_backward_s = 0.0
    total_optimizer_step_s = 0.0

    for _ in range(epochs):
        t_batch_start = time.perf_counter()
        for batch_x, batch_y in dataloader:
            t_data_end = time.perf_counter()
            total_data_loading_s += (t_data_end - t_batch_start)

            t_fwd_start = time.perf_counter()
            optimizer.zero_grad()
            out = model(batch_x)
            loss = criterion(out, batch_y)
            loss.backward()

            # Simulate heterogeneous edge/mobile compute scaling if multiplier > 1.0
            if compute_multiplier > 1.0:
                burn_cycles = int(50000 * (compute_multiplier - 1.0))
                _ = sum(i * 0.0001 for i in range(burn_cycles))

            t_fwd_end = time.perf_counter()
            total_forward_backward_s += (t_fwd_end - t_fwd_start)

            t_opt_start = time.perf_counter()
            optimizer.step()
            t_opt_end = time.perf_counter()
            total_optimizer_step_s += (t_opt_end - t_opt_start)

            t_batch_start = time.perf_counter()

    updated_params = OrderedDict((k, v.detach().clone()) for k, v in model.state_dict().items())

    # Wire encode
    t_enc_start = time.perf_counter()
    num_samples = len(dataloader.dataset)
    wire_bytes = state_dict_to_safetensors(updated_params, num_examples=num_samples)
    wire_encode_s = time.perf_counter() - t_enc_start

    client_metrics = {
        "time_data_loading_s": round(total_data_loading_s, 6),
        "time_forward_backward_s": round(total_forward_backward_s, 6),
        "time_optimizer_step_s": round(total_optimizer_step_s, 6),
        "time_wire_encode_s": round(wire_encode_s, 6),
    }

    return updated_params, client_metrics, wire_bytes


def run_benchmark(rounds: int = 3, output_path: str | None = None, verbose: bool = True) -> dict:
    torch.manual_seed(1337)

    # 1. Setup heterogeneous client configurations
    client_configs = {
        "desktop_node": {
            "num_samples": 400,
            "batch_size": 64,
            "epochs": 2,
            "compute_multiplier": 1.0,
            "seed": 101,
        },
        "edge_laptop": {
            "num_samples": 200,
            "batch_size": 32,
            "epochs": 2,
            "compute_multiplier": 1.5,
            "seed": 202,
        },
        "mobile_device": {
            "num_samples": 100,
            "batch_size": 16,
            "epochs": 2,
            "compute_multiplier": 2.5,
            "seed": 303,
        },
    }

    client_dataloaders = {}
    for cid, cfg in client_configs.items():
        ds = _generate_synthetic_dataset(cfg["num_samples"], seed=cfg["seed"])
        client_dataloaders[cid] = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True)

    # Held-out evaluation dataset
    test_ds = _generate_synthetic_dataset(150, seed=999)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    eval_model = BenchmarkClassifier()

    def evaluate_fn(server_round: int, parameters: OrderedDict[str, torch.Tensor]):
        eval_model.load_state_dict(parameters)
        eval_model.eval()
        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for bx, by in test_loader:
                out = eval_model(bx)
                total_loss += criterion(out, by).item() * len(by)
                preds = out.argmax(dim=-1)
                correct += (preds == by).sum().item()
                total += len(by)
        avg_loss = total_loss / max(total, 1)
        acc = correct / max(total, 1)
        return avg_loss, {"accuracy": acc}

    init_model = BenchmarkClassifier()
    initial_params = OrderedDict((k, v.detach().clone()) for k, v in init_model.state_dict().items())

    strategy = FedAvg(
        initial_parameters=initial_params,
        evaluate_fn=evaluate_fn,
        min_fit_clients=len(client_configs),
        clients_per_round=len(client_configs),
    )

    coordinator = FLCoordinator(
        strategy=strategy,
        min_clients_for_aggregation=len(client_configs),
        clients_per_round=len(client_configs),
        num_rounds=rounds,
        async_round_completion=False,
    )
    coordinator.set_initial_parameters(initial_params)

    # 2. Run federated rounds with full telemetry breakdown
    round_records = []

    if verbose:
        print("=" * 80)
        print("DISTRIBUTED EXECUTION TIMING PROFILING BENCHMARK (Area 5)")
        print(f"Cohorts: {list(client_configs.keys())} | Rounds: {rounds}")
        print("=" * 80)

    for r in range(1, rounds + 1):
        coordinator.start_round(r)
        current_global = coordinator.get_global_model_params()

        # Simulated distributed clients
        for cid, cfg in client_configs.items():
            loader = client_dataloaders[cid]
            params, client_metrics, wire_bytes = simulate_client_training(
                cid, current_global, loader, cfg["epochs"], cfg["compute_multiplier"]
            )

            # Server wire decode simulation
            t_dec_start = time.perf_counter()
            decoded_params, n_examples = chunks_to_parameters(wire_bytes)
            wire_decode_s = time.perf_counter() - t_dec_start

            coordinator.submit_client_update(
                client_id=cid,
                params=decoded_params,
                num_examples=n_examples,
                trained_on_round=r,
                client_metrics=client_metrics,
                wire_decode_s=wire_decode_s,
            )

        coordinator.wait_for_round_to_complete()
        round_timings = coordinator.get_latest_timings(r)
        round_eval = coordinator.latest_metrics

        rec = {
            "round": r,
            "metrics": round_eval,
            "timings": round_timings,
        }
        round_records.append(rec)

        if verbose:
            t = round_timings
            tot = sum(t.values())
            print(
                f"Round {r:02d} | Total: {tot:.4f}s | "
                f"DataLoad: {t['time_data_loading_s']:.4f}s | "
                f"FwdBwd: {t['time_forward_backward_s']:.4f}s | "
                f"OptStep: {t['time_optimizer_step_s']:.4f}s | "
                f"WireEnc: {t['time_wire_encode_s']:.4f}s | "
                f"WireDec: {t['time_wire_decode_s']:.4f}s | "
                f"Agg: {t['time_aggregation_s']:.4f}s | "
                f"Eval: {t['time_eval_s']:.4f}s"
            )

    # 3. Compute distributions across rounds
    timing_keys = [
        "time_data_loading_s",
        "time_forward_backward_s",
        "time_optimizer_step_s",
        "time_wire_encode_s",
        "time_wire_decode_s",
        "time_aggregation_s",
        "time_eval_s",
    ]

    stats = {}
    total_mean_duration = 0.0

    for k in timing_keys:
        vals = [r_rec["timings"][k] for r_rec in round_records]
        mean_v = sum(vals) / len(vals)
        var_v = sum((x - mean_v) ** 2 for x in vals) / len(vals)
        std_v = math.sqrt(var_v)
        total_mean_duration += mean_v
        stats[k] = {
            "mean_s": round(mean_v, 6),
            "std_s": round(std_v, 6),
            "min_s": round(min(vals), 6),
            "max_s": round(max(vals), 6),
        }

    for k in timing_keys:
        pct = (stats[k]["mean_s"] / max(total_mean_duration, 1e-9)) * 100.0
        stats[k]["percent_of_round"] = round(pct, 2)

    # 4. Bottleneck analysis
    dominant_key = max(timing_keys, key=lambda k: stats[k]["mean_s"])
    dominant_pct = stats[dominant_key]["percent_of_round"]

    client_compute_pct = round(
        stats["time_data_loading_s"]["percent_of_round"]
        + stats["time_forward_backward_s"]["percent_of_round"]
        + stats["time_optimizer_step_s"]["percent_of_round"],
        2,
    )
    wire_pct = round(
        stats["time_wire_encode_s"]["percent_of_round"] + stats["time_wire_decode_s"]["percent_of_round"],
        2,
    )
    server_pct = round(
        stats["time_aggregation_s"]["percent_of_round"] + stats["time_eval_s"]["percent_of_round"],
        2,
    )

    bottleneck_analysis = {
        "dominant_component": dominant_key,
        "dominant_component_share_pct": dominant_pct,
        "client_compute_total_pct": client_compute_pct,
        "wire_overhead_total_pct": wire_pct,
        "server_overhead_total_pct": server_pct,
        "summary": (
            f"The primary execution bottleneck is '{dominant_key}' accounting for {dominant_pct}% of the per-round "
            f"budget. Aggregated client-side compute represents {client_compute_pct}%, wire serialization/deserialization "
            f"represents {wire_pct}%, and server aggregation/eval represents {server_pct}%."
        ),
    }

    results = {
        "metadata": {
            "benchmark_name": "timing_profiling_benchmark",
            "model_type": "BenchmarkClassifier (2-layer MLP)",
            "num_clients": len(client_configs),
            "cohort_profiles": {cid: cfg["compute_multiplier"] for cid, cfg in client_configs.items()},
            "num_rounds": rounds,
        },
        "per_round_records": round_records,
        "component_statistics": stats,
        "total_round_duration_mean_s": round(total_mean_duration, 6),
        "bottleneck_analysis": bottleneck_analysis,
    }

    if verbose:
        print("\n" + "=" * 80)
        print("SUMMARY STATISTICS ACROSS ROUNDS")
        print("=" * 80)
        print(f"{'Component':<28} | {'Mean (s)':<10} | {'Std (s)':<10} | {'Share (%)':<10}")
        print("-" * 65)
        for k in timing_keys:
            st = stats[k]
            print(f"{k:<28} | {st['mean_s']:<10.6f} | {st['std_s']:<10.6f} | {st['percent_of_round']:<9.2f}%")
        print("-" * 65)
        print(f"Total Mean Round Duration: {total_mean_duration:.6f}s")
        print(f"Dominant Bottleneck: {dominant_key} ({dominant_pct}%)")
        print("=" * 80)

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        if verbose:
            print(f"Results saved to: {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=3, help="Number of federated rounds to profile")
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(_HERE, "results", "timing_profiling_benchmark_results.json"),
        help="Path to save benchmark JSON results",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress console output")
    args = parser.parse_args()

    run_benchmark(rounds=args.rounds, output_path=args.output, verbose=not args.quiet)


if __name__ == "__main__":
    main()
