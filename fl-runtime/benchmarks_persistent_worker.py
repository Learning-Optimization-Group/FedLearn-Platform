"""Persistent Worker Daemon vs One-Shot Process Benchmark.

Quantifies the empirical speedup and latency reduction achieved by retaining models
and dataloaders in memory across multiple federated learning rounds rather than
spawning fresh Python processes and reloading weights/datasets each round.

Writes structured JSON results to `fl-runtime/results/worker_persistence_benchmark.json`.
"""
from __future__ import annotations

import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import recipes
from worker_daemon import WorkerDaemonClient, WorkerDaemonServer


def run_benchmark(
    num_rounds: int = 5,
    model_type: str = "CNN",
    batch_size: int = 16,
    device: str = "cpu",
    output_path: Path = Path(_SCRIPT_DIR) / "results" / "worker_persistence_benchmark.json",
) -> Dict[str, Any]:
    print(f"\n{'='*70}")
    print(f"Running Persistent Worker vs One-Shot Process Benchmark")
    print(f"{'='*70}")
    print(f"Model: {model_type} | Rounds: {num_rounds} | Device: {device} | Batch Size: {batch_size}")

    dev = torch.device(device)
    recipe = recipes.get_recipe(model_type)

    # Synthetic dataset for identical compute comparison
    x = torch.randn(64, 3, 32, 32)
    y = torch.randint(0, 10, (64,))
    dataset = TensorDataset(x, y)

    # -------------------------------------------------------------------------
    # 1. ONE-SHOT SIMULATION: Fresh instantiation & model loading EVERY round
    # -------------------------------------------------------------------------
    print("\n--- 1. Measuring One-Shot Per-Round Mode ---")
    oneshot_rounds: List[Dict[str, Any]] = []
    oneshot_total_start = time.perf_counter()

    for r in range(1, num_rounds + 1):
        r_start = time.perf_counter()

        # Step A: Model construction & weights loading (reloaded every round)
        m_start = time.perf_counter()
        model = recipe.build_model(dev)
        m_load_ms = (time.perf_counter() - m_start) * 1000.0

        # Step B: Data loader creation
        d_start = time.perf_counter()
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        d_prep_ms = (time.perf_counter() - d_start) * 1000.0

        # Step C: Forward-backward local training compute
        c_start = time.perf_counter()
        model.train()
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        crit = nn.CrossEntropyLoss()
        total_loss = 0.0
        for bx, by in loader:
            bx, by = bx.to(dev), by.to(dev)
            opt.zero_grad()
            out = model(bx)
            loss = crit(out, by)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        compute_ms = (time.perf_counter() - c_start) * 1000.0

        total_round_ms = (time.perf_counter() - r_start) * 1000.0

        oneshot_rounds.append({
            "round": r,
            "model_load_ms": m_load_ms,
            "data_prep_ms": d_prep_ms,
            "compute_ms": compute_ms,
            "total_round_ms": total_round_ms,
            "loss": float(total_loss / len(loader)),
        })
        print(f"  Round {r}: total {total_round_ms:.2f} ms (model load: {m_load_ms:.2f} ms, compute: {compute_ms:.2f} ms)")

    oneshot_wall_clock_s = time.perf_counter() - oneshot_total_start

    # -------------------------------------------------------------------------
    # 2. PERSISTENT WORKER MODE: Loaded once in daemon memory, retained across rounds
    # -------------------------------------------------------------------------
    print("\n--- 2. Measuring Persistent Worker Daemon Mode ---")
    daemon_port = 8991
    server = WorkerDaemonServer(host="127.0.0.1", port=daemon_port, device=device)
    server.start_background()
    time.sleep(0.1)

    client = WorkerDaemonClient(base_url=f"http://127.0.0.1:{daemon_port}")
    persistent_rounds: List[Dict[str, Any]] = []
    persistent_total_start = time.perf_counter()

    # Step A: Initialize model once
    init_res = client.init_model(
        project_id="benchmark-persistent",
        run_id="run-persistent-1",
        model_type=model_type,
        dataset_name="synthetic",
    )
    initial_init_ms = init_res["model_load_ms"]
    print(f"  [Setup] Initial Model & Data Cache Load: {initial_init_ms:.2f} ms (done once)")

    # Step B: Execute rounds sequentially via persistent in-memory daemon
    for r in range(1, num_rounds + 1):
        r_start = time.perf_counter()
        res = client.execute_round(
            project_id="benchmark-persistent",
            run_id="run-persistent-1",
            round_number=r,
            learning_rate=0.01,
            local_epochs=1,
            batch_size=batch_size,
        )
        total_round_ms = (time.perf_counter() - r_start) * 1000.0

        persistent_rounds.append({
            "round": r,
            "model_load_ms": 0.0,  # Zero reload overhead!
            "compute_ms": res["compute_ms"],
            "total_round_ms": total_round_ms,
            "loss": res["loss"],
            "accuracy": res["accuracy"],
        })
        print(f"  Round {r}: total {total_round_ms:.2f} ms (model load: 0.00 ms, compute: {res['compute_ms']:.2f} ms)")

    persistent_wall_clock_s = time.perf_counter() - persistent_total_start
    server.shutdown()

    # -------------------------------------------------------------------------
    # 3. ANALYSIS & SUMMARY METRICS
    # -------------------------------------------------------------------------
    avg_oneshot_round_ms = sum(r["total_round_ms"] for r in oneshot_rounds) / num_rounds
    avg_persistent_round_ms = sum(r["total_round_ms"] for r in persistent_rounds) / num_rounds
    speedup_ratio = avg_oneshot_round_ms / max(avg_persistent_round_ms, 0.001)
    latency_reduction_pct = (1.0 - (avg_persistent_round_ms / avg_oneshot_round_ms)) * 100.0

    print(f"\n{'='*70}")
    print(f"BENCHMARK RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"  One-Shot Average Round Latency:    {avg_oneshot_round_ms:.2f} ms")
    print(f"  Persistent Average Round Latency:  {avg_persistent_round_ms:.2f} ms")
    print(f"  Per-Round Latency Reduction:       {latency_reduction_pct:.2f}%")
    print(f"  Speedup Ratio:                     {speedup_ratio:.2f}x")
    print(f"  One-Shot Total Wall-Clock:         {oneshot_wall_clock_s:.3f} s")
    print(f"  Persistent Total Wall-Clock:       {persistent_wall_clock_s:.3f} s")
    print(f"{'='*70}\n")

    report: Dict[str, Any] = {
        "meta": {
            "num_rounds": num_rounds,
            "model_type": model_type,
            "device": device,
            "system": platform.system(),
            "machine": platform.machine(),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
        },
        "summary": {
            "avg_oneshot_round_ms": avg_oneshot_round_ms,
            "avg_persistent_round_ms": avg_persistent_round_ms,
            "speedup_ratio": speedup_ratio,
            "latency_reduction_pct": latency_reduction_pct,
            "oneshot_wall_clock_s": oneshot_wall_clock_s,
            "persistent_wall_clock_s": persistent_wall_clock_s,
            "initial_model_load_ms": initial_init_ms,
        },
        "oneshot_rounds": oneshot_rounds,
        "persistent_rounds": persistent_rounds,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"Saved benchmark results to {output_path}")

    return report


if __name__ == "__main__":
    run_benchmark()
