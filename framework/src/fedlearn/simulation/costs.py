"""Cost accounting for simulation runs: communication, compute, two clocks, memory.

Measured wall-clock (what this machine spent, mostly Python overhead) and the simulated deployment
clock (a model of a real deployment's round time) are kept separate and labelled in every record.
"""
from __future__ import annotations

import math
import resource
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Dict, List, Tuple

from fedlearn.communication.safetensors_codec import save_safetensors


def transfer_bytes(params) -> int:
    """Exact serialized size of one model transfer on the real safetensors wire."""
    return len(save_safetensors([(k, v.detach().cpu().float().numpy()) for k, v in params.items()]))


def analytic_train_flops(layer_dims: List[Tuple[int, int]], n_samples: int, epochs: int) -> int:
    """Matmul FLOPs of full training for a stack of linear layers (the count PyTorch's FlopCounterMode
    reports): forward and weight gradient per layer, plus the input gradient for every layer after
    the first (the first layer's input — the features — needs no gradient)."""
    per_epoch = 0
    for i, (fan_in, fan_out) in enumerate(layer_dims):
        f = 2 * n_samples * fan_in * fan_out
        per_epoch += 2 * f + (f if i > 0 else 0)
    return per_epoch * epochs


def server_flops_estimate(rule: str, n: int, d: int, iterations: int = 1) -> float:
    """Order-of-magnitude aggregation cost (labelled an estimate in records)."""
    if rule in ("mean", "weighted_mean"):
        return float(n * d)
    if rule in ("trimmed_mean", "median"):
        return float(n * d * max(1.0, math.log2(n)))
    if rule in ("krum", "multi_krum", "bulyan"):
        return float(n * n * d)
    return float(2 * n * d * iterations)  # centered_clip


class StageClock:
    """Accumulates wall-clock and CPU time per named stage until ``take`` is called."""

    def __init__(self):
        self.wall: Dict[str, float] = defaultdict(float)
        self.cpu: Dict[str, float] = defaultdict(float)

    @contextmanager
    def stage(self, name: str):
        w0, c0 = time.perf_counter(), time.process_time()
        try:
            yield
        finally:
            self.wall[name] += time.perf_counter() - w0
            self.cpu[name] += time.process_time() - c0

    def take(self) -> Dict[str, Dict[str, float]]:
        snap = {"wall_s": dict(self.wall), "cpu_s": dict(self.cpu)}
        self.wall.clear()
        self.cpu.clear()
        return snap


def finish_seconds(client_flops: float, slowdown: float, down_bytes: int, up_bytes: int, network: dict) -> float:
    """Simulated time for one client to download, train and upload."""
    compute = client_flops / (float(network["device_gflops"]) * 1e9) * slowdown
    down = down_bytes * 8.0 / (float(network["downlink_mbps"]) * 1e6)
    up = up_bytes * 8.0 / (float(network["uplink_mbps"]) * 1e6)
    return compute + down + up + 2.0 * float(network["latency_ms"]) / 1000.0


def peak_rss_bytes() -> int:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(r) if sys.platform == "darwin" else int(r) * 1024
