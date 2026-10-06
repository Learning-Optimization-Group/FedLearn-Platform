# framework/tests/test_simulation_costs.py
"""Cost accounting: exact bytes, FLOPs that match PyTorch's counter, and the simulated clock."""
from collections import OrderedDict

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch.utils.flop_counter import FlopCounterMode

from fedlearn.communication.safetensors_codec import save_safetensors
from fedlearn.simulation import costs
from fedlearn.simulation.workloads import LinearHead, MLPHead


def test_transfer_bytes_is_the_codec_size():
    p = OrderedDict(w=torch.zeros(2, 512), b=torch.zeros(2))
    want = len(save_safetensors([("w", np.zeros((2, 512), np.float32)), ("b", np.zeros(2, np.float32))]))
    assert costs.transfer_bytes(p) == want


@pytest.mark.parametrize("model, dims", [
    (LinearHead(16, 2), [(16, 2)]),
    (MLPHead(16, 8, 2), [(16, 8), (8, 2)]),
])
def test_analytic_flops_match_the_pytorch_counter(model, dims):
    x, y = torch.randn(30, 16), torch.zeros(30, dtype=torch.long)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    with FlopCounterMode(display=False) as fc:
        for _ in range(3):
            opt.zero_grad()
            F.cross_entropy(model(x), y).backward()
            opt.step()
    assert fc.get_total_flops() == costs.analytic_train_flops(dims, 30, 3)


def test_stage_clock_accumulates_and_resets():
    c = costs.StageClock()
    with c.stage("train"):
        sum(range(10_000))
    with c.stage("train"):
        pass
    snap = c.take()
    assert snap["wall_s"]["train"] > 0 and "train" in snap["cpu_s"]
    assert c.take() == {"wall_s": {}, "cpu_s": {}}


def test_finish_seconds_formula():
    net = {"uplink_mbps": 8.0, "downlink_mbps": 16.0, "latency_ms": 100.0, "device_gflops": 2.0}
    # 4e9 flops / 2e9 = 2 s, x slowdown 3 = 6 s; down 2e6 bytes = 1 s; up 1e6 bytes = 1 s; 2 x 0.1 s latency
    assert costs.finish_seconds(4e9, 3.0, 2_000_000, 1_000_000, net) == pytest.approx(8.2)


def test_server_flops_estimates_scale_with_the_rule():
    assert costs.server_flops_estimate("mean", 10, 100) < costs.server_flops_estimate("trimmed_mean", 10, 100)
    assert costs.server_flops_estimate("krum", 10, 100) == pytest.approx(10 * 10 * 100)


def test_peak_rss_is_positive():
    assert costs.peak_rss_bytes() > 1_000_000
