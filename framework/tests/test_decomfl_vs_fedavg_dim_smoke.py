"""Tests for the DeComFL-vs-FedAvg dimension sweep harness.

The harness answers a production question: the phone can run the zeroth-order path on ANY
runtime (forward passes only), but the first-order path needs the ExecuTorch training
extension. So if DeComFL reaches FedAvg's quality on the frozen head, the ZO path is a
portable fallback; if it does not, first-order is mandatory. Getting that answer wrong in
either direction is expensive, so the harness itself is pinned here.

The load-bearing property is that this is a REAL DeComFL run — the framework's own strategy
and estimator, with the shared-seed invariant intact — not a local reimplementation that
would silently measure something else.
"""
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "benchmarks"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import decomfl_vs_fedavg_dim as H  # noqa: E402
from fedlearn.estimators import params as P  # noqa: E402


# --------------------------------------------------------------------------- model dimension

def test_linear_head_dim_is_the_production_1026():
    """The frozen resnet18 head is the model the phone actually trains. 512*2+2 = 1026."""
    m = H.head_model(feat_dim=512, n_classes=2, hidden=0, seed=0)
    assert P.num_trainable(m) == 1026


def test_hidden_width_is_the_dimension_knob():
    """d must vary with hidden width on a FIXED feature set — that is what isolates dimension
    from every other difference between arms."""
    dims = [P.num_trainable(H.head_model(feat_dim=512, n_classes=2, hidden=h, seed=0))
            for h in (0, 20, 200)]
    assert dims == sorted(dims)
    assert dims[0] < dims[1] < dims[2]
    # h=20: 512*20+20 (fc1) + 20*2+2 (fc2)
    assert dims[1] == 512 * 20 + 20 + 20 * 2 + 2


def test_declared_dim_matches_canonical_trainable_count():
    """The harness reports d from its own helper; if that ever drifts from the canonical
    requires_grad-filtered count, DeComFL's shared-seed perturbation would misalign."""
    for h in (0, 8, 64):
        m = H.head_model(feat_dim=512, n_classes=2, hidden=h, seed=0)
        assert H.model_dim(m) == P.num_trainable(m)


# --------------------------------------------------------------------------- byte accounting

def test_decomfl_bytes_are_dimension_free():
    """The entire claim of the paper. If this scales with d the harness is measuring a fake."""
    a = H.decomfl_bytes_per_round(K=1, P_=10, d=1026)
    b = H.decomfl_bytes_per_round(K=1, P_=10, d=10_000_000)
    assert a == b


def test_decomfl_bytes_scale_with_K_and_P():
    assert H.decomfl_bytes_per_round(K=2, P_=10, d=1026) > H.decomfl_bytes_per_round(K=1, P_=10, d=1026)
    assert H.decomfl_bytes_per_round(K=1, P_=20, d=1026) > H.decomfl_bytes_per_round(K=1, P_=10, d=1026)


def test_fedavg_bytes_scale_with_dimension():
    assert H.fedavg_bytes_per_round(10_000) == pytest.approx(4 * 10_000, rel=0.1)
    assert H.fedavg_bytes_per_round(20_000) > H.fedavg_bytes_per_round(10_000)


def test_byte_advantage_is_already_large_at_the_production_head():
    """I expected the ZO uplink win to be a large-d phenomenon and to be negligible at the
    1,026-parameter head. It is not: 80 B vs 4,104 B is a 51x advantage at the production size
    already, and it widens to ~500,000x at ResNet scale. Pinned so the report cannot understate
    it, and so the comparison is understood to turn on convergence, not on bandwidth."""
    zo = H.decomfl_bytes_per_round(K=1, P_=10, d=1026)
    assert zo == 80
    assert H.fedavg_bytes_per_round(1026) / zo == pytest.approx(51.3, rel=0.02)
    assert H.fedavg_bytes_per_round(11_000_000) / zo > 500_000


# --------------------------------------------------------------------------- the real algorithm

def _tiny_task(n=64, feat=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, feat, generator=g)
    y = (x[:, 0] + 0.5 * x[:, 1] > 0).long()
    return x, y


def test_decomfl_uses_the_framework_strategy_not_a_reimplementation():
    from fedlearn.server.decomfl_strategy import DeComFL
    assert H.DeComFL is DeComFL


def test_decomfl_uses_the_framework_estimator():
    from fedlearn.estimators.zeroth_order import ZerothOrderEstimator
    assert H.ZerothOrderEstimator is ZerothOrderEstimator


def test_one_decomfl_round_moves_the_global_model():
    x, y = _tiny_task()
    r = H.run_decomfl(train_x=x, train_y=y, test_x=x, test_y=y, feat_dim=8, n_classes=2,
                      hidden=0, clients=2, clients_per_round=2, alpha=10.0, rounds=3,
                      K=1, P_=4, lr=0.01, mu=1e-3, batch_size=16, seed=0)
    before = r["per_round"][0]["loss"]
    assert np.isfinite(before)
    assert len(r["per_round"]) == 3


def test_shared_seed_invariant_client_rebuild_matches_server():
    """DeComFL's correctness rests on every client regenerating the SAME z from the shared seed
    and thus rebuilding exactly the server's model. If the harness let those drift it would be
    benchmarking a broken algorithm and reporting it as DeComFL's quality."""
    x, y = _tiny_task()
    r = H.run_decomfl(train_x=x, train_y=y, test_x=x, test_y=y, feat_dim=8, n_classes=2,
                      hidden=0, clients=3, clients_per_round=3, alpha=10.0, rounds=4,
                      K=1, P_=4, lr=0.01, mu=1e-3, batch_size=16, seed=0,
                      check_rebuild=True)
    assert r["rebuild_max_abs_error"] == pytest.approx(0.0, abs=1e-6)


def test_decomfl_is_deterministic_under_seed():
    kw = dict(feat_dim=8, n_classes=2, hidden=0, clients=2, clients_per_round=2, alpha=10.0,
              rounds=3, K=1, P_=4, lr=0.01, mu=1e-3, batch_size=16, seed=0)
    x, y = _tiny_task()
    a = H.run_decomfl(train_x=x, train_y=y, test_x=x, test_y=y, **kw)
    b = H.run_decomfl(train_x=x, train_y=y, test_x=x, test_y=y, **kw)
    assert [p["loss"] for p in a["per_round"]] == [p["loss"] for p in b["per_round"]]


def test_different_seeds_give_different_trajectories():
    x, y = _tiny_task()
    kw = dict(train_x=x, train_y=y, test_x=x, test_y=y, feat_dim=8, n_classes=2, hidden=0,
              clients=2, clients_per_round=2, alpha=10.0, rounds=3, K=1, P_=4, lr=0.01,
              mu=1e-3, batch_size=16)
    a = H.run_decomfl(seed=0, **kw)
    b = H.run_decomfl(seed=1, **kw)
    assert [p["loss"] for p in a["per_round"]] != [p["loss"] for p in b["per_round"]]


def test_fedavg_arm_learns_the_tiny_task():
    """A control: if the first-order arm cannot learn a linearly separable task the comparison
    is meaningless regardless of what DeComFL does."""
    x, y = _tiny_task(n=256)
    r = H.run_fedavg(train_x=x, train_y=y, test_x=x, test_y=y, feat_dim=8, n_classes=2,
                     hidden=0, clients=2, clients_per_round=2, alpha=10.0, rounds=30,
                     local_epochs=1, lr=0.1, batch_size=32, seed=0)
    assert r["per_round"][-1]["auc"] > 0.9


def test_both_arms_start_from_the_same_init():
    """An unshared init would confound every between-arm difference — the exact mistake the
    FedLoRA comparison had to be rebuilt to avoid."""
    x, y = _tiny_task()
    kw = dict(train_x=x, train_y=y, test_x=x, test_y=y, feat_dim=8, n_classes=2, hidden=0,
              clients=2, clients_per_round=2, alpha=10.0, seed=0, batch_size=16)
    a = H.run_decomfl(rounds=1, K=1, P_=4, lr=0.0, mu=1e-3, **kw)
    b = H.run_fedavg(rounds=1, local_epochs=1, lr=0.0, **kw)
    assert a["init_sha"] == b["init_sha"]


def test_zero_lr_leaves_the_model_unchanged():
    """Pins that the measured movement comes from the update rule, not from bookkeeping."""
    x, y = _tiny_task()
    r = H.run_decomfl(train_x=x, train_y=y, test_x=x, test_y=y, feat_dim=8, n_classes=2,
                      hidden=0, clients=2, clients_per_round=2, alpha=10.0, rounds=3,
                      K=1, P_=4, lr=0.0, mu=1e-3, batch_size=16, seed=0)
    losses = [p["loss"] for p in r["per_round"]]
    assert losses[0] == pytest.approx(losses[-1], abs=1e-9)


# --------------------------------------------------------------------------- gradient alignment

def test_alignment_probe_is_high_in_low_dimension():
    """cos(g_hat, g_true) ~ sqrt(P/d): with P comparable to d the estimate should be well aligned."""
    x, y = _tiny_task()
    m = H.head_model(feat_dim=8, n_classes=2, hidden=0, seed=0)
    cos = H.gradient_alignment(m, x, y, P_=200, mu=1e-4, seed=0)
    assert cos > 0.5


def test_alignment_probe_collapses_in_high_dimension():
    """The mechanism the whole experiment is about, measured rather than asserted."""
    x, y = _tiny_task(feat=8)
    lo = H.gradient_alignment(H.head_model(feat_dim=8, n_classes=2, hidden=0, seed=0),
                              x, y, P_=8, mu=1e-4, seed=0)
    xb, yb = _tiny_task(feat=8)
    hi = H.gradient_alignment(H.head_model(feat_dim=8, n_classes=2, hidden=400, seed=0),
                              xb, yb, P_=8, mu=1e-4, seed=0)
    assert hi < lo


def test_alignment_is_bounded():
    x, y = _tiny_task()
    m = H.head_model(feat_dim=8, n_classes=2, hidden=0, seed=0)
    c = H.gradient_alignment(m, x, y, P_=16, mu=1e-4, seed=0)
    assert -1.0 <= c <= 1.0


# --------------------------------------------------------------------------- partitioning / io

def test_partition_covers_every_example_exactly_once():
    y = np.array([0] * 50 + [1] * 50)
    parts = H.partition(y, num_clients=5, alpha=1.0, seed=0)
    allidx = np.concatenate(parts)
    assert sorted(allidx.tolist()) == list(range(100))


def test_partition_is_deterministic():
    y = np.array([0] * 50 + [1] * 50)
    a = H.partition(y, num_clients=5, alpha=1.0, seed=0)
    b = H.partition(y, num_clients=5, alpha=1.0, seed=0)
    assert all(np.array_equal(x, z) for x, z in zip(a, b))


def test_summary_records_dimension_and_both_byte_columns():
    x, y = _tiny_task()
    r = H.run_decomfl(train_x=x, train_y=y, test_x=x, test_y=y, feat_dim=8, n_classes=2,
                      hidden=0, clients=2, clients_per_round=2, alpha=10.0, rounds=2,
                      K=1, P_=4, lr=0.01, mu=1e-3, batch_size=16, seed=0)
    for k in ("d", "bytes_per_client_round", "cum_bytes", "arm", "rounds"):
        assert k in r, k
    assert r["arm"] == "DeComFL"


def test_emit_writes_one_file_per_cell(tmp_path):
    rec = {"arm": "DeComFL", "d": 1026, "seed": 0, "hidden": 0, "per_round": []}
    p = H._emit(str(tmp_path), rec)
    assert os.path.exists(p)
    assert "DeComFL" in os.path.basename(p) and "1026" in os.path.basename(p)


def test_emit_does_not_collide_across_cells(tmp_path):
    a = H._emit(str(tmp_path), {"arm": "DeComFL", "d": 1026, "seed": 0, "hidden": 0, "per_round": []})
    b = H._emit(str(tmp_path), {"arm": "DeComFL", "d": 1026, "seed": 1, "hidden": 0, "per_round": []})
    c = H._emit(str(tmp_path), {"arm": "FedAvg", "d": 1026, "seed": 0, "hidden": 0, "per_round": []})
    assert len({a, b, c}) == 3


# --------------------------------------------------------------- device independence

def test_auc_accepts_non_cpu_dtype_restricted_tensors():
    """MPS cannot represent float64, and the AUC path used .double(). Evaluating on the compute
    device therefore crashed the whole MPS run with 'Cannot convert a MPS Tensor to float64'.
    Metrics are now computed on CPU regardless of device, which also makes them bit-identical
    across devices — a property a cross-device benchmark needs anyway."""
    logits = torch.tensor([[2.0, -1.0], [-1.0, 2.0], [0.5, 0.1], [-2.0, 1.0]])
    y = torch.tensor([0, 1, 0, 1])
    assert 0.0 <= H._auc(logits, y) <= 1.0


def test_evaluate_returns_cpu_floats_not_device_tensors():
    m = H.head_model(feat_dim=8, n_classes=2, hidden=0, seed=0)
    x, y = _tiny_task()
    out = H._evaluate(m, x, y)
    for k in ("loss", "auc", "acc"):
        assert isinstance(out[k], float), f"{k} must be a plain float, got {type(out[k])}"


def test_evaluate_is_device_independent_when_cuda_absent():
    """The CPU path must be unchanged by the fix — existing numbers have to reproduce exactly."""
    m = H.head_model(feat_dim=8, n_classes=2, hidden=0, seed=0)
    x, y = _tiny_task()
    a = H._evaluate(m, x, y)
    b = H._evaluate(m, x, y)
    assert a == b


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="no MPS device")
def test_evaluate_runs_on_mps():
    """The REAL regression test. The three tests above pass on CPU whether or not the bug is
    present, so they are vacuous for this defect — exactly the failure mode this campaign hit
    once already. This one actually exercises the crash: _evaluate on an MPS tensor raised
    'Cannot convert a MPS Tensor to float64' and killed the entire MPS sweep."""
    m = H.head_model(feat_dim=8, n_classes=2, hidden=0, seed=0).to("mps")
    x, y = _tiny_task()
    out = H._evaluate(m, x.to("mps"), y.to("mps"))
    assert 0.0 <= out["auc"] <= 1.0
    assert np.isfinite(out["loss"])


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="no MPS device")
def test_decomfl_runs_end_to_end_on_mps():
    x, y = _tiny_task()
    r = H.run_decomfl(train_x=x, train_y=y, test_x=x, test_y=y, feat_dim=8, n_classes=2,
                      hidden=0, clients=2, clients_per_round=2, alpha=10.0, rounds=3,
                      K=1, P_=4, lr=0.01, mu=1e-3, batch_size=16, seed=0, device="mps")
    assert len(r["per_round"]) == 3


# ------------------------------------------------- partition robustness at severe non-IID

def test_partition_gives_every_client_at_least_one_example():
    """At severe non-IID a Dirichlet draw can leave a client with ZERO examples, which makes
    DataLoader raise `num_samples should be a positive integer value, but got num_samples=0`
    and kills the run. Found when alpha=0.05 was used for the first time -- the harness had
    only ever been run at alpha >= 0.3, so the failure had never surfaced."""
    y = np.array([0] * 700 + [1] * 700)
    for alpha in (0.01, 0.05, 0.1):
        parts = H.partition(y, num_clients=20, alpha=alpha, seed=0)
        sizes = [len(p) for p in parts]
        assert min(sizes) >= 1, f"alpha={alpha} produced an empty client: {sizes}"


def test_partition_still_covers_every_example_exactly_once_at_severe_skew():
    y = np.array([0] * 700 + [1] * 700)
    parts = H.partition(y, num_clients=20, alpha=0.05, seed=0)
    allidx = np.concatenate(parts)
    assert sorted(allidx.tolist()) == list(range(1400))


def test_partition_still_produces_label_skew_at_low_alpha():
    """The minimum-size guarantee must not silently turn the partition into a uniform split --
    low alpha must still produce genuinely skewed label distributions."""
    y = np.array([0] * 700 + [1] * 700)
    def skew(alpha):
        parts = H.partition(y, num_clients=20, alpha=alpha, seed=0)
        return np.mean([abs((y[p] == 0).mean() - 0.5) for p in parts if len(p)])
    assert skew(0.05) > skew(10.0), "low alpha should be more label-skewed than high alpha"
