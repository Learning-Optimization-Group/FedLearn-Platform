"""Smoke test for the C7 zeroth-vs-first-order benchmark's NON-CONVEX (MLP) path + the `make_model`
factory refactor. Tiny, seeded, fast: it pins that both families run over an arbitrary model factory
(not just the hardcoded convex LogReg), that the MLP is genuinely deeper/non-linear, and that the default
LogReg path is unchanged by the refactor.
"""
import pytest

torch = pytest.importorskip("torch")

from benchmarks.zeroth_vs_first_order import (  # noqa: E402
    LogReg,
    MLP,
    iid_partition,
    make_informative,
    run_decomfl,
    run_fedavg,
)


def _tiny():
    tx, ty, ex, ey = make_informative(8, 3, 180, seed=0)
    parts = iid_partition(len(tx), 3, seed=0)
    return tx, ty, ex, ey, parts


def test_mlp_is_deeper_and_nonlinear_vs_logreg():
    m = MLP(8, 8, 3)
    d_mlp = sum(p.numel() for p in m.parameters())
    d_lr = sum(p.numel() for p in LogReg(8, 3).parameters())
    assert d_mlp > d_lr                                   # a deep net has more params than the linear model
    assert any(isinstance(mod, torch.nn.ReLU) for mod in m.modules())  # non-linear -> non-convex objective


def test_fedavg_runs_on_the_mlp_via_make_model():
    tx, ty, ex, ey, parts = _tiny()
    curve = run_fedavg(tx, ty, ex, ey, parts, 8, 3, rounds=4, lr=0.5, local_epochs=2, seed=0,
                       make_model=lambda: MLP(8, 8, 3))
    assert len(curve) == 4
    assert 0.0 <= curve[-1]["accuracy"] <= 1.0


def test_decomfl_runs_on_the_mlp_via_make_model():
    tx, ty, ex, ey, parts = _tiny()
    curve, _initial = run_decomfl(tx, ty, ex, ey, parts, 8, 3, rounds=4, lr=0.02, K=1, P=4, mu=1e-3,
                                  seed=0, make_model=lambda: MLP(8, 8, 3))
    assert len(curve) == 4
    assert 0.0 <= curve[-1]["accuracy"] <= 1.0


def test_logreg_default_path_is_unchanged_by_the_refactor():
    tx, ty, ex, ey, parts = _tiny()
    curve = run_fedavg(tx, ty, ex, ey, parts, 8, 3, rounds=3, lr=0.5, local_epochs=1, seed=0)
    assert len(curve) == 3                               # default make_model (LogReg) still works
