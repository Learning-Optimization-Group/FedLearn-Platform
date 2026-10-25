"""Optional backend integration checks against real ByzFL, using synthetic inputs only."""
from collections import OrderedDict
import importlib
import math
import subprocess
import sys

import pytest
import torch

from fedlearn.server.composed_strategy import ComposedStrategy
from fedlearn.simulation.config import ConfigError, resolve
from fedlearn.simulation.engine import run


def strategy(**kwargs):
    cls = importlib.import_module("fedlearn.simulation.byzfl_strategy").ByzFLStrategy
    return cls(OrderedDict(w=torch.zeros(1)), **kwargs)


def uploads(values, global_value=0.):
    return [(str(i), OrderedDict(w=torch.tensor([global_value + float(x)])), 1)
            for i, x in enumerate(values)]


@pytest.mark.parametrize("rule,expected", [("mean", 25.5), ("median", 1.5), ("trimmed_mean", 1.5)])
def test_real_backend_reduces_hand_computed_fixture(rule, expected):
    s = strategy(rule=rule, rule_params={"trim_ratio": 0.25})
    out = s.aggregate_fit(1, uploads([0, 1, 2, 99]))
    assert float(out["w"]) == pytest.approx(expected)


@pytest.mark.parametrize("rule,z", [("mean", 0.), ("mean", 1.3),
                                  ("median", 0.), ("trimmed_mean", 0.), ("trimmed_mean", 1.3)])
def test_baselines_match_native_across_rounds(rule, z):
    kw = dict(rule=rule, rule_params={"trim_ratio": 0.25}, clip_norm=0.5,
              central_noise_multiplier=z, noise_seed=7, clients_per_round=4)
    external = strategy(**kw)
    native = ComposedStrategy(OrderedDict(w=torch.zeros(1)), **kw)
    for round_id in range(1, 4):
        data = uploads([0.1, -0.3, 0.8, 4.], global_value=round_id * 0.1)
        a, b = external.aggregate_fit(round_id, data), native.aggregate_fit(round_id, data)
        assert torch.allclose(a["w"], b["w"], atol=1e-7, rtol=1e-6)
        assert external.last_round_stats["noise_std"] == native.last_round_stats["noise_std"]


def test_private_center_uses_previous_noisy_release_and_preview_is_read_only():
    kw = dict(rule="centered_clip", rule_params={"tau": 1., "iterations": 1},
              clip_norm=10., central_noise_multiplier=5., noise_seed=7, clients_per_round=3)
    s, control = strategy(**kw), strategy(**kw)
    data = uploads([0, 2, 4])
    assert float(s.preview(data)["w"]) == pytest.approx(2 / 3)
    first = s.aggregate_fit(1, data)
    control.aggregate_fit(1, data)
    center = float(first["w"])
    assert abs(center - 2 / 3) > 0.1
    second_data = uploads([0, 0, 0], global_value=center)
    expected_preview = center + center - max(-1., min(center, 1.))
    assert float(s.preview(second_data)["w"]) == pytest.approx(expected_preview, abs=1e-6)
    assert float(s.preview(second_data)["w"]) == pytest.approx(expected_preview, abs=1e-6)
    second = s.aggregate_fit(2, second_data)
    second_control = control.aggregate_fit(2, second_data)
    assert torch.equal(second["w"], second_control["w"])
    assert s.last_round_stats["sensitivity"] == pytest.approx(2 / 3)
    assert s.last_round_stats["noise_std"] == pytest.approx(10 / 3)


def test_centered_clipping_handles_zero_residuals_and_failed_rounds():
    s = strategy(rule="centered_clip", rule_params={"tau": 1.})
    assert torch.equal(s.aggregate_fit(1, uploads([0, 0]))["w"], torch.zeros(1))
    assert s.aggregate_fit(2, []) is None
    assert torch.isfinite(s.aggregate_fit(3, uploads([0, 0]))["w"]).all()


def test_bucketing_is_reproducible_without_advancing_global_rng():
    kw = dict(rule="mean", bucket_size=2, bucket_seed=13)
    a, b = strategy(**kw), strategy(**kw)
    data = uploads([0, 0, 0, 0, 100])
    before = torch.random.get_rng_state().clone()
    preview = a.preview(data)
    actual = a.aggregate_fit(1, data)
    assert torch.equal(torch.random.get_rng_state(), before)
    assert torch.equal(actual["w"], b.aggregate_fit(1, data)["w"])
    assert torch.equal(preview["w"], actual["w"])
    assert any(float(actual["w"]) == pytest.approx(x) for x in (100 / 6, 100 / 3))


@pytest.mark.parametrize("kwargs", [
    {"rule": "median"},
    {"rule": "centered_clip", "rule_params": {"iterations": 2}},
    {"rule": "mean", "bucket_size": 2},
])
def test_private_adapter_refuses_unsupported_bounds(kwargs):
    with pytest.raises(ValueError):
        strategy(clip_norm=1., central_noise_multiplier=1., clients_per_round=4, **kwargs)


def test_private_adapter_refuses_partial_cohort():
    s = strategy(rule="mean", clip_norm=1., central_noise_multiplier=1., clients_per_round=4)
    assert s.aggregate_fit(1, uploads([1, 2, 3])) is None
    assert s.last_round_stats["failed"]


def synthetic_config(tmp_path, rule="mean", privacy="none", bucket_size=1):
    path = tmp_path / "synthetic.pt"
    g = torch.Generator().manual_seed(12)
    x = torch.randn(80, 4, generator=g)
    torch.save({"train_x": x, "train_y": (x[:, 0] > 0).long(),
                "test_x": x[:20].clone(), "test_y": (x[:20, 0] > 0).long(),
                "feat_dim": 4, "n_classes": 2}, path)
    return {"seed": 23, "workload": {"features_path": str(path)},
            "federation": {"num_clients": 4, "rounds": 2},
            "client": {"local_epochs": 1, "lr": 0.1},
            "server": {"rule": rule, "clip_norm": 0.5,
                       "rule_params": {"tau": 0.2},
                       "external_aggregation": {"library": "byzfl", "bucket_size": bucket_size}},
            "privacy": {"mode": privacy, "epsilon": 4.}}


def test_engine_records_actual_backend_and_center_sensitivity(tmp_path):
    cfg = synthetic_config(tmp_path, rule="centered_clip", privacy="central")
    first, second = run(cfg), run(cfg)
    assert first["meta"]["aggregation_backend"]["version"] == "0.0.11"
    assert first["meta"]["aggregation_backend"]["center"] == "previous_released_delta"
    assert first["meta"]["config"]["server"]["rule"] == "centered_clip"
    assert not first["meta"]["refusals"]
    assert first["meta"]["final_digest"] == second["meta"]["final_digest"]
    assert all(r["privacy"]["sensitivity"] == pytest.approx(0.1) for r in first["rounds"])
    assert 0 < first["meta"]["accounted_epsilon"] <= 4. + 1e-6


@pytest.mark.parametrize("change", ["oracle", "bucketing", "iterations", "local", "dropout"])
def test_config_rejects_unqualified_private_combinations(tmp_path, change):
    cfg = synthetic_config(tmp_path, rule="centered_clip", privacy="central")
    if change == "oracle":
        cfg["behaviors"] = [{"type": "ipm", "fraction": 0.25}]
    elif change == "bucketing":
        cfg["server"]["external_aggregation"]["bucket_size"] = 2
    elif change == "iterations":
        cfg["server"]["rule_params"]["iterations"] = 2
    elif change == "local":
        cfg["privacy"]["mode"] = "local"
    else:
        cfg["behaviors"] = [{"type": "dropout", "fraction": 0.25}]
    with pytest.raises(ConfigError):
        resolve(cfg)


def test_native_engine_does_not_import_optional_library():
    result = subprocess.run([sys.executable, "-c", "import sys; import fedlearn.simulation.engine; "
                             "assert 'byzfl' not in sys.modules"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_existing_effusion_harness_accepts_explicit_backend_without_changing_defaults():
    import run_effusion_federated as effusion
    kw = dict(clients=80, features_path="synthetic-unused", report_sha256="0" * 64)
    native = effusion.config("control", 23, **kw)
    assert "external_aggregation" not in native["server"]
    external = effusion.config("external", 23, rule="centered_clip",
                               external_aggregation={"library": "byzfl", "bucket_size": 1}, **kw)
    assert external["server"]["external_aggregation"]["library"] == "byzfl"
    assert external["server"]["rule"] == "centered_clip"


@pytest.mark.parametrize("value", [0., -1., math.inf, math.nan])
def test_adapter_rejects_invalid_center_radius(value):
    with pytest.raises(ValueError):
        strategy(rule="centered_clip", rule_params={"tau": value})


def test_one_step_noise_uses_both_clipping_bounds():
    s = strategy(rule="centered_clip", rule_params={"tau": .2}, clip_norm=.05,
                 central_noise_multiplier=3., clients_per_round=4)
    s.aggregate_fit(1, uploads([1, -1, 2, -2]))
    assert s.last_round_stats["sensitivity"] == pytest.approx(.025)
    assert s.last_round_stats["noise_std"] == pytest.approx(.075)


def test_inner_clipping_diagnostics_are_distinct_and_preview_does_not_replace_them():
    import copy
    s = strategy(rule="centered_clip", rule_params={"tau": 1.}, clip_norm=10.)
    s.aggregate_fit(1, uploads([0, 2, 4]))
    diagnostics = s.last_round_stats["aggregation_diagnostics"]
    assert s.last_round_stats["clip_rate"] == 0
    assert diagnostics["inner_clip_rate"] == pytest.approx(2 / 3)
    assert diagnostics["center_norm"] == 0
    assert diagnostics["mean_input_norm"] == pytest.approx(2.)
    assert diagnostics["research_only"] is True
    previous = copy.deepcopy(s.last_round_stats)
    s.preview(uploads([1, 1, 1]))
    assert s.last_round_stats == previous


def test_bucket_diagnostics_report_actual_contamination(tmp_path):
    cfg = synthetic_config(tmp_path, rule="median", bucket_size=2)
    cfg["behaviors"] = [{"type": "sign_flip_scale", "fraction": .25, "scale": -4.}]
    record = run(cfg)
    for line in record["rounds"]:
        diagnostics = line["aggregation_diagnostics"]
        assert diagnostics["bucket_count"] == 2
        assert diagnostics["contaminated_bucket_count"] == 1
        assert diagnostics["contaminated_bucket_fraction"] == .5
        assert "bucket_members_by_input_index" not in diagnostics


@pytest.mark.parametrize("clip,tau", [(.05, .2), (.2, .05), (.05, .05)])
def test_neighboring_vector_uploads_respect_one_step_composed_bound(clip, tau):
    cls = importlib.import_module("fedlearn.simulation.byzfl_strategy").ByzFLStrategy
    g = torch.Generator().manual_seed(24)
    for _ in range(8):
        s = cls(OrderedDict(w=torch.zeros(3)), rule="centered_clip", clip_norm=clip,
                rule_params={"tau": tau})
        s._released_center = torch.randn(3, generator=g) * .1
        values = torch.randn(5, 3, generator=g)
        changed = values.clone()
        changed[0] = torch.randn(3, generator=g) * 10.
        def rows(x):
            return [(str(i), OrderedDict(w=v.clone()), 1) for i, v in enumerate(x)]
        a, b = s.preview(rows(values))["w"], s.preview(rows(changed))["w"]
        assert float(torch.linalg.vector_norm(a - b)) <= 2 * min(clip, tau) / 5 + 1e-7
