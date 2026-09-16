"""FR-13 integration — the ε knob. FedLoRA can be given a target ε (+ δ, round count, cohort
size) instead of a raw noise multiplier z; it solves z via the RDP accountant and exposes the
accounted (ε, δ) trace. This is the customer-tunable privacy knob (target medical range ε≈4-8).
"""
from collections import OrderedDict

import pytest
import torch

from fedlearn.privacy.dp_accountant import compute_rdp, get_epsilon
from fedlearn.server.strategy import FedLoRA


def _init_adapter():
    # A minimal FFA adapter: frozen A + trainable B + head.
    return OrderedDict([
        ("lora_A.l", torch.randn(2, 4)),
        ("lora_B.l", torch.zeros(3, 2)),
        ("head.w", torch.zeros(5)),
    ])


def _client(v):
    # A client update over the adapter keys; frozen A is re-attached by the strategy, so its
    # value here is irrelevant to the DP noise (which touches B + head only).
    return OrderedDict([
        ("lora_A.l", torch.zeros(2, 4)),
        ("lora_B.l", torch.full((3, 2), float(v))),
        ("head.w", torch.full((5,), float(v))),
    ])


def test_dp_noise_is_isolated_from_the_disclosed_global_run_seed():
    """FR-13 / DA-3 privacy invariant: the central-DP noise must be INDEPENDENT of the GLOBAL torch
    seed. fl_server.resolve_run_seed globally seeds torch for data/model-init reproducibility AND
    discloses that seed on the eval card + logs it. If the DP noise were drawn from the global
    default generator, an adversary holding the carded seed could replay the run and STRIP the
    noise, voiding (epsilon, delta)-DP. So two PRODUCTION strategies (dp_seed=None) aggregating the
    same round under the SAME global torch.manual_seed must still produce DIFFERENT noised output.
    """
    def noised_once():
        torch.manual_seed(4242)  # a disclosed run seed, exactly as resolve_run_seed sets it globally
        strat = FedLoRA(
            _init_adapter(), aggregation="FFA_LORA", clients_per_round=2,
            dp_enabled=True, dp_clip_norm=1.0, dp_noise_multiplier=1.0, dp_seed=None,
        )
        return strat.aggregate_fit(1, [("c1", _client(1.0), 10), ("c2", _client(2.0), 10)])

    out_a = noised_once()
    out_b = noised_once()
    agg_keys = [k for k in out_a if not k.startswith("lora_A")]
    identical = all(torch.equal(out_a[k], out_b[k]) for k in agg_keys)
    assert not identical, (
        "central-DP noise is deterministic from the disclosed global run seed -> an adversary can "
        "replay and strip it -> (epsilon, delta)-DP is void (DA-3 x FR-13 regression)"
    )


def test_dp_seed_still_gives_reproducible_noise_for_tests():
    """The explicit dp_seed path stays reproducible (tests/audits need determinism): two strategies
    given the SAME dp_seed produce identical noise — only the production dp_seed=None path draws
    fresh entropy."""
    def noised(seed):
        strat = FedLoRA(
            _init_adapter(), aggregation="FFA_LORA", clients_per_round=2,
            dp_enabled=True, dp_clip_norm=1.0, dp_noise_multiplier=1.0, dp_seed=seed,
        )
        return strat.aggregate_fit(1, [("c1", _client(1.0), 10), ("c2", _client(2.0), 10)])

    a, b = noised(777), noised(777)
    agg_keys = [k for k in a if not k.startswith("lora_A")]
    assert all(torch.equal(a[k], b[k]) for k in agg_keys), "explicit dp_seed must stay reproducible"


def test_target_epsilon_solves_a_noise_multiplier_within_budget():
    strat = FedLoRA(
        _init_adapter(), aggregation="FFA_LORA", clients_per_round=10,
        dp_enabled=True, dp_clip_norm=1.0,
        dp_target_epsilon=8.0, dp_delta=1e-5, dp_num_clients=100, dp_rounds=50,
    )
    # A concrete z was solved.
    assert isinstance(strat.dp_noise_multiplier, float) and strat.dp_noise_multiplier > 0.0
    # Independently confirm that z keeps the accounted ε within the target budget.
    q = 10 / 100
    eps, _ = get_epsilon(compute_rdp(q, strat.dp_noise_multiplier, 50), 1e-5)
    assert eps <= 8.0 + 1e-6


def test_accounted_epsilon_trace_is_exposed_and_within_budget():
    strat = FedLoRA(
        _init_adapter(), aggregation="FFA_LORA", clients_per_round=10,
        dp_enabled=True, dp_clip_norm=1.0,
        dp_target_epsilon=8.0, dp_delta=1e-5, dp_num_clients=100, dp_rounds=50,
    )
    assert strat.dp_accounted_epsilon is not None
    assert 0.0 < strat.dp_accounted_epsilon <= 8.0 + 1e-6


def test_exactly_one_of_noise_multiplier_or_target_epsilon_required():
    # Both provided → ambiguous → reject.
    with pytest.raises(ValueError):
        FedLoRA(_init_adapter(), dp_enabled=True, dp_clip_norm=1.0,
                dp_noise_multiplier=1.0,
                dp_target_epsilon=8.0, dp_delta=1e-5, dp_rounds=50)
    # Neither provided → nothing to calibrate → reject.
    with pytest.raises(ValueError):
        FedLoRA(_init_adapter(), dp_enabled=True, dp_clip_norm=1.0)


def test_target_epsilon_requires_delta_and_rounds():
    with pytest.raises(ValueError):
        FedLoRA(_init_adapter(), dp_enabled=True, dp_clip_norm=1.0,
                dp_target_epsilon=8.0, dp_rounds=50)  # missing delta
    with pytest.raises(ValueError):
        FedLoRA(_init_adapter(), dp_enabled=True, dp_clip_norm=1.0,
                dp_target_epsilon=8.0, dp_delta=1e-5)  # missing rounds


def test_direct_noise_multiplier_path_unchanged():
    # The raw-z path still works exactly as before (accounting trace is best-effort: None here).
    strat = FedLoRA(_init_adapter(), aggregation="FFA_LORA",
                    dp_enabled=True, dp_clip_norm=1.0, dp_noise_multiplier=1.5)
    assert strat.dp_noise_multiplier == 1.5
    assert strat.dp_accounted_epsilon is None  # no delta/rounds supplied → no trace


def test_direct_z_with_accounting_params_still_reports_a_trace():
    strat = FedLoRA(_init_adapter(), aggregation="FFA_LORA", clients_per_round=10,
                    dp_enabled=True, dp_clip_norm=1.0, dp_noise_multiplier=1.5,
                    dp_delta=1e-5, dp_num_clients=100, dp_rounds=50)
    assert strat.dp_accounted_epsilon is not None and strat.dp_accounted_epsilon > 0.0
