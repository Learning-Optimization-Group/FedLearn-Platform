"""Adversarial correctness audit for the aggregation strategies (strategy.py).

These pin behaviours that the existing suites leave a gap in:

  * FR-18 union-of-keys renormalisation at 3 clients with WEIGHTED partial overlap
    (test_fedavg_aggregator.py only exercises the 2-client, equal-weight case), plus
    the "collapses exactly to the plain weighted mean when everyone holds every key"
    reduction.
  * The MAX_SAMPLES poisoning cap under ASYMMETRIC inflation — the existing cap test caps
    two symmetric 200k clients; this pins the actual "an inflated-num_examples client cannot
    drag the aggregate past a legitimately-at-cap honest client" security claim.
  * The FedLoRA FR-23 allowlist over the JSON-string wire shape (the JSON-decode branch of
    _assert_client_keys_allowed) — the existing allowlist test only feeds an OrderedDict.
"""
from collections import OrderedDict
import json

import pytest
import torch

from fedlearn.server.strategy import FedAvgAggregator, FedLoRA


# --------------------------------------------------------------------------- FR-18 depth
def test_fedavg_three_client_weighted_partial_overlap_renormalizes_per_key():
    """A key held by a strict SUBSET of a 3-client cohort is renormalised over ONLY that
    subset's num_examples — it is neither decayed toward zero nor diluted by clients that
    never provided it.

        A: n=100 {w:1, b:10}
        B: n=300 {w:5}         (no b)
        C: n=100 {w:9, b:30}

    w is held by A,B,C -> weighted mean over 500 = (100*1 + 300*5 + 100*9)/500 = 5.0
    b is held by A,C   -> weighted mean over 200 = (100*10 + 100*30)/200        = 20.0

    If "b" were (wrongly) weighted by the cohort-wide example total (500) it would read
    (100*10 + 100*30)/500 = 8.0 — decayed. The renormalisation must give 20.0.
    """
    agg = FedAvgAggregator()
    a = (None, OrderedDict([("w", torch.tensor([1.0])), ("b", torch.tensor([10.0]))]), 100)
    b = (None, OrderedDict([("w", torch.tensor([5.0]))]), 300)          # missing "b"
    c = (None, OrderedDict([("w", torch.tensor([9.0])), ("b", torch.tensor([30.0]))]), 100)

    out = agg.aggregate([a, b, c])

    assert set(out.keys()) == {"w", "b"}
    assert torch.allclose(out["w"], torch.tensor([5.0]), atol=1e-5)
    assert torch.allclose(out["b"], torch.tensor([20.0]), atol=1e-5), (
        "subset-held key must renormalise over its own providers (200), not the cohort total (500)"
    )


def test_fedavg_full_overlap_reduces_exactly_to_plain_weighted_mean():
    """When every client holds every key, the per-key renormalisation must collapse EXACTLY
    to the ordinary num_examples-weighted mean (the FR-18 change is a no-op in that case)."""
    agg = FedAvgAggregator()
    n1, n2, n3 = 100, 300, 100
    v1, v2, v3 = 1.0, 5.0, 9.0
    a = (None, OrderedDict([("w", torch.tensor([v1])), ("b", torch.tensor([2 * v1]))]), n1)
    b = (None, OrderedDict([("w", torch.tensor([v2])), ("b", torch.tensor([2 * v2]))]), n2)
    c = (None, OrderedDict([("w", torch.tensor([v3])), ("b", torch.tensor([2 * v3]))]), n3)

    out = agg.aggregate([a, b, c])

    total = n1 + n2 + n3
    exp_w = (n1 * v1 + n2 * v2 + n3 * v3) / total
    assert torch.allclose(out["w"], torch.tensor([exp_w]), atol=1e-5)
    assert torch.allclose(out["b"], torch.tensor([2 * exp_w]), atol=1e-5)


# --------------------------------------------------------------------------- MAX_SAMPLES cap
def test_fedavg_inflated_num_examples_cannot_dominate_honest_client():
    """MAX_SAMPLES caps an inflated attacker to the SAME ceiling as a legitimately-at-cap honest
    client, so it cannot drag the aggregate arbitrarily toward its poisoned value.

    Attacker: n=10^9  value=1000.0   (capped to MAX_SAMPLES)
    Honest:   n=MAX_SAMPLES value=0.0

    Both collapse to equal weight -> mean = 500.0. Uncapped, the attacker's weight would be
    ~1.0 and the aggregate ~1000.0.
    """
    agg = FedAvgAggregator()
    cap = FedAvgAggregator.MAX_SAMPLES

    # aggregate() clears the client dicts it consumes (aggressive memory free), so build fresh
    # entries for each call rather than reusing the tuples.
    def attacker():
        return (None, OrderedDict([("x", torch.tensor([1000.0]))]), 10**9)

    def honest():
        return (None, OrderedDict([("x", torch.tensor([0.0]))]), cap)

    out = agg.aggregate([attacker(), honest()])
    assert torch.allclose(out["x"], torch.tensor([500.0]), atol=1e-4), (
        "an inflated num_examples must be capped to MAX_SAMPLES, bounding attacker weight to an "
        "honest at-cap client's"
    )

    # And the cap is order-independent (the sanitize step runs before any weighting).
    out_rev = agg.aggregate([honest(), attacker()])
    assert torch.allclose(out_rev["x"], torch.tensor([500.0]), atol=1e-4)


# --------------------------------------------------------------------------- FR-23 allowlist (JSON wire)
def _adapter() -> OrderedDict:
    return OrderedDict([
        ("lora_A.l", torch.randn(2, 4)),
        ("lora_B.l", torch.zeros(3, 2)),
        ("head.w", torch.zeros(5)),
    ])


def test_fedlora_allowlist_rejects_json_encoded_smuggled_key():
    """FR-23 over the JSON-string wire shape: a client that ships params as a JSON string (an
    accepted wire shape the shared normalizer decodes) still cannot smuggle a key outside the
    server's adapter surface. Exercises the JSON-decode branch of _assert_client_keys_allowed,
    which the OrderedDict-only allowlist test does not.
    """
    strat = FedLoRA(initial_parameters=_adapter(), aggregation="FFA_LORA", clients_per_round=1)
    smuggled = json.dumps({
        "lora_B.l": [[1, 1], [1, 1], [1, 1]],
        "head.w": [1, 1, 1, 1, 1],
        "base_model.embed_tokens.weight": [[9, 9, 9, 9]],  # poisoned base-model tensor
    })
    with pytest.raises(ValueError, match="allowlist|surface|outside|adapter"):
        strat.aggregate_fit(1, [("c1", smuggled, 10)])


def test_fedlora_assert_homogeneous_handles_json_string_params_like_its_siblings():
    """_assert_homogeneous must decode JSON-string params via normalize_update (audit fix) — like
    _client_keys / _assert_client_keys_allowed — instead of crashing with a raw str.items()
    AttributeError. Homogeneous JSON updates pass; a genuine shape mismatch raises the clean,
    attributable ValueError, not AttributeError."""
    from fedlearn.server.strategy import FedLoRA
    same = json.dumps({"score.weight": [[1.0, 2.0]]})
    # Homogeneous JSON-encoded params: no crash, no false rejection.
    FedLoRA._assert_homogeneous([("c1", same, 10), ("c2", same, 10)])
    # A genuine heterogeneous rank across JSON updates raises the clean ValueError (not AttributeError).
    other = json.dumps({"score.weight": [[1.0, 2.0, 3.0]]})
    with pytest.raises(ValueError):
        FedLoRA._assert_homogeneous([("c1", same, 10), ("c2", other, 10)])
