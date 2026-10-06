# framework/tests/test_simulation_metrics.py
"""Metrics against hand-computed fixtures."""
from collections import OrderedDict

import pytest
import torch

from fedlearn.simulation import metrics as m


def test_classification_metrics_on_a_hand_fixture():
    # 4 samples, labels [0, 0, 1, 1]; predictions [0, 1, 1, 1]
    logits = torch.tensor([[2.0, 0.0], [0.0, 1.0], [0.0, 3.0], [0.5, 1.0]])
    labels = torch.tensor([0, 0, 1, 1])
    out = m.classification_metrics(logits, labels)
    assert out["accuracy"] == pytest.approx(0.75)
    assert out["sensitivity"] == pytest.approx(1.0)      # both positives right
    assert out["specificity"] == pytest.approx(0.5)      # one of two negatives right
    assert out["balanced_accuracy"] == pytest.approx(0.75)
    assert 0.0 <= out["ece"] <= 1.0
    # P(class 1): negatives 0.119, 0.731; positives 0.953, 0.622 -> 3 of 4 pairs ordered correctly
    assert out["auc"] == pytest.approx(0.75)


def test_ece_of_a_perfectly_calibrated_confident_model_is_zero():
    logits = torch.tensor([[50.0, 0.0], [0.0, 50.0]])
    assert m.classification_metrics(logits, torch.tensor([0, 1]))["ece"] == pytest.approx(0.0, abs=1e-6)


def test_backdoor_success_counts_non_target_samples_sent_to_target():
    logits = torch.tensor([[0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])   # preds 1, 0, 1
    labels = torch.tensor([0, 0, 1])                              # target 1 -> non-target rows 0, 1
    assert m.backdoor_success(logits, labels, target=1) == pytest.approx(0.5)
    assert m.backdoor_success(logits, torch.tensor([1, 1, 1]), target=1) is None


def test_deviation_ratio_is_zero_without_attack_and_none_without_movement():
    g = OrderedDict(w=torch.zeros(2))
    honest = OrderedDict(w=torch.tensor([3.0, 4.0]))
    assert m.deviation_ratio(honest, honest, g) == pytest.approx(0.0)
    attacked = OrderedDict(w=torch.tensor([0.0, 4.0]))
    assert m.deviation_ratio(attacked, honest, g) == pytest.approx(3.0 / 5.0)
    assert m.deviation_ratio(attacked, g, g) is None


def test_mean_pairwise_cosine():
    a, b, c = torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0]), torch.tensor([1.0, 0.0])
    assert m.mean_pairwise_cosine([a, c]) == pytest.approx(1.0)
    assert m.mean_pairwise_cosine([a, b, c]) == pytest.approx(1.0 / 3.0)
    assert m.mean_pairwise_cosine([a]) is None


def test_exclusion_rates_split_by_group():
    kept = {"c0": 1.0, "c1": 0.5, "c2": 0.0}
    out = m.exclusion_rates(kept, attacker_ids={"c2"})
    assert out == {"honest": pytest.approx(0.25), "attacker": pytest.approx(1.0)}
    assert m.exclusion_rates(None, {"c2"}) is None


def test_client_fairness_mean_worst_decile_and_spread():
    accs = [0.1 * i for i in range(1, 11)]                # 0.1 .. 1.0
    out = m.client_fairness(accs + [None])                # a client with no data is skipped
    assert out["mean"] == pytest.approx(0.55)
    assert out["worst_10pct"] == pytest.approx(0.1)       # lowest ceil(10% of 10) = 1 client
    assert out["std"] == pytest.approx((99 / 12 * 0.01) ** 0.5)
    assert m.client_fairness([]) is None


def test_state_digest_matches_the_capstone_harness():
    from benchmarks.capstone_matched_study import _digest
    s = [OrderedDict(b=torch.tensor([1.0, 2.0]), a=torch.tensor([[3.0]]))]
    assert m.state_digest(s) == _digest(s)
