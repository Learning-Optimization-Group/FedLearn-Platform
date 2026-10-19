"""Stable NIH binary scoring with explicit historical probability compatibility."""
import pytest
import torch

from tests.test_capstone_nih_workload import cfg, feature_template
from fedlearn.simulation.config import ConfigError, resolve
from fedlearn.simulation.engine import run
from fedlearn.simulation.metrics import classification_metrics
from fedlearn.simulation.workloads import FeatureHeadWorkload, NihFeatureHeadWorkload


@pytest.mark.parametrize("margins", [[20., 21., 22., 23.], [-120., -119., -118., -117.]])
def test_margin_auc_preserves_order_after_positive_or_negative_probability_saturation(margins):
    logits = torch.stack((torch.zeros(4), torch.tensor(margins)), dim=1)
    labels = torch.tensor([0, 0, 1, 1])
    legacy = classification_metrics(logits, labels)
    stable = classification_metrics(logits, labels, auc_score="margin_v2")
    assert legacy["auc"] == 0.5
    assert stable["auc"] == 1.0 and stable["auc_probability"] == 0.5
    assert {k: stable[k] for k in legacy if k != "auc"} == {k: v for k, v in legacy.items() if k != "auc"}


@pytest.mark.parametrize("labels,expected", [([0, 1, 0, 1], 0.5), ([1, 1, 0, 0], 1.0)])
def test_genuine_equal_margins_remain_tied(labels, expected):
    logits = torch.tensor([[0., 1.], [2., 3.], [0., -1.], [4., 3.]])
    assert classification_metrics(logits, torch.tensor(labels), auc_score="margin_v2")["auc"] == expected


def test_ordinary_scores_and_explicit_legacy_mode_agree():
    logits = torch.tensor([[0., -2.], [0., -1.], [0., 0.], [0., 2.]])
    labels = torch.tensor([0, 1, 0, 1])
    legacy = classification_metrics(logits, labels)
    assert classification_metrics(logits, labels, auc_score="probability_v1") == legacy
    assert legacy["auc"] == classification_metrics(logits, labels, auc_score="margin_v2")["auc"] == 0.75


def test_single_class_margin_auc_is_undefined_and_multiclass_refuses_margin_mode():
    out = classification_metrics(torch.zeros(4, 2), torch.zeros(4, dtype=torch.long), auc_score="margin_v2")
    assert out["auc"] is None and out["auc_probability"] is None
    with pytest.raises(ValueError):
        classification_metrics(torch.zeros(4, 3), torch.zeros(4, dtype=torch.long), auc_score="margin_v2")
    with pytest.raises(ValueError):
        classification_metrics(torch.zeros(4, 2), torch.zeros(4, dtype=torch.long), auc_score="unknown")


@pytest.mark.parametrize("bad", [None, False, "margin", ["margin_v2"]])
def test_nih_rejects_unknown_score_versions(cfg, bad):
    cfg["workload"]["auc_score"] = bad
    with pytest.raises(ConfigError) as error:
        resolve(cfg)
    assert error.value.field == "workload.auc_score"


def test_nih_default_is_margin_and_explicit_probability_path_stays_legacy(cfg):
    cfg["workload"].pop("auc_score", None)
    resolved = resolve(cfg)
    assert resolved["workload"]["auc_score"] == "margin_v2"
    wl = NihFeatureHeadWorkload(resolved)
    assert wl.info["auc_score"] == "margin_v2"
    assert wl.info["auc_score_definition"] == "float64(logit_1)-float64(logit_0)"
    logits, labels = torch.tensor([[0., 20.], [0., 21.], [0., 22.], [0., 23.]]), torch.tensor([0, 0, 1, 1])
    assert wl._metrics(logits, labels)["auc"] == 1.0
    assert FeatureHeadWorkload._metrics(wl, logits, labels)["auc"] == 0.5
    record = run(cfg)
    assert record["meta"]["config"]["workload"]["auc_score"] == "margin_v2"
    assert record["meta"]["workload"]["auc_score"] == "margin_v2"
    assert all("auc_probability" in line for line in record["rounds"])
    cfg["workload"]["auc_score"] = "probability_v1"
    old = run(cfg)
    assert old["meta"]["workload"]["auc_score"] == "probability_v1"
    assert all("auc_probability" not in line for line in old["rounds"])


def test_legacy_dataset_does_not_accept_nih_scoring_option():
    with pytest.raises(ConfigError):
        resolve({"workload": {"features_path": "fixture.pt", "auc_score": "margin_v2"}})
