"""Prospectively fixed pilot cells, selection and record reconciliation."""
import copy
import pytest

from benchmarks.nih_validation_pilot import initial_configs, followup_configs, select_learning_rate, summarize_record, run_pilot


PIN = "0" * 64


def _record(config, auc=0.7):
    metrics = {"loss": 0.6, "auc": auc, "accuracy": 0.8, "balanced_accuracy": (0.75 + 5 / 6) / 2,
               "sensitivity": 0.75, "specificity": 5 / 6, "ece": 0.1, "refusal": None}
    return {"meta": {"config": config, "stopped_early": False, "refusals": [], "wall_seconds": 1,
                     "peak_rss_bytes": 123, "peak_rss_scope": "process",
                     "workload": {"dataset": "nih", "pretrained": True, "evaluation_split": "validation",
                                  "features_report_sha256": PIN, "parameters": 1026, "n_test": 10,
                                  "split_counts": {"validation": {"images": 10, "patients": 8,
                                                                  "positive_images": 4, "negative_images": 6}}}},
            "rounds": [{"round": i, **metrics} for i in range(1, config["federation"]["rounds"] + 1)]}


def _records():
    return [_record(config) for config in initial_configs("cache", PIN)]


def test_exact_initial_and_conditional_cells_preserve_protocol():
    initial = initial_configs("cache", PIN)
    assert [c["name"] for c in initial] == [f"{group}_lr_{lr:g}" for group in ("pooled", "federated")
                                           for lr in (0.001, 0.01, 0.1)]
    assert [c["federation"]["num_clients"] for c in initial] == [1] * 3 + [10] * 3
    follow = followup_configs("cache", PIN, 0.01)
    assert [c["name"] for c in follow] == ["federated_repeat_20261020", "federated_repeat_20261021",
                                          "federated_clip_mean", "federated_clip_trimmed", "scale_1000_two_rounds"]
    assert [c["seed"] for c in follow[:2]] == [20261020, 20261021]
    assert follow[-1]["federation"]["num_clients"] == 1000 and follow[-1]["federation"]["rounds"] == 2
    assert follow[2]["server"]["clip_norm"] == follow[3]["server"]["clip_norm"] == 0.4
    assert follow[3]["server"]["rule"] == "trimmed_mean"
    for c in initial + follow:
        assert c["workload"]["evaluation_split"] == "validation" and c["workload"]["balance"] is False
        assert c["workload"]["allow_random_features"] is False and c["privacy"]["mode"] == "none"
        assert c["workload"]["auc_score"] == "probability_v1"
        assert c["client"]["batch_size"] == 256 and c["client"]["local_epochs"] == 3
        assert not c["behaviors"]
    with pytest.raises(ValueError):
        followup_configs("cache", PIN, 0.5)


def test_lr_selection_uses_only_final_federated_auc_and_lower_lr_on_exact_ties():
    records = _records()
    for record in records[:3]:
        for line in record["rounds"]:
            line["auc"] = 0.99
    records[3]["rounds"][0]["auc"] = 0.999
    records[4]["rounds"][-1]["auc"] = records[5]["rounds"][-1]["auc"] = 0.72
    assert select_learning_rate(records) == 0.01


def test_incomplete_or_duplicate_initial_cells_cannot_select_survivors():
    records = _records()
    for bad in (records[:-1], records + [records[0]], records[:-1] + [records[0]]):
        with pytest.raises(ValueError):
            select_learning_rate(bad)


@pytest.mark.parametrize("mutation", ["test", "identity", "rounds", "refusal", "nan", "accuracy", "random"])
def test_wrong_or_incomplete_records_are_not_qualified(mutation):
    record = copy.deepcopy(_records()[0])
    if mutation == "test": record["meta"]["workload"]["evaluation_split"] = "test"
    elif mutation == "identity": record["meta"]["workload"]["features_report_sha256"] = "1" * 64
    elif mutation == "rounds": record["rounds"].pop()
    elif mutation == "refusal": record["rounds"][1]["refusal"] = "failed"
    elif mutation == "nan": record["rounds"][1]["loss"] = float("nan")
    elif mutation == "accuracy": record["rounds"][-1]["accuracy"] = 0.1
    else: record["meta"]["workload"]["pretrained"] = False
    with pytest.raises(ValueError):
        summarize_record(record)


def test_summary_reconstructs_consistent_confusion_counts_and_keeps_poor_finite_results():
    record = _records()[0]
    record["rounds"][-1]["loss"] = 1000.0
    record["rounds"][-1]["auc"] = 0.4
    summary = summarize_record(record)
    assert summary["confusion"] == {"tp": 3, "fn": 1, "tn": 5, "fp": 1}
    assert summary["majority_baseline"] == {"accuracy": 0.6, "balanced_accuracy": 0.5}
    assert summary["final"]["loss"] == 1000 and summary["final"]["auc"] == 0.4


def test_existing_pilot_directory_is_never_overwritten(tmp_path):
    output = tmp_path / "pilot"
    output.mkdir()
    (output / "sentinel").write_text("preserve")
    with pytest.raises(FileExistsError):
        run_pilot(tmp_path / "missing", PIN, output)
    assert (output / "sentinel").read_text() == "preserve"
