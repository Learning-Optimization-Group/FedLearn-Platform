"""NIH has explicit label semantics and a closed first-order execution contract."""
import json
from argparse import Namespace
from pathlib import Path

import pytest

import client
import execution_plan
import fl_server
import recipes
from fedlearn.communication.generated import execution_contract_pb2 as pb
from fedlearn.contract import parse_contract_binary, validate_contract
from nih_dataset import CLASSES


def test_nih_recipe_has_explicit_classes_and_preserves_original_pneumonia():
    recipe = recipes.get_recipe("NIH_PNEUMONIA_CNN")
    assert recipe.classes == CLASSES
    assert recipe.base_models == ["nih_pneumonia_cnn"]
    assert next(r for r in recipes.describe() if r["key"] == recipe.key)["supported_arms"] == ["FULL"]
    assert recipes.get_recipe("PNEUMONIA_CNN").classes == ["NORMAL", "PNEUMONIA"]
    assert recipe.build_model("cpu").classifier[-1].out_features == 2


@pytest.mark.parametrize("strategy", ["FedAvg", "Robust"])
def test_nih_local_plan_states_exact_preparation_classes_dropout_and_adam(strategy):
    plan = execution_plan.resolve_model_training("NIH_PNEUMONIA_CNN", strategy, "FULL", data_source="LOCAL_SNAPSHOT")
    assert plan.model_id == "nih_pneumonia_cnn"
    assert plan.data.label_schema_id == execution_plan.label_schema_id(CLASSES)
    assert list(plan.data.input_shape) == [1, 224, 224]
    assert plan.local_training.batch_order == pb.BATCH_ORDER_SEEDED_PERMUTATION_V1
    assert plan.local_training.adam.learning_rate == .001
    assert [d.rate for d in plan.dropout] == [.4]
    root = Path(__file__).resolve().parents[2]
    c = parse_contract_binary((root / "framework/tests/fixtures/execution_contract_v1/golden_tinynet_fedavg.binpb").read_bytes())
    plan.model_revision = c.model_training.model_revision
    plan.initial_state_sha256 = c.model_training.initial_state_sha256
    plan.artifacts.extend(c.model_training.artifacts)
    c.model_training.CopyFrom(plan);c.recipe = pb.RECIPE_NIH_PNEUMONIA_CNN
    c.strategy = pb.STRATEGY_FEDAVG if strategy == "FedAvg" else pb.STRATEGY_ROBUST
    assert validate_contract(c, reader_protocol_version=2) == []
    assert execution_plan.check_contract(c, "NIH_PNEUMONIA_CNN", strategy, "FULL", data_source="LOCAL_SNAPSHOT") == []
    c.model_training.data.label_schema_id = execution_plan.label_schema_id(["NORMAL", "PNEUMONIA"])
    assert execution_plan.check_contract(c, "NIH_PNEUMONIA_CNN", strategy, "FULL", data_source="LOCAL_SNAPSHOT")


@pytest.mark.parametrize("strategy,arm,source", [("FedAvg", "FULL", "FIXTURE"), ("FedOpt", "FULL", "LOCAL_SNAPSHOT"),
    ("FedProx", "FULL", "LOCAL_SNAPSHOT"), ("DeComFL", "FULL", "LOCAL_SNAPSHOT"), ("FedAvg", "FROZEN_HEAD", "LOCAL_SNAPSHOT")])
def test_nih_unsupported_execution_is_refused(strategy, arm, source):
    with pytest.raises(execution_plan.NotRepresentable):
        execution_plan.resolve_model_training("NIH_PNEUMONIA_CNN", strategy, arm, data_source=source)


def test_nih_client_requires_explicit_contract_before_any_data_is_read(tmp_path):
    args = Namespace(execution_contract=None, dataset_path=None)
    with pytest.raises(SystemExit, match="contract"):
        client.enforce_execution_contract(args, "NIH_PNEUMONIA_CNN", "FULL")


def test_nih_registry_card_records_validation_sample_and_binary_metrics():
    args = Namespace(model_type="NIH_PNEUMONIA_CNN", strategy="FedAvg", num_rounds=2, training_arm="FULL",
        nih_evaluation_provenance={"split": "validation", "record_count": 512, "total_split_images": 17565})
    card = json.loads(fl_server.build_eval_card(args, [(2, {"loss": .12, "accuracy": 99.0,
        "balanced_accuracy": .5, "auroc": .6, "average_precision": .02, "majority_accuracy": .99})]))
    assert card["nih"]["classes"] == CLASSES
    assert card["nih"]["evaluation"]["split"] == "validation"
    assert card["nih"]["evaluation"]["record_count"] == 512
    assert card["nih"]["metrics"]["balanced_accuracy"] == .5
    assert card["nih"]["metrics"]["majority_accuracy"] == .99


def test_nih_benchmark_has_no_generic_accuracy_target_and_exposes_majority_baseline():
    import benchmarks
    from nih_runtime import binary_metrics
    args = Namespace(model_type="NIH_PNEUMONIA_CNN", dataset="cb")
    assert fl_server.benchmark_accuracy_target(args) is None
    metrics = binary_metrics([0, 0, 0, 1], [.1, .2, .3, .4])
    record = benchmarks.build_round_record(1, quality={},
        target_accuracy=fl_server.benchmark_accuracy_target(args), extra=fl_server.nih_benchmark_metrics(metrics))
    assert "targetAccuracy" not in record
    assert record["extraMetrics"]["majorityAccuracy"] == .75
    assert record["extraMetrics"]["averagePrecision"] == 1.0


@pytest.mark.parametrize("strategy", ["FedOpt", "FedProx", "DeComFL", "FedLoRA", "unknown"])
def test_direct_nih_server_refuses_unsupported_training_before_strategy_creation(strategy):
    with pytest.raises(ValueError, match="NIH"):
        fl_server.select_strategy(Namespace(model_type="NIH_PNEUMONIA_CNN", strategy=strategy), {}, None)
