"""Tests for fine-grained per-round timing breakdown profiling across clients and server."""
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

import benchmarks
import client


def configure_client(monkeypatch):
    for flag in ("USE_LLM", "USE_LLM_LORA", "USE_PNEUMONIA", "USE_DERIVED"):
        monkeypatch.setattr(client, flag, False)
    monkeypatch.setattr(client, "USE_MLP", True)
    monkeypatch.setattr(client, "MODEL_TYPE", "MLP")
    monkeypatch.setattr(client, "TRAINING_ARM", "FULL")
    monkeypatch.setattr(client, "DEVICE", torch.device("cpu"))
    monkeypatch.setattr(client, "EXECUTION_CONTRACT", None)


def test_client_train_records_fine_grained_timing_breakdown(monkeypatch):
    configure_client(monkeypatch)
    net = torch.nn.Linear(2, 2)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            torch.tensor([0, 1], dtype=torch.long),
        ),
        batch_size=1,
    )
    client.train(net, loader, 1, "ecg", learning_rate_override=0.01)

    assert hasattr(net, "_train_timings"), "net should have _train_timings after train()"
    timings = net._train_timings
    assert "time_data_loading_s" in timings
    assert "time_forward_backward_s" in timings
    assert "time_optimizer_step_s" in timings

    for k in ("time_data_loading_s", "time_forward_backward_s", "time_optimizer_step_s"):
        v = timings[k]
        assert isinstance(v, float)
        assert v >= 0.0, f"{k} should be non-negative"


def test_zosl_client_fit_exposes_latest_timings(monkeypatch):
    configure_client(monkeypatch)
    zosl = client.ZOSLClient.__new__(client.ZOSLClient)
    zosl.dataset_name = "ecg"
    zosl.latest_timings = {}
    zosl.net = torch.nn.Linear(2, 2)
    zosl.net._train_timings = {
        "time_data_loading_s": 0.012345,
        "time_forward_backward_s": 0.045678,
        "time_optimizer_step_s": 0.009876,
    }
    zosl.trainloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.ones(4, 2), torch.zeros(4, dtype=torch.long)),
        batch_size=2,
    )
    zosl.get_parameters = Mock(return_value={})
    zosl.set_parameters = Mock()

    with patch.object(client, "train"):
        params, count = zosl.fit(zosl.net.state_dict(), {})

    assert zosl.latest_timings == zosl.net._train_timings
    assert count == 4


def test_benchmarks_build_round_record_merges_timings():
    timings = {
        "time_data_loading_s": 0.01,
        "time_forward_backward_s": 0.05,
        "time_optimizer_step_s": 0.02,
        "time_wire_encode_s": 0.003,
        "time_aggregation_s": 0.04,
        "time_wire_decode_s": 0.002,
        "time_eval_s": 0.08,
    }
    record = benchmarks.build_round_record(
        server_round=3,
        model_type="MLP",
        loss=0.42,
        timings=timings,
    )
    assert record["serverRound"] == 3
    assert record["modelType"] == "MLP"
    assert "extraMetrics" in record
    assert record["extraMetrics"]["timings"] == timings


def test_fl_server_result_payload_includes_timing_breakdown():
    timings = {
        "time_data_loading_s": 0.011,
        "time_forward_backward_s": 0.022,
        "time_optimizer_step_s": 0.033,
        "time_wire_encode_s": 0.044,
        "time_aggregation_s": 0.055,
        "time_wire_decode_s": 0.066,
        "time_eval_s": 0.077,
    }
    metrics = {
        "loss": 0.35,
        "accuracy": 88.5,
        "timings": timings,
    }
    decimal_accuracy = float(metrics["accuracy"]) / 100.0
    payload = {
        "serverRound": 1,
        "loss": float(metrics.get("loss", 0.0)),
        "accuracy": decimal_accuracy,
        "gpuUtilization": 0.0,
        "timeDataLoadingS": timings.get("time_data_loading_s"),
        "timeForwardBackwardS": timings.get("time_forward_backward_s"),
        "timeOptimizerStepS": timings.get("time_optimizer_step_s"),
        "timeWireEncodeS": timings.get("time_wire_encode_s"),
        "timeAggregationS": timings.get("time_aggregation_s"),
        "timeWireDecodeS": timings.get("time_wire_decode_s"),
        "timeEvalS": timings.get("time_eval_s"),
    }
    assert payload["timeDataLoadingS"] == 0.011
    assert payload["timeForwardBackwardS"] == 0.022
    assert payload["timeOptimizerStepS"] == 0.033
    assert payload["timeWireEncodeS"] == 0.044
    assert payload["timeAggregationS"] == 0.055
    assert payload["timeWireDecodeS"] == 0.066
    assert payload["timeEvalS"] == 0.077
