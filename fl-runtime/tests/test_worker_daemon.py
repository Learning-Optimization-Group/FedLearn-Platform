"""Unit and integration tests for fl-runtime persistent worker daemon."""
from __future__ import annotations

import time
import pytest
import torch

from worker_daemon import (
    DaemonState,
    WorkerDaemonClient,
    WorkerDaemonServer,
    deserialize_state_dict,
    serialize_state_dict,
)


@pytest.fixture(scope="module")
def daemon_server():
    """Starts a live WorkerDaemonServer on a local test port."""
    port = 8995
    server = WorkerDaemonServer(host="127.0.0.1", port=port, device="cpu")
    server.start_background()
    time.sleep(0.1)
    yield server
    server.shutdown()


@pytest.fixture
def daemon_client(daemon_server):
    """Provides a client connected to the test daemon server."""
    client = WorkerDaemonClient(base_url=f"http://127.0.0.1:{daemon_server.port}")
    # Ensure clean state before each test
    client.reset_state()
    return client


def test_state_dict_serialization_roundtrip():
    """Verifies PyTorch state dict serialize/deserialize preserves bitwise tensor values."""
    original = {
        "conv1.weight": torch.randn(4, 3, 3, 3),
        "fc.bias": torch.tensor([1.0, -2.5, 3.14159]),
    }
    b64 = serialize_state_dict(original)
    recovered = deserialize_state_dict(b64)

    assert set(recovered.keys()) == set(original.keys())
    for k in original:
        assert torch.equal(recovered[k], original[k])


def test_daemon_health_and_ping(daemon_client):
    """Verifies health check and readiness responses."""
    assert daemon_client.ping() is True
    status = daemon_client.get_status()
    assert status["status"] == "IDLE"
    assert status["model_loaded"] is False
    assert status["active_project_id"] is None
    assert "memory" in status


def test_model_initialization_and_fencing(daemon_client):
    """Verifies model loading into memory and project/run state isolation fencing."""
    # 1. Initialize for Project A, Run 1
    res1 = daemon_client.init_model(
        project_id="project-A",
        run_id="run-1",
        model_type="CNN",
        training_arm="FULL",
        dataset_name="synthetic",
        partition_id=0,
    )
    assert res1["status"] == "INITIALIZED"
    assert res1["model_type"] == "CNN"
    assert res1["param_count"] > 0
    assert res1["model_load_ms"] > 0

    # 2. Re-initialization with same project and run is idempotent (reused)
    res_same = daemon_client.init_model(
        project_id="project-A",
        run_id="run-1",
        model_type="CNN",
        dataset_name="synthetic",
    )
    assert res_same["status"] == "REUSED"

    # 3. Conflict: attempting to initialize Project B while Project A is active must fail
    with pytest.raises(RuntimeError, match="Project/Run state conflict"):
        daemon_client.init_model(
            project_id="project-B",
            run_id="run-2",
            model_type="CNN",
            dataset_name="synthetic",
            force_reset=False,
        )

    # 4. With force_reset=True, can re-bind to Project B safely
    res_reset = daemon_client.init_model(
        project_id="project-B",
        run_id="run-2",
        model_type="CNN",
        dataset_name="synthetic",
        force_reset=True,
    )
    assert res_reset["status"] == "INITIALIZED"
    assert res_reset["project_id"] == "project-B"
    assert res_reset["run_id"] == "run-2"


def test_round_execution_across_consecutive_rounds(daemon_client):
    """Verifies multi-round execution retains model in memory across rounds."""
    init_res = daemon_client.init_model(
        project_id="proj-multi",
        run_id="run-multi",
        model_type="CNN",
        dataset_name="synthetic",
    )
    assert init_res["status"] == "INITIALIZED"

    # Execute Round 1
    r1 = daemon_client.execute_round(
        project_id="proj-multi",
        run_id="run-multi",
        round_number=1,
        learning_rate=0.01,
        local_epochs=1,
    )
    assert r1["status"] == "COMPLETED"
    assert r1["round_number"] == 1
    assert "weights_b64" in r1
    assert r1["compute_ms"] > 0

    # Execute Round 2 using weights from Round 1
    r2 = daemon_client.execute_round(
        project_id="proj-multi",
        run_id="run-multi",
        round_number=2,
        weights_b64=r1["weights_b64"],
        learning_rate=0.01,
        local_epochs=1,
    )
    assert r2["status"] == "COMPLETED"
    assert r2["round_number"] == 2
    assert r2["cumulative_compute_ms"] > r1["compute_ms"]

    # Fencing violation during round execution
    with pytest.raises(RuntimeError, match="Fencing violation"):
        daemon_client.execute_round(
            project_id="proj-foreign",
            run_id="run-foreign",
            round_number=3,
        )


def test_reset_state(daemon_client):
    """Verifies reset_state clears cache and returns worker to IDLE."""
    daemon_client.init_model(
        project_id="proj-to-clear",
        run_id="run-to-clear",
        model_type="CNN",
        dataset_name="synthetic",
    )
    status_before = daemon_client.get_status()
    assert status_before["model_loaded"] is True

    res = daemon_client.reset_state()
    assert res["status"] == "IDLE"

    status_after = daemon_client.get_status()
    assert status_after["model_loaded"] is False
    assert status_after["active_project_id"] is None
