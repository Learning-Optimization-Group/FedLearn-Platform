"""The training plan an execution contract publishes must be what the laptop client actually executes.

``execution_plan.resolve_model_training`` states, for a run's recipe, strategy and arm, the part of the
contract Python owns: objective, update protocol, ordered trainable layout, local training (optimizer
and every hyperparameter, step budget, batching) and the data requirement. These tests compare that
statement with the objects ``client.py`` really builds, so the plan cannot drift from the runtime.
"""
import hashlib
import json
import os
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import execution_plan  # noqa: E402
import recipes  # noqa: E402
from fedlearn.communication.generated import execution_contract_pb2 as pb  # noqa: E402
from fedlearn.contract import parse_contract_binary, validate_contract  # noqa: E402

GOLDEN_CONTRACT = os.path.join(os.path.dirname(__file__), "..", "..", "framework", "tests", "fixtures",
                               "execution_contract_v1", "golden_tinynet_fedavg.binpb")


def _tinynet_fedavg_client(monkeypatch):
    import torch
    import client
    monkeypatch.setattr(client, "USE_LLM_LORA", False, raising=False)
    monkeypatch.setattr(client, "USE_LLM", False, raising=False)
    monkeypatch.setattr(client, "USE_MLP", False, raising=False)
    monkeypatch.setattr(client, "USE_PNEUMONIA", False, raising=False)
    monkeypatch.setattr(client, "USE_DERIVED", False, raising=False)
    monkeypatch.setattr(client, "MODEL_TYPE", "TINYNET_GOLDEN", raising=False)
    monkeypatch.setattr(client, "TRAINING_ARM", "FULL", raising=False)
    monkeypatch.setattr(client, "USE_WIRE_SUBSET",
                        recipes.federates_trainable_subset("TINYNET_GOLDEN", "FULL"), raising=False)
    monkeypatch.setattr(client, "DEVICE", torch.device("cpu"), raising=False)
    return client.ZOSLClient(partition_id=1, dataset_name="cb", num_clients=4)


def _recording_sgd(monkeypatch):
    import torch
    created = []
    real_sgd = torch.optim.SGD

    class RecordingSgd(real_sgd):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.steps = 0
            created.append(self)

        def step(self, *args, **kwargs):
            self.steps += 1
            return super().step(*args, **kwargs)

    monkeypatch.setattr(torch.optim, "SGD", RecordingSgd)
    return created


def test_tinynet_fedavg_optimizer_is_the_one_the_client_builds(monkeypatch):
    plan = execution_plan.resolve_model_training("TINYNET_GOLDEN", "FedAvg", "FULL")
    participant = _tinynet_fedavg_client(monkeypatch)
    created = _recording_sgd(monkeypatch)

    participant.fit(participant.get_parameters(), {})   # FedAvg sends no training settings

    local = plan.local_training
    assert local.WhichOneof("optimizer") == "sgd"
    assert len(created) == 1
    group = created[0].param_groups[0]
    assert group["lr"] == local.sgd.learning_rate
    assert group["momentum"] == local.sgd.momentum
    assert group["dampening"] == local.sgd.dampening
    assert group["weight_decay"] == local.sgd.weight_decay
    assert group["nesterov"] is local.sgd.nesterov
    assert group["maximize"] is False
    assert created[0].steps == local.local_epochs * len(participant.trainloader)
    assert not local.HasField("max_local_steps")


def test_tinynet_fedavg_optimizer_state_starts_fresh_every_round(monkeypatch):
    plan = execution_plan.resolve_model_training("TINYNET_GOLDEN", "FedAvg", "FULL")
    participant = _tinynet_fedavg_client(monkeypatch)
    created = _recording_sgd(monkeypatch)

    participant.fit(participant.get_parameters(), {})
    participant.fit(participant.get_parameters(), {})

    assert plan.local_training.reset_optimizer_each_round is True
    assert len(created) == 2 and created[0] is not created[1]


def test_tinynet_fedavg_does_not_clip_gradients(monkeypatch):
    import torch
    plan = execution_plan.resolve_model_training("TINYNET_GOLDEN", "FedAvg", "FULL")
    participant = _tinynet_fedavg_client(monkeypatch)

    def refuse(*_args, **_kwargs):
        raise AssertionError("TinyNet training clipped gradients")

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", refuse)
    participant.fit(participant.get_parameters(), {})
    assert not plan.local_training.HasField("gradient_clip_norm")


def test_tinynet_fedavg_batching_is_the_client_loader(monkeypatch):
    from torch.utils.data import RandomSampler
    plan = execution_plan.resolve_model_training("TINYNET_GOLDEN", "FedAvg", "FULL")
    loader = _tinynet_fedavg_client(monkeypatch).trainloader

    local = plan.local_training
    assert loader.batch_size == local.batch_size
    assert loader.drop_last is local.drop_last
    assert isinstance(loader.sampler, RandomSampler)
    assert local.batch_order == pb.BATCH_ORDER_SHUFFLED_EACH_EPOCH


def test_tinynet_layout_is_the_models_trainable_parameters_in_order(monkeypatch):
    plan = execution_plan.resolve_model_training("TINYNET_GOLDEN", "FedAvg", "FULL")
    net = _tinynet_fedavg_client(monkeypatch).net

    expected = [(name, list(p.shape)) for name, p in net.named_parameters() if p.requires_grad]
    assert [(t.name, list(t.shape)) for t in plan.trainable] == expected
    assert all(t.dtype == pb.DTYPE_F32 for t in plan.trainable)


def test_tinynet_data_requirement_is_the_client_data(monkeypatch):
    plan = execution_plan.resolve_model_training("TINYNET_GOLDEN", "FedAvg", "FULL")
    x, _ = next(iter(_tinynet_fedavg_client(monkeypatch).trainloader))

    data = plan.data
    assert list(data.input_shape) == list(x.shape[1:])
    assert data.input_dtype == pb.DTYPE_F32
    classes = recipes.get_recipe("TINYNET_GOLDEN").classes
    assert data.class_count == len(classes)
    digest = hashlib.sha256(json.dumps(classes, separators=(",", ":")).encode()).hexdigest()
    assert data.label_schema_id == "labels-sha256:" + digest
    assert [t.identity_vector.width for t in data.transforms] == [x.shape[1]]


def test_tinynet_fedavg_identity_follows_the_recipe():
    plan = execution_plan.resolve_model_training("TINYNET_GOLDEN", "FedAvg", "FULL")
    assert plan.model_id == "tinynet_golden"
    assert plan.arm == pb.ARM_FULL
    assert plan.task == pb.TASK_VECTOR_CLASSIFICATION
    assert recipes.ARM_OBJECTIVES["FULL"] == "cross_entropy" and plan.objective == pb.OBJECTIVE_CROSS_ENTROPY
    assert plan.update_protocol == pb.UPDATE_TRAINABLE_STATE_F32


def test_the_plan_completes_a_valid_contract():
    with open(GOLDEN_CONTRACT, "rb") as fh:
        contract = parse_contract_binary(fh.read())
    plan = execution_plan.resolve_model_training("TINYNET_GOLDEN", "FedAvg", "FULL")
    training = contract.model_training
    for field in ("model_id", "arm", "task", "objective", "update_protocol"):
        setattr(training, field, getattr(plan, field))
    training.ClearField("trainable")
    training.trainable.extend(plan.trainable)
    training.local_training.CopyFrom(plan.local_training)
    training.data.CopyFrom(plan.data)
    assert validate_contract(contract, reader_protocol_version=2) == []


@pytest.mark.parametrize("recipe, strategy, arm", [
    ("CNN", "FedAvg", "FULL"),
    ("TINYNET_GOLDEN", "FedOpt", "FULL"),
    ("TINYNET_GOLDEN", "DeComFL", "FULL"),
    ("TINYNET_GOLDEN", "FedProx", "FULL"),
])
def test_a_run_without_a_v1_plan_is_not_representable(recipe, strategy, arm):
    with pytest.raises(execution_plan.NotRepresentable):
        execution_plan.resolve_model_training(recipe, strategy, arm)


def _cli(*args):
    script = os.path.join(os.path.dirname(__file__), "..", "execution_plan.py")
    done = subprocess.run([sys.executable, script, *args], capture_output=True, text=True, check=True)
    return json.loads(done.stdout)


def test_the_cli_prints_the_plan_as_protojson():
    from google.protobuf import json_format
    out = _cli("--recipe", "TINYNET_GOLDEN", "--strategy", "FedAvg", "--training-arm", "FULL")
    assert out["representable"] is True
    printed = json_format.ParseDict(out["modelTraining"], pb.ModelTraining())
    assert printed == execution_plan.resolve_model_training("TINYNET_GOLDEN", "FedAvg", "FULL")


def test_the_cli_reports_an_unrepresentable_run_without_failing():
    out = _cli("--recipe", "CNN", "--strategy", "FedAvg", "--training-arm", "FULL")
    assert out["representable"] is False
    assert "CNN" in out["reason"]


# --- state digests ------------------------------------------------------------------------------------

ZO_STATE = os.path.join(os.path.dirname(__file__), "..", "..", "framework", "tests", "fixtures",
                        "decomfl_golden", "zo_state.safetensors")


def _canonical_digest(pairs):
    from fedlearn.communication.safetensors_codec import save_safetensors
    return hashlib.sha256(save_safetensors(list(pairs))).hexdigest()


def _save_like_init_model(path, state):
    """The .npz format init_model.py writes and fl_server.py loads its initial model from."""
    import numpy as np
    np.savez(path, **{k.replace(".", "__DOT__"): v for k, v in state.items()})


def _tinynet_trainable_state():
    from fedlearn.estimators.params import trainable_state
    model = recipes.get_recipe("TINYNET_GOLDEN").build_model("cpu")
    return {k: v.detach().numpy() for k, v in trainable_state(model).items()}


def test_the_frozen_state_digest_covers_the_recipe_models_frozen_tensors():
    plan = execution_plan.resolve_model_training("TINYNET_GOLDEN", "FedAvg", "FULL")
    model = recipes.get_recipe("TINYNET_GOLDEN").build_model("cpu")
    trainable = {t.name for t in plan.trainable}
    frozen = [(n, t.detach().numpy()) for n, t in model.state_dict().items() if n not in trainable]

    assert [n for n, _ in frozen] == ["fc2.weight", "fc2.bias"]
    assert plan.frozen_state_sha256 == _canonical_digest(frozen)


def test_the_initial_state_digest_covers_the_servers_initial_model_in_contract_order(tmp_path):
    path = tmp_path / "model.npz"
    state = _tinynet_trainable_state()
    _save_like_init_model(path, dict(reversed(list(state.items()))))   # file order must not matter

    plan = execution_plan.resolve_model_training("TINYNET_GOLDEN", "FedAvg", "FULL", initial_state_path=str(path))

    assert plan.initial_state_sha256 == _canonical_digest(state.items())


def test_the_phones_golden_state_is_the_servers_initial_state(tmp_path):
    from fedlearn.communication.safetensors_codec import load_safetensors
    with open(ZO_STATE, "rb") as fh:
        golden, _metadata = load_safetensors(fh.read())
    path = tmp_path / "model.npz"
    _save_like_init_model(path, _tinynet_trainable_state())

    plan = execution_plan.resolve_model_training("TINYNET_GOLDEN", "FedAvg", "FULL", initial_state_path=str(path))

    assert plan.initial_state_sha256 == _canonical_digest(golden)


@pytest.mark.parametrize("change", ["missing", "reshaped", "extra_float", "renamed"])
def test_an_initial_model_that_disagrees_with_the_layout_is_not_representable(tmp_path, change):
    import numpy as np
    state = _tinynet_trainable_state()
    if change == "missing":
        del state["fc1.bias"]
    elif change == "reshaped":
        state["fc1.bias"] = np.zeros((6,), dtype=np.float32)
    elif change == "extra_float":
        state["fc2.weight"] = np.zeros((3, 5), dtype=np.float32)
    else:
        state["fc1.b"] = state.pop("fc1.bias")
    path = tmp_path / "model.npz"
    _save_like_init_model(path, state)

    with pytest.raises(execution_plan.NotRepresentable):
        execution_plan.resolve_model_training("TINYNET_GOLDEN", "FedAvg", "FULL", initial_state_path=str(path))


def test_a_non_float_tensor_the_server_withholds_is_ignored(tmp_path):
    import numpy as np
    state = _tinynet_trainable_state()
    expected = _canonical_digest(state.items())
    state["num_batches_tracked"] = np.array(3, dtype=np.int64)
    path = tmp_path / "model.npz"
    _save_like_init_model(path, state)

    plan = execution_plan.resolve_model_training("TINYNET_GOLDEN", "FedAvg", "FULL", initial_state_path=str(path))

    assert plan.initial_state_sha256 == expected


def test_the_cli_adds_the_initial_state_digest(tmp_path):
    path = tmp_path / "model.npz"
    _save_like_init_model(path, _tinynet_trainable_state())
    out = _cli("--recipe", "TINYNET_GOLDEN", "--strategy", "FedAvg", "--training-arm", "FULL",
               "--initial-state", str(path))
    assert out["modelTraining"]["initialStateSha256"] == _canonical_digest(_tinynet_trainable_state().items())


def test_the_golden_contract_carries_tinynets_canonical_state_digests(tmp_path):
    with open(GOLDEN_CONTRACT, "rb") as fh:
        golden = parse_contract_binary(fh.read()).model_training
    path = tmp_path / "model.npz"
    _save_like_init_model(path, _tinynet_trainable_state())

    plan = execution_plan.resolve_model_training("TINYNET_GOLDEN", "FedAvg", "FULL", initial_state_path=str(path))

    assert golden.frozen_state_sha256 == plan.frozen_state_sha256
    assert golden.initial_state_sha256 == plan.initial_state_sha256
