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
from collections import OrderedDict

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


def _config_the_server_sends(strategy_name):
    """The per-round client config the real FL server ships for ``strategy_name`` on TinyNet.

    Built by fl_server.select_strategy exactly as a spawned server builds it, and read through the
    coordinator's own _strategy_client_config, so the test sees what a client receives on the wire.
    """
    import types
    from argparse import Namespace
    import fl_server
    from fedlearn.server.coordinator import FLCoordinator
    initial = OrderedDict((name, p.detach().clone())
                          for name, p in recipes.get_recipe("TINYNET_GOLDEN").build_model("cpu").named_parameters()
                          if p.requires_grad)
    strategy = fl_server.select_strategy(
        Namespace(strategy=strategy_name, min_clients=4, clients_per_round=4, dataset="cb",
                  aggregation="FFA_LORA"),
        initial, None)
    return FLCoordinator._strategy_client_config(types.SimpleNamespace(strategy=strategy))


@pytest.mark.parametrize("strategy, server_name", [("FedAvg", "fedavg"), ("FedOpt", "fedopt"),
                                                   ("Robust", "robust"), ("FedProx", "fedprox")])
def test_tinynet_first_order_plan_is_what_the_client_trains_under_its_servers_config(
        monkeypatch, strategy, server_name):
    """For each first-order strategy: the server's per-round config -> what client.py builds from it -> the
    plan. FedOpt ships its own client rate and epochs; FedAvg and Robust ship nothing."""
    plan = execution_plan.resolve_model_training("TINYNET_GOLDEN", strategy, "FULL")
    config = _config_the_server_sends(server_name)
    participant = _tinynet_fedavg_client(monkeypatch)
    created = _recording_sgd(monkeypatch)

    participant.fit(participant.get_parameters(), config)

    local = plan.local_training
    assert len(created) == 1
    group = created[0].param_groups[0]
    assert group["lr"] == local.sgd.learning_rate
    assert (group["momentum"], group["dampening"], group["weight_decay"], group["nesterov"]) == (
        local.sgd.momentum, local.sgd.dampening, local.sgd.weight_decay, local.sgd.nesterov)
    assert created[0].steps == local.local_epochs * len(participant.trainloader)


def test_robust_changes_nothing_a_client_executes():
    fedavg = execution_plan.resolve_model_training("TINYNET_GOLDEN", "FedAvg", "FULL")
    robust = execution_plan.resolve_model_training("TINYNET_GOLDEN", "Robust", "FULL")
    assert robust == fedavg


def test_fedopt_trains_at_a_different_rate_than_fedavg():
    """Guards the parametrized test above against passing vacuously: FedOpt's server really does move the
    client rate, so the FedOpt plan must follow it rather than copy FedAvg's."""
    fedavg = execution_plan.resolve_model_training("TINYNET_GOLDEN", "FedAvg", "FULL")
    fedopt = execution_plan.resolve_model_training("TINYNET_GOLDEN", "FedOpt", "FULL")
    assert fedopt.local_training.sgd.learning_rate != fedavg.local_training.sgd.learning_rate


def test_the_fedprox_plan_states_the_mu_its_server_sends_and_the_client_applies(monkeypatch):
    """FedProx's only client-side difference is the proximal term, so its plan must state the server's mu -- the
    value client.py hands the proximal gradient every step."""
    import client
    plan = execution_plan.resolve_model_training("TINYNET_GOLDEN", "FedProx", "FULL")
    config = _config_the_server_sends("fedprox")
    participant = _tinynet_fedavg_client(monkeypatch)
    applied = []
    real = client._apply_proximal_gradient
    monkeypatch.setattr(client, "_apply_proximal_gradient",
                        lambda net, anchor, mu: (applied.append(mu), real(net, anchor, mu)))

    participant.fit(participant.get_parameters(), config)

    assert plan.HasField("fedprox_mu")
    assert plan.fedprox_mu == float(config["proximal_mu"]) > 0.0
    assert applied and set(applied) == {plan.fedprox_mu}


def test_the_phones_fedprox_golden_uses_the_servers_proximal_coefficient():
    """The native proximal term is proven against a framework golden; it must be the coefficient the contract states."""
    import strategy_client_settings
    path = os.path.join(os.path.dirname(GOLDEN_CONTRACT), "..", "decomfl_golden", "fedprox_local_manifest.json")
    with open(path) as fh:
        assert json.load(fh)["proximal_mu"] == strategy_client_settings.FEDPROX_CLIENT_PROXIMAL_MU


@pytest.mark.parametrize("strategy", ["FedAvg", "FedOpt", "Robust", "DeComFL"])
def test_only_the_fedprox_plan_states_a_mu(strategy):
    assert not execution_plan.resolve_model_training("TINYNET_GOLDEN", strategy, "FULL").HasField("fedprox_mu")


def test_fedprox_trains_what_fedopts_clients_train_apart_from_the_proximal_term():
    """TinyNet's data is one batch, so one epoch is one step, and at a round's first step w == w_global: the proximal
    gradient is exactly zero. At the server's real settings a FedProx round trains exactly what a FedOpt client
    does. Pinned so the equivalence is known, not rediscovered from a live run that cannot tell them apart."""
    fedopt = execution_plan.resolve_model_training("TINYNET_GOLDEN", "FedOpt", "FULL")
    fedprox = execution_plan.resolve_model_training("TINYNET_GOLDEN", "FedProx", "FULL")
    assert fedprox.local_training == fedopt.local_training
    assert fedprox.local_training.local_epochs == 1 and fedprox.local_training.batch_size == 8


def _decomfl_round_the_server_sends():
    """What the real FL server sends a DeComFL client for round 1, read off the wire.

    fl_server.select_strategy builds the strategy exactly as a spawned server does; a real coordinator and the real
    servicer answer GetDeComFLConfig with it.
    """
    from argparse import Namespace
    import fl_server
    from fedlearn.communication.generated import fedlearn_pb2
    from fedlearn.server.coordinator import FLCoordinator
    from fedlearn.server.grpc_servicer import FederatedLearningServiceServicer
    initial = OrderedDict((name, p.detach().clone())
                          for name, p in recipes.get_recipe("TINYNET_GOLDEN").build_model("cpu").named_parameters()
                          if p.requires_grad)
    strategy = fl_server.select_strategy(
        Namespace(strategy="decomfl", min_clients=4, clients_per_round=4, dataset="cb", aggregation="FFA_LORA"),
        initial, None)
    coordinator = FLCoordinator(strategy=strategy, min_clients_for_aggregation=4, clients_per_round=4)
    coordinator.set_initial_parameters(initial)
    return FederatedLearningServiceServicer(coordinator).GetDeComFLConfig(
        fedlearn_pb2.GetDeComFLConfigRequest(client_id="c"), None)


def test_tinynet_decomfl_plan_is_what_the_decomfl_server_asks_clients_to_train():
    plan = execution_plan.resolve_model_training("TINYNET_GOLDEN", "DeComFL", "FULL")
    response = _decomfl_round_the_server_sends()
    seeds = [list(step.seeds) for step in response.current_seeds.local_steps]

    zo = plan.local_training.zeroth_order_sgd
    assert plan.local_training.WhichOneof("optimizer") == "zeroth_order_sgd"
    assert zo.learning_rate == float(response.config["learning_rate"])
    assert zo.smoothing == float(response.config["smoothing_param"])
    assert zo.num_local_steps == int(response.config["num_local_steps"]) == len(seeds)
    assert zo.num_perturbations == int(response.config["num_perturbations"]) == len(seeds[0])
    assert zo.estimator == {"forward": pb.ESTIMATOR_FORWARD, "central": pb.ESTIMATOR_CENTRAL}[
        response.grad_estimate_method]
    assert plan.update_protocol == pb.UPDATE_DECOMFL_SCALAR
    assert plan.local_training.local_epochs == 0 and not plan.local_training.HasField("max_local_steps")


def test_tinynet_decomfl_draws_perturbations_with_the_generator_the_contract_names():
    import torch
    from fedlearn.estimators.perturbation import canonical_perturbation
    plan = execution_plan.resolve_model_training("TINYNET_GOLDEN", "DeComFL", "FULL")
    assert plan.local_training.zeroth_order_sgd.rng == pb.RNG_TORCH_CPU_RANDN_F32
    size = sum(t.shape[0] * (t.shape[1] if len(t.shape) > 1 else 1) for t in plan.trainable)
    for seed in (0, 42, 2_147_483_646):
        expected = torch.randn(size, generator=torch.Generator(device="cpu").manual_seed(seed), dtype=torch.float32)
        assert torch.equal(canonical_perturbation(seed, size), expected)


def test_the_decomfl_plan_completes_a_valid_contract():
    with open(GOLDEN_CONTRACT, "rb") as fh:
        contract = parse_contract_binary(fh.read())
    contract.strategy = pb.STRATEGY_DECOMFL
    contract.model_training.CopyFrom(execution_plan.resolve_model_training("TINYNET_GOLDEN", "DeComFL", "FULL"))
    golden = parse_contract_binary(open(GOLDEN_CONTRACT, "rb").read()).model_training
    contract.model_training.model_revision = golden.model_revision
    contract.model_training.initial_state_sha256 = golden.initial_state_sha256
    contract.model_training.artifacts.extend(golden.artifacts)
    assert validate_contract(contract, reader_protocol_version=2) == []


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


@pytest.mark.parametrize("strategy", ["FedAvg", "FedOpt", "Robust", "FedProx", "DeComFL"])
def test_tinynet_trains_its_committed_fixture_data(strategy):
    """Every TinyNet participant trains the recipe's committed batch, served by the run's server: a fixture run, which
    release phones refuse. A run on participants' own data will state DATA_SOURCE_LOCAL_SNAPSHOT."""
    plan = execution_plan.resolve_model_training("TINYNET_GOLDEN", strategy, "FULL")
    assert plan.data.source == pb.DATA_SOURCE_FIXTURE


@pytest.mark.parametrize("strategy", ["FedAvg", "FedOpt", "Robust", "FedProx", "DeComFL"])
def test_a_run_on_participants_own_data_states_a_local_snapshot(strategy):
    """The run's intent decides the data source; everything else in the plan is the same run."""
    fixture = execution_plan.resolve_model_training("TINYNET_GOLDEN", strategy, "FULL")
    local = execution_plan.resolve_model_training("TINYNET_GOLDEN", strategy, "FULL", data_source="LOCAL_SNAPSHOT")
    assert local.data.source == pb.DATA_SOURCE_LOCAL_SNAPSHOT
    local.data.source = pb.DATA_SOURCE_FIXTURE
    local.local_training.batch_order = fixture.local_training.batch_order
    assert local == fixture


@pytest.mark.parametrize("strategy", ["FedAvg", "FedOpt", "Robust", "FedProx"])
def test_a_first_order_run_on_participants_own_data_trains_in_the_reproducible_batch_order(strategy):
    """A device's own dataset can exceed one batch; the seeded order keeps every multi-batch update replayable."""
    local = execution_plan.resolve_model_training("TINYNET_GOLDEN", strategy, "FULL", data_source="LOCAL_SNAPSHOT")
    assert local.local_training.batch_order == pb.BATCH_ORDER_SEEDED_PERMUTATION_V1
    assert local.local_training.drop_last is False


def test_a_fixture_run_keeps_the_laptops_shuffled_order():
    fixture = execution_plan.resolve_model_training("TINYNET_GOLDEN", "FedAvg", "FULL")
    assert fixture.local_training.batch_order == pb.BATCH_ORDER_SHUFFLED_EACH_EPOCH


def test_a_decomfl_run_on_participants_own_data_keeps_one_batch():
    """The phone trains DeComFL's dataset as one batch, so its order does not change to the minibatch one."""
    local = execution_plan.resolve_model_training("TINYNET_GOLDEN", "DeComFL", "FULL", data_source="LOCAL_SNAPSHOT")
    assert local.local_training.batch_order == pb.BATCH_ORDER_SHUFFLED_EACH_EPOCH


def test_an_unknown_data_source_is_refused():
    with pytest.raises(ValueError, match="data source"):
        execution_plan.resolve_model_training("TINYNET_GOLDEN", "FedAvg", "FULL", data_source="UNSPECIFIED")


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
    ("CNN", "FedProx", "FULL"),
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


def test_the_cli_passes_the_runs_data_source():
    out = _cli("--recipe", "TINYNET_GOLDEN", "--strategy", "FedAvg", "--training-arm", "FULL",
               "--data-source", "LOCAL_SNAPSHOT")
    assert out["modelTraining"]["data"]["source"] == "DATA_SOURCE_LOCAL_SNAPSHOT"


def test_the_cli_refuses_a_data_source_outside_its_vocabulary():
    script = os.path.join(os.path.dirname(__file__), "..", "execution_plan.py")
    done = subprocess.run([sys.executable, script, "--recipe", "TINYNET_GOLDEN", "--strategy", "FedAvg",
                           "--training-arm", "FULL", "--data-source", "UNSPECIFIED"], capture_output=True, text=True)
    assert done.returncode != 0


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


def test_the_backend_wrapper_runs_the_resolver_with_the_backends_argv(tmp_path):
    """ScriptExecutionPlanResolver runs this wrapper and reads the last line it prints."""
    path = tmp_path / "model.npz"
    _save_like_init_model(path, _tinynet_trainable_state())
    wrapper = os.path.join(os.path.dirname(__file__), "..", "run_execution_plan.sh")
    env = dict(os.environ, FEDLEARN_PYTHON=sys.executable)
    done = subprocess.run(["bash", wrapper, "--recipe", "TINYNET_GOLDEN", "--strategy", "FedAvg",
                           "--training-arm", "FULL", f"--initial-state={path}"],
                          capture_output=True, text=True, check=True, env=env)
    out = json.loads(done.stdout.strip().splitlines()[-1])
    assert out["representable"] is True
    assert out["modelTraining"]["initialStateSha256"] == _canonical_digest(_tinynet_trainable_state().items())


# --- the laptop client checks a published contract against what it executes ----------------------------

def _contract_for_this_client():
    """A published-style contract whose Python-owned part is this client's own plan."""
    with open(GOLDEN_CONTRACT, "rb") as fh:
        contract = parse_contract_binary(fh.read())
    plan = execution_plan.resolve_model_training("TINYNET_GOLDEN", "FedAvg", "FULL")
    training = contract.model_training
    for field in ("model_id", "arm", "task", "objective", "update_protocol", "frozen_state_sha256"):
        setattr(training, field, getattr(plan, field))
    training.ClearField("trainable")
    training.trainable.extend(plan.trainable)
    training.local_training.CopyFrom(plan.local_training)
    training.data.CopyFrom(plan.data)
    return contract


def _check(contract, **kwargs):
    kwargs.setdefault("project_id", contract.project_id)
    return execution_plan.check_contract(contract, "TINYNET_GOLDEN", "FedAvg", "FULL", **kwargs)


def test_a_contract_matching_this_clients_execution_is_accepted():
    assert _check(_contract_for_this_client()) == []


def test_this_client_refuses_a_run_on_participants_own_data():
    """A laptop client trains the recipe's fixture batch; it has no dataset snapshot, so it cannot join such a run."""
    contract = _contract_for_this_client()
    contract.model_training.data.source = pb.DATA_SOURCE_LOCAL_SNAPSHOT
    assert _check(contract) == ["modelTraining.data differs from what this client executes"]


def test_a_contract_for_different_training_is_refused():
    contract = _contract_for_this_client()
    contract.model_training.local_training.sgd.learning_rate = 0.02
    assert _check(contract) == ["modelTraining.localTraining differs from what this client executes"]


def test_a_contract_with_a_different_layout_is_refused():
    contract = _contract_for_this_client()
    first, second = list(contract.model_training.trainable)
    contract.model_training.ClearField("trainable")
    contract.model_training.trainable.extend([second, first])
    assert _check(contract) == ["modelTraining.trainable differs from what this client executes"]


def test_a_contract_for_a_different_frozen_backbone_is_refused():
    contract = _contract_for_this_client()
    contract.model_training.frozen_state_sha256 = "0" * 64
    assert _check(contract) == ["modelTraining.frozenStateSha256 differs from what this client executes"]


def test_a_contract_for_another_recipe_or_strategy_is_refused():
    assert execution_plan.check_contract(_contract_for_this_client(), "CNN", "FedAvg", "FULL",
                                         project_id=_contract_for_this_client().project_id)
    problems = execution_plan.check_contract(_contract_for_this_client(), "TINYNET_GOLDEN", "FedOpt", "FULL",
                                             project_id=_contract_for_this_client().project_id)
    assert problems == ["strategy STRATEGY_FEDAVG is not this client's FedOpt"]


def _fedprox_contract():
    contract = _contract_for_this_client()
    contract.strategy = pb.STRATEGY_FEDPROX
    plan = execution_plan.resolve_model_training("TINYNET_GOLDEN", "FedProx", "FULL")
    contract.model_training.local_training.CopyFrom(plan.local_training)
    contract.model_training.fedprox_mu = plan.fedprox_mu
    return contract


def test_a_fedprox_contract_stating_this_clients_proximal_term_is_accepted():
    contract = _fedprox_contract()
    assert execution_plan.check_contract(contract, "TINYNET_GOLDEN", "FedProx", "FULL",
                                         project_id=contract.project_id) == []


def test_a_fedprox_contract_with_another_proximal_coefficient_is_refused():
    contract = _fedprox_contract()
    contract.model_training.fedprox_mu = 0.2
    assert execution_plan.check_contract(contract, "TINYNET_GOLDEN", "FedProx", "FULL",
                                         project_id=contract.project_id) == [
        "modelTraining.fedproxMu differs from what this client executes"]


def test_an_invalid_contract_is_refused_with_its_issues():
    contract = _contract_for_this_client()
    contract.num_rounds = 0
    assert _check(contract) == ["ISSUE_OUT_OF_RANGE at numRounds"]


def test_a_contract_for_another_project_is_refused():
    contract = _contract_for_this_client()
    assert _check(contract, project_id="00000000-0000-4000-8000-000000000001") == [
        "ISSUE_IDENTITY_MISMATCH at projectId"]


def test_the_client_refuses_before_training_when_its_contract_disagrees(tmp_path, monkeypatch):
    import client
    from google.protobuf import json_format
    contract = _contract_for_this_client()
    contract.model_training.local_training.batch_size = 4
    path = tmp_path / "contract.json"
    path.write_text(json_format.MessageToJson(contract))
    args = client.parse_args(["--project-id", contract.project_id, "--server-address", "localhost:50000",
                              "--partition-id", "0", "--model-type", "TINYNET_GOLDEN", "--strategy", "FedAvg",
                              "--execution-contract", str(path)])
    with pytest.raises(SystemExit) as refused:
        client.enforce_execution_contract(args, "TINYNET_GOLDEN", "FULL")
    assert "localTraining" in str(refused.value)


def test_the_client_accepts_its_own_contract_and_ignores_its_absence(tmp_path):
    import client
    from google.protobuf import json_format
    contract = _contract_for_this_client()
    path = tmp_path / "contract.json"
    path.write_text(json_format.MessageToJson(contract))
    base = ["--project-id", contract.project_id, "--server-address", "localhost:50000", "--partition-id", "0",
            "--model-type", "TINYNET_GOLDEN", "--strategy", "FedAvg"]
    assert client.enforce_execution_contract(
        client.parse_args(base + ["--execution-contract", str(path)]), "TINYNET_GOLDEN", "FULL") == contract
    assert client.enforce_execution_contract(client.parse_args(base), "TINYNET_GOLDEN", "FULL") is None


def test_the_golden_contract_is_one_this_client_accepts_unmodified():
    """The published fixture must be a contract a real client trains under, not a template.

    The other golden-contract tests overwrite its plan fields with the resolved plan before checking it, so
    they pass whatever the fixture states. Every client now decides from the contract as published -- the
    phone projects it into the settings it trains with -- so the fixture itself has to be acceptable.
    """
    with open(GOLDEN_CONTRACT, "rb") as fh:
        contract = parse_contract_binary(fh.read())
    assert execution_plan.check_contract(
        contract, "TINYNET_GOLDEN", "FedAvg", "FULL",
        project_id=contract.project_id, run_id=contract.run_id) == []


# --- a contracted client trains only what its contract states, round by round -----------------------------

def _contract_for(strategy_name, pb_strategy):
    contract = _contract_for_this_client()
    contract.strategy = pb_strategy
    contract.model_training.local_training.CopyFrom(
        execution_plan.resolve_model_training("TINYNET_GOLDEN", strategy_name, "FULL").local_training)
    return contract


@pytest.mark.parametrize("config", [
    {"learning_rate": "0.02", "local_epochs": "1", "proximal_mu": "0.0"},
    {"learning_rate": "0.01", "local_epochs": "2", "proximal_mu": "0.0"},
], ids=["other_rate", "other_epochs"])
def test_a_contracted_client_refuses_a_round_the_contract_does_not_state(monkeypatch, config):
    """The server's per-round settings used to override the client's own. Under a contract they are a cross-check:
    a server asking for different training refuses the round before any optimizer exists."""
    import client
    monkeypatch.setattr(client, "EXECUTION_CONTRACT", _contract_for("FedOpt", pb.STRATEGY_FEDOPT))
    participant = _tinynet_fedavg_client(monkeypatch)
    created = _recording_sgd(monkeypatch)

    with pytest.raises(ValueError, match="execution contract"):
        participant.fit(participant.get_parameters(), config)
    assert created == []


def test_a_contracted_client_trains_when_the_server_confirms_its_contract(monkeypatch):
    import client
    contract = _contract_for("FedOpt", pb.STRATEGY_FEDOPT)
    monkeypatch.setattr(client, "EXECUTION_CONTRACT", contract)
    participant = _tinynet_fedavg_client(monkeypatch)
    created = _recording_sgd(monkeypatch)

    participant.fit(participant.get_parameters(), _config_the_server_sends("fedopt"))

    assert created[0].param_groups[0]["lr"] == contract.model_training.local_training.sgd.learning_rate


def test_a_contracted_fedavg_client_refuses_a_server_that_sends_another_rate(monkeypatch):
    import client
    monkeypatch.setattr(client, "EXECUTION_CONTRACT", _contract_for("FedAvg", pb.STRATEGY_FEDAVG))
    participant = _tinynet_fedavg_client(monkeypatch)

    with pytest.raises(ValueError, match="execution contract"):
        participant.fit(participant.get_parameters(), _config_the_server_sends("fedopt"))


@pytest.mark.parametrize("strategy_name, pb_strategy, config", [
    ("FedProx", pb.STRATEGY_FEDPROX, {"learning_rate": "0.01", "local_epochs": "1", "proximal_mu": "0.2"}),
    ("FedProx", pb.STRATEGY_FEDPROX, {"learning_rate": "0.01", "local_epochs": "1"}),
    ("FedAvg", pb.STRATEGY_FEDAVG, {"proximal_mu": "0.1"}),
    ("FedOpt", pb.STRATEGY_FEDOPT, {"learning_rate": "0.01", "local_epochs": "1", "proximal_mu": "0.1"}),
], ids=["fedprox_other_mu", "fedprox_server_sends_no_mu", "fedavg_server_sends_a_mu", "fedopt_server_sends_a_mu"])
def test_a_contracted_client_refuses_a_proximal_term_the_contract_does_not_state(
        monkeypatch, strategy_name, pb_strategy, config):
    """The proximal coefficient is a cross-check like the rate: absent means zero on both sides, so a FedProx round
    without the contract's mu, or a proximal term on any other strategy, is refused before any optimizer exists."""
    import client
    contract = _fedprox_contract() if strategy_name == "FedProx" else _contract_for(strategy_name, pb_strategy)
    monkeypatch.setattr(client, "EXECUTION_CONTRACT", contract)
    participant = _tinynet_fedavg_client(monkeypatch)
    created = _recording_sgd(monkeypatch)

    with pytest.raises(ValueError, match="execution contract"):
        participant.fit(participant.get_parameters(), config)
    assert created == []


def test_a_contracted_fedprox_client_trains_the_proximal_term_its_server_confirms(monkeypatch):
    import client
    contract = _fedprox_contract()
    monkeypatch.setattr(client, "EXECUTION_CONTRACT", contract)
    participant = _tinynet_fedavg_client(monkeypatch)
    created = _recording_sgd(monkeypatch)
    applied = []
    real = client._apply_proximal_gradient
    monkeypatch.setattr(client, "_apply_proximal_gradient",
                        lambda net, anchor, mu: (applied.append(mu), real(net, anchor, mu)))

    participant.fit(participant.get_parameters(), _config_the_server_sends("fedprox"))

    assert created[0].param_groups[0]["lr"] == contract.model_training.local_training.sgd.learning_rate
    assert applied and set(applied) == {contract.model_training.fedprox_mu}


def test_without_a_contract_the_server_rate_still_applies(monkeypatch):
    import client
    monkeypatch.setattr(client, "EXECUTION_CONTRACT", None)
    participant = _tinynet_fedavg_client(monkeypatch)
    created = _recording_sgd(monkeypatch)

    participant.fit(participant.get_parameters(), {"learning_rate": "0.02", "local_epochs": "1"})

    assert created[0].param_groups[0]["lr"] == 0.02


def test_accepting_a_contract_keeps_it_for_every_round(tmp_path, monkeypatch):
    import client
    from google.protobuf import json_format
    monkeypatch.setattr(client, "EXECUTION_CONTRACT", None)
    contract = _contract_for_this_client()
    path = tmp_path / "contract.json"
    path.write_text(json_format.MessageToJson(contract))
    args = client.parse_args(["--project-id", contract.project_id, "--server-address", "localhost:50000",
                              "--partition-id", "0", "--model-type", "TINYNET_GOLDEN", "--strategy", "FedAvg",
                              "--execution-contract", str(path)])

    client.enforce_execution_contract(args, "TINYNET_GOLDEN", "FULL")

    assert client.EXECUTION_CONTRACT == contract


# --- a contracted DeComFL client holds each round's server config to the contract ------------------------------

def _decomfl_contract():
    contract = _contract_for_this_client()
    contract.strategy = pb.STRATEGY_DECOMFL
    plan = execution_plan.resolve_model_training("TINYNET_GOLDEN", "DeComFL", "FULL")
    contract.model_training.update_protocol = plan.update_protocol
    contract.model_training.local_training.CopyFrom(plan.local_training)
    return contract


def _decomfl_training_config():
    """What the laptop's DeComFL fit() receives for round 1: the server's config, its seeds, its estimator."""
    response = _decomfl_round_the_server_sends()
    config = dict(response.config)
    config["seeds"] = [list(step.seeds) for step in response.current_seeds.local_steps]
    config["grad_estimate_method"] = response.grad_estimate_method
    return config


def test_the_decomfl_guard_accepts_the_round_the_real_server_sends(monkeypatch):
    import client
    monkeypatch.setattr(client, "EXECUTION_CONTRACT", _decomfl_contract())
    client._refuse_decomfl_round_outside_contract(_decomfl_training_config())


@pytest.mark.parametrize("mutate", [
    lambda c: c.update({"learning_rate": "0.01"}),
    lambda c: c.update({"smoothing_param": "0.01"}),
    lambda c: c["seeds"].append(list(c["seeds"][0])),        # one more local step than the contract states
    lambda c: c["seeds"][0].pop(),                            # one fewer perturbation
    lambda c: c.update({"grad_estimate_method": "central"}),
    lambda c: c.pop("learning_rate"),                         # the client would fall back to its own default
], ids=["rate", "smoothing", "local_steps", "perturbations", "estimator", "rate_missing"])
def test_the_decomfl_guard_refuses_a_round_the_contract_does_not_state(monkeypatch, mutate):
    import client
    monkeypatch.setattr(client, "EXECUTION_CONTRACT", _decomfl_contract())
    config = _decomfl_training_config()
    mutate(config)
    with pytest.raises(ValueError, match="execution contract"):
        client._refuse_decomfl_round_outside_contract(config)


def test_without_a_contract_the_decomfl_guard_does_nothing(monkeypatch):
    import client
    monkeypatch.setattr(client, "EXECUTION_CONTRACT", None)
    client._refuse_decomfl_round_outside_contract({"learning_rate": "5"})


def test_a_contracted_decomfl_client_holds_its_rounds_to_the_contract(monkeypatch):
    import client
    net = recipes.get_recipe("TINYNET_GOLDEN").build_model("cpu")
    monkeypatch.setattr(client, "EXECUTION_CONTRACT", _decomfl_contract())
    assert client.build_decomfl_client(net, [], 0.001).round_check is client._refuse_decomfl_round_outside_contract
    monkeypatch.setattr(client, "EXECUTION_CONTRACT", None)
    assert client.build_decomfl_client(net, [], 0.001).round_check is None


# --- the catalog declares which recipes participants can train on their own data ----------------------------

def _declared_sources(meta):
    return meta.get("supported_data_sources", ["FIXTURE"])


@pytest.mark.parametrize("meta", recipes.RECIPE_METADATA, ids=lambda m: m["key"])
def test_every_declared_data_source_is_one_the_contract_can_state(meta):
    assert set(_declared_sources(meta)) <= set(execution_plan.DATA_SOURCES)
    assert "FIXTURE" in _declared_sources(meta)


@pytest.mark.parametrize("meta", recipes.RECIPE_METADATA, ids=lambda m: m["key"])
def test_a_recipe_offers_participants_own_data_only_with_a_plan_that_states_it(meta):
    """The picker offers a run on participants' own data from this declaration; a recipe without an execution plan
    has no contract to tell a device what data to bring, so it must not declare one."""
    planned = [strategy for (key, strategy, arm) in execution_plan._PLANS if key == meta["key"]]
    if "LOCAL_SNAPSHOT" in _declared_sources(meta):
        assert planned, f"{meta['key']} declares LOCAL_SNAPSHOT but has no execution plan"
        for strategy in planned:
            plan = execution_plan.resolve_model_training(meta["key"], strategy, "FULL", data_source="LOCAL_SNAPSHOT")
            assert plan.data.source == pb.DATA_SOURCE_LOCAL_SNAPSHOT
    else:
        assert not planned, f"{meta['key']} has an execution plan but does not offer participants' own data"


def test_the_catalog_describes_the_data_sources_a_recipe_supports():
    tinynet = next(r for r in recipes.describe() if r["key"] == "TINYNET_GOLDEN")
    assert tinynet["supported_data_sources"] == ["FIXTURE", "LOCAL_SNAPSHOT"]
