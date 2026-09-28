"""Resolve the training plan an execution contract publishes for a run.

The backend builds a run's execution contract (``proto/fedlearn/contract/v1``) from its recorded intent, its
staged artifacts and this plan: the part Python owns because Python runs it -- objective, update protocol,
the ordered trainable layout, local training (optimizer, every hyperparameter, step budget and batching) and
the data requirement, plus the canonical frozen- and initial-state digests. The plan states what ``client.py`` actually executes; ``tests/test_execution_plan.py``
compares it with the objects the client builds, so a change to either fails the tests instead of drifting.

A run is representable only when a plan for its recipe, strategy and arm is written below; anything else
raises NotRepresentable rather than being guessed.

    python execution_plan.py --recipe TINYNET_GOLDEN --strategy FedAvg --training-arm FULL \
        [--initial-state <model file the FL server loads>]

prints ``{"representable": true, "modelTraining": <ProtoJSON ModelTraining>}``, or
``{"representable": false, "reason": "..."}`` for a run that has no v1 plan.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys

from google.protobuf import json_format

import recipes
import strategy_client_settings
from fedlearn.communication.generated import execution_contract_pb2 as pb

_OBJECTIVES = {"cross_entropy": pb.OBJECTIVE_CROSS_ENTROPY, "one_vs_all": pb.OBJECTIVE_ONE_VS_ALL}
_STRATEGIES = {"DeComFL": pb.STRATEGY_DECOMFL, "FedAvg": pb.STRATEGY_FEDAVG, "FedProx": pb.STRATEGY_FEDPROX,
               "FedOpt": pb.STRATEGY_FEDOPT, "Robust": pb.STRATEGY_ROBUST}

# The fedlearn.v2 control/round protocol this client speaks.
CLIENT_PROTOCOL_VERSION = 2

# The contract fields the plan states: a client refuses a contract that differs in any of them from its own plan.
_PLAN_FIELDS = (("model_id", "modelId"), ("arm", "arm"), ("task", "task"), ("objective", "objective"),
                ("update_protocol", "updateProtocol"), ("trainable", "trainable"),
                ("frozen_state_sha256", "frozenStateSha256"), ("local_training", "localTraining"),
                ("data", "data"), ("fedprox_mu", "fedproxMu"))


class NotRepresentable(Exception):
    """The run has no v1 execution plan."""


def label_schema_id(classes) -> str:
    """A label-schema identifier bound to the recipe's ordered class names."""
    canonical = json.dumps(list(classes), separators=(",", ":")).encode("utf-8")
    return "labels-sha256:" + hashlib.sha256(canonical).hexdigest()


def state_sha256(tensors) -> str:
    """The canonical digest of an ordered state: SHA-256 of its float32 safetensors encoding, no metadata."""
    from fedlearn.communication.safetensors_codec import save_safetensors
    return hashlib.sha256(save_safetensors(list(tensors))).hexdigest()


def _frozen_state_sha256(model, layout) -> str:
    """Digest of every state tensor outside the trainable layout, in state_dict order."""
    import torch
    trainable = {spec.name for spec in layout}
    frozen = []
    for name, tensor in model.state_dict().items():
        if name in trainable:
            continue
        if tensor.dtype != torch.float32:
            raise NotRepresentable(f"frozen tensor {name} is {tensor.dtype}; v1 describes float32 state only")
        frozen.append((name, tensor.detach().cpu().numpy()))
    return state_sha256(frozen)


def _initial_state_sha256(path, recipe_key, training_arm, layout) -> str:
    """Digest of the initial federated state the FL server loads from ``path``, in contract order.

    The server loads the .npz that init_model.py wrote (or the registry head on a continued run), keeps its
    float32 tensors and the arm's prefixes, and federates the rest of the file as the global model. That set
    must be exactly the trainable layout with the same shapes, or the contract would describe a different
    model than the server distributes.
    """
    from collections import OrderedDict

    import numpy as np
    import torch
    from fedlearn.estimators.params import federable_state

    with np.load(path, allow_pickle=False) as npz:
        loaded = OrderedDict((key.replace("__DOT__", "."), torch.from_numpy(npz[key])) for key in npz.files)
    federated = federable_state(loaded)
    prefixes = recipes.trainable_prefixes(recipe_key, training_arm)
    if prefixes is not None:
        federated = OrderedDict((k, v) for k, v in federated.items() if k.startswith(tuple(prefixes)))
    expected = [spec.name for spec in layout]
    if set(federated) != set(expected):
        raise NotRepresentable(
            f"the server's initial model federates {sorted(federated)}, but the trainable layout is {expected}")
    for spec in layout:
        if list(federated[spec.name].shape) != list(spec.shape):
            raise NotRepresentable(
                f"initial tensor {spec.name} has shape {list(federated[spec.name].shape)}, "
                f"not {list(spec.shape)}")
    return state_sha256((spec.name, federated[spec.name].numpy()) for spec in layout)


def _trainable_layout(model):
    layout = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if str(param.dtype) != "torch.float32":
            raise NotRepresentable(f"trainable parameter {name} is {param.dtype}; the v1 wire is float32 only")
        layout.append(pb.TensorSpec(name=name, shape=list(param.shape), dtype=pb.DTYPE_F32))
    return layout


def _tinynet_first_order_full(learning_rate: float, local_epochs: int):
    """TINYNET_GOLDEN on the FULL arm under a first-order strategy whose client trains ``local_epochs`` epochs
    of SGD at ``learning_rate``: the client default for FedAvg and Robust, whose servers send no training
    settings, and the server's settings for FedOpt."""
    return lambda initial_state_path=None: _tinynet_first_order(learning_rate, local_epochs, initial_state_path)


def _tinynet_fedprox_full(initial_state_path=None) -> pb.ModelTraining:
    """TINYNET_GOLDEN under FedProx on the FULL arm: the first-order training the FedProx server sends (its rate and
    epochs), plus its proximal coefficient. client.py adds mu * (w - w_global) to every trainable gradient after
    backward and before the SGD step, with w_global the round's downloaded model."""
    plan = _tinynet_first_order(strategy_client_settings.FEDPROX_CLIENT_LEARNING_RATE,
                                strategy_client_settings.FEDPROX_CLIENT_LOCAL_EPOCHS, initial_state_path)
    plan.fedprox_mu = strategy_client_settings.FEDPROX_CLIENT_PROXIMAL_MU
    return plan


def _tinynet_decomfl_full(initial_state_path=None) -> pb.ModelTraining:
    """TINYNET_GOLDEN under DeComFL on the FULL arm: zeroth-order training, gradient scalars instead of weights.

    Every number is what the DeComFL server fl_server.py builds sends each round (``config.get_decomfl_config``,
    'default' because the backend passes no --dataset): the rate and smoothing, K local steps and P perturbations
    (the shape of the seed matrix), and the forward estimator. Perturbations come from
    fedlearn.estimators.perturbation.canonical_perturbation: seeded CPU torch.randn, float32.
    """
    from config import get_decomfl_config
    settings = get_decomfl_config("default")
    local = pb.LocalTraining(
        zeroth_order_sgd=pb.ZerothOrderSgd(
            learning_rate=settings.learning_rate, smoothing=settings.smoothing_param,
            num_local_steps=settings.num_local_steps, num_perturbations=settings.num_perturbations,
            estimator=pb.ESTIMATOR_FORWARD, rng=pb.RNG_TORCH_CPU_RANDN_F32),
        # No optimizer state exists to carry: each step is a fresh estimate from the round's seeds.
        reset_optimizer_each_round=True,
        batch_size=8,
        drop_last=False,
        batch_order=pb.BATCH_ORDER_SHUFFLED_EACH_EPOCH,
    )
    return _tinynet_full(pb.UPDATE_DECOMFL_SCALAR, local, initial_state_path)


def _tinynet_first_order(learning_rate: float, local_epochs: int, initial_state_path=None) -> pb.ModelTraining:
    """TINYNET_GOLDEN under a first-order strategy on the FULL arm, as client.py runs it.

    - Optimizer: client.train() builds torch.optim.SGD over the trainable parameters, anew on every call, with
      PyTorch's defaults for everything but the rate: CNN_LEARNING_RATE unless the server sends a
      learning_rate, which then overrides it.
    - Step budget: ZOSLClient.fit() trains the server's local_epochs, or one epoch when none is sent.
    - Batching: build_tinynet_golden_decomfl_loader() yields batches of 8, reshuffled every epoch, and
      keeps an incomplete final batch.
    - Layout: the recipe's model, which freezes fc2 by construction. Its seeded build is also the frozen
      state every peer rebuilds, so the frozen digest comes from it.
    """
    local = pb.LocalTraining(
        local_epochs=local_epochs,
        sgd=pb.Sgd(learning_rate=learning_rate, momentum=0.0, dampening=0.0, weight_decay=0.0, nesterov=False),
        reset_optimizer_each_round=True,
        batch_size=8,
        drop_last=False,
        batch_order=pb.BATCH_ORDER_SHUFFLED_EACH_EPOCH,
    )
    return _tinynet_full(pb.UPDATE_TRAINABLE_STATE_F32, local, initial_state_path)


def _tinynet_full(update_protocol, local_training, initial_state_path=None) -> pb.ModelTraining:
    """The parts of a TINYNET_GOLDEN FULL-arm plan every strategy shares: the recipe's model, which freezes fc2 by
    construction (its seeded build is also the frozen state every peer rebuilds), its trainable layout, and its data."""
    recipe = recipes.get_recipe("TINYNET_GOLDEN")
    model = recipe.build_model("cpu")
    width = model.fc1.in_features
    if model.fc2.out_features != len(recipe.classes):
        raise NotRepresentable("TINYNET_GOLDEN's output width disagrees with its class list")
    layout = _trainable_layout(model)
    plan = pb.ModelTraining(
        model_id=recipe.base_models[0],
        arm=pb.ARM_FULL,
        task=pb.TASK_VECTOR_CLASSIFICATION,
        objective=_OBJECTIVES[recipes.ARM_OBJECTIVES["FULL"]],
        update_protocol=update_protocol,
        trainable=layout,
        frozen_state_sha256=_frozen_state_sha256(model, layout),
        local_training=local_training,
        data=pb.DataRequirement(
            task=pb.TASK_VECTOR_CLASSIFICATION,
            input_shape=[width],
            input_dtype=pb.DTYPE_F32,
            class_count=len(recipe.classes),
            label_schema_id=label_schema_id(recipe.classes),
            transforms=[pb.Transform(identity_vector=pb.IdentityVector(width=width))],
            # By default every TinyNet client trains the recipe's committed fixture batch
            # (build_tinynet_golden_decomfl_loader), served to phones by the run's server: a test/demo run.
            # resolve_model_training restates it when the run trains on participants' own snapshots.
            source=pb.DATA_SOURCE_FIXTURE,
        ),
    )
    if initial_state_path is not None:
        plan.initial_state_sha256 = _initial_state_sha256(initial_state_path, "TINYNET_GOLDEN", "FULL", layout)
    return plan


# client.py's CNN_LEARNING_RATE and one epoch: what a client trains when its server sends no settings.
_CLIENT_DEFAULT_RATE, _CLIENT_DEFAULT_EPOCHS = 0.001, 1

_PLANS = {
    ("TINYNET_GOLDEN", "FedAvg", "FULL"): _tinynet_first_order_full(_CLIENT_DEFAULT_RATE, _CLIENT_DEFAULT_EPOCHS),
    # Robust aggregates on the server and sends clients nothing, so they train exactly as under FedAvg.
    ("TINYNET_GOLDEN", "Robust", "FULL"): _tinynet_first_order_full(_CLIENT_DEFAULT_RATE, _CLIENT_DEFAULT_EPOCHS),
    ("TINYNET_GOLDEN", "FedOpt", "FULL"): _tinynet_first_order_full(
        strategy_client_settings.FEDOPT_CLIENT_LEARNING_RATE, strategy_client_settings.FEDOPT_CLIENT_LOCAL_EPOCHS),
    # FedProx: first-order training plus the proximal term the contract's fedprox_mu states.
    ("TINYNET_GOLDEN", "FedProx", "FULL"): _tinynet_fedprox_full,
    # DeComFL: zeroth-order training; the update is gradient scalars.
    ("TINYNET_GOLDEN", "DeComFL", "FULL"): _tinynet_decomfl_full,
}


# The run intent's TrainingDataSource names, as the backend passes them.
DATA_SOURCES = {"FIXTURE": pb.DATA_SOURCE_FIXTURE, "LOCAL_SNAPSHOT": pb.DATA_SOURCE_LOCAL_SNAPSHOT}


def resolve_model_training(recipe_key: str, strategy: str, training_arm: str,
                           initial_state_path: str | None = None, data_source: str = "FIXTURE") -> pb.ModelTraining:
    """The contract's ModelTraining fields that Python owns, for one run configuration.

    ``initial_state_path`` is the model file the FL server loads its initial global model from; when given,
    its digest is included. ``data_source`` is the run's: the recipe's committed fixture batch, or each
    participant's own dataset snapshot, which must match the plan's data requirement. Model revision, artifacts
    and strategy settings are added by the publisher from the staged bundle and the run record.
    """
    if data_source not in DATA_SOURCES:
        raise ValueError(f"unknown data source {data_source!r}")
    build = _PLANS.get((recipe_key, strategy, training_arm))
    if build is None:
        raise NotRepresentable(
            f"no execution contract v1 plan for recipe {recipe_key} with strategy {strategy} on arm "
            f"{training_arm}")
    plan = pb.ModelTraining()
    plan.CopyFrom(build(initial_state_path))
    plan.data.source = DATA_SOURCES[data_source]
    return plan


def check_contract(contract: pb.ExecutionContract, recipe_key: str, strategy: str, training_arm: str, *,
                   project_id: str | None = None, run_id: str | None = None) -> list[str]:
    """Why this client must refuse ``contract``; empty when it may train under it.

    The contract must be valid for this client and this run, must name the client's recipe and strategy, and
    must state exactly the plan the client executes -- training, layout, frozen state and data.
    """
    from fedlearn.contract import validate_contract

    issues = validate_contract(contract, reader_protocol_version=CLIENT_PROTOCOL_VERSION,
                               expected_run_id=run_id, expected_project_id=project_id)
    if issues:
        return sorted(f"{pb.ContractIssueCode.Name(i.code)} at {i.path or 'the root'}" for i in issues)
    problems = []
    if pb.Recipe.Name(contract.recipe) != "RECIPE_" + recipe_key:
        problems.append(f"recipe {pb.Recipe.Name(contract.recipe)} is not this client's {recipe_key}")
    if _STRATEGIES.get(strategy) != contract.strategy:
        problems.append(f"strategy {pb.Strategy.Name(contract.strategy)} is not this client's {strategy}")
    if problems:
        return problems
    try:
        plan = resolve_model_training(recipe_key, strategy, training_arm)
    except NotRepresentable as exc:
        return [str(exc)]
    return [f"modelTraining.{json_name} differs from what this client executes"
            for field, json_name in _PLAN_FIELDS
            if getattr(contract.model_training, field) != getattr(plan, field)]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--training-arm", required=True)
    parser.add_argument("--initial-state", help="the model file the FL server loads its initial model from")
    parser.add_argument("--data-source", choices=sorted(DATA_SOURCES), default="FIXTURE",
                        help="where participants' training data comes from")
    args = parser.parse_args(argv)
    try:
        plan = resolve_model_training(args.recipe, args.strategy, args.training_arm, args.initial_state,
                                      args.data_source)
    except NotRepresentable as exc:
        print(json.dumps({"representable": False, "reason": str(exc)}))
        return 0
    print(json.dumps({"representable": True, "modelTraining": json_format.MessageToDict(plan)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
