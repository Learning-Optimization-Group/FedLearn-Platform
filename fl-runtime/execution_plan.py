"""Resolve the training plan an execution contract publishes for a run.

The backend builds a run's execution contract (``proto/fedlearn/contract/v1``) from its recorded intent, its
staged artifacts and this plan: the part Python owns because Python runs it -- objective, update protocol,
the ordered trainable layout, local training (optimizer, every hyperparameter, step budget and batching) and
the data requirement. The plan states what ``client.py`` actually executes; ``tests/test_execution_plan.py``
compares it with the objects the client builds, so a change to either fails the tests instead of drifting.

A run is representable only when a plan for its recipe, strategy and arm is written below; anything else
raises NotRepresentable rather than being guessed.

    python execution_plan.py --recipe TINYNET_GOLDEN --strategy FedAvg --training-arm FULL

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
from fedlearn.communication.generated import execution_contract_pb2 as pb

_OBJECTIVES = {"cross_entropy": pb.OBJECTIVE_CROSS_ENTROPY, "one_vs_all": pb.OBJECTIVE_ONE_VS_ALL}


class NotRepresentable(Exception):
    """The run has no v1 execution plan."""


def label_schema_id(classes) -> str:
    """A label-schema identifier bound to the recipe's ordered class names."""
    canonical = json.dumps(list(classes), separators=(",", ":")).encode("utf-8")
    return "labels-sha256:" + hashlib.sha256(canonical).hexdigest()


def _trainable_layout(model):
    layout = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if str(param.dtype) != "torch.float32":
            raise NotRepresentable(f"trainable parameter {name} is {param.dtype}; the v1 wire is float32 only")
        layout.append(pb.TensorSpec(name=name, shape=list(param.shape), dtype=pb.DTYPE_F32))
    return layout


def _tinynet_fedavg_full() -> pb.ModelTraining:
    """TINYNET_GOLDEN under FedAvg on the FULL arm, as client.py runs it.

    - Optimizer: client.train() builds torch.optim.SGD over the trainable parameters with
      CNN_LEARNING_RATE and PyTorch's defaults for everything else, anew on every call.
    - Step budget: the FedAvg server sends no local_epochs, so ZOSLClient.fit() trains one epoch.
    - Batching: build_tinynet_golden_decomfl_loader() yields batches of 8, reshuffled every epoch, and
      keeps an incomplete final batch.
    - Layout: the recipe's model, which freezes fc2 by construction.
    """
    recipe = recipes.get_recipe("TINYNET_GOLDEN")
    model = recipe.build_model("cpu")
    width = model.fc1.in_features
    if model.fc2.out_features != len(recipe.classes):
        raise NotRepresentable("TINYNET_GOLDEN's output width disagrees with its class list")
    return pb.ModelTraining(
        model_id=recipe.base_models[0],
        arm=pb.ARM_FULL,
        task=pb.TASK_VECTOR_CLASSIFICATION,
        objective=_OBJECTIVES[recipes.ARM_OBJECTIVES["FULL"]],
        update_protocol=pb.UPDATE_TRAINABLE_STATE_F32,
        trainable=_trainable_layout(model),
        local_training=pb.LocalTraining(
            local_epochs=1,
            sgd=pb.Sgd(learning_rate=0.001, momentum=0.0, dampening=0.0, weight_decay=0.0, nesterov=False),
            reset_optimizer_each_round=True,
            batch_size=8,
            drop_last=False,
            batch_order=pb.BATCH_ORDER_SHUFFLED_EACH_EPOCH,
        ),
        data=pb.DataRequirement(
            task=pb.TASK_VECTOR_CLASSIFICATION,
            input_shape=[width],
            input_dtype=pb.DTYPE_F32,
            class_count=len(recipe.classes),
            label_schema_id=label_schema_id(recipe.classes),
            transforms=[pb.Transform(identity_vector=pb.IdentityVector(width=width))],
        ),
    )


_PLANS = {
    ("TINYNET_GOLDEN", "FedAvg", "FULL"): _tinynet_fedavg_full,
}


def resolve_model_training(recipe_key: str, strategy: str, training_arm: str) -> pb.ModelTraining:
    """The contract's ModelTraining fields that Python owns, for one run configuration.

    Model revision, state digests, artifacts and strategy settings are added by the publisher from the
    staged bundle and the run record.
    """
    build = _PLANS.get((recipe_key, strategy, training_arm))
    if build is None:
        raise NotRepresentable(
            f"no execution contract v1 plan for recipe {recipe_key} with strategy {strategy} on arm "
            f"{training_arm}")
    return build()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--training-arm", required=True)
    args = parser.parse_args(argv)
    try:
        plan = resolve_model_training(args.recipe, args.strategy, args.training_arm)
    except NotRepresentable as exc:
        print(json.dumps({"representable": False, "reason": str(exc)}))
        return 0
    print(json.dumps({"representable": True, "modelTraining": json_format.MessageToDict(plan)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
