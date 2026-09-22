"""Freeze the execution contract v1 golden fixtures (the cross-language schema contract).

Every language's generated reader (Python, Java, TypeScript) decodes these files and must recover
the same contract. Re-run only when the golden contract deliberately changes:

    cd framework && PYTHONPATH=src python tests/fixtures/execution_contract_v1/generate.py

It rewrites ``golden_tinynet_fedavg.binpb`` (canonical protobuf bytes) and
``golden_tinynet_fedavg.json`` (ProtoJSON). ``framework/tests/test_execution_contract_v1_fixtures.py``
fails if the committed files differ from this script's output.

The golden is a schema fixture for a TinyNet FedAvg run, not a published contract. Values that name
real committed artifacts are real: the three ``.pte`` paths, sizes and SHA-256 digests come from
``../decomfl_golden/``, ``required_operators`` is the union of those programs' operator tables, and
``initial_state_sha256`` is the digest of ``zo_state.safetensors``. Values for artifacts that do not
exist yet are fixture values: ``frozen_state_sha256`` is the SHA-256 of the text
``FROZEN_STATE_FIXTURE``, and the declared resource envelope is illustrative. The publication slices
replace both with measured, staged values.
"""
from __future__ import annotations

import hashlib
import json
import os

from google.protobuf import json_format

from fedlearn.communication.generated import execution_contract_pb2 as pb

HERE = os.path.dirname(os.path.abspath(__file__))
FROZEN_STATE_FIXTURE = hashlib.sha256(b"FROZEN_STATE_FIXTURE").hexdigest()

# Extracted with executorch.exir._serialize._program.deserialize_pte_binary from the three programs
# below; sorted, duplicates removed.
TINYNET_OPERATORS = [
    "aten::_log_softmax.out", "aten::addmm.out", "aten::alias_copy.out", "aten::argmax.out",
    "aten::div.out", "aten::exp.out", "aten::full_like.out", "aten::gather.out",
    "aten::le.Scalar_out", "aten::mm.out", "aten::mul.out", "aten::ne.Scalar_out", "aten::neg.out",
    "aten::permute_copy.out", "aten::relu.out", "aten::scalar_tensor.out",
    "aten::scatter.value_out", "aten::slice_copy.Tensor_out", "aten::squeeze_copy.dims_out",
    "aten::sub.out", "aten::sum.IntList_out", "aten::unsqueeze_copy.out", "aten::view_copy.out",
    "aten::where.self_out", "dim_order_ops::_to_dim_order_copy.out",
]


def build_golden() -> pb.ExecutionContract:
    """The TinyNet FedAvg golden contract, with every behavioral field set explicitly."""
    return pb.ExecutionContract(
        contract_version=1,
        min_client_protocol_version=2,
        run_id="4f2c8a1e-7b3d-4c59-9e21-6a0d5b8f3c17",
        project_id="9b1e6d3a-2c47-4f85-a0d9-3e7c1b5a8f64",
        recipe=pb.RECIPE_TINYNET_GOLDEN,
        strategy=pb.STRATEGY_FEDAVG,
        num_rounds=3,
        clients_per_round=4,
        partitioning=pb.PARTITIONING_SHARDED,
        seed=42,
        round=pb.RoundPolicy(
            timeout_ms=900_000,
            one_accepted_update_per_round=True,
            max_transient_retries=3,
            retry_backoff_ms=1_000,
        ),
        security=pb.SecurityPolicy(
            transport=pb.TRANSPORT_TLS_REQUIRED,
            client_auth=pb.CLIENT_AUTH_CONNECTION_TOKEN,
            secure_aggregation=pb.SECAGG_NONE,
        ),
        model_training=pb.ModelTraining(
            model_id="tinynet_golden",
            model_revision="sha256:5dfd5242f551900d213a4f5a13dcbffd184c7a166fec96458f6f9fd3558e0656",
            arm=pb.ARM_FULL,
            task=pb.TASK_VECTOR_CLASSIFICATION,
            objective=pb.OBJECTIVE_CROSS_ENTROPY,
            update_protocol=pb.UPDATE_TRAINABLE_STATE_F32,
            trainable=[
                pb.TensorSpec(name="fc1.weight", shape=[5, 4], dtype=pb.DTYPE_F32),
                pb.TensorSpec(name="fc1.bias", shape=[5], dtype=pb.DTYPE_F32),
            ],
            frozen_state_sha256=FROZEN_STATE_FIXTURE,
            initial_state_sha256="4b1016fca301c00ba84cdf10ee9402e6d226e1471c1b698129e7f9e8c1f98179",
            local_training=pb.LocalTraining(
                local_epochs=5,
                sgd=pb.Sgd(learning_rate=0.001, momentum=0.0, dampening=0.0,
                           weight_decay=0.0, nesterov=False),
                reset_optimizer_each_round=True,
                batch_size=8,
                drop_last=False,
                batch_order=pb.BATCH_ORDER_SHUFFLED_EACH_EPOCH,
            ),
            data=pb.DataRequirement(
                task=pb.TASK_VECTOR_CLASSIFICATION,
                input_shape=[4],
                input_dtype=pb.DTYPE_F32,
                class_count=3,
                label_schema_id="tinynet_golden.labels.v1",
                transforms=[pb.Transform(identity_vector=pb.IdentityVector(width=4))],
            ),
            artifacts=[
                pb.ArtifactVariant(
                    variant_id="executorch-cpu-arm64-v8a",
                    backend=pb.BACKEND_EXECUTORCH_CPU,
                    abi="arm64-v8a",
                    files=[
                        pb.ArtifactRef(
                            relative_path="zo_model_tiny.pte",
                            sha256="2eca3c02e2084383f038494d6ecf7c20a1e7e0a1dcc6d7ce2b6e11e7d82f1c56",
                            byte_size=5836),
                        pb.ArtifactRef(
                            relative_path="zo_model_tiny_infer.pte",
                            sha256="cf8744b9579d78f14bbb82e2d4ce98dcaffc8d2c6ed2253349c39342de546746",
                            byte_size=2892),
                        pb.ArtifactRef(
                            relative_path="tinynet_trainable.pte",
                            sha256="ff398410f7339172295386dfc6220c5f46f21eddfb8ea145daf54e6a15dae412",
                            byte_size=12004),
                    ],
                    required_operators=TINYNET_OPERATORS,
                    declared_peak_memory_bytes=8 * 1024 * 1024,
                    declared_storage_bytes=64 * 1024,
                    declared_probe_ms=1_000,
                    declared_train_ms=5_000,
                ),
            ],
        ),
    )


def to_binary(contract: pb.ExecutionContract) -> bytes:
    return contract.SerializeToString(deterministic=True)


def to_json_text(contract: pb.ExecutionContract) -> str:
    return json.dumps(json_format.MessageToDict(contract), indent=2) + "\n"


def main() -> None:
    contract = build_golden()
    with open(os.path.join(HERE, "golden_tinynet_fedavg.binpb"), "wb") as fh:
        fh.write(to_binary(contract))
    with open(os.path.join(HERE, "golden_tinynet_fedavg.json"), "w", encoding="utf-8") as fh:
        fh.write(to_json_text(contract))


if __name__ == "__main__":
    main()
