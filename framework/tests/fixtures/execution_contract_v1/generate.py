"""Freeze the execution contract v1 golden fixtures (the cross-language schema contract).

Every language's generated reader (Python, Java, TypeScript) decodes these files and must recover
the same contract. Re-run only when the golden contract deliberately changes:

    cd framework && PYTHONPATH=src python tests/fixtures/execution_contract_v1/generate.py

It rewrites ``golden_tinynet_fedavg.binpb`` (canonical protobuf bytes),
``golden_tinynet_fedavg.json`` (ProtoJSON) and ``conformance.json`` (inputs with the exact issues
every reader must report; see ``README.md``). The framework tests fail if the committed files differ
from this script's output.

The golden is a schema fixture for a TinyNet FedAvg run, not a published contract. Values that name
real committed artifacts are real: the three ``.pte`` paths, sizes and SHA-256 digests come from
``../decomfl_golden/``, ``required_operators`` is the union of those programs' operator tables, and
``initial_state_sha256`` is the digest of ``zo_state.safetensors``. Values for artifacts that do not
exist yet are fixture values: ``frozen_state_sha256`` is the SHA-256 of the text
``FROZEN_STATE_FIXTURE``, and the declared resource envelope is illustrative. The publication slices
replace both with measured, staged values.
"""
from __future__ import annotations

import base64
import copy
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


# --- conformance corpus --------------------------------------------------------------------------

READER_PROTOCOL_VERSION = 2
GOLDEN_RUN_ID = "4f2c8a1e-7b3d-4c59-9e21-6a0d5b8f3c17"
GOLDEN_PROJECT_ID = "9b1e6d3a-2c47-4f85-a0d9-3e7c1b5a8f64"
OTHER_UUID = "00000000-0000-4000-8000-000000000001"

MT = "modelTraining"
LT = "modelTraining.localTraining"
DATA = "modelTraining.data"
ART = "modelTraining.artifacts"


def _issue(code: str, path: str) -> dict:
    return {"code": "ISSUE_" + code, "path": path}


def _case(case_id, description, issues, *, json_doc=None, json_text=None, binary=None, context=None):
    case = {"id": case_id, "description": description}
    if json_doc is not None:
        case["json"] = json_doc
    elif json_text is not None:
        case["jsonText"] = json_text
    else:
        case["binaryBase64"] = base64.b64encode(binary).decode("ascii")
    if context is not None:
        case["context"] = context
    case["issues"] = sorted((_issue(code, path) for code, path in issues),
                            key=lambda i: (i["path"], i["code"]))
    return case


def _golden_doc() -> dict:
    return json_format.MessageToDict(build_golden())


def _mutated(mutate) -> dict:
    doc = _golden_doc()
    mutate(doc)
    return doc


def _json(case_id, description, mutate, *issues, context=None):
    return _case(case_id, description, issues, json_doc=_mutated(mutate), context=context)


def _decomfl_secagg(doc, threshold=2):
    doc["strategy"] = "STRATEGY_DECOMFL"
    doc[MT]["updateProtocol"] = "UPDATE_DECOMFL_SCALAR"
    doc["security"]["secureAggregation"] = "SECAGG_LIGHTSECAGG_SCALAR"
    if threshold is not None:
        doc["security"]["secureAggThreshold"] = threshold


def _set_optimizer(doc, name, settings):
    local = doc[MT]["localTraining"]
    local.pop("sgd")
    local[name] = settings


def _binary_with(mutate) -> bytes:
    contract = build_golden()
    mutate(contract)
    return to_binary(contract)


VALID_ADAM = {"learningRate": 0.001, "beta1": 0.9, "beta2": 0.999, "epsilon": 1e-08,
              "weightDecay": 0.0, "amsgrad": False}
VALID_RMSPROP = {"learningRate": 0.01, "alpha": 0.99, "epsilon": 1e-08, "weightDecay": 0.0,
                 "momentum": 0.0, "centered": False}
VALID_TOKENIZER = {"relativePath": "tokenizer/tokenizer.json", "sha256": "a" * 64, "byteSize": "1024"}


def build_conformance() -> dict:
    """Valid and invalid inputs paired with the exact issue set a v1 reader must report."""
    golden_bytes = to_binary(build_golden())
    cases = [
        # --- accepted inputs --------------------------------------------------------------------
        _json("golden_json", "The golden ProtoJSON document is valid.", lambda d: None),
        _case("golden_binary", "The golden protobuf bytes are valid.", [], binary=golden_bytes),
        _json("context_matches", "Enrollment identities equal the contract's.", lambda d: None,
              context={"runId": GOLDEN_RUN_ID, "projectId": GOLDEN_PROJECT_ID}),
        _json("unknown_json_field_ignored", "An unknown ProtoJSON field is ignored.",
              lambda d: d.update({"futureDisplayLabel": "x"})),
        _case("unknown_binary_field_ignored", "An unknown protobuf field is ignored.", [],
              binary=golden_bytes + bytes([0x98, 0x06, 0x01])),
        _json("numeric_enum_json", "ProtoJSON accepts an enum as its number.",
              lambda d: d.update({"recipe": 1})),
        _json("original_field_name_json", "ProtoJSON accepts the proto field name.",
              lambda d: d.update({"num_rounds": d.pop("numRounds")})),
        _json("uint64_as_json_number", "ProtoJSON accepts a 64-bit integer as a JSON number.",
              lambda d: d["round"].update({"timeoutMs": 900000})),
        _json("zero_retries_explicit", "An explicit zero retry budget is legal.",
              lambda d: d["round"].update({"maxTransientRetries": 0})),
        _json("central_dp_valid", "A valid central-DP disclosure.",
              lambda d: d["security"].update(
                  {"centralDp": {"targetEpsilon": 1.0, "delta": 1e-05, "clipNorm": 1.0}})),
        _json("adam_valid", "Adam with explicit settings.",
              lambda d: _set_optimizer(d, "adam", dict(VALID_ADAM))),
        _json("adamw_valid", "AdamW with explicit settings.",
              lambda d: _set_optimizer(d, "adamw", dict(VALID_ADAM, weightDecay=0.01))),
        _json("rmsprop_valid", "RMSprop with explicit settings.",
              lambda d: _set_optimizer(d, "rmsprop", dict(VALID_RMSPROP))),
        _json("sgd_nesterov_valid", "Nesterov SGD with positive momentum and zero dampening.",
              lambda d: d[MT]["localTraining"]["sgd"].update({"momentum": 0.9, "nesterov": True})),
        _json("step_cap_and_clip_valid", "Optional step cap and clip norm when present and positive.",
              lambda d: d[MT]["localTraining"].update({"maxLocalSteps": 10,
                                                       "gradientClipNorm": 1.0})),
        _json("nested_artifact_path_valid", "A nested relative artifact path.",
              lambda d: d[MT]["artifacts"][0]["files"][0].update(
                  {"relativePath": "graphs/cpu/zo_model_tiny.pte"})),
        # --- parse failures -----------------------------------------------------------------------
        _case("json_not_object", "ProtoJSON must be an object.", [("MALFORMED", "")],
              json_text="[]"),
        _case("json_syntax_error", "Invalid JSON text.", [("MALFORMED", "")], json_text="{"),
        _case("json_bare_nan", "NaN is only valid as the ProtoJSON string \"NaN\".",
              [("MALFORMED", "")],
              json_text=json.dumps(_golden_doc()).replace('"learningRate": 0.001',
                                                          '"learningRate": NaN')),
        _json("json_wrong_type", "A string where an integer is required.",
              lambda d: d.update({"numRounds": "three"}), ("MALFORMED", "")),
        _json("json_negative_unsigned", "A negative unsigned integer.",
              lambda d: d.update({"numRounds": -1}), ("MALFORMED", "")),
        _json("json_uint32_overflow", "A uint32 above 2^32 - 1.",
              lambda d: d.update({"numRounds": 4294967296}), ("MALFORMED", "")),
        _json("json_fractional_integer", "A fractional integer.",
              lambda d: d.update({"numRounds": 1.5}), ("MALFORMED", "")),
        _case("binary_truncated", "Truncated protobuf bytes.", [("MALFORMED", "")],
              binary=golden_bytes[:-3]),
        _case("binary_zero_tag", "Field number zero is never valid.", [("MALFORMED", "")],
              binary=golden_bytes + b"\x00"),
        # --- version ------------------------------------------------------------------------------
        _json("version_2_stops_validation", "Another version reports only the version.",
              lambda d: d.update({"contractVersion": 2, "numRounds": 0}),
              ("UNSUPPORTED_CONTRACT_VERSION", "contractVersion")),
        _json("version_absent", "An absent version is not v1.",
              lambda d: d.pop("contractVersion"),
              ("UNSUPPORTED_CONTRACT_VERSION", "contractVersion")),
        # --- contract root ------------------------------------------------------------------------
        _json("min_protocol_absent", "A zero minimum protocol.",
              lambda d: d.pop("minClientProtocolVersion"),
              ("UNSUPPORTED_CLIENT_PROTOCOL", "minClientProtocolVersion")),
        _json("min_protocol_newer", "A minimum protocol newer than the reader.",
              lambda d: d.update({"minClientProtocolVersion": 3}),
              ("UNSUPPORTED_CLIENT_PROTOCOL", "minClientProtocolVersion")),
        _json("run_id_uppercase", "UUIDs are canonical lowercase.",
              lambda d: d.update({"runId": GOLDEN_RUN_ID.upper()}), ("INVALID_IDENTIFIER", "runId")),
        _json("run_id_trailing_newline", "Patterns match the whole string.",
              lambda d: d.update({"runId": GOLDEN_RUN_ID + "\n"}), ("INVALID_IDENTIFIER", "runId")),
        _json("project_id_not_uuid", "A project ID that is not a UUID.",
              lambda d: d.update({"projectId": "project"}), ("INVALID_IDENTIFIER", "projectId")),
        _json("run_id_mismatch", "The run differs from the enrollment.", lambda d: None,
              ("IDENTITY_MISMATCH", "runId"),
              context={"runId": OTHER_UUID, "projectId": GOLDEN_PROJECT_ID}),
        _json("project_id_mismatch", "The project differs from the enrollment.", lambda d: None,
              ("IDENTITY_MISMATCH", "projectId"), context={"projectId": OTHER_UUID}),
        _json("recipe_unknown_name", "An enum name this reader does not define.",
              lambda d: d.update({"recipe": "RECIPE_FUTURE"}), ("UNKNOWN_ENUM", "recipe")),
        _json("recipe_unknown_number", "An enum number this reader does not define.",
              lambda d: d.update({"recipe": 99}), ("UNKNOWN_ENUM", "recipe")),
        _json("strategy_unspecified", "An absent strategy.",
              lambda d: d.pop("strategy"), ("UNKNOWN_ENUM", "strategy")),
        _case("partitioning_unknown_number_binary", "An undefined enum number in protobuf bytes.",
              [("UNKNOWN_ENUM", "partitioning")],
              binary=_binary_with(lambda c: setattr(c, "partitioning", 7))),
        _json("num_rounds_absent", "Zero rounds.",
              lambda d: d.pop("numRounds"), ("OUT_OF_RANGE", "numRounds")),
        _json("num_rounds_max_uint32", "The largest uint32 is compared as unsigned.",
              lambda d: d.update({"numRounds": 4294967295}), ("OUT_OF_RANGE", "numRounds")),
        _json("clients_per_round_over_limit", "Clients per round above the v1 limit.",
              lambda d: d.update({"clientsPerRound": 10001}), ("OUT_OF_RANGE", "clientsPerRound")),
        _json("round_absent", "No round policy.", lambda d: d.pop("round"),
              ("MISSING_FIELD", "round")),
        _json("security_absent", "No security policy.", lambda d: d.pop("security"),
              ("MISSING_FIELD", "security")),
        _json("model_training_absent", "No workload; the matrix is not evaluated.",
              lambda d: d.pop(MT), ("MISSING_FIELD", MT)),
        _json("unsupported_recipe", "A recipe outside the v1 matrix.",
              lambda d: d.update({"recipe": "RECIPE_CNN"}), ("UNSUPPORTED_COMBINATION", "")),
        _json("unsupported_strategy_fedopt", "FedOpt is not yet in the v1 matrix.",
              lambda d: d.update({"strategy": "STRATEGY_FEDOPT"}), ("UNSUPPORTED_COMBINATION", "")),
        _json("decomfl_with_weight_update", "DeComFL paired with a weight update.",
              lambda d: d.update({"strategy": "STRATEGY_DECOMFL"}), ("UNSUPPORTED_COMBINATION", "")),
        _json("fedprox_without_mu", "FedProx requires its coefficient.",
              lambda d: d.update({"strategy": "STRATEGY_FEDPROX"}),
              ("UNSUPPORTED_COMBINATION", ""), ("MISSING_FIELD", MT + ".fedproxMu")),
        _json("fedprox_negative_mu", "A negative proximal coefficient.",
              lambda d: (d.update({"strategy": "STRATEGY_FEDPROX"}),
                         d[MT].update({"fedproxMu": -0.1})),
              ("UNSUPPORTED_COMBINATION", ""), ("OUT_OF_RANGE", MT + ".fedproxMu")),
        _json("fedavg_with_explicit_zero_mu", "Any proximal coefficient on FedAvg, even zero.",
              lambda d: d[MT].update({"fedproxMu": 0.0}),
              ("INVALID_STRATEGY_SETTINGS", MT + ".fedproxMu")),
        # --- round policy -------------------------------------------------------------------------
        _json("timeout_absent", "A zero timeout.", lambda d: d["round"].pop("timeoutMs"),
              ("OUT_OF_RANGE", "round.timeoutMs")),
        _json("timeout_max_uint64", "The largest uint64 is compared as unsigned.",
              lambda d: d["round"].update({"timeoutMs": "18446744073709551615"}),
              ("OUT_OF_RANGE", "round.timeoutMs")),
        _json("one_update_per_round_false", "v1 requires one accepted update per round.",
              lambda d: d["round"].update({"oneAcceptedUpdatePerRound": False}),
              ("OUT_OF_RANGE", "round.oneAcceptedUpdatePerRound")),
        _json("retries_absent", "The retry budget is explicit.",
              lambda d: d["round"].pop("maxTransientRetries"),
              ("MISSING_FIELD", "round.maxTransientRetries")),
        _json("retries_over_limit", "More retries than v1 allows.",
              lambda d: d["round"].update({"maxTransientRetries": 11}),
              ("OUT_OF_RANGE", "round.maxTransientRetries")),
        _json("backoff_over_limit", "A backoff above one hour.",
              lambda d: d["round"].update({"retryBackoffMs": "3600001"}),
              ("OUT_OF_RANGE", "round.retryBackoffMs")),
        # --- security -----------------------------------------------------------------------------
        _json("transport_unspecified", "An absent transport.",
              lambda d: d["security"].pop("transport"), ("UNKNOWN_ENUM", "security.transport")),
        _json("client_auth_unknown_number", "An undefined client-auth number.",
              lambda d: d["security"].update({"clientAuth": 5}),
              ("UNKNOWN_ENUM", "security.clientAuth")),
        _json("secagg_on_weight_updates", "LightSecAgg claimed for FedAvg weight updates.",
              lambda d: d["security"].update({"secureAggregation": "SECAGG_LIGHTSECAGG_SCALAR",
                                              "secureAggThreshold": 2}),
              ("INVALID_SECURITY", "security.secureAggregation")),
        _json("secagg_decomfl_security_valid",
              "DeComFL LightSecAgg is consistent; only the matrix refuses it in v1.",
              _decomfl_secagg, ("UNSUPPORTED_COMBINATION", "")),
        _json("secagg_without_threshold", "LightSecAgg without a threshold.",
              lambda d: _decomfl_secagg(d, threshold=None),
              ("UNSUPPORTED_COMBINATION", ""), ("MISSING_FIELD", "security.secureAggThreshold")),
        _json("secagg_threshold_above_clients", "A threshold above clients per round.",
              lambda d: _decomfl_secagg(d, threshold=5),
              ("UNSUPPORTED_COMBINATION", ""), ("OUT_OF_RANGE", "security.secureAggThreshold")),
        _json("secagg_threshold_one", "A threshold below two.",
              lambda d: _decomfl_secagg(d, threshold=1),
              ("UNSUPPORTED_COMBINATION", ""), ("OUT_OF_RANGE", "security.secureAggThreshold")),
        _json("threshold_without_secagg", "A threshold on a run without secure aggregation.",
              lambda d: d["security"].update({"secureAggThreshold": 2}),
              ("INVALID_SECURITY", "security.secureAggThreshold")),
        _json("central_dp_epsilon_nan", "A NaN privacy budget.",
              lambda d: d["security"].update(
                  {"centralDp": {"targetEpsilon": "NaN", "delta": 1e-05, "clipNorm": 1.0}}),
              ("OUT_OF_RANGE", "security.centralDp.targetEpsilon")),
        _json("central_dp_delta_one", "Delta must be below one.",
              lambda d: d["security"].update(
                  {"centralDp": {"targetEpsilon": 1.0, "delta": 1.0, "clipNorm": 1.0}}),
              ("OUT_OF_RANGE", "security.centralDp.delta")),
        _json("central_dp_clip_infinite", "An infinite clip norm.",
              lambda d: d["security"].update(
                  {"centralDp": {"targetEpsilon": 1.0, "delta": 1e-05, "clipNorm": "Infinity"}}),
              ("OUT_OF_RANGE", "security.centralDp.clipNorm")),
        _json("central_dp_empty", "A present but empty central-DP message.",
              lambda d: d["security"].update({"centralDp": {}}),
              ("OUT_OF_RANGE", "security.centralDp.clipNorm"),
              ("OUT_OF_RANGE", "security.centralDp.delta"),
              ("OUT_OF_RANGE", "security.centralDp.targetEpsilon")),
        # --- model training -----------------------------------------------------------------------
        _json("model_id_with_space", "A model ID outside its format.",
              lambda d: d[MT].update({"modelId": "tinynet golden"}),
              ("INVALID_IDENTIFIER", MT + ".modelId")),
        _json("model_revision_absent", "An empty revision.",
              lambda d: d[MT].pop("modelRevision"), ("INVALID_IDENTIFIER", MT + ".modelRevision")),
        _json("arm_unknown_skips_matrix", "An unknown arm; the matrix is not evaluated.",
              lambda d: d[MT].update({"arm": "ARM_FUTURE"}), ("UNKNOWN_ENUM", MT + ".arm")),
        _json("objective_one_vs_all", "One-vs-all is not in the v1 matrix.",
              lambda d: d[MT].update({"objective": "OBJECTIVE_ONE_VS_ALL"}),
              ("UNSUPPORTED_COMBINATION", "")),
        _json("frozen_hash_uppercase", "Digests are lowercase.",
              lambda d: d[MT].update({"frozenStateSha256": d[MT]["frozenStateSha256"].upper()}),
              ("INVALID_HASH", MT + ".frozenStateSha256")),
        _json("initial_hash_short", "A 63-character digest.",
              lambda d: d[MT].update({"initialStateSha256": d[MT]["initialStateSha256"][:-1]}),
              ("INVALID_HASH", MT + ".initialStateSha256")),
        _json("local_training_absent", "No local-training settings.",
              lambda d: d[MT].pop("localTraining"), ("MISSING_FIELD", LT)),
        _json("data_absent", "No data requirement.",
              lambda d: d[MT].pop("data"), ("MISSING_FIELD", DATA)),
        # --- trainable layout ---------------------------------------------------------------------
        _json("trainable_empty", "No trainable tensors.",
              lambda d: d[MT].pop("trainable"), ("MALFORMED_LAYOUT", MT + ".trainable")),
        _json("trainable_too_many", "More tensors than v1 allows; elements are not inspected.",
              lambda d: d[MT].update({"trainable": [{}] * 4097}),
              ("MALFORMED_LAYOUT", MT + ".trainable")),
        _json("trainable_duplicate_name", "A repeated tensor name.",
              lambda d: d[MT]["trainable"][1].update({"name": "fc1.weight"}),
              ("MALFORMED_LAYOUT", MT + ".trainable[1].name")),
        _json("trainable_empty_name_segment", "A tensor name with an empty segment.",
              lambda d: d[MT]["trainable"][0].update({"name": "fc1..weight"}),
              ("MALFORMED_LAYOUT", MT + ".trainable[0].name")),
        _json("trainable_name_too_long", "A 257-character tensor name.",
              lambda d: d[MT]["trainable"][0].update({"name": "w" * 257}),
              ("MALFORMED_LAYOUT", MT + ".trainable[0].name")),
        _json("trainable_name_at_limit_valid", "A 256-character tensor name.",
              lambda d: d[MT]["trainable"][0].update({"name": "w" * 256})),
        _json("trainable_scalar_shape", "A tensor without dimensions.",
              lambda d: d[MT]["trainable"][1].pop("shape"),
              ("MALFORMED_LAYOUT", MT + ".trainable[1].shape")),
        _json("trainable_zero_extent", "A zero extent.",
              lambda d: d[MT]["trainable"][0].update({"shape": ["5", "0"]}),
              ("MALFORMED_LAYOUT", MT + ".trainable[0].shape")),
        _json("trainable_rank_nine", "More dimensions than v1 allows.",
              lambda d: d[MT]["trainable"][0].update({"shape": ["1"] * 9}),
              ("MALFORMED_LAYOUT", MT + ".trainable[0].shape")),
        _json("trainable_tensor_too_large", "One tensor above MAX_ELEMENTS.",
              lambda d: d[MT]["trainable"][0].update({"shape": ["65536", "65536"]}),
              ("MALFORMED_LAYOUT", MT + ".trainable[0].shape")),
        _json("trainable_product_overflows_uint64", "An element count that overflows 64 bits.",
              lambda d: d[MT]["trainable"][0].update({"shape": ["2147483647"] * 8}),
              ("MALFORMED_LAYOUT", MT + ".trainable[0].shape")),
        _json("trainable_total_too_large", "Valid tensors whose total exceeds MAX_ELEMENTS.",
              lambda d: (d[MT]["trainable"][0].update({"shape": ["2147483647"]}),
                         d[MT]["trainable"][1].update({"shape": ["1"]})),
              ("MALFORMED_LAYOUT", MT + ".trainable")),
        _json("trainable_dtype_unspecified", "An absent tensor dtype.",
              lambda d: d[MT]["trainable"][0].pop("dtype"),
              ("UNKNOWN_ENUM", MT + ".trainable[0].dtype")),
        # --- local training -----------------------------------------------------------------------
        _json("local_epochs_absent", "Zero local epochs.",
              lambda d: d[MT]["localTraining"].pop("localEpochs"),
              ("OUT_OF_RANGE", LT + ".localEpochs")),
        _json("local_epochs_over_limit", "More epochs than v1 allows.",
              lambda d: d[MT]["localTraining"].update({"localEpochs": 1001}),
              ("OUT_OF_RANGE", LT + ".localEpochs")),
        _json("max_local_steps_explicit_zero", "A present step cap must be positive.",
              lambda d: d[MT]["localTraining"].update({"maxLocalSteps": 0}),
              ("OUT_OF_RANGE", LT + ".maxLocalSteps")),
        _json("clip_norm_negative", "A negative clip norm.",
              lambda d: d[MT]["localTraining"].update({"gradientClipNorm": -1.0}),
              ("OUT_OF_RANGE", LT + ".gradientClipNorm")),
        _json("optimizer_absent", "No optimizer.",
              lambda d: d[MT]["localTraining"].pop("sgd"), ("MISSING_FIELD", LT + ".optimizer")),
        _json("reset_rule_absent", "The optimizer-state rule is explicit.",
              lambda d: d[MT]["localTraining"].pop("resetOptimizerEachRound"),
              ("MISSING_FIELD", LT + ".resetOptimizerEachRound")),
        _json("drop_last_absent", "The final-batch rule is explicit.",
              lambda d: d[MT]["localTraining"].pop("dropLast"),
              ("MISSING_FIELD", LT + ".dropLast")),
        _json("batch_size_absent", "A zero batch size.",
              lambda d: d[MT]["localTraining"].pop("batchSize"),
              ("OUT_OF_RANGE", LT + ".batchSize")),
        _json("batch_size_over_limit", "A batch above v1's limit.",
              lambda d: d[MT]["localTraining"].update({"batchSize": 65537}),
              ("OUT_OF_RANGE", LT + ".batchSize")),
        _json("batch_order_unknown", "An undefined batch order.",
              lambda d: d[MT]["localTraining"].update({"batchOrder": 9}),
              ("UNKNOWN_ENUM", LT + ".batchOrder")),
        _json("sgd_learning_rate_absent", "A zero learning rate.",
              lambda d: d[MT]["localTraining"]["sgd"].pop("learningRate"),
              ("OUT_OF_RANGE", LT + ".sgd.learningRate")),
        _json("sgd_learning_rate_negative_infinity", "A non-finite learning rate.",
              lambda d: d[MT]["localTraining"]["sgd"].update({"learningRate": "-Infinity"}),
              ("OUT_OF_RANGE", LT + ".sgd.learningRate")),
        _json("sgd_momentum_absent", "Momentum is explicit even when zero.",
              lambda d: d[MT]["localTraining"]["sgd"].pop("momentum"),
              ("MISSING_FIELD", LT + ".sgd.momentum")),
        _json("sgd_weight_decay_negative", "Negative weight decay.",
              lambda d: d[MT]["localTraining"]["sgd"].update({"weightDecay": -0.01}),
              ("OUT_OF_RANGE", LT + ".sgd.weightDecay")),
        _json("sgd_nesterov_absent", "Nesterov is explicit even when false.",
              lambda d: d[MT]["localTraining"]["sgd"].pop("nesterov"),
              ("MISSING_FIELD", LT + ".sgd.nesterov")),
        _json("sgd_nesterov_without_momentum", "Nesterov needs positive momentum.",
              lambda d: d[MT]["localTraining"]["sgd"].update({"nesterov": True}),
              ("INVALID_OPTIMIZER", LT + ".sgd.nesterov")),
        _json("sgd_nesterov_with_dampening", "Nesterov needs zero dampening.",
              lambda d: d[MT]["localTraining"]["sgd"].update(
                  {"momentum": 0.9, "dampening": 0.1, "nesterov": True}),
              ("INVALID_OPTIMIZER", LT + ".sgd.nesterov")),
        _json("adam_beta1_one", "Beta1 must be below one.",
              lambda d: _set_optimizer(d, "adam", dict(VALID_ADAM, beta1=1.0)),
              ("OUT_OF_RANGE", LT + ".adam.beta1")),
        _json("adam_epsilon_absent", "A zero epsilon.",
              lambda d: _set_optimizer(d, "adam", {k: v for k, v in VALID_ADAM.items()
                                                   if k != "epsilon"}),
              ("OUT_OF_RANGE", LT + ".adam.epsilon")),
        _json("adam_amsgrad_absent", "AMSGrad is explicit even when false.",
              lambda d: _set_optimizer(d, "adam", {k: v for k, v in VALID_ADAM.items()
                                                   if k != "amsgrad"}),
              ("MISSING_FIELD", LT + ".adam.amsgrad")),
        _json("adamw_weight_decay_absent", "AdamW weight decay is explicit.",
              lambda d: _set_optimizer(d, "adamw", {k: v for k, v in VALID_ADAM.items()
                                                    if k != "weightDecay"}),
              ("MISSING_FIELD", LT + ".adamw.weightDecay")),
        _json("rmsprop_alpha_absent", "A zero smoothing constant.",
              lambda d: _set_optimizer(d, "rmsprop", {k: v for k, v in VALID_RMSPROP.items()
                                                      if k != "alpha"}),
              ("OUT_OF_RANGE", LT + ".rmsprop.alpha")),
        _json("rmsprop_centered_absent", "Centered is explicit even when false.",
              lambda d: _set_optimizer(d, "rmsprop", {k: v for k, v in VALID_RMSPROP.items()
                                                      if k != "centered"}),
              ("MISSING_FIELD", LT + ".rmsprop.centered")),
        # --- data requirement ---------------------------------------------------------------------
        _json("data_task_differs", "The data task disagrees with the training task.",
              lambda d: d[MT]["data"].update({"task": "TASK_IMAGE_CLASSIFICATION"}),
              ("INVALID_DATA_REQUIREMENT", DATA + ".task"),
              ("INVALID_DATA_REQUIREMENT", DATA + ".transforms[0].identityVector.width")),
        _json("data_task_unknown", "An unknown data task; task-dependent rules are skipped.",
              lambda d: d[MT]["data"].update({"task": "TASK_FUTURE"}),
              ("UNKNOWN_ENUM", DATA + ".task")),
        _json("input_shape_absent", "A sample without dimensions.",
              lambda d: d[MT]["data"].pop("inputShape"),
              ("INVALID_DATA_REQUIREMENT", DATA + ".inputShape")),
        _json("input_width_differs", "The identity width must equal the input width.",
              lambda d: d[MT]["data"].update({"inputShape": ["5"]}),
              ("INVALID_DATA_REQUIREMENT", DATA + ".transforms[0].identityVector.width")),
        _json("input_shape_rank_two", "An identity vector needs a rank-1 sample.",
              lambda d: d[MT]["data"].update({"inputShape": ["2", "2"]}),
              ("INVALID_DATA_REQUIREMENT", DATA + ".transforms[0].identityVector.width")),
        _json("input_dtype_absent", "An absent input dtype.",
              lambda d: d[MT]["data"].pop("inputDtype"), ("UNKNOWN_ENUM", DATA + ".inputDtype")),
        _json("class_count_one", "Classification needs at least two classes.",
              lambda d: d[MT]["data"].update({"classCount": 1}),
              ("OUT_OF_RANGE", DATA + ".classCount")),
        _json("label_schema_absent", "An empty label schema ID.",
              lambda d: d[MT]["data"].pop("labelSchemaId"),
              ("INVALID_IDENTIFIER", DATA + ".labelSchemaId")),
        _json("transforms_absent", "No transform.",
              lambda d: d[MT]["data"].pop("transforms"),
              ("INVALID_DATA_REQUIREMENT", DATA + ".transforms")),
        _json("transforms_too_many", "More transforms than v1 allows; elements are not inspected.",
              lambda d: d[MT]["data"].update({"transforms": [{}] * 17}),
              ("INVALID_DATA_REQUIREMENT", DATA + ".transforms")),
        _json("transform_without_operation", "A transform this reader does not understand.",
              lambda d: d[MT]["data"].update({"transforms": [{}]}),
              ("MISSING_FIELD", DATA + ".transforms[0].operation")),
        _json("identity_width_absent", "A zero identity width.",
              lambda d: d[MT]["data"].update({"transforms": [{"identityVector": {}}]}),
              ("OUT_OF_RANGE", DATA + ".transforms[0].identityVector.width")),
        _json("tokenizer_on_vector_task", "A tokenizer where no text is involved.",
              lambda d: d[MT]["data"].update({"tokenizer": dict(VALID_TOKENIZER)}),
              ("INVALID_DATA_REQUIREMENT", DATA + ".tokenizer")),
        _json("tokenizer_reference_invalid", "Tokenizer references obey artifact rules.",
              lambda d: d[MT]["data"].update(
                  {"tokenizer": {"relativePath": "../tokenizer.json", "sha256": "zz"}}),
              ("INVALID_DATA_REQUIREMENT", DATA + ".tokenizer"),
              ("INVALID_HASH", DATA + ".tokenizer.sha256"),
              ("INVALID_PATH", DATA + ".tokenizer.relativePath"),
              ("OUT_OF_RANGE", DATA + ".tokenizer.byteSize")),
        _json("causal_lm_rules", "Causal-LM data rules, outside the v1 matrix.",
              lambda d: (d[MT].update({"task": "TASK_CAUSAL_LM",
                                       "objective": "OBJECTIVE_CAUSAL_LM"}),
                         d[MT]["data"].update({"task": "TASK_CAUSAL_LM"})),
              ("UNSUPPORTED_COMBINATION", ""),
              ("INVALID_DATA_REQUIREMENT", DATA + ".classCount"),
              ("MISSING_FIELD", DATA + ".tokenizer"),
              ("INVALID_DATA_REQUIREMENT", DATA + ".transforms[0].identityVector.width")),
        # --- artifacts ----------------------------------------------------------------------------
        _json("artifacts_absent", "No artifact variant.",
              lambda d: d[MT].pop("artifacts"), ("MISSING_ARTIFACT", ART)),
        _json("artifacts_without_cpu", "The portable CPU baseline is mandatory.",
              lambda d: d[MT]["artifacts"][0].update({"backend": "BACKEND_EXECUTORCH_GPU"}),
              ("MISSING_ARTIFACT", ART)),
        _json("artifacts_too_many", "More variants than v1 allows; variants are not inspected.",
              lambda d: d[MT].update({"artifacts": [{}] * 17}), ("INVALID_ARTIFACT", ART)),
        _json("artifact_duplicate_variant", "A repeated variant ID.",
              lambda d: d[MT]["artifacts"].append(copy.deepcopy(d[MT]["artifacts"][0])),
              ("INVALID_ARTIFACT", ART + "[1].variantId")),
        _json("artifact_variant_id_invalid", "A variant ID outside its format.",
              lambda d: d[MT]["artifacts"][0].update({"variantId": "-cpu"}),
              ("INVALID_IDENTIFIER", ART + "[0].variantId")),
        _json("artifact_backend_unknown", "An unknown backend is also not a CPU baseline.",
              lambda d: d[MT]["artifacts"][0].update({"backend": 9}),
              ("UNKNOWN_ENUM", ART + "[0].backend"), ("MISSING_ARTIFACT", ART)),
        _json("artifact_abi_uppercase", "ABIs are lowercase tokens.",
              lambda d: d[MT]["artifacts"][0].update({"abi": "ARM64-V8A"}),
              ("INVALID_IDENTIFIER", ART + "[0].abi")),
        _json("artifact_files_absent", "A variant with no files.",
              lambda d: d[MT]["artifacts"][0].pop("files"),
              ("MISSING_ARTIFACT", ART + "[0].files")),
        _json("artifact_files_too_many", "More files than v1 allows; files are not inspected.",
              lambda d: d[MT]["artifacts"][0].update({"files": [{}] * 65}),
              ("INVALID_ARTIFACT", ART + "[0].files")),
        _json("artifact_path_traversal", "A parent-directory segment.",
              lambda d: d[MT]["artifacts"][0]["files"][0].update(
                  {"relativePath": "../zo_model_tiny.pte"}),
              ("INVALID_PATH", ART + "[0].files[0].relativePath")),
        _json("artifact_path_absolute", "An absolute path.",
              lambda d: d[MT]["artifacts"][0]["files"][0].update(
                  {"relativePath": "/data/zo_model_tiny.pte"}),
              ("INVALID_PATH", ART + "[0].files[0].relativePath")),
        _json("artifact_path_backslash", "A backslash separator.",
              lambda d: d[MT]["artifacts"][0]["files"][0].update(
                  {"relativePath": "graphs\\zo_model_tiny.pte"}),
              ("INVALID_PATH", ART + "[0].files[0].relativePath")),
        _json("artifact_path_hidden", "A segment starting with a dot.",
              lambda d: d[MT]["artifacts"][0]["files"][0].update({"relativePath": ".pte"}),
              ("INVALID_PATH", ART + "[0].files[0].relativePath")),
        _json("artifact_path_empty_segment", "A doubled separator.",
              lambda d: d[MT]["artifacts"][0]["files"][0].update(
                  {"relativePath": "graphs//zo_model_tiny.pte"}),
              ("INVALID_PATH", ART + "[0].files[0].relativePath")),
        _json("artifact_path_duplicate_ignoring_case", "Paths collide on case-insensitive storage.",
              lambda d: d[MT]["artifacts"][0]["files"][1].update(
                  {"relativePath": "ZO_MODEL_TINY.PTE"}),
              ("INVALID_PATH", ART + "[0].files[1].relativePath")),
        _json("artifact_path_empty", "An empty path.",
              lambda d: d[MT]["artifacts"][0]["files"][0].pop("relativePath"),
              ("INVALID_PATH", ART + "[0].files[0].relativePath")),
        _json("artifact_path_too_long", "A 256-character path.",
              lambda d: d[MT]["artifacts"][0]["files"][0].update({"relativePath": "a" * 256}),
              ("INVALID_PATH", ART + "[0].files[0].relativePath")),
        _json("artifact_path_limits_valid", "A 255-character path with eight segments.",
              lambda d: d[MT]["artifacts"][0]["files"][0].update(
                  {"relativePath": "/".join(["a"] * 7 + ["b" * 241])})),
        _json("artifact_path_too_deep", "Nine path segments.",
              lambda d: d[MT]["artifacts"][0]["files"][0].update(
                  {"relativePath": "/".join(["a"] * 8 + ["b.pte"])}),
              ("INVALID_PATH", ART + "[0].files[0].relativePath")),
        _json("artifact_hash_invalid", "A file digest outside its format.",
              lambda d: d[MT]["artifacts"][0]["files"][0].update({"sha256": "x" * 64}),
              ("INVALID_HASH", ART + "[0].files[0].sha256")),
        _json("artifact_size_absent", "A zero-byte file.",
              lambda d: d[MT]["artifacts"][0]["files"][0].pop("byteSize"),
              ("OUT_OF_RANGE", ART + "[0].files[0].byteSize")),
        _json("artifact_size_over_limit", "A file above MAX_FILE_BYTES; the storage sum is skipped.",
              lambda d: d[MT]["artifacts"][0]["files"][0].update({"byteSize": "68719476737"}),
              ("OUT_OF_RANGE", ART + "[0].files[0].byteSize")),
        _json("artifact_storage_below_files", "Declared storage smaller than the files.",
              lambda d: d[MT]["artifacts"][0].update({"declaredStorageBytes": "1000"}),
              ("INVALID_ARTIFACT", ART + "[0].declaredStorageBytes")),
        _json("artifact_operators_absent", "No required operators.",
              lambda d: d[MT]["artifacts"][0].pop("requiredOperators"),
              ("INVALID_ARTIFACT", ART + "[0].requiredOperators")),
        _json("artifact_operators_too_many", "More operators than v1 allows; not inspected.",
              lambda d: d[MT]["artifacts"][0].update({"requiredOperators": ["x"] * 4097}),
              ("INVALID_ARTIFACT", ART + "[0].requiredOperators")),
        _json("artifact_operator_wildcard", "A wildcard operator.",
              lambda d: d[MT]["artifacts"][0]["requiredOperators"].__setitem__(0, "aten::*"),
              ("INVALID_ARTIFACT", ART + "[0].requiredOperators[0]")),
        _json("artifact_operator_too_long", "A 129-character operator.",
              lambda d: d[MT]["artifacts"][0]["requiredOperators"].__setitem__(
                  0, "aten::" + "o" * 123),
              ("INVALID_ARTIFACT", ART + "[0].requiredOperators[0]")),
        _json("artifact_operator_duplicate", "A repeated operator.",
              lambda d: d[MT]["artifacts"][0]["requiredOperators"].__setitem__(
                  1, d[MT]["artifacts"][0]["requiredOperators"][0]),
              ("INVALID_ARTIFACT", ART + "[0].requiredOperators[1]")),
        _json("artifact_probe_absent", "A zero probe budget.",
              lambda d: d[MT]["artifacts"][0].pop("declaredProbeMs"),
              ("OUT_OF_RANGE", ART + "[0].declaredProbeMs")),
        _json("artifact_memory_over_limit", "Declared memory above MAX_DECLARED_BYTES.",
              lambda d: d[MT]["artifacts"][0].update({"declaredPeakMemoryBytes": "1099511627777"}),
              ("OUT_OF_RANGE", ART + "[0].declaredPeakMemoryBytes")),
    ]
    ids = [case["id"] for case in cases]
    assert len(ids) == len(set(ids)), "conformance case IDs must be unique"
    return {"readerProtocolVersion": READER_PROTOCOL_VERSION, "cases": cases}


def to_conformance_text(corpus: dict) -> str:
    """One case per line, so a changed case is a one-line diff."""
    lines = [json.dumps(case, separators=(",", ":")) for case in corpus["cases"]]
    return ('{"readerProtocolVersion":%d,"cases":[\n' % corpus["readerProtocolVersion"]
            + ",\n".join(lines) + "\n]}\n")


def main() -> None:
    contract = build_golden()
    with open(os.path.join(HERE, "golden_tinynet_fedavg.binpb"), "wb") as fh:
        fh.write(to_binary(contract))
    with open(os.path.join(HERE, "golden_tinynet_fedavg.json"), "w", encoding="utf-8") as fh:
        fh.write(to_json_text(contract))
    with open(os.path.join(HERE, "conformance.json"), "w", encoding="utf-8") as fh:
        fh.write(to_conformance_text(build_conformance()))


if __name__ == "__main__":
    main()
