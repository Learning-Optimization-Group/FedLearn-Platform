"""Parse and validate execution contract v1.

The rules, limits and issue paths are specified in
``framework/tests/fixtures/execution_contract_v1/README.md``; the conformance corpus beside it is
shared with the Java and TypeScript readers. A successful parse is never acceptance: callers must
validate before downloading a model or opening local data.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Optional

from google.protobuf import json_format
from google.protobuf.message import DecodeError

from fedlearn.communication.generated import execution_contract_pb2 as pb

CONTRACT_VERSION = 1

MAX_ROUNDS = 10_000
MAX_CLIENTS_PER_ROUND = 10_000
MAX_TIMEOUT_MS = 86_400_000
MAX_DECLARED_MS = 86_400_000
MAX_TRANSIENT_RETRIES = 10
MAX_RETRY_BACKOFF_MS = 3_600_000
MAX_TENSORS = 4096
MAX_RANK = 8
MAX_ELEMENTS = 2_147_483_647
MAX_LOCAL_EPOCHS = 1000
MAX_LOCAL_STEPS = 1_000_000
MAX_PERTURBATIONS = 10_000
MAX_BATCH_SIZE = 65_536
MAX_CLASSES = 1_000_000
MAX_TRANSFORMS = 16
MAX_VARIANTS = 16
MAX_FILES = 64
MAX_OPERATORS = 4096
MAX_FILE_BYTES = 1 << 36
MAX_DECLARED_BYTES = 1 << 40
MAX_TENSOR_NAME_LENGTH = 256
MAX_OPERATOR_LENGTH = 128
MAX_PATH_LENGTH = 255
MAX_PATH_SEGMENTS = 8

_UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
_MODEL_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_REVISION = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}")
_VARIANT_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
_ABI = re.compile(r"[a-z0-9][a-z0-9_-]{0,31}")
_TENSOR_NAME = re.compile(r"[A-Za-z0-9_]+(\.[A-Za-z0-9_]+)*")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_PATH_SEGMENT = re.compile(r"[A-Za-z0-9_-][A-Za-z0-9._-]*")
_OPERATOR = re.compile(r"[A-Za-z_][A-Za-z0-9_]*::[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)?")

APPROVED_MATRIX = frozenset({
    (pb.RECIPE_TINYNET_GOLDEN, strategy, pb.ARM_FULL, pb.TASK_VECTOR_CLASSIFICATION,
     pb.OBJECTIVE_CROSS_ENTROPY, pb.UPDATE_TRAINABLE_STATE_F32)
    # FedOpt and Robust are first-order client training too; their server-side work is not client behavior.
    # FedProx is first-order client training plus the proximal term its fedprox_mu states.
    for strategy in (pb.STRATEGY_FEDAVG, pb.STRATEGY_FEDOPT, pb.STRATEGY_ROBUST, pb.STRATEGY_FEDPROX)
} | {
    # DeComFL: zeroth-order training, gradient scalars instead of weights.
    (pb.RECIPE_TINYNET_GOLDEN, pb.STRATEGY_DECOMFL, pb.ARM_FULL, pb.TASK_VECTOR_CLASSIFICATION,
     pb.OBJECTIVE_CROSS_ENTROPY, pb.UPDATE_DECOMFL_SCALAR),
})

_CLASSIFICATION_TASKS = frozenset({pb.TASK_VECTOR_CLASSIFICATION, pb.TASK_IMAGE_CLASSIFICATION,
                                   pb.TASK_SEQUENCE_CLASSIFICATION})
_TEXT_TASKS = frozenset({pb.TASK_SEQUENCE_CLASSIFICATION, pb.TASK_CAUSAL_LM})


@dataclass(frozen=True)
class ContractIssue:
    """One reason a reader refuses a contract: a ContractIssueCode value and a ProtoJSON path."""
    code: int
    path: str


class MalformedContractError(ValueError):
    """The input does not parse as an ExecutionContract."""


def parse_contract_binary(data: bytes) -> pb.ExecutionContract:
    contract = pb.ExecutionContract()
    try:
        contract.ParseFromString(data)
    except DecodeError as exc:
        raise MalformedContractError("contract bytes do not parse") from exc
    return contract


def parse_contract_json(text: str) -> pb.ExecutionContract:
    try:
        # json_format.Parse accepts any top-level JSON value (a list reads as an empty message);
        # a ProtoJSON message must be an object.
        if not isinstance(json.loads(text), dict):
            raise MalformedContractError("contract ProtoJSON is not an object")
        return json_format.Parse(text, pb.ExecutionContract(), ignore_unknown_fields=True)
    except (json_format.ParseError, ValueError, TypeError) as exc:
        raise MalformedContractError("contract ProtoJSON does not parse") from exc


def validate_contract(contract: pb.ExecutionContract, *, reader_protocol_version: int,
                      expected_run_id: Optional[str] = None,
                      expected_project_id: Optional[str] = None) -> list[ContractIssue]:
    """Every v1 issue in ``contract``; an empty list means the contract is accepted."""
    if contract.contract_version != CONTRACT_VERSION:
        return [ContractIssue(pb.ISSUE_UNSUPPORTED_CONTRACT_VERSION, "contractVersion")]
    return _Validator(contract, reader_protocol_version, expected_run_id,
                      expected_project_id).run()


def _matches(pattern: re.Pattern, value: str) -> bool:
    return pattern.fullmatch(value) is not None


def _known(enum_type, value: int) -> bool:
    return value != 0 and value in enum_type.values_by_number


def _positive_finite(value: float) -> bool:
    return math.isfinite(value) and value > 0


def _nonnegative_finite(value: float) -> bool:
    return math.isfinite(value) and value >= 0


def _open_unit(value: float) -> bool:
    return math.isfinite(value) and 0 < value < 1


def _element_count(shape) -> Optional[int]:
    """The element count of a valid shape, or None when the shape is not valid."""
    if not 1 <= len(shape) <= MAX_RANK:
        return None
    count = 1
    for extent in shape:
        if not 1 <= extent <= MAX_ELEMENTS:
            return None
        count *= extent
        if count > MAX_ELEMENTS:
            return None
    return count


def _valid_path(path: str) -> bool:
    if not 1 <= len(path) <= MAX_PATH_LENGTH:
        return False
    segments = path.split("/")
    return (len(segments) <= MAX_PATH_SEGMENTS
            and all(_matches(_PATH_SEGMENT, segment) for segment in segments))


class _Validator:
    def __init__(self, contract, reader_protocol_version, expected_run_id, expected_project_id):
        self.c = contract
        self.reader_protocol_version = reader_protocol_version
        self.expected_run_id = expected_run_id
        self.expected_project_id = expected_project_id
        self.issues: list[ContractIssue] = []

    def add(self, code: int, path: str) -> None:
        self.issues.append(ContractIssue(code, path))

    def enum(self, enum_type, value: int, path: str) -> None:
        if not _known(enum_type, value):
            self.add(pb.ISSUE_UNKNOWN_ENUM, path)

    def bounded(self, value: int, low: int, high: int, path: str) -> None:
        if not low <= value <= high:
            self.add(pb.ISSUE_OUT_OF_RANGE, path)

    def check(self, ok: bool, code: int, path: str) -> None:
        if not ok:
            self.add(code, path)

    def run(self) -> list[ContractIssue]:
        c = self.c
        if c.min_client_protocol_version == 0 or \
                c.min_client_protocol_version > self.reader_protocol_version:
            self.add(pb.ISSUE_UNSUPPORTED_CLIENT_PROTOCOL, "minClientProtocolVersion")
        self.identity(c.run_id, self.expected_run_id, "runId")
        self.identity(c.project_id, self.expected_project_id, "projectId")
        self.enum(pb.Recipe.DESCRIPTOR, c.recipe, "recipe")
        self.enum(pb.Strategy.DESCRIPTOR, c.strategy, "strategy")
        self.enum(pb.Partitioning.DESCRIPTOR, c.partitioning, "partitioning")
        self.bounded(c.num_rounds, 1, MAX_ROUNDS, "numRounds")
        self.bounded(c.clients_per_round, 1, MAX_CLIENTS_PER_ROUND, "clientsPerRound")
        if c.HasField("round"):
            self.round_policy(c.round)
        else:
            self.add(pb.ISSUE_MISSING_FIELD, "round")
        if c.HasField("security"):
            self.security(c.security)
        else:
            self.add(pb.ISSUE_MISSING_FIELD, "security")
        if c.HasField("model_training"):
            self.model_training(c.model_training)
            self.matrix(c.model_training)
        else:
            self.add(pb.ISSUE_MISSING_FIELD, "modelTraining")
        return self.issues

    def identity(self, value: str, expected: Optional[str], path: str) -> None:
        if not _matches(_UUID, value):
            self.add(pb.ISSUE_INVALID_IDENTIFIER, path)
        elif expected is not None and value != expected:
            self.add(pb.ISSUE_IDENTITY_MISMATCH, path)

    def matrix(self, mt) -> None:
        c = self.c
        fields = ((pb.Recipe.DESCRIPTOR, c.recipe), (pb.Strategy.DESCRIPTOR, c.strategy),
                  (pb.Arm.DESCRIPTOR, mt.arm), (pb.Task.DESCRIPTOR, mt.task),
                  (pb.Objective.DESCRIPTOR, mt.objective),
                  (pb.UpdateProtocol.DESCRIPTOR, mt.update_protocol))
        if all(_known(enum_type, value) for enum_type, value in fields) and \
                tuple(value for _, value in fields) not in APPROVED_MATRIX:
            self.add(pb.ISSUE_UNSUPPORTED_COMBINATION, "")

    def round_policy(self, r) -> None:
        self.bounded(r.timeout_ms, 1, MAX_TIMEOUT_MS, "round.timeoutMs")
        self.check(r.one_accepted_update_per_round, pb.ISSUE_OUT_OF_RANGE,
                   "round.oneAcceptedUpdatePerRound")
        if not r.HasField("max_transient_retries"):
            self.add(pb.ISSUE_MISSING_FIELD, "round.maxTransientRetries")
        else:
            self.bounded(r.max_transient_retries, 0, MAX_TRANSIENT_RETRIES,
                         "round.maxTransientRetries")
        self.bounded(r.retry_backoff_ms, 1, MAX_RETRY_BACKOFF_MS, "round.retryBackoffMs")

    def security(self, s) -> None:
        self.enum(pb.Transport.DESCRIPTOR, s.transport, "security.transport")
        self.enum(pb.ClientAuth.DESCRIPTOR, s.client_auth, "security.clientAuth")
        self.enum(pb.SecureAggregation.DESCRIPTOR, s.secure_aggregation,
                  "security.secureAggregation")
        if s.secure_aggregation == pb.SECAGG_LIGHTSECAGG_SCALAR:
            if not s.HasField("secure_agg_threshold"):
                self.add(pb.ISSUE_MISSING_FIELD, "security.secureAggThreshold")
            else:
                self.bounded(s.secure_agg_threshold, 2, self.c.clients_per_round,
                             "security.secureAggThreshold")
            if _known(pb.Strategy.DESCRIPTOR, self.c.strategy) and \
                    self.c.strategy != pb.STRATEGY_DECOMFL:
                self.add(pb.ISSUE_INVALID_SECURITY, "security.secureAggregation")
        elif s.secure_aggregation == pb.SECAGG_NONE and s.HasField("secure_agg_threshold"):
            self.add(pb.ISSUE_INVALID_SECURITY, "security.secureAggThreshold")
        if s.HasField("central_dp"):
            dp = s.central_dp
            self.check(_positive_finite(dp.target_epsilon), pb.ISSUE_OUT_OF_RANGE,
                       "security.centralDp.targetEpsilon")
            self.check(_open_unit(dp.delta), pb.ISSUE_OUT_OF_RANGE, "security.centralDp.delta")
            self.check(_positive_finite(dp.clip_norm), pb.ISSUE_OUT_OF_RANGE,
                       "security.centralDp.clipNorm")

    def model_training(self, mt) -> None:
        p = "modelTraining"
        self.check(_matches(_MODEL_ID, mt.model_id), pb.ISSUE_INVALID_IDENTIFIER, p + ".modelId")
        self.check(_matches(_REVISION, mt.model_revision), pb.ISSUE_INVALID_IDENTIFIER,
                   p + ".modelRevision")
        self.enum(pb.Arm.DESCRIPTOR, mt.arm, p + ".arm")
        self.enum(pb.Task.DESCRIPTOR, mt.task, p + ".task")
        self.enum(pb.Objective.DESCRIPTOR, mt.objective, p + ".objective")
        self.enum(pb.UpdateProtocol.DESCRIPTOR, mt.update_protocol, p + ".updateProtocol")
        self.trainable(mt.trainable)
        self.check(_matches(_SHA256, mt.frozen_state_sha256), pb.ISSUE_INVALID_HASH,
                   p + ".frozenStateSha256")
        self.check(_matches(_SHA256, mt.initial_state_sha256), pb.ISSUE_INVALID_HASH,
                   p + ".initialStateSha256")
        if mt.HasField("local_training"):
            self.local_training(mt.local_training)
            # DeComFL scalars come only from zeroth-order training, and zeroth-order training produces nothing else.
            optimizer = mt.local_training.WhichOneof("optimizer")
            if optimizer is not None and _known(pb.UpdateProtocol.DESCRIPTOR, mt.update_protocol) and \
                    (optimizer == "zeroth_order_sgd") != (mt.update_protocol == pb.UPDATE_DECOMFL_SCALAR):
                self.add(pb.ISSUE_INVALID_STRATEGY_SETTINGS, p + ".updateProtocol")
        else:
            self.add(pb.ISSUE_MISSING_FIELD, p + ".localTraining")
        if mt.HasField("data"):
            self.data(mt.data, mt.task)
        else:
            self.add(pb.ISSUE_MISSING_FIELD, p + ".data")
        self.artifacts(mt.artifacts)
        if self.c.strategy == pb.STRATEGY_FEDPROX:
            if not mt.HasField("fedprox_mu"):
                self.add(pb.ISSUE_MISSING_FIELD, p + ".fedproxMu")
            elif not _nonnegative_finite(mt.fedprox_mu):
                self.add(pb.ISSUE_OUT_OF_RANGE, p + ".fedproxMu")
        elif _known(pb.Strategy.DESCRIPTOR, self.c.strategy) and mt.HasField("fedprox_mu"):
            self.add(pb.ISSUE_INVALID_STRATEGY_SETTINGS, p + ".fedproxMu")

    def trainable(self, tensors) -> None:
        p = "modelTraining.trainable"
        if not 1 <= len(tensors) <= MAX_TENSORS:
            self.add(pb.ISSUE_MALFORMED_LAYOUT, p)
            return
        seen: set[str] = set()
        total = 0
        for i, tensor in enumerate(tensors):
            at = f"{p}[{i}]"
            name_ok = len(tensor.name) <= MAX_TENSOR_NAME_LENGTH and \
                _matches(_TENSOR_NAME, tensor.name)
            self.check(name_ok and tensor.name not in seen, pb.ISSUE_MALFORMED_LAYOUT,
                       at + ".name")
            seen.add(tensor.name)
            count = _element_count(tensor.shape)
            if count is None:
                self.add(pb.ISSUE_MALFORMED_LAYOUT, at + ".shape")
            else:
                total += count
            self.enum(pb.DType.DESCRIPTOR, tensor.dtype, at + ".dtype")
        if total > MAX_ELEMENTS:
            self.add(pb.ISSUE_MALFORMED_LAYOUT, p)

    def local_training(self, lt) -> None:
        p = "modelTraining.localTraining"
        optimizer = lt.WhichOneof("optimizer")
        if optimizer == "zeroth_order_sgd":
            # Zeroth-order training counts its own steps; an epoch budget or a step cap would be a second one.
            self.check(lt.local_epochs == 0, pb.ISSUE_INVALID_STRATEGY_SETTINGS, p + ".localEpochs")
            self.check(not lt.HasField("max_local_steps"), pb.ISSUE_INVALID_STRATEGY_SETTINGS,
                       p + ".maxLocalSteps")
        else:
            self.bounded(lt.local_epochs, 1, MAX_LOCAL_EPOCHS, p + ".localEpochs")
            if lt.HasField("max_local_steps"):
                self.bounded(lt.max_local_steps, 1, MAX_LOCAL_STEPS, p + ".maxLocalSteps")
        if lt.HasField("gradient_clip_norm"):
            self.check(_positive_finite(lt.gradient_clip_norm), pb.ISSUE_OUT_OF_RANGE,
                       p + ".gradientClipNorm")
        if optimizer is None:
            self.add(pb.ISSUE_MISSING_FIELD, p + ".optimizer")
        elif optimizer == "sgd":
            self.sgd(lt.sgd, p + ".sgd")
        elif optimizer == "rmsprop":
            self.rmsprop(lt.rmsprop, p + ".rmsprop")
        elif optimizer == "zeroth_order_sgd":
            self.zeroth_order_sgd(lt.zeroth_order_sgd, p + ".zerothOrderSgd")
        else:
            # A oneof case added to the schema must get its own rules here, never fall into Adam's.
            assert optimizer in ("adam", "adamw"), f"no rules for optimizer {optimizer}"
            self.adam(getattr(lt, optimizer), f"{p}.{optimizer}")
        self.present(lt, "reset_optimizer_each_round", p + ".resetOptimizerEachRound")
        self.bounded(lt.batch_size, 1, MAX_BATCH_SIZE, p + ".batchSize")
        self.present(lt, "drop_last", p + ".dropLast")
        self.enum(pb.BatchOrder.DESCRIPTOR, lt.batch_order, p + ".batchOrder")

    def present(self, message, field: str, path: str) -> bool:
        if message.HasField(field):
            return True
        self.add(pb.ISSUE_MISSING_FIELD, path)
        return False

    def nonnegative(self, message, field: str, path: str) -> bool:
        """Checks a required explicit-presence value; True when present and valid."""
        if not self.present(message, field, path):
            return False
        if not _nonnegative_finite(getattr(message, field)):
            self.add(pb.ISSUE_OUT_OF_RANGE, path)
            return False
        return True

    def sgd(self, sgd, p: str) -> None:
        self.check(_positive_finite(sgd.learning_rate), pb.ISSUE_OUT_OF_RANGE, p + ".learningRate")
        momentum_ok = self.nonnegative(sgd, "momentum", p + ".momentum")
        dampening_ok = self.nonnegative(sgd, "dampening", p + ".dampening")
        self.nonnegative(sgd, "weight_decay", p + ".weightDecay")
        if self.present(sgd, "nesterov", p + ".nesterov") and sgd.nesterov and \
                momentum_ok and dampening_ok and not (sgd.momentum > 0 and sgd.dampening == 0):
            self.add(pb.ISSUE_INVALID_OPTIMIZER, p + ".nesterov")

    def zeroth_order_sgd(self, zo, p: str) -> None:
        self.check(_positive_finite(zo.learning_rate), pb.ISSUE_OUT_OF_RANGE, p + ".learningRate")
        self.check(_positive_finite(zo.smoothing), pb.ISSUE_OUT_OF_RANGE, p + ".smoothing")
        self.bounded(zo.num_local_steps, 1, MAX_LOCAL_STEPS, p + ".numLocalSteps")
        self.bounded(zo.num_perturbations, 1, MAX_PERTURBATIONS, p + ".numPerturbations")
        self.enum(pb.GradientEstimator.DESCRIPTOR, zo.estimator, p + ".estimator")
        self.enum(pb.PerturbationRng.DESCRIPTOR, zo.rng, p + ".rng")

    def adam(self, adam, p: str) -> None:
        self.check(_positive_finite(adam.learning_rate), pb.ISSUE_OUT_OF_RANGE,
                   p + ".learningRate")
        self.check(_open_unit(adam.beta1), pb.ISSUE_OUT_OF_RANGE, p + ".beta1")
        self.check(_open_unit(adam.beta2), pb.ISSUE_OUT_OF_RANGE, p + ".beta2")
        self.check(_positive_finite(adam.epsilon), pb.ISSUE_OUT_OF_RANGE, p + ".epsilon")
        self.nonnegative(adam, "weight_decay", p + ".weightDecay")
        self.present(adam, "amsgrad", p + ".amsgrad")

    def rmsprop(self, rms, p: str) -> None:
        self.check(_positive_finite(rms.learning_rate), pb.ISSUE_OUT_OF_RANGE, p + ".learningRate")
        self.check(_open_unit(rms.alpha), pb.ISSUE_OUT_OF_RANGE, p + ".alpha")
        self.check(_positive_finite(rms.epsilon), pb.ISSUE_OUT_OF_RANGE, p + ".epsilon")
        self.nonnegative(rms, "weight_decay", p + ".weightDecay")
        self.nonnegative(rms, "momentum", p + ".momentum")
        self.present(rms, "centered", p + ".centered")

    def data(self, data, training_task: int) -> None:
        p = "modelTraining.data"
        task_known = _known(pb.Task.DESCRIPTOR, data.task)
        if not task_known:
            self.add(pb.ISSUE_UNKNOWN_ENUM, p + ".task")
        elif _known(pb.Task.DESCRIPTOR, training_task) and training_task != data.task:
            self.add(pb.ISSUE_INVALID_DATA_REQUIREMENT, p + ".task")
        shape_ok = _element_count(data.input_shape) is not None
        self.check(shape_ok, pb.ISSUE_INVALID_DATA_REQUIREMENT, p + ".inputShape")
        self.enum(pb.DType.DESCRIPTOR, data.input_dtype, p + ".inputDtype")
        if data.task in _CLASSIFICATION_TASKS:
            self.bounded(data.class_count, 2, MAX_CLASSES, p + ".classCount")
        elif data.task == pb.TASK_CAUSAL_LM and data.class_count != 0:
            self.add(pb.ISSUE_INVALID_DATA_REQUIREMENT, p + ".classCount")
        self.check(_matches(_REVISION, data.label_schema_id), pb.ISSUE_INVALID_IDENTIFIER,
                   p + ".labelSchemaId")
        if not 1 <= len(data.transforms) <= MAX_TRANSFORMS:
            self.add(pb.ISSUE_INVALID_DATA_REQUIREMENT, p + ".transforms")
        else:
            for i, transform in enumerate(data.transforms):
                at = f"{p}.transforms[{i}]"
                if transform.WhichOneof("operation") is None:
                    self.add(pb.ISSUE_MISSING_FIELD, at + ".operation")
                    continue
                width = transform.identity_vector.width
                width_path = at + ".identityVector.width"
                if not 1 <= width <= MAX_ELEMENTS:
                    self.add(pb.ISSUE_OUT_OF_RANGE, width_path)
                elif task_known and (
                        data.task != pb.TASK_VECTOR_CLASSIFICATION
                        or (shape_ok and list(data.input_shape) != [width])):
                    self.add(pb.ISSUE_INVALID_DATA_REQUIREMENT, width_path)
        has_tokenizer = data.HasField("tokenizer")
        if data.task in _TEXT_TASKS and not has_tokenizer:
            self.add(pb.ISSUE_MISSING_FIELD, p + ".tokenizer")
        elif task_known and data.task not in _TEXT_TASKS and has_tokenizer:
            self.add(pb.ISSUE_INVALID_DATA_REQUIREMENT, p + ".tokenizer")
        if has_tokenizer:
            self.artifact_ref(data.tokenizer, p + ".tokenizer")

    def artifact_ref(self, ref, p: str) -> bool:
        """Checks one reference; True when its byte size is in range."""
        self.check(_valid_path(ref.relative_path), pb.ISSUE_INVALID_PATH, p + ".relativePath")
        self.check(_matches(_SHA256, ref.sha256), pb.ISSUE_INVALID_HASH, p + ".sha256")
        size_ok = 1 <= ref.byte_size <= MAX_FILE_BYTES
        self.check(size_ok, pb.ISSUE_OUT_OF_RANGE, p + ".byteSize")
        return size_ok

    def artifacts(self, variants) -> None:
        p = "modelTraining.artifacts"
        if not variants:
            self.add(pb.ISSUE_MISSING_ARTIFACT, p)
            return
        if len(variants) > MAX_VARIANTS:
            self.add(pb.ISSUE_INVALID_ARTIFACT, p)
            return
        seen_ids: set[str] = set()
        for i, variant in enumerate(variants):
            at = f"{p}[{i}]"
            if not _matches(_VARIANT_ID, variant.variant_id):
                self.add(pb.ISSUE_INVALID_IDENTIFIER, at + ".variantId")
            elif variant.variant_id in seen_ids:
                self.add(pb.ISSUE_INVALID_ARTIFACT, at + ".variantId")
            seen_ids.add(variant.variant_id)
            self.enum(pb.ArtifactBackend.DESCRIPTOR, variant.backend, at + ".backend")
            self.check(_matches(_ABI, variant.abi), pb.ISSUE_INVALID_IDENTIFIER, at + ".abi")
            sizes_ok = self.files(variant.files, at + ".files")
            self.operators(variant.required_operators, at + ".requiredOperators")
            self.bounded(variant.declared_peak_memory_bytes, 1, MAX_DECLARED_BYTES,
                         at + ".declaredPeakMemoryBytes")
            storage_ok = 1 <= variant.declared_storage_bytes <= MAX_DECLARED_BYTES
            self.check(storage_ok, pb.ISSUE_OUT_OF_RANGE, at + ".declaredStorageBytes")
            self.bounded(variant.declared_probe_ms, 1, MAX_DECLARED_MS, at + ".declaredProbeMs")
            self.bounded(variant.declared_train_ms, 1, MAX_DECLARED_MS, at + ".declaredTrainMs")
            if storage_ok and sizes_ok and \
                    sum(f.byte_size for f in variant.files) > variant.declared_storage_bytes:
                self.add(pb.ISSUE_INVALID_ARTIFACT, at + ".declaredStorageBytes")
        if not any(variant.backend == pb.BACKEND_EXECUTORCH_CPU for variant in variants):
            self.add(pb.ISSUE_MISSING_ARTIFACT, p)

    def files(self, files, p: str) -> bool:
        """Checks a variant's files; True when there are 1..MAX_FILES, all with sizes in range."""
        if not files:
            self.add(pb.ISSUE_MISSING_ARTIFACT, p)
            return False
        if len(files) > MAX_FILES:
            self.add(pb.ISSUE_INVALID_ARTIFACT, p)
            return False
        seen_paths: set[str] = set()
        all_sizes_ok = True
        for j, ref in enumerate(files):
            at = f"{p}[{j}]"
            all_sizes_ok = self.artifact_ref(ref, at) and all_sizes_ok
            if _valid_path(ref.relative_path):
                folded = ref.relative_path.lower()
                if folded in seen_paths:
                    self.add(pb.ISSUE_INVALID_PATH, at + ".relativePath")
                seen_paths.add(folded)
        return all_sizes_ok

    def operators(self, operators, p: str) -> None:
        if not 1 <= len(operators) <= MAX_OPERATORS:
            self.add(pb.ISSUE_INVALID_ARTIFACT, p)
            return
        seen: set[str] = set()
        for k, operator in enumerate(operators):
            valid = len(operator) <= MAX_OPERATOR_LENGTH and _matches(_OPERATOR, operator)
            self.check(valid and operator not in seen, pb.ISSUE_INVALID_ARTIFACT, f"{p}[{k}]")
            seen.add(operator)
