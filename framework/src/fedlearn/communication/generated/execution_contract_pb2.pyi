from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Recipe(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    RECIPE_UNSPECIFIED: _ClassVar[Recipe]
    RECIPE_TINYNET_GOLDEN: _ClassVar[Recipe]
    RECIPE_MLP: _ClassVar[Recipe]
    RECIPE_CNN: _ClassVar[Recipe]
    RECIPE_PNEUMONIA_CNN: _ClassVar[Recipe]
    RECIPE_CIFAR_RESNET18: _ClassVar[Recipe]
    RECIPE_TRANSFORMER: _ClassVar[Recipe]
    RECIPE_LLM_LORA: _ClassVar[Recipe]

class Strategy(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    STRATEGY_UNSPECIFIED: _ClassVar[Strategy]
    STRATEGY_DECOMFL: _ClassVar[Strategy]
    STRATEGY_FEDAVG: _ClassVar[Strategy]
    STRATEGY_FEDPROX: _ClassVar[Strategy]
    STRATEGY_FEDOPT: _ClassVar[Strategy]
    STRATEGY_ROBUST: _ClassVar[Strategy]

class Partitioning(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    PARTITIONING_UNSPECIFIED: _ClassVar[Partitioning]
    PARTITIONING_SHARDED: _ClassVar[Partitioning]
    PARTITIONING_LOCAL: _ClassVar[Partitioning]

class Arm(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    ARM_UNSPECIFIED: _ClassVar[Arm]
    ARM_FULL: _ClassVar[Arm]
    ARM_FROZEN_HEAD: _ClassVar[Arm]
    ARM_OVA_LP: _ClassVar[Arm]

class Task(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    TASK_UNSPECIFIED: _ClassVar[Task]
    TASK_VECTOR_CLASSIFICATION: _ClassVar[Task]
    TASK_IMAGE_CLASSIFICATION: _ClassVar[Task]
    TASK_SEQUENCE_CLASSIFICATION: _ClassVar[Task]
    TASK_CAUSAL_LM: _ClassVar[Task]

class Objective(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    OBJECTIVE_UNSPECIFIED: _ClassVar[Objective]
    OBJECTIVE_CROSS_ENTROPY: _ClassVar[Objective]
    OBJECTIVE_ONE_VS_ALL: _ClassVar[Objective]
    OBJECTIVE_CAUSAL_LM: _ClassVar[Objective]

class UpdateProtocol(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    UPDATE_UNSPECIFIED: _ClassVar[UpdateProtocol]
    UPDATE_DECOMFL_SCALAR: _ClassVar[UpdateProtocol]
    UPDATE_TRAINABLE_STATE_F32: _ClassVar[UpdateProtocol]

class Transport(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    TRANSPORT_UNSPECIFIED: _ClassVar[Transport]
    TRANSPORT_TLS_REQUIRED: _ClassVar[Transport]
    TRANSPORT_PLAINTEXT_DEV: _ClassVar[Transport]

class ClientAuth(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    CLIENT_AUTH_UNSPECIFIED: _ClassVar[ClientAuth]
    CLIENT_AUTH_CONNECTION_TOKEN: _ClassVar[ClientAuth]
    CLIENT_AUTH_DISABLED_DEV: _ClassVar[ClientAuth]

class SecureAggregation(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    SECAGG_UNSPECIFIED: _ClassVar[SecureAggregation]
    SECAGG_NONE: _ClassVar[SecureAggregation]
    SECAGG_LIGHTSECAGG_SCALAR: _ClassVar[SecureAggregation]

class ArtifactBackend(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    BACKEND_UNSPECIFIED: _ClassVar[ArtifactBackend]
    BACKEND_EXECUTORCH_CPU: _ClassVar[ArtifactBackend]
    BACKEND_EXECUTORCH_GPU: _ClassVar[ArtifactBackend]
    BACKEND_EXECUTORCH_VENDOR: _ClassVar[ArtifactBackend]

class DType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    DTYPE_UNSPECIFIED: _ClassVar[DType]
    DTYPE_F32: _ClassVar[DType]

class BatchOrder(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    BATCH_ORDER_UNSPECIFIED: _ClassVar[BatchOrder]
    BATCH_ORDER_SEQUENTIAL: _ClassVar[BatchOrder]
    BATCH_ORDER_SHUFFLED_EACH_EPOCH: _ClassVar[BatchOrder]

class GradientEstimator(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    ESTIMATOR_UNSPECIFIED: _ClassVar[GradientEstimator]
    ESTIMATOR_FORWARD: _ClassVar[GradientEstimator]
    ESTIMATOR_CENTRAL: _ClassVar[GradientEstimator]

class PerturbationRng(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    RNG_UNSPECIFIED: _ClassVar[PerturbationRng]
    RNG_TORCH_CPU_RANDN_F32: _ClassVar[PerturbationRng]

class ContractIssueCode(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    ISSUE_UNSPECIFIED: _ClassVar[ContractIssueCode]
    ISSUE_MALFORMED: _ClassVar[ContractIssueCode]
    ISSUE_UNSUPPORTED_CONTRACT_VERSION: _ClassVar[ContractIssueCode]
    ISSUE_UNSUPPORTED_CLIENT_PROTOCOL: _ClassVar[ContractIssueCode]
    ISSUE_UNKNOWN_ENUM: _ClassVar[ContractIssueCode]
    ISSUE_MISSING_FIELD: _ClassVar[ContractIssueCode]
    ISSUE_OUT_OF_RANGE: _ClassVar[ContractIssueCode]
    ISSUE_INVALID_IDENTIFIER: _ClassVar[ContractIssueCode]
    ISSUE_IDENTITY_MISMATCH: _ClassVar[ContractIssueCode]
    ISSUE_INVALID_HASH: _ClassVar[ContractIssueCode]
    ISSUE_INVALID_PATH: _ClassVar[ContractIssueCode]
    ISSUE_MALFORMED_LAYOUT: _ClassVar[ContractIssueCode]
    ISSUE_UNSUPPORTED_COMBINATION: _ClassVar[ContractIssueCode]
    ISSUE_INVALID_OPTIMIZER: _ClassVar[ContractIssueCode]
    ISSUE_INVALID_SECURITY: _ClassVar[ContractIssueCode]
    ISSUE_INVALID_STRATEGY_SETTINGS: _ClassVar[ContractIssueCode]
    ISSUE_MISSING_ARTIFACT: _ClassVar[ContractIssueCode]
    ISSUE_INVALID_ARTIFACT: _ClassVar[ContractIssueCode]
    ISSUE_INVALID_DATA_REQUIREMENT: _ClassVar[ContractIssueCode]
RECIPE_UNSPECIFIED: Recipe
RECIPE_TINYNET_GOLDEN: Recipe
RECIPE_MLP: Recipe
RECIPE_CNN: Recipe
RECIPE_PNEUMONIA_CNN: Recipe
RECIPE_CIFAR_RESNET18: Recipe
RECIPE_TRANSFORMER: Recipe
RECIPE_LLM_LORA: Recipe
STRATEGY_UNSPECIFIED: Strategy
STRATEGY_DECOMFL: Strategy
STRATEGY_FEDAVG: Strategy
STRATEGY_FEDPROX: Strategy
STRATEGY_FEDOPT: Strategy
STRATEGY_ROBUST: Strategy
PARTITIONING_UNSPECIFIED: Partitioning
PARTITIONING_SHARDED: Partitioning
PARTITIONING_LOCAL: Partitioning
ARM_UNSPECIFIED: Arm
ARM_FULL: Arm
ARM_FROZEN_HEAD: Arm
ARM_OVA_LP: Arm
TASK_UNSPECIFIED: Task
TASK_VECTOR_CLASSIFICATION: Task
TASK_IMAGE_CLASSIFICATION: Task
TASK_SEQUENCE_CLASSIFICATION: Task
TASK_CAUSAL_LM: Task
OBJECTIVE_UNSPECIFIED: Objective
OBJECTIVE_CROSS_ENTROPY: Objective
OBJECTIVE_ONE_VS_ALL: Objective
OBJECTIVE_CAUSAL_LM: Objective
UPDATE_UNSPECIFIED: UpdateProtocol
UPDATE_DECOMFL_SCALAR: UpdateProtocol
UPDATE_TRAINABLE_STATE_F32: UpdateProtocol
TRANSPORT_UNSPECIFIED: Transport
TRANSPORT_TLS_REQUIRED: Transport
TRANSPORT_PLAINTEXT_DEV: Transport
CLIENT_AUTH_UNSPECIFIED: ClientAuth
CLIENT_AUTH_CONNECTION_TOKEN: ClientAuth
CLIENT_AUTH_DISABLED_DEV: ClientAuth
SECAGG_UNSPECIFIED: SecureAggregation
SECAGG_NONE: SecureAggregation
SECAGG_LIGHTSECAGG_SCALAR: SecureAggregation
BACKEND_UNSPECIFIED: ArtifactBackend
BACKEND_EXECUTORCH_CPU: ArtifactBackend
BACKEND_EXECUTORCH_GPU: ArtifactBackend
BACKEND_EXECUTORCH_VENDOR: ArtifactBackend
DTYPE_UNSPECIFIED: DType
DTYPE_F32: DType
BATCH_ORDER_UNSPECIFIED: BatchOrder
BATCH_ORDER_SEQUENTIAL: BatchOrder
BATCH_ORDER_SHUFFLED_EACH_EPOCH: BatchOrder
ESTIMATOR_UNSPECIFIED: GradientEstimator
ESTIMATOR_FORWARD: GradientEstimator
ESTIMATOR_CENTRAL: GradientEstimator
RNG_UNSPECIFIED: PerturbationRng
RNG_TORCH_CPU_RANDN_F32: PerturbationRng
ISSUE_UNSPECIFIED: ContractIssueCode
ISSUE_MALFORMED: ContractIssueCode
ISSUE_UNSUPPORTED_CONTRACT_VERSION: ContractIssueCode
ISSUE_UNSUPPORTED_CLIENT_PROTOCOL: ContractIssueCode
ISSUE_UNKNOWN_ENUM: ContractIssueCode
ISSUE_MISSING_FIELD: ContractIssueCode
ISSUE_OUT_OF_RANGE: ContractIssueCode
ISSUE_INVALID_IDENTIFIER: ContractIssueCode
ISSUE_IDENTITY_MISMATCH: ContractIssueCode
ISSUE_INVALID_HASH: ContractIssueCode
ISSUE_INVALID_PATH: ContractIssueCode
ISSUE_MALFORMED_LAYOUT: ContractIssueCode
ISSUE_UNSUPPORTED_COMBINATION: ContractIssueCode
ISSUE_INVALID_OPTIMIZER: ContractIssueCode
ISSUE_INVALID_SECURITY: ContractIssueCode
ISSUE_INVALID_STRATEGY_SETTINGS: ContractIssueCode
ISSUE_MISSING_ARTIFACT: ContractIssueCode
ISSUE_INVALID_ARTIFACT: ContractIssueCode
ISSUE_INVALID_DATA_REQUIREMENT: ContractIssueCode

class ExecutionContract(_message.Message):
    __slots__ = ("contract_version", "min_client_protocol_version", "run_id", "project_id", "recipe", "strategy", "num_rounds", "clients_per_round", "partitioning", "seed", "round", "security", "model_training")
    CONTRACT_VERSION_FIELD_NUMBER: _ClassVar[int]
    MIN_CLIENT_PROTOCOL_VERSION_FIELD_NUMBER: _ClassVar[int]
    RUN_ID_FIELD_NUMBER: _ClassVar[int]
    PROJECT_ID_FIELD_NUMBER: _ClassVar[int]
    RECIPE_FIELD_NUMBER: _ClassVar[int]
    STRATEGY_FIELD_NUMBER: _ClassVar[int]
    NUM_ROUNDS_FIELD_NUMBER: _ClassVar[int]
    CLIENTS_PER_ROUND_FIELD_NUMBER: _ClassVar[int]
    PARTITIONING_FIELD_NUMBER: _ClassVar[int]
    SEED_FIELD_NUMBER: _ClassVar[int]
    ROUND_FIELD_NUMBER: _ClassVar[int]
    SECURITY_FIELD_NUMBER: _ClassVar[int]
    MODEL_TRAINING_FIELD_NUMBER: _ClassVar[int]
    contract_version: int
    min_client_protocol_version: int
    run_id: str
    project_id: str
    recipe: Recipe
    strategy: Strategy
    num_rounds: int
    clients_per_round: int
    partitioning: Partitioning
    seed: int
    round: RoundPolicy
    security: SecurityPolicy
    model_training: ModelTraining
    def __init__(self, contract_version: _Optional[int] = ..., min_client_protocol_version: _Optional[int] = ..., run_id: _Optional[str] = ..., project_id: _Optional[str] = ..., recipe: _Optional[_Union[Recipe, str]] = ..., strategy: _Optional[_Union[Strategy, str]] = ..., num_rounds: _Optional[int] = ..., clients_per_round: _Optional[int] = ..., partitioning: _Optional[_Union[Partitioning, str]] = ..., seed: _Optional[int] = ..., round: _Optional[_Union[RoundPolicy, _Mapping]] = ..., security: _Optional[_Union[SecurityPolicy, _Mapping]] = ..., model_training: _Optional[_Union[ModelTraining, _Mapping]] = ...) -> None: ...

class RoundPolicy(_message.Message):
    __slots__ = ("timeout_ms", "one_accepted_update_per_round", "max_transient_retries", "retry_backoff_ms")
    TIMEOUT_MS_FIELD_NUMBER: _ClassVar[int]
    ONE_ACCEPTED_UPDATE_PER_ROUND_FIELD_NUMBER: _ClassVar[int]
    MAX_TRANSIENT_RETRIES_FIELD_NUMBER: _ClassVar[int]
    RETRY_BACKOFF_MS_FIELD_NUMBER: _ClassVar[int]
    timeout_ms: int
    one_accepted_update_per_round: bool
    max_transient_retries: int
    retry_backoff_ms: int
    def __init__(self, timeout_ms: _Optional[int] = ..., one_accepted_update_per_round: bool = ..., max_transient_retries: _Optional[int] = ..., retry_backoff_ms: _Optional[int] = ...) -> None: ...

class SecurityPolicy(_message.Message):
    __slots__ = ("transport", "client_auth", "secure_aggregation", "secure_agg_threshold", "central_dp")
    TRANSPORT_FIELD_NUMBER: _ClassVar[int]
    CLIENT_AUTH_FIELD_NUMBER: _ClassVar[int]
    SECURE_AGGREGATION_FIELD_NUMBER: _ClassVar[int]
    SECURE_AGG_THRESHOLD_FIELD_NUMBER: _ClassVar[int]
    CENTRAL_DP_FIELD_NUMBER: _ClassVar[int]
    transport: Transport
    client_auth: ClientAuth
    secure_aggregation: SecureAggregation
    secure_agg_threshold: int
    central_dp: CentralDp
    def __init__(self, transport: _Optional[_Union[Transport, str]] = ..., client_auth: _Optional[_Union[ClientAuth, str]] = ..., secure_aggregation: _Optional[_Union[SecureAggregation, str]] = ..., secure_agg_threshold: _Optional[int] = ..., central_dp: _Optional[_Union[CentralDp, _Mapping]] = ...) -> None: ...

class CentralDp(_message.Message):
    __slots__ = ("target_epsilon", "delta", "clip_norm")
    TARGET_EPSILON_FIELD_NUMBER: _ClassVar[int]
    DELTA_FIELD_NUMBER: _ClassVar[int]
    CLIP_NORM_FIELD_NUMBER: _ClassVar[int]
    target_epsilon: float
    delta: float
    clip_norm: float
    def __init__(self, target_epsilon: _Optional[float] = ..., delta: _Optional[float] = ..., clip_norm: _Optional[float] = ...) -> None: ...

class ModelTraining(_message.Message):
    __slots__ = ("model_id", "model_revision", "arm", "task", "objective", "update_protocol", "trainable", "frozen_state_sha256", "initial_state_sha256", "local_training", "data", "artifacts", "fedprox_mu")
    MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    MODEL_REVISION_FIELD_NUMBER: _ClassVar[int]
    ARM_FIELD_NUMBER: _ClassVar[int]
    TASK_FIELD_NUMBER: _ClassVar[int]
    OBJECTIVE_FIELD_NUMBER: _ClassVar[int]
    UPDATE_PROTOCOL_FIELD_NUMBER: _ClassVar[int]
    TRAINABLE_FIELD_NUMBER: _ClassVar[int]
    FROZEN_STATE_SHA256_FIELD_NUMBER: _ClassVar[int]
    INITIAL_STATE_SHA256_FIELD_NUMBER: _ClassVar[int]
    LOCAL_TRAINING_FIELD_NUMBER: _ClassVar[int]
    DATA_FIELD_NUMBER: _ClassVar[int]
    ARTIFACTS_FIELD_NUMBER: _ClassVar[int]
    FEDPROX_MU_FIELD_NUMBER: _ClassVar[int]
    model_id: str
    model_revision: str
    arm: Arm
    task: Task
    objective: Objective
    update_protocol: UpdateProtocol
    trainable: _containers.RepeatedCompositeFieldContainer[TensorSpec]
    frozen_state_sha256: str
    initial_state_sha256: str
    local_training: LocalTraining
    data: DataRequirement
    artifacts: _containers.RepeatedCompositeFieldContainer[ArtifactVariant]
    fedprox_mu: float
    def __init__(self, model_id: _Optional[str] = ..., model_revision: _Optional[str] = ..., arm: _Optional[_Union[Arm, str]] = ..., task: _Optional[_Union[Task, str]] = ..., objective: _Optional[_Union[Objective, str]] = ..., update_protocol: _Optional[_Union[UpdateProtocol, str]] = ..., trainable: _Optional[_Iterable[_Union[TensorSpec, _Mapping]]] = ..., frozen_state_sha256: _Optional[str] = ..., initial_state_sha256: _Optional[str] = ..., local_training: _Optional[_Union[LocalTraining, _Mapping]] = ..., data: _Optional[_Union[DataRequirement, _Mapping]] = ..., artifacts: _Optional[_Iterable[_Union[ArtifactVariant, _Mapping]]] = ..., fedprox_mu: _Optional[float] = ...) -> None: ...

class TensorSpec(_message.Message):
    __slots__ = ("name", "shape", "dtype")
    NAME_FIELD_NUMBER: _ClassVar[int]
    SHAPE_FIELD_NUMBER: _ClassVar[int]
    DTYPE_FIELD_NUMBER: _ClassVar[int]
    name: str
    shape: _containers.RepeatedScalarFieldContainer[int]
    dtype: DType
    def __init__(self, name: _Optional[str] = ..., shape: _Optional[_Iterable[int]] = ..., dtype: _Optional[_Union[DType, str]] = ...) -> None: ...

class LocalTraining(_message.Message):
    __slots__ = ("local_epochs", "max_local_steps", "gradient_clip_norm", "sgd", "adam", "adamw", "rmsprop", "zeroth_order_sgd", "reset_optimizer_each_round", "batch_size", "drop_last", "batch_order")
    LOCAL_EPOCHS_FIELD_NUMBER: _ClassVar[int]
    MAX_LOCAL_STEPS_FIELD_NUMBER: _ClassVar[int]
    GRADIENT_CLIP_NORM_FIELD_NUMBER: _ClassVar[int]
    SGD_FIELD_NUMBER: _ClassVar[int]
    ADAM_FIELD_NUMBER: _ClassVar[int]
    ADAMW_FIELD_NUMBER: _ClassVar[int]
    RMSPROP_FIELD_NUMBER: _ClassVar[int]
    ZEROTH_ORDER_SGD_FIELD_NUMBER: _ClassVar[int]
    RESET_OPTIMIZER_EACH_ROUND_FIELD_NUMBER: _ClassVar[int]
    BATCH_SIZE_FIELD_NUMBER: _ClassVar[int]
    DROP_LAST_FIELD_NUMBER: _ClassVar[int]
    BATCH_ORDER_FIELD_NUMBER: _ClassVar[int]
    local_epochs: int
    max_local_steps: int
    gradient_clip_norm: float
    sgd: Sgd
    adam: Adam
    adamw: AdamW
    rmsprop: Rmsprop
    zeroth_order_sgd: ZerothOrderSgd
    reset_optimizer_each_round: bool
    batch_size: int
    drop_last: bool
    batch_order: BatchOrder
    def __init__(self, local_epochs: _Optional[int] = ..., max_local_steps: _Optional[int] = ..., gradient_clip_norm: _Optional[float] = ..., sgd: _Optional[_Union[Sgd, _Mapping]] = ..., adam: _Optional[_Union[Adam, _Mapping]] = ..., adamw: _Optional[_Union[AdamW, _Mapping]] = ..., rmsprop: _Optional[_Union[Rmsprop, _Mapping]] = ..., zeroth_order_sgd: _Optional[_Union[ZerothOrderSgd, _Mapping]] = ..., reset_optimizer_each_round: bool = ..., batch_size: _Optional[int] = ..., drop_last: bool = ..., batch_order: _Optional[_Union[BatchOrder, str]] = ...) -> None: ...

class Sgd(_message.Message):
    __slots__ = ("learning_rate", "momentum", "dampening", "weight_decay", "nesterov")
    LEARNING_RATE_FIELD_NUMBER: _ClassVar[int]
    MOMENTUM_FIELD_NUMBER: _ClassVar[int]
    DAMPENING_FIELD_NUMBER: _ClassVar[int]
    WEIGHT_DECAY_FIELD_NUMBER: _ClassVar[int]
    NESTEROV_FIELD_NUMBER: _ClassVar[int]
    learning_rate: float
    momentum: float
    dampening: float
    weight_decay: float
    nesterov: bool
    def __init__(self, learning_rate: _Optional[float] = ..., momentum: _Optional[float] = ..., dampening: _Optional[float] = ..., weight_decay: _Optional[float] = ..., nesterov: bool = ...) -> None: ...

class Adam(_message.Message):
    __slots__ = ("learning_rate", "beta1", "beta2", "epsilon", "weight_decay", "amsgrad")
    LEARNING_RATE_FIELD_NUMBER: _ClassVar[int]
    BETA1_FIELD_NUMBER: _ClassVar[int]
    BETA2_FIELD_NUMBER: _ClassVar[int]
    EPSILON_FIELD_NUMBER: _ClassVar[int]
    WEIGHT_DECAY_FIELD_NUMBER: _ClassVar[int]
    AMSGRAD_FIELD_NUMBER: _ClassVar[int]
    learning_rate: float
    beta1: float
    beta2: float
    epsilon: float
    weight_decay: float
    amsgrad: bool
    def __init__(self, learning_rate: _Optional[float] = ..., beta1: _Optional[float] = ..., beta2: _Optional[float] = ..., epsilon: _Optional[float] = ..., weight_decay: _Optional[float] = ..., amsgrad: bool = ...) -> None: ...

class AdamW(_message.Message):
    __slots__ = ("learning_rate", "beta1", "beta2", "epsilon", "weight_decay", "amsgrad")
    LEARNING_RATE_FIELD_NUMBER: _ClassVar[int]
    BETA1_FIELD_NUMBER: _ClassVar[int]
    BETA2_FIELD_NUMBER: _ClassVar[int]
    EPSILON_FIELD_NUMBER: _ClassVar[int]
    WEIGHT_DECAY_FIELD_NUMBER: _ClassVar[int]
    AMSGRAD_FIELD_NUMBER: _ClassVar[int]
    learning_rate: float
    beta1: float
    beta2: float
    epsilon: float
    weight_decay: float
    amsgrad: bool
    def __init__(self, learning_rate: _Optional[float] = ..., beta1: _Optional[float] = ..., beta2: _Optional[float] = ..., epsilon: _Optional[float] = ..., weight_decay: _Optional[float] = ..., amsgrad: bool = ...) -> None: ...

class Rmsprop(_message.Message):
    __slots__ = ("learning_rate", "alpha", "epsilon", "weight_decay", "momentum", "centered")
    LEARNING_RATE_FIELD_NUMBER: _ClassVar[int]
    ALPHA_FIELD_NUMBER: _ClassVar[int]
    EPSILON_FIELD_NUMBER: _ClassVar[int]
    WEIGHT_DECAY_FIELD_NUMBER: _ClassVar[int]
    MOMENTUM_FIELD_NUMBER: _ClassVar[int]
    CENTERED_FIELD_NUMBER: _ClassVar[int]
    learning_rate: float
    alpha: float
    epsilon: float
    weight_decay: float
    momentum: float
    centered: bool
    def __init__(self, learning_rate: _Optional[float] = ..., alpha: _Optional[float] = ..., epsilon: _Optional[float] = ..., weight_decay: _Optional[float] = ..., momentum: _Optional[float] = ..., centered: bool = ...) -> None: ...

class ZerothOrderSgd(_message.Message):
    __slots__ = ("learning_rate", "smoothing", "num_local_steps", "num_perturbations", "estimator", "rng")
    LEARNING_RATE_FIELD_NUMBER: _ClassVar[int]
    SMOOTHING_FIELD_NUMBER: _ClassVar[int]
    NUM_LOCAL_STEPS_FIELD_NUMBER: _ClassVar[int]
    NUM_PERTURBATIONS_FIELD_NUMBER: _ClassVar[int]
    ESTIMATOR_FIELD_NUMBER: _ClassVar[int]
    RNG_FIELD_NUMBER: _ClassVar[int]
    learning_rate: float
    smoothing: float
    num_local_steps: int
    num_perturbations: int
    estimator: GradientEstimator
    rng: PerturbationRng
    def __init__(self, learning_rate: _Optional[float] = ..., smoothing: _Optional[float] = ..., num_local_steps: _Optional[int] = ..., num_perturbations: _Optional[int] = ..., estimator: _Optional[_Union[GradientEstimator, str]] = ..., rng: _Optional[_Union[PerturbationRng, str]] = ...) -> None: ...

class DataRequirement(_message.Message):
    __slots__ = ("task", "input_shape", "input_dtype", "class_count", "label_schema_id", "transforms", "tokenizer")
    TASK_FIELD_NUMBER: _ClassVar[int]
    INPUT_SHAPE_FIELD_NUMBER: _ClassVar[int]
    INPUT_DTYPE_FIELD_NUMBER: _ClassVar[int]
    CLASS_COUNT_FIELD_NUMBER: _ClassVar[int]
    LABEL_SCHEMA_ID_FIELD_NUMBER: _ClassVar[int]
    TRANSFORMS_FIELD_NUMBER: _ClassVar[int]
    TOKENIZER_FIELD_NUMBER: _ClassVar[int]
    task: Task
    input_shape: _containers.RepeatedScalarFieldContainer[int]
    input_dtype: DType
    class_count: int
    label_schema_id: str
    transforms: _containers.RepeatedCompositeFieldContainer[Transform]
    tokenizer: ArtifactRef
    def __init__(self, task: _Optional[_Union[Task, str]] = ..., input_shape: _Optional[_Iterable[int]] = ..., input_dtype: _Optional[_Union[DType, str]] = ..., class_count: _Optional[int] = ..., label_schema_id: _Optional[str] = ..., transforms: _Optional[_Iterable[_Union[Transform, _Mapping]]] = ..., tokenizer: _Optional[_Union[ArtifactRef, _Mapping]] = ...) -> None: ...

class Transform(_message.Message):
    __slots__ = ("identity_vector",)
    IDENTITY_VECTOR_FIELD_NUMBER: _ClassVar[int]
    identity_vector: IdentityVector
    def __init__(self, identity_vector: _Optional[_Union[IdentityVector, _Mapping]] = ...) -> None: ...

class IdentityVector(_message.Message):
    __slots__ = ("width",)
    WIDTH_FIELD_NUMBER: _ClassVar[int]
    width: int
    def __init__(self, width: _Optional[int] = ...) -> None: ...

class ArtifactRef(_message.Message):
    __slots__ = ("relative_path", "sha256", "byte_size")
    RELATIVE_PATH_FIELD_NUMBER: _ClassVar[int]
    SHA256_FIELD_NUMBER: _ClassVar[int]
    BYTE_SIZE_FIELD_NUMBER: _ClassVar[int]
    relative_path: str
    sha256: str
    byte_size: int
    def __init__(self, relative_path: _Optional[str] = ..., sha256: _Optional[str] = ..., byte_size: _Optional[int] = ...) -> None: ...

class ArtifactVariant(_message.Message):
    __slots__ = ("variant_id", "backend", "abi", "files", "required_operators", "declared_peak_memory_bytes", "declared_storage_bytes", "declared_probe_ms", "declared_train_ms")
    VARIANT_ID_FIELD_NUMBER: _ClassVar[int]
    BACKEND_FIELD_NUMBER: _ClassVar[int]
    ABI_FIELD_NUMBER: _ClassVar[int]
    FILES_FIELD_NUMBER: _ClassVar[int]
    REQUIRED_OPERATORS_FIELD_NUMBER: _ClassVar[int]
    DECLARED_PEAK_MEMORY_BYTES_FIELD_NUMBER: _ClassVar[int]
    DECLARED_STORAGE_BYTES_FIELD_NUMBER: _ClassVar[int]
    DECLARED_PROBE_MS_FIELD_NUMBER: _ClassVar[int]
    DECLARED_TRAIN_MS_FIELD_NUMBER: _ClassVar[int]
    variant_id: str
    backend: ArtifactBackend
    abi: str
    files: _containers.RepeatedCompositeFieldContainer[ArtifactRef]
    required_operators: _containers.RepeatedScalarFieldContainer[str]
    declared_peak_memory_bytes: int
    declared_storage_bytes: int
    declared_probe_ms: int
    declared_train_ms: int
    def __init__(self, variant_id: _Optional[str] = ..., backend: _Optional[_Union[ArtifactBackend, str]] = ..., abi: _Optional[str] = ..., files: _Optional[_Iterable[_Union[ArtifactRef, _Mapping]]] = ..., required_operators: _Optional[_Iterable[str]] = ..., declared_peak_memory_bytes: _Optional[int] = ..., declared_storage_bytes: _Optional[int] = ..., declared_probe_ms: _Optional[int] = ..., declared_train_ms: _Optional[int] = ...) -> None: ...
