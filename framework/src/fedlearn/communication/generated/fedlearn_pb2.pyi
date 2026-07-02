from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Tensor(_message.Message):
    __slots__ = ("data", "dims", "dtype")
    DATA_FIELD_NUMBER: _ClassVar[int]
    DIMS_FIELD_NUMBER: _ClassVar[int]
    DTYPE_FIELD_NUMBER: _ClassVar[int]
    data: bytes
    dims: _containers.RepeatedScalarFieldContainer[int]
    dtype: str
    def __init__(self, data: _Optional[bytes] = ..., dims: _Optional[_Iterable[int]] = ..., dtype: _Optional[str] = ...) -> None: ...

class ModelParameters(_message.Message):
    __slots__ = ("tensors", "num_examples_trained")
    class TensorsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: Tensor
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[Tensor, _Mapping]] = ...) -> None: ...
    TENSORS_FIELD_NUMBER: _ClassVar[int]
    NUM_EXAMPLES_TRAINED_FIELD_NUMBER: _ClassVar[int]
    tensors: _containers.MessageMap[str, Tensor]
    num_examples_trained: int
    def __init__(self, tensors: _Optional[_Mapping[str, Tensor]] = ..., num_examples_trained: _Optional[int] = ...) -> None: ...

class RegisterClientRequest(_message.Message):
    __slots__ = ("client_id", "run_id", "protocol_version", "enrollment_token")
    CLIENT_ID_FIELD_NUMBER: _ClassVar[int]
    RUN_ID_FIELD_NUMBER: _ClassVar[int]
    PROTOCOL_VERSION_FIELD_NUMBER: _ClassVar[int]
    ENROLLMENT_TOKEN_FIELD_NUMBER: _ClassVar[int]
    client_id: str
    run_id: str
    protocol_version: int
    enrollment_token: str
    def __init__(self, client_id: _Optional[str] = ..., run_id: _Optional[str] = ..., protocol_version: _Optional[int] = ..., enrollment_token: _Optional[str] = ...) -> None: ...

class RegisterClientResponse(_message.Message):
    __slots__ = ("status", "message", "assigned_round", "protocol_version")
    class Status(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        STATUS_UNSPECIFIED: _ClassVar[RegisterClientResponse.Status]
        ACCEPTED: _ClassVar[RegisterClientResponse.Status]
        REJECTED: _ClassVar[RegisterClientResponse.Status]
    STATUS_UNSPECIFIED: RegisterClientResponse.Status
    ACCEPTED: RegisterClientResponse.Status
    REJECTED: RegisterClientResponse.Status
    STATUS_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    ASSIGNED_ROUND_FIELD_NUMBER: _ClassVar[int]
    PROTOCOL_VERSION_FIELD_NUMBER: _ClassVar[int]
    status: RegisterClientResponse.Status
    message: str
    assigned_round: int
    protocol_version: int
    def __init__(self, status: _Optional[_Union[RegisterClientResponse.Status, str]] = ..., message: _Optional[str] = ..., assigned_round: _Optional[int] = ..., protocol_version: _Optional[int] = ...) -> None: ...

class GetServerStatusRequest(_message.Message):
    __slots__ = ("run_id",)
    RUN_ID_FIELD_NUMBER: _ClassVar[int]
    run_id: str
    def __init__(self, run_id: _Optional[str] = ...) -> None: ...

class GetServerStatusResponse(_message.Message):
    __slots__ = ("server_state", "current_round", "required_clients_for_round", "received_updates_this_round", "active_clients", "round_deadline_unix_ms")
    class ServerState(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        STATE_UNSPECIFIED: _ClassVar[GetServerStatusResponse.ServerState]
        INITIALIZING: _ClassVar[GetServerStatusResponse.ServerState]
        WAITING_FOR_CLIENTS: _ClassVar[GetServerStatusResponse.ServerState]
        TRAINING: _ClassVar[GetServerStatusResponse.ServerState]
        AGGREGATING: _ClassVar[GetServerStatusResponse.ServerState]
        TRAINING_COMPLETE: _ClassVar[GetServerStatusResponse.ServerState]
        FAILED: _ClassVar[GetServerStatusResponse.ServerState]
    STATE_UNSPECIFIED: GetServerStatusResponse.ServerState
    INITIALIZING: GetServerStatusResponse.ServerState
    WAITING_FOR_CLIENTS: GetServerStatusResponse.ServerState
    TRAINING: GetServerStatusResponse.ServerState
    AGGREGATING: GetServerStatusResponse.ServerState
    TRAINING_COMPLETE: GetServerStatusResponse.ServerState
    FAILED: GetServerStatusResponse.ServerState
    SERVER_STATE_FIELD_NUMBER: _ClassVar[int]
    CURRENT_ROUND_FIELD_NUMBER: _ClassVar[int]
    REQUIRED_CLIENTS_FOR_ROUND_FIELD_NUMBER: _ClassVar[int]
    RECEIVED_UPDATES_THIS_ROUND_FIELD_NUMBER: _ClassVar[int]
    ACTIVE_CLIENTS_FIELD_NUMBER: _ClassVar[int]
    ROUND_DEADLINE_UNIX_MS_FIELD_NUMBER: _ClassVar[int]
    server_state: GetServerStatusResponse.ServerState
    current_round: int
    required_clients_for_round: int
    received_updates_this_round: int
    active_clients: int
    round_deadline_unix_ms: int
    def __init__(self, server_state: _Optional[_Union[GetServerStatusResponse.ServerState, str]] = ..., current_round: _Optional[int] = ..., required_clients_for_round: _Optional[int] = ..., received_updates_this_round: _Optional[int] = ..., active_clients: _Optional[int] = ..., round_deadline_unix_ms: _Optional[int] = ...) -> None: ...

class HeartbeatRequest(_message.Message):
    __slots__ = ("client_id", "run_id", "status", "current_step", "total_steps", "current_round")
    CLIENT_ID_FIELD_NUMBER: _ClassVar[int]
    RUN_ID_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    CURRENT_STEP_FIELD_NUMBER: _ClassVar[int]
    TOTAL_STEPS_FIELD_NUMBER: _ClassVar[int]
    CURRENT_ROUND_FIELD_NUMBER: _ClassVar[int]
    client_id: str
    run_id: str
    status: str
    current_step: int
    total_steps: int
    current_round: int
    def __init__(self, client_id: _Optional[str] = ..., run_id: _Optional[str] = ..., status: _Optional[str] = ..., current_step: _Optional[int] = ..., total_steps: _Optional[int] = ..., current_round: _Optional[int] = ...) -> None: ...

class HeartbeatResponse(_message.Message):
    __slots__ = ("acknowledged", "should_stop", "message")
    ACKNOWLEDGED_FIELD_NUMBER: _ClassVar[int]
    SHOULD_STOP_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    acknowledged: bool
    should_stop: bool
    message: str
    def __init__(self, acknowledged: bool = ..., should_stop: bool = ..., message: _Optional[str] = ...) -> None: ...

class GetGlobalModelRequest(_message.Message):
    __slots__ = ("client_id", "run_id")
    CLIENT_ID_FIELD_NUMBER: _ClassVar[int]
    RUN_ID_FIELD_NUMBER: _ClassVar[int]
    client_id: str
    run_id: str
    def __init__(self, client_id: _Optional[str] = ..., run_id: _Optional[str] = ...) -> None: ...

class GetGlobalModelResponse(_message.Message):
    __slots__ = ("parameters", "current_round", "config", "total_bytes")
    class ConfigEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    PARAMETERS_FIELD_NUMBER: _ClassVar[int]
    CURRENT_ROUND_FIELD_NUMBER: _ClassVar[int]
    CONFIG_FIELD_NUMBER: _ClassVar[int]
    TOTAL_BYTES_FIELD_NUMBER: _ClassVar[int]
    parameters: ModelParameters
    current_round: int
    config: _containers.ScalarMap[str, str]
    total_bytes: int
    def __init__(self, parameters: _Optional[_Union[ModelParameters, _Mapping]] = ..., current_round: _Optional[int] = ..., config: _Optional[_Mapping[str, str]] = ..., total_bytes: _Optional[int] = ...) -> None: ...

class ModelChunk(_message.Message):
    __slots__ = ("chunk_index", "total_chunks", "chunk_data", "is_final_chunk", "current_round", "config", "codec", "compressed", "total_bytes", "sha256")
    class ConfigEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    CHUNK_INDEX_FIELD_NUMBER: _ClassVar[int]
    TOTAL_CHUNKS_FIELD_NUMBER: _ClassVar[int]
    CHUNK_DATA_FIELD_NUMBER: _ClassVar[int]
    IS_FINAL_CHUNK_FIELD_NUMBER: _ClassVar[int]
    CURRENT_ROUND_FIELD_NUMBER: _ClassVar[int]
    CONFIG_FIELD_NUMBER: _ClassVar[int]
    CODEC_FIELD_NUMBER: _ClassVar[int]
    COMPRESSED_FIELD_NUMBER: _ClassVar[int]
    TOTAL_BYTES_FIELD_NUMBER: _ClassVar[int]
    SHA256_FIELD_NUMBER: _ClassVar[int]
    chunk_index: int
    total_chunks: int
    chunk_data: bytes
    is_final_chunk: bool
    current_round: int
    config: _containers.ScalarMap[str, str]
    codec: str
    compressed: bool
    total_bytes: int
    sha256: str
    def __init__(self, chunk_index: _Optional[int] = ..., total_chunks: _Optional[int] = ..., chunk_data: _Optional[bytes] = ..., is_final_chunk: bool = ..., current_round: _Optional[int] = ..., config: _Optional[_Mapping[str, str]] = ..., codec: _Optional[str] = ..., compressed: bool = ..., total_bytes: _Optional[int] = ..., sha256: _Optional[str] = ...) -> None: ...

class SubmitModelUpdateRequest(_message.Message):
    __slots__ = ("client_id", "run_id", "parameters", "trained_on_round")
    CLIENT_ID_FIELD_NUMBER: _ClassVar[int]
    RUN_ID_FIELD_NUMBER: _ClassVar[int]
    PARAMETERS_FIELD_NUMBER: _ClassVar[int]
    TRAINED_ON_ROUND_FIELD_NUMBER: _ClassVar[int]
    client_id: str
    run_id: str
    parameters: ModelParameters
    trained_on_round: int
    def __init__(self, client_id: _Optional[str] = ..., run_id: _Optional[str] = ..., parameters: _Optional[_Union[ModelParameters, _Mapping]] = ..., trained_on_round: _Optional[int] = ...) -> None: ...

class SubmitModelUpdateResponse(_message.Message):
    __slots__ = ("received", "bytes_received")
    RECEIVED_FIELD_NUMBER: _ClassVar[int]
    BYTES_RECEIVED_FIELD_NUMBER: _ClassVar[int]
    received: bool
    bytes_received: int
    def __init__(self, received: bool = ..., bytes_received: _Optional[int] = ...) -> None: ...

class ModelUpdateChunk(_message.Message):
    __slots__ = ("client_id", "run_id", "trained_on_round", "chunk_index", "total_chunks", "chunk_data", "is_final_chunk", "num_examples", "codec", "compressed", "total_bytes", "sha256")
    CLIENT_ID_FIELD_NUMBER: _ClassVar[int]
    RUN_ID_FIELD_NUMBER: _ClassVar[int]
    TRAINED_ON_ROUND_FIELD_NUMBER: _ClassVar[int]
    CHUNK_INDEX_FIELD_NUMBER: _ClassVar[int]
    TOTAL_CHUNKS_FIELD_NUMBER: _ClassVar[int]
    CHUNK_DATA_FIELD_NUMBER: _ClassVar[int]
    IS_FINAL_CHUNK_FIELD_NUMBER: _ClassVar[int]
    NUM_EXAMPLES_FIELD_NUMBER: _ClassVar[int]
    CODEC_FIELD_NUMBER: _ClassVar[int]
    COMPRESSED_FIELD_NUMBER: _ClassVar[int]
    TOTAL_BYTES_FIELD_NUMBER: _ClassVar[int]
    SHA256_FIELD_NUMBER: _ClassVar[int]
    client_id: str
    run_id: str
    trained_on_round: int
    chunk_index: int
    total_chunks: int
    chunk_data: bytes
    is_final_chunk: bool
    num_examples: int
    codec: str
    compressed: bool
    total_bytes: int
    sha256: str
    def __init__(self, client_id: _Optional[str] = ..., run_id: _Optional[str] = ..., trained_on_round: _Optional[int] = ..., chunk_index: _Optional[int] = ..., total_chunks: _Optional[int] = ..., chunk_data: _Optional[bytes] = ..., is_final_chunk: bool = ..., num_examples: _Optional[int] = ..., codec: _Optional[str] = ..., compressed: bool = ..., total_bytes: _Optional[int] = ..., sha256: _Optional[str] = ...) -> None: ...

class PerturbationSeeds(_message.Message):
    __slots__ = ("local_steps",)
    LOCAL_STEPS_FIELD_NUMBER: _ClassVar[int]
    local_steps: _containers.RepeatedCompositeFieldContainer[LocalStepSeeds]
    def __init__(self, local_steps: _Optional[_Iterable[_Union[LocalStepSeeds, _Mapping]]] = ...) -> None: ...

class LocalStepSeeds(_message.Message):
    __slots__ = ("seeds",)
    SEEDS_FIELD_NUMBER: _ClassVar[int]
    seeds: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, seeds: _Optional[_Iterable[int]] = ...) -> None: ...

class GradientScalars(_message.Message):
    __slots__ = ("local_steps",)
    LOCAL_STEPS_FIELD_NUMBER: _ClassVar[int]
    local_steps: _containers.RepeatedCompositeFieldContainer[LocalStepGradients]
    def __init__(self, local_steps: _Optional[_Iterable[_Union[LocalStepGradients, _Mapping]]] = ...) -> None: ...

class LocalStepGradients(_message.Message):
    __slots__ = ("scalars",)
    SCALARS_FIELD_NUMBER: _ClassVar[int]
    scalars: _containers.RepeatedScalarFieldContainer[float]
    def __init__(self, scalars: _Optional[_Iterable[float]] = ...) -> None: ...

class RebuildHistory(_message.Message):
    __slots__ = ("rounds",)
    ROUNDS_FIELD_NUMBER: _ClassVar[int]
    rounds: _containers.RepeatedCompositeFieldContainer[RoundHistory]
    def __init__(self, rounds: _Optional[_Iterable[_Union[RoundHistory, _Mapping]]] = ...) -> None: ...

class RoundHistory(_message.Message):
    __slots__ = ("round_number", "seeds", "average_gradients")
    ROUND_NUMBER_FIELD_NUMBER: _ClassVar[int]
    SEEDS_FIELD_NUMBER: _ClassVar[int]
    AVERAGE_GRADIENTS_FIELD_NUMBER: _ClassVar[int]
    round_number: int
    seeds: PerturbationSeeds
    average_gradients: GradientScalars
    def __init__(self, round_number: _Optional[int] = ..., seeds: _Optional[_Union[PerturbationSeeds, _Mapping]] = ..., average_gradients: _Optional[_Union[GradientScalars, _Mapping]] = ...) -> None: ...

class GetDeComFLConfigRequest(_message.Message):
    __slots__ = ("client_id", "run_id")
    CLIENT_ID_FIELD_NUMBER: _ClassVar[int]
    RUN_ID_FIELD_NUMBER: _ClassVar[int]
    client_id: str
    run_id: str
    def __init__(self, client_id: _Optional[str] = ..., run_id: _Optional[str] = ...) -> None: ...

class GetDeComFLConfigResponse(_message.Message):
    __slots__ = ("current_round", "current_seeds", "rebuild_history", "config", "torch_version", "grad_estimate_method", "golden_vector_sha256")
    class ConfigEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    CURRENT_ROUND_FIELD_NUMBER: _ClassVar[int]
    CURRENT_SEEDS_FIELD_NUMBER: _ClassVar[int]
    REBUILD_HISTORY_FIELD_NUMBER: _ClassVar[int]
    CONFIG_FIELD_NUMBER: _ClassVar[int]
    TORCH_VERSION_FIELD_NUMBER: _ClassVar[int]
    GRAD_ESTIMATE_METHOD_FIELD_NUMBER: _ClassVar[int]
    GOLDEN_VECTOR_SHA256_FIELD_NUMBER: _ClassVar[int]
    current_round: int
    current_seeds: PerturbationSeeds
    rebuild_history: RebuildHistory
    config: _containers.ScalarMap[str, str]
    torch_version: str
    grad_estimate_method: str
    golden_vector_sha256: str
    def __init__(self, current_round: _Optional[int] = ..., current_seeds: _Optional[_Union[PerturbationSeeds, _Mapping]] = ..., rebuild_history: _Optional[_Union[RebuildHistory, _Mapping]] = ..., config: _Optional[_Mapping[str, str]] = ..., torch_version: _Optional[str] = ..., grad_estimate_method: _Optional[str] = ..., golden_vector_sha256: _Optional[str] = ...) -> None: ...

class SubmitGradientScalarsRequest(_message.Message):
    __slots__ = ("client_id", "run_id", "trained_on_round", "gradients", "num_examples", "perturbation_seeds")
    CLIENT_ID_FIELD_NUMBER: _ClassVar[int]
    RUN_ID_FIELD_NUMBER: _ClassVar[int]
    TRAINED_ON_ROUND_FIELD_NUMBER: _ClassVar[int]
    GRADIENTS_FIELD_NUMBER: _ClassVar[int]
    NUM_EXAMPLES_FIELD_NUMBER: _ClassVar[int]
    PERTURBATION_SEEDS_FIELD_NUMBER: _ClassVar[int]
    client_id: str
    run_id: str
    trained_on_round: int
    gradients: GradientScalars
    num_examples: int
    perturbation_seeds: PerturbationSeeds
    def __init__(self, client_id: _Optional[str] = ..., run_id: _Optional[str] = ..., trained_on_round: _Optional[int] = ..., gradients: _Optional[_Union[GradientScalars, _Mapping]] = ..., num_examples: _Optional[int] = ..., perturbation_seeds: _Optional[_Union[PerturbationSeeds, _Mapping]] = ...) -> None: ...

class SubmitGradientScalarsResponse(_message.Message):
    __slots__ = ("received", "bytes_received")
    RECEIVED_FIELD_NUMBER: _ClassVar[int]
    BYTES_RECEIVED_FIELD_NUMBER: _ClassVar[int]
    received: bool
    bytes_received: int
    def __init__(self, received: bool = ..., bytes_received: _Optional[int] = ...) -> None: ...

class ReportClientMetricsRequest(_message.Message):
    __slots__ = ("client_id", "run_id", "round", "loss", "accuracy", "current_step", "total_steps", "client_type", "compute_ms")
    CLIENT_ID_FIELD_NUMBER: _ClassVar[int]
    RUN_ID_FIELD_NUMBER: _ClassVar[int]
    ROUND_FIELD_NUMBER: _ClassVar[int]
    LOSS_FIELD_NUMBER: _ClassVar[int]
    ACCURACY_FIELD_NUMBER: _ClassVar[int]
    CURRENT_STEP_FIELD_NUMBER: _ClassVar[int]
    TOTAL_STEPS_FIELD_NUMBER: _ClassVar[int]
    CLIENT_TYPE_FIELD_NUMBER: _ClassVar[int]
    COMPUTE_MS_FIELD_NUMBER: _ClassVar[int]
    client_id: str
    run_id: str
    round: int
    loss: float
    accuracy: float
    current_step: int
    total_steps: int
    client_type: str
    compute_ms: int
    def __init__(self, client_id: _Optional[str] = ..., run_id: _Optional[str] = ..., round: _Optional[int] = ..., loss: _Optional[float] = ..., accuracy: _Optional[float] = ..., current_step: _Optional[int] = ..., total_steps: _Optional[int] = ..., client_type: _Optional[str] = ..., compute_ms: _Optional[int] = ...) -> None: ...

class ReportClientMetricsResponse(_message.Message):
    __slots__ = ("acknowledged",)
    ACKNOWLEDGED_FIELD_NUMBER: _ClassVar[int]
    acknowledged: bool
    def __init__(self, acknowledged: bool = ...) -> None: ...
