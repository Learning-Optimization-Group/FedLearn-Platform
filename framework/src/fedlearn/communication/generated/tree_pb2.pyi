from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class TreeRegistration(_message.Message):
    __slots__ = ("client_id", "run_id", "contract_id", "protocol_version", "partition_id")
    CLIENT_ID_FIELD_NUMBER: _ClassVar[int]
    RUN_ID_FIELD_NUMBER: _ClassVar[int]
    CONTRACT_ID_FIELD_NUMBER: _ClassVar[int]
    PROTOCOL_VERSION_FIELD_NUMBER: _ClassVar[int]
    PARTITION_ID_FIELD_NUMBER: _ClassVar[int]
    client_id: str
    run_id: str
    contract_id: str
    protocol_version: int
    partition_id: int
    def __init__(self, client_id: _Optional[str] = ..., run_id: _Optional[str] = ..., contract_id: _Optional[str] = ..., protocol_version: _Optional[int] = ..., partition_id: _Optional[int] = ...) -> None: ...

class TreeRegistrationAck(_message.Message):
    __slots__ = ("protocol_version",)
    PROTOCOL_VERSION_FIELD_NUMBER: _ClassVar[int]
    protocol_version: int
    def __init__(self, protocol_version: _Optional[int] = ...) -> None: ...

class TreeModelRequest(_message.Message):
    __slots__ = ("client_id", "run_id", "contract_id")
    CLIENT_ID_FIELD_NUMBER: _ClassVar[int]
    RUN_ID_FIELD_NUMBER: _ClassVar[int]
    CONTRACT_ID_FIELD_NUMBER: _ClassVar[int]
    client_id: str
    run_id: str
    contract_id: str
    def __init__(self, client_id: _Optional[str] = ..., run_id: _Optional[str] = ..., contract_id: _Optional[str] = ...) -> None: ...

class TreeModel(_message.Message):
    __slots__ = ("state", "server_round", "parent_sha256", "model_json", "contract_id", "deadline_unix_ms", "initial_model_ubj")
    class State(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        STATE_UNSPECIFIED: _ClassVar[TreeModel.State]
        WAITING: _ClassVar[TreeModel.State]
        TRAINING: _ClassVar[TreeModel.State]
        FINALIZING: _ClassVar[TreeModel.State]
        COMPLETE: _ClassVar[TreeModel.State]
        FAILED: _ClassVar[TreeModel.State]
    STATE_UNSPECIFIED: TreeModel.State
    WAITING: TreeModel.State
    TRAINING: TreeModel.State
    FINALIZING: TreeModel.State
    COMPLETE: TreeModel.State
    FAILED: TreeModel.State
    STATE_FIELD_NUMBER: _ClassVar[int]
    SERVER_ROUND_FIELD_NUMBER: _ClassVar[int]
    PARENT_SHA256_FIELD_NUMBER: _ClassVar[int]
    MODEL_JSON_FIELD_NUMBER: _ClassVar[int]
    CONTRACT_ID_FIELD_NUMBER: _ClassVar[int]
    DEADLINE_UNIX_MS_FIELD_NUMBER: _ClassVar[int]
    INITIAL_MODEL_UBJ_FIELD_NUMBER: _ClassVar[int]
    state: TreeModel.State
    server_round: int
    parent_sha256: str
    model_json: bytes
    contract_id: str
    deadline_unix_ms: int
    initial_model_ubj: bytes
    def __init__(self, state: _Optional[_Union[TreeModel.State, str]] = ..., server_round: _Optional[int] = ..., parent_sha256: _Optional[str] = ..., model_json: _Optional[bytes] = ..., contract_id: _Optional[str] = ..., deadline_unix_ms: _Optional[int] = ..., initial_model_ubj: _Optional[bytes] = ...) -> None: ...

class TreeContribution(_message.Message):
    __slots__ = ("client_id", "run_id", "contract_id", "server_round", "parent_sha256", "delta_sha256", "new_trees_json", "num_examples")
    CLIENT_ID_FIELD_NUMBER: _ClassVar[int]
    RUN_ID_FIELD_NUMBER: _ClassVar[int]
    CONTRACT_ID_FIELD_NUMBER: _ClassVar[int]
    SERVER_ROUND_FIELD_NUMBER: _ClassVar[int]
    PARENT_SHA256_FIELD_NUMBER: _ClassVar[int]
    DELTA_SHA256_FIELD_NUMBER: _ClassVar[int]
    NEW_TREES_JSON_FIELD_NUMBER: _ClassVar[int]
    NUM_EXAMPLES_FIELD_NUMBER: _ClassVar[int]
    client_id: str
    run_id: str
    contract_id: str
    server_round: int
    parent_sha256: str
    delta_sha256: str
    new_trees_json: bytes
    num_examples: int
    def __init__(self, client_id: _Optional[str] = ..., run_id: _Optional[str] = ..., contract_id: _Optional[str] = ..., server_round: _Optional[int] = ..., parent_sha256: _Optional[str] = ..., delta_sha256: _Optional[str] = ..., new_trees_json: _Optional[bytes] = ..., num_examples: _Optional[int] = ...) -> None: ...

class TreeContributionAck(_message.Message):
    __slots__ = ("result",)
    class Result(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        RESULT_UNSPECIFIED: _ClassVar[TreeContributionAck.Result]
        ACCEPTED: _ClassVar[TreeContributionAck.Result]
        ALREADY_ACCEPTED: _ClassVar[TreeContributionAck.Result]
    RESULT_UNSPECIFIED: TreeContributionAck.Result
    ACCEPTED: TreeContributionAck.Result
    ALREADY_ACCEPTED: TreeContributionAck.Result
    RESULT_FIELD_NUMBER: _ClassVar[int]
    result: TreeContributionAck.Result
    def __init__(self, result: _Optional[_Union[TreeContributionAck.Result, str]] = ...) -> None: ...
