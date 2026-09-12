from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class SubmitReasoningTraceRequest(_message.Message):
    __slots__ = ("client_id", "round", "trace_json")
    CLIENT_ID_FIELD_NUMBER: _ClassVar[int]
    ROUND_FIELD_NUMBER: _ClassVar[int]
    TRACE_JSON_FIELD_NUMBER: _ClassVar[int]
    client_id: str
    round: int
    trace_json: str
    def __init__(self, client_id: _Optional[str] = ..., round: _Optional[int] = ..., trace_json: _Optional[str] = ...) -> None: ...

class SubmitReasoningTraceResponse(_message.Message):
    __slots__ = ("accepted", "reason")
    ACCEPTED_FIELD_NUMBER: _ClassVar[int]
    REASON_FIELD_NUMBER: _ClassVar[int]
    accepted: bool
    reason: str
    def __init__(self, accepted: bool = ..., reason: _Optional[str] = ...) -> None: ...

class GetInsightLibraryRequest(_message.Message):
    __slots__ = ("client_id", "known_version")
    CLIENT_ID_FIELD_NUMBER: _ClassVar[int]
    KNOWN_VERSION_FIELD_NUMBER: _ClassVar[int]
    client_id: str
    known_version: int
    def __init__(self, client_id: _Optional[str] = ..., known_version: _Optional[int] = ...) -> None: ...

class GetInsightLibraryResponse(_message.Message):
    __slots__ = ("unchanged", "version", "library_json")
    UNCHANGED_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    LIBRARY_JSON_FIELD_NUMBER: _ClassVar[int]
    unchanged: bool
    version: int
    library_json: str
    def __init__(self, unchanged: bool = ..., version: _Optional[int] = ..., library_json: _Optional[str] = ...) -> None: ...
