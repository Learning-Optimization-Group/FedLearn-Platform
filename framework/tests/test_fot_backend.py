"""Tests for the FoT AgentBackend seam + DeterministicStubBackend."""
import inspect

import pytest

from fedlearn.fot.backend import (
    AgentBackend,
    BackendError,
    ChatMessage,
    DeterministicStubBackend,
    get_backend,
)


def _user(text):
    return [ChatMessage("user", text)]


def test_stub_keyed_and_default_and_records_calls():
    b = DeterministicStubBackend(keyed={"solve": "S", "reflect": "R"}, default="D")
    assert b.complete(_user("please solve this")) == "S"
    assert b.complete(_user("now reflect on it")) == "R"
    assert b.complete(_user("unrelated")) == "D"
    assert len(b.calls) == 3
    assert b.calls[0][0].content == "please solve this"


def test_stub_is_deterministic():
    a = DeterministicStubBackend(keyed={"x": "1"}, default="0")
    b = DeterministicStubBackend(keyed={"x": "1"}, default="0")
    assert a.complete(_user("x marks")) == b.complete(_user("x marks")) == "1"


def test_stub_scripted_by_index_then_exhaustion():
    b = DeterministicStubBackend(scripted=["a", "b"])
    assert b.complete(_user("q1")) == "a"
    assert b.complete(_user("q2")) == "b"
    with pytest.raises(BackendError):
        b.complete(_user("q3"))


def test_stub_responder_callable():
    b = DeterministicStubBackend(responder=lambda msgs: f"got:{len(msgs)}")
    assert b.complete(_user("hi")) == "got:1"


def test_complete_temperature_default_is_zero():
    sig = inspect.signature(DeterministicStubBackend.complete)
    assert sig.parameters["temperature"].default == 0.0


def test_get_backend_stub_and_errors():
    assert isinstance(get_backend("stub"), DeterministicStubBackend)
    with pytest.raises(BackendError):
        get_backend("local-http")  # local adapter is a documented, unimplemented seam
    with pytest.raises(BackendError):
        get_backend("gpt-9")


def test_stub_satisfies_protocol():
    assert isinstance(DeterministicStubBackend(), AgentBackend)


def test_stub_opens_no_socket(monkeypatch):
    import socket

    def boom(*a, **k):
        raise AssertionError("FoT stub backend must not open a socket")

    monkeypatch.setattr(socket.socket, "connect", boom)
    b = DeterministicStubBackend(default="ok")
    assert b.complete(_user("hello")) == "ok"
