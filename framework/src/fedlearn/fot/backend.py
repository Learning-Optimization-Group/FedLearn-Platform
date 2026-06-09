"""LLM/agent backend seam for the FoT module.

Every model call in fedlearn.fot routes through the AgentBackend protocol so the whole package
is exercisable offline. DeterministicStubBackend opens no socket and returns scripted/derived
text, which is what makes FoT testable with no LLM, network, or GPU.

Torch-free and independent of the gradient (DeComFL/FedAvg) path.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Union, runtime_checkable

try:  # Protocol is stdlib on 3.8+, but guard for safety.
    from typing import Protocol
except ImportError:  # pragma: no cover
    from typing_extensions import Protocol  # type: ignore


@dataclass(frozen=True)
class ChatMessage:
    """A single chat turn. role is one of: 'system' | 'user' | 'assistant'."""

    role: str
    content: str


class BackendError(RuntimeError):
    """Raised when a backend cannot satisfy a request."""


@runtime_checkable
class AgentBackend(Protocol):
    """The single LLM seam. A real impl calls a local model server; tests use the stub."""

    def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        response_format: Optional[str] = None,
    ) -> str:
        ...


Responder = Callable[[Sequence[ChatMessage]], str]


class DeterministicStubBackend:
    """Offline AgentBackend for tests/examples — never opens a socket.

    Resolution order per complete() call:
      1. ``responder`` callable, if given -> responder(messages)
      2. ``scripted`` list, if given -> next entry by call index (BackendError if exhausted)
      3. ``keyed`` substrings matched against the last user message (first match wins)
      4. ``default``

    Every call's messages are appended to ``.calls`` for prompt-shape assertions.
    """

    def __init__(
        self,
        scripted: Optional[Sequence[str]] = None,
        *,
        responder: Optional[Responder] = None,
        keyed: Optional[Dict[str, str]] = None,
        default: str = "{}",
    ) -> None:
        self._scripted: Optional[List[str]] = list(scripted) if scripted is not None else None
        self._responder = responder
        self._keyed: Dict[str, str] = dict(keyed) if keyed else {}
        self._default = default
        self._i = 0
        self.calls: List[List[ChatMessage]] = []

    def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        response_format: Optional[str] = None,
    ) -> str:
        self.calls.append(list(messages))
        if self._responder is not None:
            return self._responder(messages)
        if self._scripted is not None:
            if self._i >= len(self._scripted):
                raise BackendError("DeterministicStubBackend: scripted responses exhausted")
            out = self._scripted[self._i]
            self._i += 1
            return out
        last_user = next(
            (m.content for m in reversed(list(messages)) if m.role == "user"), ""
        )
        for key, val in self._keyed.items():
            if key in last_user:
                return val
        return self._default


def get_backend(name: str = "stub", **kwargs) -> AgentBackend:
    """Factory. Only the offline stub is wired; a real local adapter is a documented seam.

    FoT is a local-LLM-only research mode — there is intentionally no hosted-API backend here
    (that would defeat the on-device framing). Tests and the example use ``name='stub'``.
    """
    if name == "stub":
        return DeterministicStubBackend(**kwargs)
    if name in ("local-http", "vllm", "ollama"):
        raise BackendError(
            f"Backend '{name}' is not implemented in this build. Wire a LOCAL OpenAI-compatible "
            "adapter (e.g. a vLLM/Ollama server on localhost) here; do not call a hosted API. "
            "Tests/examples use the 'stub' backend."
        )
    raise BackendError(f"Unknown backend '{name}'")
