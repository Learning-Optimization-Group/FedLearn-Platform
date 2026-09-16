"""FoT client reasoning agent: solve -> reflect -> extract -> (redact) -> ReasoningTrace.

The local task/problem stays on the client; only abstracted, redacted insights are emitted. The
LLM is behind the AgentBackend seam, so the agent runs fully offline in tests. Torch-free.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, Optional

from fedlearn.fot.backend import AgentBackend, ChatMessage
from fedlearn.fot.model import INSIGHT_KEY_PREFIX, ReasoningTrace
from fedlearn.fot.redaction import TraceRedactor

_SYSTEM = ChatMessage(
    "system",
    "You are a careful problem-solving agent. You extract GENERAL, reusable reasoning insights "
    "and never restate the specific problem or its data.",
)


@dataclass
class Task:
    """A local problem. `prompt` stays client-local and is never uploaded."""

    task_id: str
    prompt: str


def _coerce_insight_key(name: str) -> str:
    slug = "".join(ch if ch.isalnum() else "_" for ch in name.strip().lower()).strip("_") or "insight"
    return slug if slug.startswith(INSIGHT_KEY_PREFIX) else INSIGHT_KEY_PREFIX + slug


def _parse_insights(raw: str) -> Dict[str, str]:
    try:
        data = json.loads(raw)
    except Exception:
        return {}
    if isinstance(data, dict) and isinstance(data.get("insights"), dict):
        data = data["insights"]
    if not isinstance(data, dict):
        return {}
    out: Dict[str, str] = {}
    for key, val in data.items():
        if isinstance(val, str) and val.strip():
            out[_coerce_insight_key(str(key))] = " ".join(val.split())
    return out


class ReasoningAgent:
    def __init__(
        self,
        backend: AgentBackend,
        client_id: str,
        run_id: str,
        *,
        redactor: Optional[TraceRedactor] = None,
    ) -> None:
        self.backend = backend
        self.client_id = client_id
        self.run_id = run_id
        self.redactor = redactor

    def run(self, task: Task, *, round_index: int, library=None) -> ReasoningTrace:
        context = ""
        if library is not None and len(library) > 0:
            context = (
                "Known insights so far (use them):\n"
                + "\n".join(f"- {s}" for s in library.statements())
                + "\n\n"
            )
        # Stage 1: solve (the raw problem never leaves the client).
        solution = self.backend.complete(
            [_SYSTEM, ChatMessage("user", f"{context}Solve this task:\n{task.prompt}")]
        )
        # Stage 2: reflect on the general technique that worked.
        reflection = self.backend.complete(
            [_SYSTEM, ChatMessage("user",
                "Reflect on how you solved it. What general, transferable technique worked?\n"
                "Solution:\n" + solution)]
        )
        # Stage 3: extract abstract insights as JSON.
        extract_user = (
            'Extract reusable, ABSTRACT insights as JSON of the form '
            '{"insight_<name>": "<general statement, no problem specifics>"}.\n'
            "Reflection:\n" + reflection
        )
        raw = self.backend.complete(
            [_SYSTEM, ChatMessage("user", extract_user)], response_format="json"
        )
        insights = _parse_insights(raw)
        # Pre-egress redaction (empirical leakage control), if a redactor is wired.
        if self.redactor is not None and insights:
            insights = self.redactor.redact(insights).kept
        return ReasoningTrace(
            trace_id=f"{self.client_id}-r{round_index}-{task.task_id}",
            client_id=self.client_id,
            run_id=self.run_id,
            round=round_index,
            task_id=task.task_id,
            insights=insights,
        )
