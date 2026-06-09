"""Frozen data models for FoT.

ReasoningTrace is the client->server upload (a dict of named abstracted insights, never raw
problems). Insight + InsightLibrary are the server artifact — plain JSON/markdown, no vector
store, which is the FoT deliverable. Torch-free.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import List, Mapping, Tuple

INSIGHT_KEY_PREFIX = "insight_"


@dataclass(frozen=True)
class ReasoningTrace:
    """One client's abstracted reasoning for one task in one round."""

    trace_id: str
    client_id: str
    run_id: str
    round: int
    task_id: str
    insights: Mapping[str, str]  # {"insight_<name>": "<abstracted text>"}
    schema_version: str = "1"

    def validate(self) -> List[str]:
        """Return a list of problems (empty list == valid)."""
        problems: List[str] = []
        if not self.client_id:
            problems.append("client_id is empty")
        if self.round < 0:
            problems.append("round is negative")
        if not isinstance(self.insights, Mapping) or len(self.insights) == 0:
            problems.append("insights must be a non-empty mapping")
            return problems
        for key, text in self.insights.items():
            if not isinstance(key, str) or not key.startswith(INSIGHT_KEY_PREFIX):
                problems.append(f"insight key '{key}' must start with '{INSIGHT_KEY_PREFIX}'")
            if not isinstance(text, str) or not text.strip():
                problems.append(f"insight '{key}' has empty text")
        return problems

    def to_json(self) -> str:
        return json.dumps(
            {
                "trace_id": self.trace_id,
                "client_id": self.client_id,
                "run_id": self.run_id,
                "round": self.round,
                "task_id": self.task_id,
                "insights": dict(self.insights),
                "schema_version": self.schema_version,
            },
            sort_keys=True,
        )

    @classmethod
    def from_json(cls, s: str) -> "ReasoningTrace":
        d = json.loads(s)
        return cls(
            trace_id=d["trace_id"],
            client_id=d["client_id"],
            run_id=d["run_id"],
            round=int(d["round"]),
            task_id=d["task_id"],
            insights=dict(d["insights"]),
            schema_version=str(d.get("schema_version", "1")),
        )


@dataclass(frozen=True)
class Insight:
    """A distilled, canonical insight in the shared library."""

    insight_id: str
    statement: str
    support_count: int = 1
    source_client_ids: Tuple[str, ...] = ()
    tags: Tuple[str, ...] = ()


@dataclass(frozen=True)
class InsightLibrary:
    """The interpretable FoT artifact — plain data, no embeddings/vector store."""

    insights: Tuple[Insight, ...] = ()
    version: int = 1

    def __len__(self) -> int:
        return len(self.insights)

    def statements(self) -> List[str]:
        return [i.statement for i in self.insights]

    def to_json(self) -> str:
        return json.dumps(
            {
                "version": self.version,
                "insights": [
                    {
                        "insight_id": i.insight_id,
                        "statement": i.statement,
                        "support_count": i.support_count,
                        "source_client_ids": list(i.source_client_ids),
                        "tags": list(i.tags),
                    }
                    for i in self.insights
                ],
            },
            sort_keys=True,
        )

    @classmethod
    def from_json(cls, s: str) -> "InsightLibrary":
        d = json.loads(s)
        return cls(
            version=int(d.get("version", 1)),
            insights=tuple(
                Insight(
                    insight_id=x["insight_id"],
                    statement=x["statement"],
                    support_count=int(x.get("support_count", 1)),
                    source_client_ids=tuple(x.get("source_client_ids", ())),
                    tags=tuple(x.get("tags", ())),
                )
                for x in d.get("insights", [])
            ),
        )

    def render_markdown(self) -> str:
        lines = [f"# Insight Library (v{self.version}) — {len(self.insights)} insights", ""]
        for i in self.insights:
            src = ", ".join(i.source_client_ids) if i.source_client_ids else "?"
            lines.append(f"## {i.insight_id}")
            lines.append(i.statement)
            lines.append(f"_support: {i.support_count} · sources: {src}_")
            lines.append("")
        return "\n".join(lines)

    def sha256(self) -> str:
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()
