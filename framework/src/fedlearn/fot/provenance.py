"""Cross-client provenance/quorum ledger (hallucination-propagation control).

FoT's own limitation: an LLM hallucination in one client's trace can be baked into the shared
library as authoritative text. The InsightLedger tracks how many DISTINCT clients support each
(normalized) statement, so the distiller can promote only quorum-backed insights and merely flag
single-source ones. Torch-free.
"""
from __future__ import annotations

import hashlib
from typing import Dict, List, Set

from fedlearn.fot.model import Insight


def normalize(statement: str) -> str:
    # Collapse whitespace + lowercase, then strip surrounding punctuation so statements that
    # differ only by trailing punctuation ("validate inputs early" vs "Validate inputs early.")
    # are treated as the same insight for quorum/dedup.
    return " ".join(statement.lower().split()).strip(".!?,;: \t\n\"'")


def insight_id(statement: str) -> str:
    return "i_" + hashlib.sha256(normalize(statement).encode("utf-8")).hexdigest()[:10]


class InsightLedger:
    def __init__(self, *, quorum: int = 2) -> None:
        if quorum < 1:
            raise ValueError("quorum must be >= 1")
        self.quorum = quorum
        self._sources: Dict[str, Set[str]] = {}
        self._display: Dict[str, str] = {}

    def record(self, statement: str, client_id: str) -> None:
        key = normalize(statement)
        if not key:
            return
        self._sources.setdefault(key, set()).add(client_id)
        self._display.setdefault(key, statement.strip())

    def _build(self, key: str) -> Insight:
        srcs = tuple(sorted(self._sources[key]))
        return Insight(
            insight_id=insight_id(self._display[key]),
            statement=self._display[key],
            support_count=len(srcs),
            source_client_ids=srcs,
        )

    def promoted(self) -> List[Insight]:
        out = [self._build(k) for k, s in self._sources.items() if len(s) >= self.quorum]
        return sorted(out, key=lambda i: (-i.support_count, i.statement))

    def flagged(self) -> List[Insight]:
        out = [self._build(k) for k, s in self._sources.items() if len(s) < self.quorum]
        return sorted(out, key=lambda i: i.statement)
