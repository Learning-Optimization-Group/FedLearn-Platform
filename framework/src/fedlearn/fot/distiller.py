"""FoT trace -> insight-library distiller (server side).

Faithful to FoT's server pipeline: collect raw traces -> cluster near-duplicate insights (an LLM
step) -> knowledge-extract canonical statements. Crucially, provenance/quorum is computed HERE
from the REAL traces (never trusted to the LLM), so a hallucination in one client's trace is not
promoted into the shared library unless `quorum` distinct clients independently support it.

Torch-free; the LLM is behind the AgentBackend seam, so distillation is fully testable offline.
"""
from __future__ import annotations

import json
from typing import Dict, List, Optional, Sequence, Set

from fedlearn.fot.backend import AgentBackend, ChatMessage
from fedlearn.fot.model import Insight, InsightLibrary, ReasoningTrace
from fedlearn.fot.provenance import insight_id, normalize

_SYSTEM = ChatMessage(
    "system",
    "You are the aggregation server in a Federation-over-Text system. You cluster near-duplicate "
    "reasoning insights. You never invent new insights.",
)


def _cluster_prompt(texts: List[str]) -> str:
    numbered = "\n".join(f"{i}: {t}" for i, t in enumerate(texts))
    return (
        "Group the following reasoning insights into clusters of near-duplicates. Return ONLY JSON "
        '{"clusters": [[<indices>], ...]} where each inner list holds the 0-based indices of '
        "insights expressing the same idea; every index appears in exactly one cluster.\n\n"
        + numbered
    )


class TraceDistiller:
    def __init__(self, backend: AgentBackend, *, quorum: int = 2, max_insights: int = 37) -> None:
        if quorum < 1:
            raise ValueError("quorum must be >= 1")
        self.backend = backend
        self.quorum = quorum
        self.max_insights = max_insights

    def distill(
        self, traces: Sequence[ReasoningTrace], *, prior: Optional[InsightLibrary] = None
    ) -> InsightLibrary:
        sources: Dict[str, Set[str]] = {}
        display: Dict[str, str] = {}
        order: List[str] = []
        floor: Dict[str, int] = {}  # support carried from the prior library, as a lower bound

        def _register(text: str):
            key = normalize(text)
            if not key:
                return None
            if key not in sources:
                sources[key] = set()
                display[key] = text.strip()
                order.append(key)
            return key

        def _add(text: str, client_id: str) -> None:
            key = _register(text)
            if key is not None:
                sources[key].add(client_id)

        # Fold the prior library back in so the library grows then stabilizes across rounds. Honor
        # each prior insight's recorded support_count as a FLOOR so an already-authoritative entry
        # that lacks per-client provenance (e.g. a seeded or JSON-deserialized library) is not
        # silently erased on the next round.
        if prior is not None:
            for ins in prior.insights:
                if ins.source_client_ids:
                    for cid in ins.source_client_ids:
                        _add(ins.statement, cid)
                else:
                    _register(ins.statement)
                key = normalize(ins.statement)
                if key:
                    floor[key] = max(floor.get(key, 0), ins.support_count)

        for tr in traces:
            for text in tr.insights.values():
                _add(text, tr.client_id)

        if not order:
            prior_insights = prior.insights if prior is not None else ()
            prior_version = prior.version if prior is not None else 1
            return InsightLibrary(insights=prior_insights, version=prior_version)

        clusters = self._cluster(order, [display[k] for k in order])

        built: List[Insight] = []
        for member_keys in clusters:
            member_keys = [k for k in member_keys if k in sources]
            if not member_keys:
                continue
            srcs: Set[str] = set()
            for k in member_keys:
                srcs |= sources[k]
            srcs.discard("")
            canonical = max((display[k] for k in member_keys), key=len)  # richest phrasing wins
            support = max(len(srcs), max((floor.get(k, 0) for k in member_keys), default=0))
            built.append(Insight(insight_id(canonical), canonical, support, tuple(sorted(srcs))))

        promoted = [i for i in built if i.support_count >= self.quorum]
        promoted.sort(key=lambda i: (-i.support_count, i.statement))
        new_insights = tuple(promoted[: self.max_insights])

        # Bump the version ONLY when the content actually changes, so GetInsightLibrary's
        # `unchanged` short-circuit works and clients don't re-download an identical library.
        # Compare on content (statement/support/sources), not the derived insight_id.
        def _sig(insights):
            return tuple((i.statement, i.support_count, i.source_client_ids) for i in insights)

        if prior is None:
            version = 1
        elif _sig(new_insights) == _sig(prior.insights):
            version = prior.version
        else:
            version = prior.version + 1
        return InsightLibrary(insights=new_insights, version=version)

    def _cluster(self, keys: List[str], texts: List[str]) -> List[List[str]]:
        """Cluster near-duplicates via the LLM; deterministic singleton fallback on any failure."""
        try:
            raw = self.backend.complete(
                [_SYSTEM, ChatMessage("user", _cluster_prompt(texts))],
                response_format="json",
            )
            groups = json.loads(raw)["clusters"]
            seen: Set[int] = set()
            clusters: List[List[str]] = []
            for group in groups:
                members: List[str] = []
                for idx in group:
                    i = int(idx)
                    if 0 <= i < len(keys) and i not in seen:
                        seen.add(i)
                        members.append(keys[i])
                if members:
                    clusters.append(members)
            for i, key in enumerate(keys):  # any unclustered index -> its own singleton
                if i not in seen:
                    clusters.append([key])
            return clusters
        except Exception:
            return [[key] for key in keys]
