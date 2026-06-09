"""In-process FoT round loop.

Runs one full Federation-over-Text round entirely in-process (no network): each (agent, task)
pair emits a ReasoningTrace, the TraceValidator quarantines unsafe/adversarial traces at ingest,
and the TraceDistiller folds the survivors into the updated InsightLibrary. The same loop backs
the gRPC server (fot_server) — that path just sources traces over the wire instead. Torch-free.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from fedlearn.fot.agent import ReasoningAgent, Task
from fedlearn.fot.distiller import TraceDistiller
from fedlearn.fot.model import InsightLibrary, ReasoningTrace
from fedlearn.fot.trace_guard import TraceValidator


def run_fot_round(
    pairs: Sequence[Tuple[ReasoningAgent, Task]],
    distiller: TraceDistiller,
    *,
    round_index: int,
    library: Optional[InsightLibrary] = None,
    validator: Optional[TraceValidator] = None,
) -> InsightLibrary:
    validator = validator or TraceValidator()
    safe_traces: List[ReasoningTrace] = []
    for agent, task in pairs:
        trace = agent.run(task, round_index=round_index, library=library)
        if validator.is_safe(trace):
            safe_traces.append(trace)
    return distiller.distill(safe_traces, prior=library)
