#!/usr/bin/env python3
"""Local, offline demo of Federation over Text (no LLM, no network, no GPU).

Runs stub ReasoningAgents -> TraceDistiller -> InsightLibrary across two rounds and prints the
interpretable insight library. The cross-client QUORUM is the visible mechanic: an insight is
promoted only once >= `quorum` distinct clients independently surface it, so a single client's
hallucination is not baked into the shared library.

Swap DeterministicStubBackend for a real LOCAL-LLM adapter (fedlearn.fot.backend.get_backend) to
run against an actual model. FoT is a SEPARATE, local-LLM-only, non-PHI research mode — it does
NOT touch, and is not a replacement for, the DeComFL gradient path.

Run:  PYTHONPATH=../../src python run_fot.py     (from this directory)
"""
from fedlearn.fot.agent import ReasoningAgent, Task
from fedlearn.fot.backend import DeterministicStubBackend
from fedlearn.fot.distiller import TraceDistiller
from fedlearn.fot.round import run_fot_round


def _agent(client_id: str, extract_json: str) -> ReasoningAgent:
    # The stub scripts the agent's three stages (solve, reflect, extract) so the demo is offline.
    return ReasoningAgent(
        DeterministicStubBackend(scripted=["(solved)", "(reflected)", extract_json]),
        client_id,
        "demo-run",
    )


def main() -> None:
    distiller = TraceDistiller(DeterministicStubBackend(default="{}"), quorum=2)

    # Round 0: clients A and B independently surface the SAME general insight (-> promoted);
    # client C surfaces a one-off that lacks quorum (-> not promoted).
    round0 = [
        (_agent("clientA", '{"insight_symmetry": "Exploit symmetry to simplify integrals."}'), Task("t0", "...")),
        (_agent("clientB", '{"insight_symmetry": "exploit symmetry to simplify integrals"}'), Task("t0", "...")),
        (_agent("clientC", '{"insight_solo": "A one-off trick only one client has found."}'), Task("t0", "...")),
    ]
    library = run_fot_round(round0, distiller, round_index=0)
    print("=== Insight library after round 0 ===")
    print(library.render_markdown())

    # Round 1: a second client corroborates the one-off -> it reaches quorum and is promoted,
    # while the prior insight carries forward. The library grows as consensus builds.
    round1 = [
        (_agent("clientC", '{"insight_solo": "A one-off trick only one client has found."}'), Task("t1", "...")),
        (_agent("clientA", '{"insight_solo": "a one-off trick only one client has found"}'), Task("t1", "...")),
    ]
    library = run_fot_round(round1, distiller, round_index=1, library=library)
    print("=== Insight library after round 1 ===")
    print(library.render_markdown())


if __name__ == "__main__":
    main()
