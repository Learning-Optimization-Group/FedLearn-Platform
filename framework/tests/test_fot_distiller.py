"""Tests for the FoT trace -> insight-library distiller."""
from fedlearn.fot.backend import DeterministicStubBackend
from fedlearn.fot.distiller import TraceDistiller
from fedlearn.fot.model import ReasoningTrace


def _tr(cid, insights, rnd=0):
    return ReasoningTrace(f"{cid}-{rnd}", cid, "run", rnd, "task", insights)


def test_promotes_quorum_drops_singletons():
    # default "{}" makes the cluster step fall back to singletons (deterministic).
    d = TraceDistiller(DeterministicStubBackend(default="{}"), quorum=2)
    lib = d.distill([
        _tr("c1", {"insight_a": "Validate inputs early."}),
        _tr("c2", {"insight_a": "validate inputs early"}),  # same normalized, distinct client
        _tr("c1", {"insight_b": "A unique single-client idea."}),
    ])
    assert len(lib) == 1
    assert lib.insights[0].support_count == 2
    assert "unique single-client" not in " ".join(lib.statements()).lower()


def test_clusters_near_duplicates_via_llm():
    stub = DeterministicStubBackend(keyed={"Group the following": '{"clusters": [[0, 1]]}'}, default="{}")
    d = TraceDistiller(stub, quorum=2)
    lib = d.distill([
        _tr("c1", {"insight_x": "Use symmetry to simplify integrals."}),
        _tr("c2", {"insight_y": "Exploit symmetry when integrating."}),
    ])
    assert len(lib) == 1  # two phrasings merged into one cluster, 2 distinct clients -> promoted
    assert lib.insights[0].support_count == 2


def test_grows_then_stabilizes_with_prior():
    d = TraceDistiller(DeterministicStubBackend(default="{}"), quorum=1)
    r1 = d.distill([_tr("c1", {"insight_a": "Idea A."})])
    assert len(r1) == 1 and r1.version == 1
    r2 = d.distill([_tr("c2", {"insight_b": "Idea B."})], prior=r1)
    assert len(r2) == 2 and r2.version == 2  # grew -> version bumped
    r3 = d.distill([_tr("c1", {"insight_a": "Idea A."})], prior=r2)
    assert len(r3) == 2 and r3.version == 2  # unchanged content -> version HELD (not bumped)


def test_version_bumps_on_change_holds_on_no_change():
    d = TraceDistiller(DeterministicStubBackend(default="{}"), quorum=1)
    r1 = d.distill([_tr("c1", {"insight_a": "Alpha."})])
    r2 = d.distill([_tr("c2", {"insight_b": "Beta."})], prior=r1)
    assert r2.version == r1.version + 1  # content changed
    r3 = d.distill([_tr("c1", {"insight_a": "Alpha."})], prior=r2)
    assert r3.version == r2.version  # no new distinct insight -> held


def test_prior_insight_without_provenance_is_preserved():
    from fedlearn.fot.model import Insight, InsightLibrary

    d = TraceDistiller(DeterministicStubBackend(default="{}"), quorum=2)
    prior = InsightLibrary(
        insights=(Insight("i1", "Seeded authoritative lemma.", support_count=5, source_client_ids=()),),
        version=4,
    )
    lib = d.distill([], prior=prior)  # no new traces, no per-client provenance
    assert any(i.statement == "Seeded authoritative lemma." for i in lib.insights)
    assert lib.insights[0].support_count == 5  # support floor honored, not erased to 0
    assert lib.version == 4  # content unchanged -> version held


def test_max_insights_cap():
    d = TraceDistiller(DeterministicStubBackend(default="{}"), quorum=1, max_insights=2)
    traces = [_tr(f"c{i}", {f"insight_{i}": f"Idea number {i}."}) for i in range(5)]
    assert len(d.distill(traces)) == 2


def test_canonical_statement_is_deterministic_under_cluster_member_order():
    # Same clustering with members in different index order must yield the SAME canonical statement
    # and library sha256: the canonical (longest, ties broken lexicographically) must not depend on
    # the LLM's cluster-member ordering, or insight_id / the artifact sha256 silently flip run-to-run.
    def run(order_json):
        stub = DeterministicStubBackend(keyed={"Group the following": order_json}, default="{}")
        lib = TraceDistiller(stub, quorum=1).distill([
            ReasoningTrace("c1", "c1", "r", 0, "t", {"insight_a": "AAA equal len stmt X"}),
            ReasoningTrace("c2", "c2", "r", 0, "t", {"insight_b": "BBB equal len stmt Y"})])
        return lib.insights[0].statement, lib.sha256()
    s01, h01 = run('{"clusters": [[0, 1]]}')
    s10, h10 = run('{"clusters": [[1, 0]]}')
    assert s01 == s10  # canonical independent of member order
    assert h01 == h10  # -> stable insight_id / library sha256
