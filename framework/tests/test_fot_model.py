"""Tests for FoT data models: ReasoningTrace, Insight, InsightLibrary."""
from fedlearn.fot.model import Insight, InsightLibrary, ReasoningTrace


def _trace(**kw):
    base = dict(
        trace_id="t1",
        client_id="c1",
        run_id="run1",
        round=0,
        task_id="task1",
        insights={"insight_x": "Do X when Y holds."},
    )
    base.update(kw)
    return ReasoningTrace(**base)


def test_trace_validate_ok():
    assert _trace().validate() == []


def test_trace_validate_flags_bad_key_and_empty_text():
    t = _trace(insights={"bad": "txt", "insight_ok": "   "})
    problems = t.validate()
    assert any("must start with" in p for p in problems)
    assert any("empty text" in p for p in problems)


def test_trace_validate_empty_insights():
    assert any("non-empty" in p for p in _trace(insights={}).validate())


def test_trace_validate_negative_round():
    assert any("round is negative" in p for p in _trace(round=-1).validate())


def test_trace_json_roundtrip():
    t = _trace(insights={"insight_a": "A.", "insight_b": "B."})
    assert ReasoningTrace.from_json(t.to_json()) == t


def test_library_json_roundtrip():
    lib = InsightLibrary(
        insights=(
            Insight("i1", "Always validate inputs.", 2, ("c1", "c2"), ("safety",)),
            Insight("i2", "Prefer X over Y.", 1, ("c1",)),
        ),
        version=3,
    )
    assert InsightLibrary.from_json(lib.to_json()) == lib
    assert len(lib) == 2
    assert lib.statements() == ["Always validate inputs.", "Prefer X over Y."]


def test_library_markdown_and_sha_stable():
    lib = InsightLibrary(insights=(Insight("i1", "Always validate inputs.", 2, ("c1", "c2")),), version=3)
    md = lib.render_markdown()
    assert "i1" in md and "Always validate inputs." in md and "v3" in md
    assert lib.sha256() == InsightLibrary.from_json(lib.to_json()).sha256()


def test_library_sha_changes_with_content():
    a = InsightLibrary(insights=(Insight("i1", "A"),))
    b = InsightLibrary(insights=(Insight("i1", "B"),))
    assert a.sha256() != b.sha256()
