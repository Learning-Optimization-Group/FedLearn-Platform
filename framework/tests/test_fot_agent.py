"""Tests for the FoT client reasoning agent (solve -> reflect -> extract -> redact)."""
from fedlearn.fot.agent import ReasoningAgent, Task
from fedlearn.fot.backend import DeterministicStubBackend
from fedlearn.fot.redaction import LeakageScanner, TraceRedactor


def test_emits_abstracted_trace_with_coerced_keys():
    stub = DeterministicStubBackend(scripted=[
        "solution text",
        "reflection: exploited symmetry",
        '{"symmetryTrick": "Exploit symmetry to simplify integrals."}',
    ])
    tr = ReasoningAgent(stub, "c1", "run").run(Task("task1", "compute integral of x^2"), round_index=2)
    assert tr.client_id == "c1" and tr.round == 2 and tr.task_id == "task1"
    assert tr.trace_id == "c1-r2-task1"
    assert list(tr.insights.keys()) == ["insight_symmetrytrick"]
    assert tr.validate() == []


def test_parses_insights_wrapper_object():
    stub = DeterministicStubBackend(scripted=["s", "r", '{"insights": {"insight_a": "Reusable idea."}}'])
    tr = ReasoningAgent(stub, "c1", "run").run(Task("t", "p"), round_index=0)
    assert tr.insights == {"insight_a": "Reusable idea."}


def test_redacts_verbatim_leak_keeps_abstracted():
    raw = "the secret patient identifier is 12345 and the diagnosis is extremely rare"
    redactor = TraceRedactor(LeakageScanner([raw], n=4), max_overlap=0.4)
    stub = DeterministicStubBackend(scripted=[
        "s", "r",
        '{"insight_leak": "the secret patient identifier is 12345 and the diagnosis", '
        '"insight_ok": "Anonymize identifiers before reasoning."}',
    ])
    tr = ReasoningAgent(stub, "c1", "run", redactor=redactor).run(Task("t", raw), round_index=0)
    assert "insight_ok" in tr.insights
    assert "insight_leak" not in tr.insights


def test_bad_json_yields_empty_insights():
    stub = DeterministicStubBackend(scripted=["s", "r", "not json at all"])
    tr = ReasoningAgent(stub, "c1", "run").run(Task("t", "p"), round_index=0)
    assert tr.insights == {}
