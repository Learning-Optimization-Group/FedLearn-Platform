"""Tests for the FoT pre-egress leakage guard."""
from fedlearn.fot.redaction import LeakageScanner, TraceRedactor


def test_overlap_detects_verbatim_and_ignores_novel():
    corpus = ["The patient presented with acute chest pain and shortness of breath today."]
    sc = LeakageScanner(corpus, n=4)
    assert sc.overlap("acute chest pain and shortness of breath") > 0.5
    assert sc.overlap("Prefer a differential diagnosis before ordering imaging.") < 0.5


def test_overlap_zero_for_short_text_not_in_corpus():
    # Short text (< n tokens) that is NOT a verbatim run in the corpus -> 0.0 (no false positive).
    sc = LeakageScanner(["a b c d e"], n=4)
    assert sc.overlap("x y") == 0.0


def test_short_secret_is_not_waved_through():
    # A candidate shorter than n tokens must NOT fail open to 0.0 overlap; it falls back to a
    # verbatim-containment check so a short echoed secret is still caught.
    sc = LeakageScanner(["the secret code is alpha bravo charlie"], n=4)
    assert sc.overlap("secret code is") == 1.0  # 3 tokens, contiguous verbatim in corpus
    assert sc.overlap("unrelated short phrase") == 0.0
    red = TraceRedactor(sc, max_overlap=0.5)
    res = red.redact({"insight_leak": "secret code is", "insight_ok": "unrelated short phrase"})
    assert "insight_leak" in res.dropped
    assert "insight_ok" in res.kept


def test_redactor_drops_high_overlap_keeps_abstracted():
    corpus = ["compute the integral of x squared from zero to one exactly"]
    red = TraceRedactor(LeakageScanner(corpus, n=4), max_overlap=0.5)
    insights = {
        "insight_leak": "the integral of x squared from zero to one",
        "insight_ok": "Use symmetry to simplify definite integrals.",
    }
    result = red.redact(insights)
    assert "insight_ok" in result.kept
    assert "insight_leak" in result.dropped
    assert result.dropped["insight_leak"] > 0.5
