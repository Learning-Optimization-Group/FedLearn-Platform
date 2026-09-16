import os, sys, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import fl_server  # noqa: E402


def test_perplexity_from_avg_loss():
    assert math.isclose(fl_server.perplexity_from_loss(0.0), 1.0, rel_tol=1e-6)
    assert math.isclose(fl_server.perplexity_from_loss(1.0), math.e, rel_tol=1e-6)
    assert fl_server.perplexity_from_loss(1000.0) == float("inf")  # overflow-guarded
