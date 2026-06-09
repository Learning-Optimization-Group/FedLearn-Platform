"""Tests for the FoT cross-client quorum ledger."""
from fedlearn.fot.provenance import InsightLedger


def test_quorum_promotes_multi_client_flags_single():
    led = InsightLedger(quorum=2)
    led.record("Validate inputs early.", "c1")
    led.record("validate inputs early", "c2")  # normalized duplicate, distinct client
    led.record("Single source idea.", "c1")
    promoted = led.promoted()
    assert len(promoted) == 1
    assert promoted[0].support_count == 2
    assert promoted[0].source_client_ids == ("c1", "c2")
    assert any(f.statement == "Single source idea." for f in led.flagged())


def test_same_client_twice_counts_once():
    led = InsightLedger(quorum=2)
    led.record("Idea.", "c1")
    led.record("idea", "c1")
    assert led.promoted() == []
    assert led.flagged()[0].support_count == 1


def test_quorum_one_promotes_everything():
    led = InsightLedger(quorum=1)
    led.record("A.", "c1")
    assert len(led.promoted()) == 1
