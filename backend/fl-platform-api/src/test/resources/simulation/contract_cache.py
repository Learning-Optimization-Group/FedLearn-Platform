"""Probe fixture for process-contract tests, not NIH data validation."""
import json


def load_nih_cache(path, *, expected_report_sha256):
    report = json.loads((path / "report.json").read_text())
    if report["pin"] != expected_report_sha256:
        raise ValueError("/private/invalid-cache")
    return {"provenance": report}
