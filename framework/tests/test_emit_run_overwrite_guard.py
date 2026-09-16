"""A cell must never silently overwrite a different cell.

Commit `21699bc` fixed the *second* instance of one bug class:

* a multi-alpha sweep overwrote its own per-cell files, because ``alpha`` was missing from the
  filename (fixed by adding ``alpha``);
* frozen cells were all stamped ``resnet18``, so two backbones collided (fixed by propagating
  ``backbone_name``) — "this campaign has already lost arm-B cells to exactly that".

Both fixes enumerate a field. Enumeration cannot prevent the third instance: the next sweep axis
added to ``run_arm``'s meta and forgotten in ``_emit_run``'s name reproduces the failure exactly,
and does so *silently* — a shorter sweep looks complete because the file count is right, while the
surviving file holds whichever cell happened to run last.

So this pins the property instead of another field: **if two runs differ anywhere in their meta,
they must not land in one file.** That holds for every axis, including axes that do not exist yet.
The guard is at the write point because that is the only place with both the incoming cell and the
one already on disk.

A re-run of the *same* cell must still overwrite — resuming an interrupted sweep depends on it.
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from benchmarks import frozen_vs_finetune_xray as fx  # noqa: E402


def _cell(**meta):
    m = {"per_client": 10, "alpha": 1.0, "backbone_name": "resnet18",
         "norm": "batch", "rounds": 30, "seed": 0}
    m.update(meta)
    return {"arm": meta.pop("arm", "B"), "meta": m, "final_auc": 0.8}


class TestEmitRunRefusesSilentOverwrite:
    def test_a_differing_cell_cannot_overwrite_an_existing_one(self, tmp_path):
        """THE regression, stated generally: any meta difference must not share a file."""
        fx._emit_run(str(tmp_path), _cell(backbone_name="resnet18"))
        # Simulate the bug: a cell whose meta differs but whose NAME collides, which is what a
        # forgotten axis produces. Force the collision by making the differing field one the
        # filename does not use.
        other = _cell(backbone_name="resnet18")
        other["meta"]["lr"] = 0.05          # a NEW axis, absent from the filename template
        with pytest.raises(ValueError, match="would overwrite"):
            fx._emit_run(str(tmp_path), other)

    def test_the_error_identifies_the_field_that_differs(self, tmp_path):
        """A bare 'collision' tells you nothing; the fix is to add THAT field to the name."""
        fx._emit_run(str(tmp_path), _cell())
        other = _cell()
        other["meta"]["lr"] = 0.05
        with pytest.raises(ValueError) as exc:
            fx._emit_run(str(tmp_path), other)
        assert "lr" in str(exc.value), "the guard must name the axis missing from the filename"

    def test_the_original_cell_survives_a_refused_write(self, tmp_path):
        """Refusing must not corrupt what is already there — the whole point is not losing cells."""
        p = fx._emit_run(str(tmp_path), _cell(backbone_name="resnet18"))
        other = _cell(backbone_name="resnet18")
        other["meta"]["lr"] = 0.05
        with pytest.raises(ValueError):
            fx._emit_run(str(tmp_path), other)
        assert json.load(open(p))["meta"].get("lr") is None, "the refused write still clobbered it"

    def test_rerunning_the_identical_cell_still_overwrites(self, tmp_path):
        """Resuming an interrupted sweep re-runs cells; that must stay idempotent, not raise."""
        fx._emit_run(str(tmp_path), _cell())
        fx._emit_run(str(tmp_path), _cell())          # must not raise

    def test_cells_differing_in_a_named_axis_are_unaffected(self, tmp_path):
        """Axes already in the filename separate normally — the guard adds no false positives."""
        a = fx._emit_run(str(tmp_path), _cell(backbone_name="resnet18"))
        b = fx._emit_run(str(tmp_path), _cell(backbone_name="timm:resnet50_gn.a1h_in1k"))
        c = fx._emit_run(str(tmp_path), _cell(seed=1))
        assert len({a, b, c}) == 3

    def test_a_measured_field_differing_does_not_block_a_rerun(self, tmp_path):
        """The guard's most dangerous failure mode is a FALSE positive.

        ``meta`` mixes configuration with measurements, and the measured ones — ``total_sec``,
        ``peak_rss_mb`` — differ on literally every re-run. Comparing meta wholesale would raise
        every time an interrupted sweep resumed, halting hours of compute over a timing jitter.
        That is worse than the silent overwrite it is meant to prevent, so only *configuration*
        may be compared.
        """
        c = _cell()
        c["meta"]["total_sec"] = 812.4
        c["meta"]["peak_rss_mb"] = 1904.0
        fx._emit_run(str(tmp_path), c)

        again = _cell()
        again["meta"]["total_sec"] = 799.1      # same config, re-run: different wall clock
        again["meta"]["peak_rss_mb"] = 1888.5
        fx._emit_run(str(tmp_path), again)      # must not raise

    def test_a_result_difference_alone_does_not_block_a_rerun(self, tmp_path):
        """Same configuration, different measured value (nondeterminism, longer run) is a legitimate
        overwrite. Only *configuration* differences indicate a missing filename axis."""
        fx._emit_run(str(tmp_path), _cell())
        again = _cell()
        again["final_auc"] = 0.9123
        fx._emit_run(str(tmp_path), again)            # must not raise
