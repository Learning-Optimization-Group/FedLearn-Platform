"""A frozen arm's recorded backbone must be the backbone that actually ran.

THE DEFECT THIS PINS
--------------------
``_run_one`` declares ``backbone_name`` as a *named* parameter, so it never appears in ``**kw``.
The frozen branch then calls ``run_arm(..., **_accepted_kwargs(run_arm, kw))`` — and since
``backbone_name`` is not in ``kw``, ``run_arm`` silently fell back to its default ``"resnet18"``.

Consequences, both real rather than hypothetical:

1. **Wrong provenance.** A frozen sweep on ``timm:resnet50_gn.a1h_in1k`` recorded
   ``backbone_name="resnet18"`` in every cell's meta block.
2. **Silent data loss.** ``_emit_run`` builds the per-cell filename from that field, so cells from
   two different backbones collide. This campaign has already lost data to exactly this class of
   incident once — a resnet50 sweep clobbered the resnet18 arm-B shard-70 cells, and the surviving
   file was identifiable only because it reported ``feat_dim=2048``. The comment at the
   ``backbone_name`` assignment records that incident and was intended to prevent a recurrence, but
   it fixed the *callee* while the *caller* still dropped the value.

The full arms were unaffected: ``run_full_arm`` takes ``backbone_name`` explicitly.
"""
import pytest

torch = pytest.importorskip("torch")

import benchmarks.frozen_vs_finetune_xray as fx  # noqa: E402


def test_emit_run_separates_backbones_in_the_filename(tmp_path):
    """Two cells identical but for the backbone must not share a filename."""
    def cell(bb):
        return {"arm": "B", "final_auc": 0.9, "final_accuracy": 0.9,
                "meta": {"per_client": 10, "alpha": 1.0, "backbone_name": bb,
                         "norm": "batch", "rounds": 400, "seed": 0}}

    a = fx._emit_run(str(tmp_path), cell("resnet18"))
    b = fx._emit_run(str(tmp_path), cell("timm:resnet50_gn.a1h_in1k"))
    assert a != b, "cells from different backbones collided on one filename — one overwrote the other"


def test_frozen_arm_propagates_backbone_name_to_run_arm(monkeypatch):
    """The regression: `_run_one` must hand the real backbone to the frozen runner.

    Both collaborators are stubbed, so this asserts the wiring rather than re-running training.
    """
    seen = {}

    def fake_extract_features(data_dir, *, backbone, pretrained, img_size, device, backbone_seed):
        seen["extract_backbone"] = backbone
        n, d = 8, 2048
        return {"train_x": torch.zeros(n, d), "train_y": torch.zeros(n, dtype=torch.long),
                "test_x": torch.zeros(n, d), "test_y": torch.zeros(n, dtype=torch.long)}

    def fake_run_arm(arm, **kw):
        seen["run_arm_backbone"] = kw.get("backbone_name", "<NOT PASSED>")
        return {"arm": arm, "meta": {"backbone_name": kw.get("backbone_name", "<NOT PASSED>")}}

    import benchmarks.dp_on_head_xray as dp
    monkeypatch.setattr(dp, "extract_features", fake_extract_features)
    monkeypatch.setattr(fx, "run_arm", fake_run_arm)

    fx._run_one("B", data_dir="/nonexistent", backbone_name="timm:resnet50_gn.a1h_in1k",
                device="cpu", feature_cache={}, seed=0, rounds=1)

    assert seen["extract_backbone"] == "timm:resnet50_gn.a1h_in1k", (
        "feature extraction used the wrong backbone")
    assert seen["run_arm_backbone"] == "timm:resnet50_gn.a1h_in1k", (
        f"run_arm received {seen['run_arm_backbone']!r} instead of the backbone that actually ran — "
        f"the cell will be mislabelled and may overwrite another backbone's cell")


def test_run_arm_default_backbone_does_not_silently_stand_in(monkeypatch):
    """Guard the shape of the bug rather than one instance of it.

    Any future named parameter of `_run_one` that `run_arm` also declares can be dropped the same
    way. This asserts the specific field that names the output file.
    """
    def fake_extract_features(data_dir, **kw):
        n, d = 4, 512
        return {"train_x": torch.zeros(n, d), "train_y": torch.zeros(n, dtype=torch.long),
                "test_x": torch.zeros(n, d), "test_y": torch.zeros(n, dtype=torch.long)}

    captured = {}

    def fake_run_arm(arm, **kw):
        captured.update(kw)
        return {"arm": arm, "meta": {}}

    import benchmarks.dp_on_head_xray as dp
    monkeypatch.setattr(dp, "extract_features", fake_extract_features)
    monkeypatch.setattr(fx, "run_arm", fake_run_arm)

    fx._run_one("B", data_dir="/nonexistent", backbone_name="resnet34",
                device="cpu", feature_cache={}, seed=0, rounds=1)
    assert captured.get("backbone_name") == "resnet34"


def test_frozen_backbone_bytes_accepts_timm_and_differs_from_resnet18():
    """The one-shot delivery term must be measured on the backbone that is actually shipped.

    This bug was MASKED by the propagation bug above: because `backbone_name` never reached
    `run_arm`, `frozen_backbone_bytes` only ever saw torchvision names. Fixing the propagation
    exposed it. A timm frozen sweep was therefore charging itself resnet18's one-shot bytes.
    """
    pytest.importorskip("timm")
    r18 = fx.frozen_backbone_bytes("resnet18")
    gn50 = fx.frozen_backbone_bytes("timm:resnet50_gn.a1h_in1k")
    assert gn50 > r18, (
        f"resnet50_gn ({gn50:,} B) must be larger than resnet18 ({r18:,} B); equal values mean the "
        f"timm name silently fell back to a torchvision default")
