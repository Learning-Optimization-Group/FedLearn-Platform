"""The frozen-arm feature extractor must accept a ``timm:`` backbone, not only torchvision.

WHY THIS EXISTS
---------------
The campaign established that the measured "GroupNorm penalty" is the cost of *converting* a
BatchNorm-pretrained network into GroupNorm, not a property of GroupNorm — and that a genuinely
GN-*pretrained* backbone beats BatchNorm outright. torchvision ships no GN-pretrained model at any
depth, so that finding rests on timm (``resnet50_gn.a1h_in1k``).

``frozen_vs_finetune_xray.build_model`` already understands the ``timm:`` prefix, but only on the
FULL arms. The FROZEN arms route through ``dp_on_head_xray.extract_features`` →
``_build_backbone``, which was torchvision-only and raised ``ValueError`` on any name outside
``_FEAT_DIMS``. The consequence was that the one contrast the frozen-vs-full verdict names as its
open hole — *does a GN-pretrained backbone's full fine-tune beat its own frozen head?* — could not be
run at all, because the frozen half of it could not be built.

These tests pin the prefix contract on the extractor so the two arms stay buildable from the same
backbone spec.
"""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from benchmarks.dp_on_head_xray import _FEAT_DIMS, _build_backbone  # noqa: E402


def test_torchvision_backbones_still_build():
    """The timm path must not regress the existing torchvision path."""
    model, feat_dim = _build_backbone("resnet18", pretrained=False, seed=0)
    assert feat_dim == _FEAT_DIMS["resnet18"] == 512
    out = model(torch.zeros(1, 3, 64, 64))
    assert out.shape == (1, 512), f"expected pooled features, got {tuple(out.shape)}"


def test_unknown_torchvision_backbone_still_raises():
    """Adding a prefix must not turn a typo into a silent download attempt."""
    with pytest.raises(ValueError, match="unsupported backbone"):
        _build_backbone("resnet19", pretrained=False, seed=0)


def test_timm_prefix_is_accepted_and_returns_pooled_features():
    """The contract: a `timm:` name builds, and yields (module, feat_dim) like torchvision does.

    Run with pretrained=False so the test needs no network and no checkpoint download.
    """
    timm = pytest.importorskip("timm")
    del timm

    model, feat_dim = _build_backbone("timm:resnet50_gn.a1h_in1k", pretrained=False, seed=0)
    assert isinstance(feat_dim, int) and feat_dim > 0
    out = model(torch.zeros(1, 3, 64, 64))
    assert out.ndim == 2 and out.shape[0] == 1, (
        f"a feature extractor must emit [B, feat_dim]; got {tuple(out.shape)}")
    assert out.shape[1] == feat_dim, (
        f"declared feat_dim {feat_dim} disagrees with the actual output width {out.shape[1]} — the "
        f"head would be built at the wrong size")


def test_timm_backbone_is_groupnorm_and_frozen():
    """The whole point of reaching for timm is a GN-PRETRAINED network. Assert we got one.

    A silent fallback to a BatchNorm model would reintroduce the exact confound the experiment
    exists to remove, and would do so invisibly.
    """
    pytest.importorskip("timm")
    model, _ = _build_backbone("timm:resnet50_gn.a1h_in1k", pretrained=False, seed=0)
    kinds = {type(m).__name__ for m in model.modules()}
    assert "GroupNorm" in kinds, f"expected GroupNorm layers, found {sorted(kinds)[:12]}"
    assert "BatchNorm2d" not in kinds, "a GN-pretrained backbone must carry no BatchNorm layers"
    assert all(not p.requires_grad for p in model.parameters()), "the backbone must be frozen"


def test_cache_key_separates_timm_from_torchvision():
    """Both names must not collide in the feature cache.

    `resnet50` and `timm:resnet50_gn.a1h_in1k` produce different features; sharing a cache file
    would serve one experiment's features to the other and silently invalidate the comparison.
    """
    from benchmarks.dp_on_head_xray import _cache_key
    import os
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        for split in ("train", "test"):
            os.makedirs(os.path.join(d, split, "cls"), exist_ok=True)
        a = _cache_key(d, "resnet50", True, 224, None, 0)
        b = _cache_key(d, "timm:resnet50_gn.a1h_in1k", True, 224, None, 0)
    assert a != b, "distinct backbones must not share a feature-cache key"
