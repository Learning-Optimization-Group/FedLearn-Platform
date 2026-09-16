"""Loading a backbone from timm, so a GroupNorm-PRETRAINED model can be federated.

Why this is needed. The centralized screen
(`research/notes/frozen-backbone/2026-08-08-gn-penalty-EXPLAINED-it-is-the-conversion.md`) showed the
~0.008 AUC "GroupNorm penalty" is the cost of CONVERTING a BatchNorm-pretrained network, not a property
of GroupNorm: given weights actually trained for it, GroupNorm beat BatchNorm by +0.0077 (sign-consistent
3/3). That was measured centrally. Confirming it federated needs the harness to load
``timm/resnet50_gn.a1h_in1k`` — torchvision ships no GroupNorm-pretrained model at any depth.

The contract: a ``timm:`` prefix on ``--backbone`` routes to timm; anything else stays on torchvision so
every committed result reproduces unchanged.
"""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from benchmarks.frozen_vs_finetune_xray import build_model  # noqa: E402


def _census(net):
    bn = sum(1 for m in net.modules() if isinstance(m, torch.nn.BatchNorm2d))
    gn = sum(1 for m in net.modules() if isinstance(m, torch.nn.GroupNorm))
    return bn, gn


def test_torchvision_backbones_are_unchanged():
    """Regression guard: every committed result used torchvision. The default path must not move."""
    net = build_model("C", feat_dim=0, n_classes=2, backbone_name="resnet18", seed=0)
    bn, gn = _census(net)
    assert bn > 0 and gn == 0


def test_timm_prefix_loads_a_timm_model():
    timm = pytest.importorskip("timm")  # noqa: F841
    net = build_model("C", feat_dim=0, n_classes=2,
                      backbone_name="timm:resnet50.a1h_in1k", seed=0)
    bn, gn = _census(net)
    assert bn > 0 and gn == 0
    assert sum(p.numel() for p in net.parameters()) > 20_000_000, "resnet50-scale"


def test_timm_groupnorm_pretrained_arrives_with_groupnorm_intact():
    """The whole point: this model must reach the federation as GroupNorm, NOT be converted."""
    pytest.importorskip("timm")
    net = build_model("C", feat_dim=0, n_classes=2,
                      backbone_name="timm:resnet50_gn.a1h_in1k", seed=0)
    bn, gn = _census(net)
    assert bn == 0, "a GN-pretrained model must contain no BatchNorm"
    assert gn > 0


def test_a_gn_pretrained_backbone_is_not_re_converted():
    """--norm group on an already-GroupNorm model must be a no-op, not a destructive second pass that
    replaces the PRETRAINED GroupNorm layers with freshly initialised ones — which would silently
    reintroduce the exact defect this experiment exists to avoid."""
    pytest.importorskip("timm")
    a = build_model("C", feat_dim=0, n_classes=2,
                    backbone_name="timm:resnet50_gn.a1h_in1k", norm="batch", seed=0)
    b = build_model("C", feat_dim=0, n_classes=2,
                    backbone_name="timm:resnet50_gn.a1h_in1k", norm="group", seed=0)

    ga = next(m for m in a.modules() if isinstance(m, torch.nn.GroupNorm))
    gb = next(m for m in b.modules() if isinstance(m, torch.nn.GroupNorm))
    assert torch.equal(ga.weight, gb.weight), "pretrained GroupNorm affine must survive --norm group"
    assert not torch.equal(gb.weight, torch.ones_like(gb.weight)), "affine must not be reset to 1.0"


def test_timm_and_torchvision_agree_on_the_classifier_head():
    """Both paths must expose n_classes outputs, or the arms are not comparable."""
    pytest.importorskip("timm")
    tv = build_model("C", feat_dim=0, n_classes=2, backbone_name="resnet50", seed=0)
    tm = build_model("C", feat_dim=0, n_classes=2, backbone_name="timm:resnet50.a1h_in1k", seed=0)

    x = torch.randn(2, 3, 64, 64)
    assert tv(x).shape == tm(x).shape == (2, 2)


def test_every_parameter_is_trainable_on_the_timm_path():
    """Arm C federates the full model; a frozen timm parameter would silently shrink the wire."""
    pytest.importorskip("timm")
    net = build_model("C", feat_dim=0, n_classes=2,
                      backbone_name="timm:resnet50_gn.a1h_in1k", seed=0)
    assert all(p.requires_grad for p in net.parameters())


def test_unknown_timm_model_fails_loudly():
    """A typo must not fall back to a torchvision model and silently run the wrong architecture."""
    pytest.importorskip("timm")
    with pytest.raises(Exception):
        build_model("C", feat_dim=0, n_classes=2,
                    backbone_name="timm:definitely_not_a_model_xyz", seed=0)
