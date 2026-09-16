"""How the GroupNorm affine parameters are initialised when converting a pretrained BatchNorm model.

The 400-round matrix measured a BatchNorm->GroupNorm penalty of **0.0082 ± 0.0011 AUC**, sign-consistent
6/6 across both shard sizes — the only effect in the whole experiment that is robust everywhere
(`research/notes/frozen-backbone/2026-08-07-400round-shard10-norm-swap-verdict.md`). Its mechanism was
never tested. This is the test.

The hypothesis: the penalty is not GroupNorm, it is the *init*. ``convert_bn_to_gn`` keeps the pretrained
conv weights but constructs a fresh ``GroupNorm``, whose affine parameters default to gamma=1, beta=0 —
discarding whatever the pretrained BatchNorm had learned. On torchvision's ResNet-18 the pretrained
gamma averages **0.258** (sd 0.123), so resetting it to 1.0 applies a roughly 4x per-channel rescale on
top of weights tuned for the original scale.

BatchNorm and GroupNorm affine parameters have identical shape ``(C,)`` and identical semantics — a
per-channel scale and shift applied after normalisation. Only the statistics they normalise against
differ. So the learned pair can be carried across directly, which isolates the hypothesised mechanism
while holding architecture, conv weights, data and federation identical.

These tests pin the mechanics. They assert nothing about accuracy — that is what the benchmark measures.
"""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from benchmarks.frozen_vs_finetune_xray import (  # noqa: E402
    GN_INITS,
    build_model,
    convert_bn_to_gn,
)


def _first_bn(net):
    return next(m for m in net.modules() if isinstance(m, torch.nn.BatchNorm2d))


def _gn_list(net):
    return [m for m in net.modules() if isinstance(m, torch.nn.GroupNorm)]


def test_default_init_still_discards_the_pretrained_affine():
    """Regression guard: the committed C(group) results were produced with the default init, so it must
    not move. gamma=1, beta=0 everywhere."""
    import torchvision

    net = torchvision.models.resnet18(weights=None)
    convert_bn_to_gn(net)

    for g in _gn_list(net):
        assert torch.equal(g.weight, torch.ones_like(g.weight))
        assert torch.equal(g.bias, torch.zeros_like(g.bias))


def test_from_bn_init_carries_the_learned_scale_and_shift():
    """The whole point: gamma and beta come from the BatchNorm they replace, not from (1, 0)."""
    import torchvision

    net = torchvision.models.resnet18(weights="DEFAULT")
    bn_affine = [(m.weight.detach().clone(), m.bias.detach().clone())
                 for m in net.modules() if isinstance(m, torch.nn.BatchNorm2d)]

    convert_bn_to_gn(net, copy_affine=True)

    gn = _gn_list(net)
    assert len(gn) == len(bn_affine)
    for g, (w, b) in zip(gn, bn_affine):
        assert torch.equal(g.weight, w), "gamma must be the pretrained BatchNorm gamma"
        assert torch.equal(g.bias, b), "beta must be the pretrained BatchNorm beta"


def test_the_two_inits_actually_differ_on_a_pretrained_model():
    """Guards against a silently no-op option: on a PRETRAINED net the copied affine must not happen to
    equal the default. Pretrained ResNet-18 gamma averages ~0.26, so it does not."""
    import torchvision

    a = convert_bn_to_gn(torchvision.models.resnet18(weights="DEFAULT"))
    b = convert_bn_to_gn(torchvision.models.resnet18(weights="DEFAULT"), copy_affine=True)

    assert not torch.equal(_gn_list(a)[0].weight, _gn_list(b)[0].weight)


def test_copying_affine_leaves_conv_weights_untouched():
    """Only the norm init may change between the two arms, or the contrast is confounded."""
    import torchvision

    net = torchvision.models.resnet18(weights="DEFAULT")
    before = {n: p.detach().clone() for n, p in net.named_parameters() if "conv" in n}

    convert_bn_to_gn(net, copy_affine=True)

    after = dict(net.named_parameters())
    for name, w in before.items():
        assert torch.equal(after[name], w), f"{name} changed"


def test_copied_affine_is_a_copy_not_an_alias():
    """A view onto the dead BatchNorm's storage would make the two arms share state and would not
    survive the module being garbage collected."""
    import torchvision

    net = torchvision.models.resnet18(weights="DEFAULT")
    convert_bn_to_gn(net, copy_affine=True)

    g = _gn_list(net)[0]
    assert g.weight.requires_grad, "the affine must remain trainable"
    assert g.weight.is_leaf


def test_build_model_exposes_the_init_choice():
    for init in GN_INITS:
        net = build_model("C", feat_dim=0, n_classes=2, norm="group", gn_init=init, seed=0)
        assert not [m for m in net.modules() if isinstance(m, torch.nn.BatchNorm2d)]
        assert _gn_list(net)


def test_from_bn_only_differs_from_default_for_the_pretrained_arm():
    """Arm D is randomly initialised, so there is no learned affine to carry — the option must be a
    no-op there rather than silently doing something else."""
    d_def = build_model("D", feat_dim=0, n_classes=2, norm="group", gn_init="default", seed=0)
    d_bn = build_model("D", feat_dim=0, n_classes=2, norm="group", gn_init="from-bn", seed=0)

    assert len(_gn_list(d_def)) == len(_gn_list(d_bn))


def test_unknown_gn_init_is_rejected():
    with pytest.raises((ValueError, KeyError)):
        build_model("C", feat_dim=0, n_classes=2, norm="group", gn_init="xavier", seed=0)


def test_gn_init_reaches_the_cell_filename_and_meta():
    """Fourth instance of this class in this campaign, after alpha, backbone and round budget. A
    from-bn cell must not overwrite the default-init cell it is being compared against."""
    from benchmarks.frozen_vs_finetune_xray import _emit_run
    import tempfile

    base = {"arm": "C", "final_auc": 0.9,
            "meta": {"per_client": 10, "alpha": 1.0, "backbone_name": "resnet18",
                     "norm": "group", "rounds": 400, "seed": 0}}
    a = dict(base, meta=dict(base["meta"], gn_init="default"))
    b = dict(base, meta=dict(base["meta"], gn_init="from-bn"))

    with tempfile.TemporaryDirectory() as d:
        assert _emit_run(d, a) != _emit_run(d, b)


def test_batchnorm_arm_is_unaffected_by_gn_init():
    """--norm batch must ignore the option entirely."""
    a = build_model("C", feat_dim=0, n_classes=2, norm="batch", gn_init="default", seed=0)
    b = build_model("C", feat_dim=0, n_classes=2, norm="batch", gn_init="from-bn", seed=0)

    assert all(isinstance(m, torch.nn.BatchNorm2d) or not isinstance(m, torch.nn.GroupNorm)
               for m in a.modules())
    assert torch.equal(_first_bn(a).weight, _first_bn(b).weight)
