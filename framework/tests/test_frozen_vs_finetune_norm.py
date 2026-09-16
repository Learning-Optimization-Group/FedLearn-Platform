"""Normalisation-layer selection for the full-fine-tune arms (C/D).

Why this exists: ExecuTorch's trainable export **rejects BatchNorm outright** —
``_native_batch_norm_legit_functional`` is not in the Core ATen opset, so a stock torchvision
ResNet-18 cannot be exported as a trainable graph at all. GroupNorm exports cleanly. That makes the
norm layer the difference between a full-fine-tune arm that can run on every client platform and one
that can only ever run on a server.

The substitution is not merely a runtime workaround. BatchNorm's running statistics are estimated
per-client under non-IID data and then averaged, which is a known failure mode in federated learning
(Hsieh et al. 2020, *The Non-IID Data Quagmire of Decentralized Machine Learning*); GroupNorm has no
running statistics at all. So the GN arm is the *methodologically* correct federated configuration
independently of ExecuTorch.

These tests pin the conversion. They deliberately do NOT assert anything about accuracy — that is
what the benchmark measures, and pinning it here would be circular.
"""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from benchmarks.frozen_vs_finetune_xray import (  # noqa: E402
    build_model,
    convert_bn_to_gn,
    run_full_arm,
)


def _count(module, cls):
    return sum(1 for m in module.modules() if isinstance(m, cls))


def test_conversion_removes_every_batchnorm():
    """A single surviving BatchNorm is enough to fail the trainable export, so this must be total."""
    import torchvision

    net = torchvision.models.resnet18(weights=None)
    assert _count(net, torch.nn.BatchNorm2d) > 0, "fixture must actually contain BatchNorm"

    convert_bn_to_gn(net)

    assert _count(net, torch.nn.BatchNorm2d) == 0
    assert _count(net, torch.nn.GroupNorm) > 0


def test_conversion_preserves_channel_count_at_every_site():
    """GroupNorm must normalise the same channels the BatchNorm did, or the graph is malformed."""
    import torchvision

    net = torchvision.models.resnet18(weights=None)
    before = [m.num_features for m in net.modules() if isinstance(m, torch.nn.BatchNorm2d)]

    convert_bn_to_gn(net)

    after = [m.num_channels for m in net.modules() if isinstance(m, torch.nn.GroupNorm)]
    assert before == after


def test_group_count_divides_channels_everywhere():
    """GroupNorm raises if num_channels % num_groups != 0. ResNet-18 has 64/128/256/512-channel
    stages, so a fixed group count is only safe if it divides all of them — assert the harness
    picked per-site rather than assuming."""
    import torchvision

    net = torchvision.models.resnet18(weights=None)
    convert_bn_to_gn(net)

    for m in net.modules():
        if isinstance(m, torch.nn.GroupNorm):
            assert m.num_channels % m.num_groups == 0
            assert m.num_groups >= 1


def test_conversion_leaves_conv_weights_untouched():
    """The whole point of arm C is a PRETRAINED backbone. Swapping the norm layers must not perturb
    the conv weights, or the arm silently stops being pretrained."""
    import torchvision

    net = torchvision.models.resnet18(weights=None)
    torch.manual_seed(0)
    before = {n: p.detach().clone() for n, p in net.named_parameters() if "conv" in n}

    convert_bn_to_gn(net)

    after = dict(net.named_parameters())
    for name, w in before.items():
        assert torch.equal(after[name], w), f"{name} changed"


def test_group_norm_model_has_no_running_statistics():
    """The federated hazard BatchNorm creates is its running_mean/running_var buffers, which get
    estimated on non-IID shards and then averaged. Assert they are genuinely gone, not just renamed."""
    import torchvision

    net = torchvision.models.resnet18(weights=None)
    convert_bn_to_gn(net)

    names = [n for n, _ in net.named_buffers()]
    assert not [n for n in names if "running_mean" in n or "running_var" in n]


def test_build_model_group_norm_yields_an_exportable_surface():
    """The harness-level entry point: arm C with norm='group' must contain no BatchNorm."""
    net = build_model("C", feat_dim=0, n_classes=2, norm="group", seed=0)
    assert _count(net, torch.nn.BatchNorm2d) == 0
    assert _count(net, torch.nn.GroupNorm) > 0
    assert all(p.requires_grad for p in net.parameters()), "arm C trains every parameter"


def test_build_model_defaults_to_batchnorm_unchanged():
    """Regression guard: the committed B-vs-C numbers were produced with BatchNorm. The default must
    not move, or adding this option silently invalidates the existing record."""
    net = build_model("C", feat_dim=0, n_classes=2, seed=0)
    assert _count(net, torch.nn.BatchNorm2d) > 0
    assert _count(net, torch.nn.GroupNorm) == 0


def test_frozen_arms_are_unaffected_by_the_norm_option():
    """Arms A/B consume pre-extracted features and train only a Linear head — there is no norm layer
    to swap, and passing the option must not change their trainable surface."""
    a = build_model("B", feat_dim=512, n_classes=2, seed=0)
    b = build_model("B", feat_dim=512, n_classes=2, norm="group", seed=0)
    assert isinstance(a, torch.nn.Linear) and isinstance(b, torch.nn.Linear)
    assert a.weight.shape == b.weight.shape


def test_unknown_norm_is_rejected_loudly():
    """A typo must not silently fall through to BatchNorm and produce an un-exportable arm that
    looks like it ran with GroupNorm."""
    with pytest.raises((ValueError, KeyError)):
        build_model("C", feat_dim=0, n_classes=2, norm="layer", seed=0)


def test_group_norm_arm_runs_end_to_end(tmp_path):
    """The regression this file exists for.

    ``build_model`` being norm-aware is not enough — every construction site inside the federated
    loop must pass the option through. The server model and each client model are built separately,
    and the client loads the server's ``state_dict`` STRICTLY. If a client is built with the default
    BatchNorm while the server is GroupNorm, the run dies on missing ``running_mean``/``running_var``
    keys. That is exactly what happened on the first launch of the GroupNorm sweep, several minutes
    into a multi-hour job.

    A unit test on ``build_model`` alone cannot catch it; only exercising the loop can.
    """
    pytest.importorskip("torchvision")
    from tests.test_frozen_vs_finetune_xray_smoke import _tiny_imagefolder

    root = _tiny_imagefolder(str(tmp_path / "ds_gn"))

    out = run_full_arm("C", data_dir=root, clients=2, clients_per_round=2, alpha=1.0,
                       rounds=1, local_epochs=1, img_size=32, batch_size=8, seed=0,
                       device="cpu", norm="group")

    assert out["meta"]["norm"] == "group", "the norm factor must be recorded in the run's provenance"
    assert out["meta"]["backbone_changed"] is True, "arm C must still move the backbone under GroupNorm"


def test_norm_is_recorded_so_batch_and_group_cells_cannot_overwrite_each_other(tmp_path):
    """``_emit_run`` names a cell from the factors that vary across the sweep. norm is now one of
    them, and a previous incident in this campaign lost cells to exactly this omission (alpha was
    missing from the name and a multi-alpha sweep silently overwrote itself)."""
    from benchmarks.frozen_vs_finetune_xray import _emit_run

    base = {"arm": "C", "final_auc": 0.9,
            "meta": {"per_client": 70, "alpha": 1.0, "backbone_name": "resnet18", "seed": 0}}
    bn = dict(base, meta=dict(base["meta"], norm="batch"))
    gn = dict(base, meta=dict(base["meta"], norm="group"))

    p_bn = _emit_run(str(tmp_path / "cells"), bn)
    p_gn = _emit_run(str(tmp_path / "cells"), gn)

    assert p_bn != p_gn, "batch and group cells must not collide on disk"
