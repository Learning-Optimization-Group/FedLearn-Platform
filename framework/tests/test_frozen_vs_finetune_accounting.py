"""Full communication + compute accounting for the frozen-vs-finetune arms.

The B-vs-C headline is a TRADE — accuracy against communication — so the communication side has to be
measured as carefully as the accuracy side. It was not. Three defects this file pins:

1. **Uplink only.** Both runners recorded ``cum_wire_bytes_up`` and nothing else. A federated round is
   bidirectional: the server broadcasts the model to every selected client before they train. Omitting
   it understates BOTH arms, and it understates them by different factors, so the *ratio* the paper
   quotes is affected too.

2. **The accumulator was wrong.** ``cum_wire_bytes_up = wire_bytes * len(updates) * rnd`` multiplies
   the CURRENT round's participant count by the round index, which is only correct if participation
   never varies. Clients with empty shards are skipped, so it does vary — the cumulative figure was
   silently wrong whenever it did.

3. **The frozen arm's one-shot cost was invisible.** Arm B's 4,276 B/round is real, but the client
   cannot compute features without the frozen backbone, which has to reach the device once. Reporting
   the per-round figure alone flatters the frozen design by hiding a 44.7 MB delivery. DeComFL's
   accounting already reports its one-shot download separately; this brings the frozen arm in line.

These assert structure and internal consistency, never a specific accuracy — that is the measurement.
"""
import pytest

torch = pytest.importorskip("torch")

from benchmarks.frozen_vs_finetune_xray import (  # noqa: E402
    frozen_backbone_bytes,
    round_wire_bytes,
    run_arm,
    run_full_arm,
)


def _separable(n=80, dim=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    y = torch.randint(0, 2, (n,), generator=g)
    x = torch.randn(n, dim, generator=g) + y.unsqueeze(1).float() * 2.0
    return x, y


def _run_frozen(rounds=3, clients=4):
    x, y = _separable()
    return run_arm("B", train_x=x, train_y=y, test_x=x, test_y=y,
                   clients=clients, clients_per_round=clients, alpha=1.0,
                   rounds=rounds, local_epochs=1, seed=0)


def test_every_round_reports_both_directions():
    """Uplink alone cannot express the cost of a round."""
    out = _run_frozen()

    for r in out["per_round"]:
        for key in ("bytes_up_round", "bytes_down_round", "cum_bytes_up",
                    "cum_bytes_down", "cum_bytes_total"):
            assert key in r, f"per-round record missing {key}"


def test_cumulative_totals_are_the_running_sum_of_the_per_round_figures():
    """The defect: multiplying the current participant count by the round index is not a sum."""
    out = _run_frozen()

    up = down = 0
    for r in out["per_round"]:
        up += r["bytes_up_round"]
        down += r["bytes_down_round"]
        assert r["cum_bytes_up"] == up
        assert r["cum_bytes_down"] == down
        assert r["cum_bytes_total"] == up + down


def test_cumulative_accounting_survives_varying_participation():
    """Clients with empty shards are skipped, so participation is not constant. With per_client set
    small and many clients, some rounds have fewer contributors — the running sum must still hold."""
    x, y = _separable(n=60)
    out = run_arm("B", train_x=x, train_y=y, test_x=x, test_y=y,
                  clients=10, clients_per_round=6, alpha=0.3, rounds=4, local_epochs=1, seed=3)

    counts = {r["participants"] for r in out["per_round"]}
    up = 0
    for r in out["per_round"]:
        up += r["bytes_up_round"]
        assert r["cum_bytes_up"] == up, f"broke at round {r['round']} (participants seen: {counts})"


def test_per_round_bytes_scale_with_the_number_of_participants():
    """Bytes are per client-round, so a round with n contributors moves n payloads each way."""
    out = _run_frozen()
    m = out["meta"]

    for r in out["per_round"]:
        assert r["bytes_up_round"] == m["wire_bytes_up_per_client_round"] * r["participants"]
        assert r["bytes_down_round"] == m["wire_bytes_down_per_client_round"] * r["participants"]


def test_meta_reports_both_directions_and_the_oneshot_cost():
    out = _run_frozen()
    m = out["meta"]

    assert m["wire_bytes_up_per_client_round"] > 0
    assert m["wire_bytes_down_per_client_round"] > 0
    assert "oneshot_backbone_download_bytes" in m


def test_the_frozen_arm_declares_a_oneshot_backbone_cost_and_the_full_arm_does_not(tmp_path):
    """Arm B's tiny per-round figure is only honest alongside the one-time backbone delivery. Arm C
    ships the whole model every round anyway, so it has no separate one-shot term."""
    pytest.importorskip("torchvision")
    from tests.test_frozen_vs_finetune_xray_smoke import _tiny_imagefolder

    frozen = _run_frozen()
    full = run_full_arm("C", data_dir=_tiny_imagefolder(str(tmp_path / "ds_acct")),
                        clients=2, clients_per_round=2, alpha=1.0, rounds=1, local_epochs=1,
                        img_size=32, batch_size=8, seed=0, device="cpu")

    assert frozen["meta"]["oneshot_backbone_download_bytes"] > 0
    assert full["meta"]["oneshot_backbone_download_bytes"] == 0


def test_frozen_backbone_bytes_is_measured_with_the_production_codec():
    """Not a parameter-count estimate — the same safetensors path the socket uses, so the one-shot
    figure is comparable to the per-round ones."""
    pytest.importorskip("torchvision")

    b = frozen_backbone_bytes("resnet18")

    assert b > 10_000_000, "a resnet18 backbone is tens of MB; this looks like a count, not bytes"
    net = torch.nn.Linear(4, 3)
    assert round_wire_bytes(net) > 0


def test_compute_cost_is_recorded_per_round_and_in_total(tmp_path):
    """The other half of the trade. Wall-clock per round plus peak memory — the axes that decide
    whether an arm can run on a client at all."""
    pytest.importorskip("torchvision")
    from tests.test_frozen_vs_finetune_xray_smoke import _tiny_imagefolder

    out = run_full_arm("C", data_dir=_tiny_imagefolder(str(tmp_path / "ds_cost")),
                       clients=2, clients_per_round=2, alpha=1.0, rounds=2, local_epochs=1,
                       img_size=32, batch_size=8, seed=0, device="cpu")

    assert all("round_sec" in r for r in out["per_round"])
    assert out["meta"]["total_sec"] > 0
    assert out["meta"]["peak_rss_mb"] > 0
    assert out["meta"]["trainable_params"] > 0


def test_both_arms_report_the_same_accounting_fields(tmp_path):
    """B-vs-C is only a valid comparison if both arms are measured on identical axes."""
    pytest.importorskip("torchvision")
    from tests.test_frozen_vs_finetune_xray_smoke import _tiny_imagefolder

    frozen = _run_frozen()
    full = run_full_arm("C", data_dir=_tiny_imagefolder(str(tmp_path / "ds_par")),
                        clients=2, clients_per_round=2, alpha=1.0, rounds=1, local_epochs=1,
                        img_size=32, batch_size=8, seed=0, device="cpu")

    for key in ("wire_bytes_up_per_client_round", "wire_bytes_down_per_client_round",
                "oneshot_backbone_download_bytes", "peak_rss_mb", "trainable_params"):
        assert key in frozen["meta"], f"frozen arm missing {key}"
        assert key in full["meta"], f"full arm missing {key}"

    common = {"round", "auc", "accuracy", "participants",
              "bytes_up_round", "bytes_down_round", "cum_bytes_up", "cum_bytes_down",
              "cum_bytes_total"}
    assert common <= set(frozen["per_round"][0])
    assert common <= set(full["per_round"][0])
