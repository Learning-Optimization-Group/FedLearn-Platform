"""Smoke-verify the three-regime communication-cost benchmark end to end.

Pins the comparative contract: (a) full-model FedAvg, (b) head-only frozen-backbone FedAvg, and
(c) DeComFL zeroth-order — every byte a REAL measurement from the production wire_bytes codecs.
"""
from benchmarks.comms_regimes import regime_bytes


def test_full_gt_head_gt_zero_for_a_large_backbone():
    # For a backbone-dominant model, the full-model payload dwarfs the head, and both are real bytes.
    r = regime_bytes([(8192, 1024, 10)], num_local_steps=10, num_perturbations=10)[-1]
    assert r["full_bytes"] > r["head_bytes"] > 0
    # DeComFL's per-round upload is a real, positive, and *tiny* payload...
    assert r["decomfl_bytes"] > 0
    assert r["decomfl_bytes"] < r["head_bytes"]          # even smaller than the frozen-backbone head
    assert r["decomfl_bytes"] < r["full_bytes"] / 100    # >100x smaller than the full first-order model


def test_decomfl_upload_is_independent_of_model_size():
    # Same (K, P) => byte-identical DeComFL upload regardless of the backbone dimension d.
    rows = regime_bytes(
        [(256, 128, 4), (4096, 512, 8), (8192, 2048, 10)],
        num_local_steps=10,
        num_perturbations=10,
    )
    assert len({r["decomfl_bytes"] for r in rows}) == 1
    # ...while the first-order payloads balloon with the backbone size.
    assert rows[-1]["full_bytes"] > rows[0]["full_bytes"] * 10
    assert rows[-1]["head_bytes"] > rows[0]["head_bytes"]


def test_decomfl_upload_scales_with_K_times_P_not_model_size():
    small_kp = regime_bytes([(4096, 512, 8)], num_local_steps=5, num_perturbations=5)[-1]
    big_kp = regime_bytes([(4096, 512, 8)], num_local_steps=10, num_perturbations=10)[-1]
    # Identical model, more (local steps x perturbations) => a bigger DeComFL upload.
    assert big_kp["decomfl_bytes"] > small_kp["decomfl_bytes"]
    # ...and the full/head first-order bytes are unchanged by (K, P): they depend only on the model.
    assert big_kp["full_bytes"] == small_kp["full_bytes"]
    assert big_kp["head_bytes"] == small_kp["head_bytes"]


def test_all_ratios_exceed_one_and_grow_with_model_size():
    rows = regime_bytes(
        [(256, 64, 4), (1024, 256, 4), (4096, 512, 8), (8192, 1024, 10)],
        num_local_steps=10,
        num_perturbations=10,
    )
    for r in rows:
        assert r["ratio_full_head"] > 1.0
        assert r["ratio_full_decomfl"] > 1.0
        assert r["ratio_head_decomfl"] > 1.0
    # A bigger frozen backbone is a bigger communication win on every comparison axis.
    for key in ("ratio_full_head", "ratio_full_decomfl", "ratio_head_decomfl"):
        vals = [r[key] for r in rows]
        assert all(b > a for a, b in zip(vals, vals[1:])), f"{key} not strictly growing: {vals}"


def test_decomfl_oneshot_download_is_the_full_model_reported_separately():
    # The honest DeComFL caveat: it still pays a ONE-SHOT O(d) initial model download (the full
    # model, once) — accounted separately from the O(K*P) per-round upload, never folded into it.
    r = regime_bytes([(4096, 512, 8)], num_local_steps=10, num_perturbations=10)[-1]
    assert r["decomfl_oneshot_download_bytes"] == r["full_bytes"]
    assert r["decomfl_oneshot_download_bytes"] > r["decomfl_bytes"] * 100
