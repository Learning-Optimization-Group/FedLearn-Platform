"""DA-14 Ph3.0/Ph3.3: smoke-verify the frozen-backbone head-only FL benchmark end to end."""
from benchmarks.frozen_backbone_fl import head_wire_win, run_head_federation


def test_head_only_wire_win_is_real_and_grows_with_backbone_size():
    wins = head_wire_win([(64, 32, 3), (512, 256, 3)])
    # The head is genuinely smaller than the full model (real safetensors bytes)...
    assert all(w["head_bytes"] < w["full_bytes"] for w in wins)
    assert all(w["ratio"] > 1.0 for w in wins)
    # ...and freezing a bigger backbone is a bigger communication win.
    assert wins[-1]["ratio"] > wins[0]["ratio"]


def test_head_federation_learns_and_preserves_the_frozen_backbone():
    r = run_head_federation(rounds=12, clients=3, seed=0)
    assert r["final_acc"] > r["initial_acc"] + 0.15   # the federated head actually learns
    assert r["final_acc"] > 0.6
    assert r["backbone_unchanged"] is True            # frozen backbone never moves across rounds
    assert r["wire_is_head_only"] is True             # only the head on the wire, every round
    assert len(r["per_round_acc"]) == 12
