"""FR-15: a dropped DeComFL submission must not cause the client to double-apply a round.

Root cause (audited 2026-07-17): the server advances ``client_last_round`` only when a client's
submission is *aggregated* (``decomfl_strategy.aggregate_fit``), but the client mutates its local
model at config-fetch time (``rebuild_model``). Whenever a submission is dropped after the client
already fetched+applied that round's history — the normal straggler path when
``clients_per_round < live clients``, plus non-finite drops and restarts — the next
``get_rebuild_history`` re-hands an already-applied round and the client replays it on top of a
model that already includes it. The client permanently diverges and then poisons every aggregate.

The contract: rebuilding is idempotent — replaying an already-applied round is a no-op — so the
client's trajectory equals a clean single-application of each round regardless of overlap.
"""
from collections import OrderedDict

import torch
import torch.nn as nn

from fedlearn.server.decomfl_strategy import DeComFL
from fedlearn.client.decomfl_client import DeComFLClient


class TinyNet(nn.Module):
    """Linear(3, 1) -> fc.weight [1,3] + fc.bias [1] (d=4)."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(3, 1)

    def forward(self, x):  # noqa: D401
        return self.fc(x)


def _make_strategy(model: nn.Module, K: int, P: int, eta: float) -> DeComFL:
    init = OrderedDict((k, v.clone()) for k, v in model.state_dict().items())
    return DeComFL(
        initial_parameters=init,
        evaluate_fn=None,
        min_fit_clients=1,
        clients_per_round=2,
        num_local_steps=K,
        num_perturbations=P,
        learning_rate=eta,
        smoothing_param=0.001,
        seed=123,
    )


def test_dropped_submission_does_not_double_apply_rebuild():
    """Client C fetches round r-1's history, is dropped in round r, then re-fetches at r+1.

    The server, having never aggregated C, re-hands round r-1 alongside round r (overlapping
    range). The client must NOT apply round r-1 twice.
    """
    torch.manual_seed(0)
    model = TinyNet()
    # Snapshot the pristine x0 BEFORE any rebuild — rebuild_model writes x_current back into
    # self.model, so the reference client must be built from this snapshot, not a post-mutation one.
    x0_state = OrderedDict((k, v.detach().clone()) for k, v in model.state_dict().items())
    K, P, eta = 1, 1, 0.05
    strat = _make_strategy(model, K, P, eta)

    # Populate the server's per-round seed + averaged-gradient history for rounds 1 and 2.
    for r in (1, 2):
        strat.seed_history[r] = strat.generate_seeds(r)
        strat.gradient_history[r] = [[0.1 * r + 0.01 * p for p in range(P)] for _ in range(K)]

    # Client C was last aggregated in round 1 -> the server records it synced through round 0.
    strat.client_last_round["C"] = 0

    client = DeComFLClient(model=model, train_loader=None, device="cpu")

    # Round 2: C fetches and applies its rebuild history (just round 1), trains, then is DROPPED
    # (straggler). Because C is not aggregated, the server never advances client_last_round["C"].
    hist_r2 = strat.get_rebuild_history("C", current_round=2)
    assert [h["round_number"] for h in hist_r2] == [1]
    client.rebuild_model(hist_r2, learning_rate=eta)

    # Round 3: the server advanced (other clients completed round 2). C re-fetches: the server
    # still thinks C is synced through 0, so it hands rounds 1 AND 2 — round 1 overlaps.
    hist_r3 = strat.get_rebuild_history("C", current_round=3)
    assert [h["round_number"] for h in hist_r3] == [1, 2]
    client.rebuild_model(hist_r3, learning_rate=eta)

    got = client.x_current.detach().cpu().clone()

    # Reference: a clean client at the same pristine x0 that applies rounds 1 and 2 exactly once.
    ref_model = TinyNet()
    ref_model.load_state_dict(x0_state)
    ref_client = DeComFLClient(model=ref_model, train_loader=None, device="cpu")
    ref_client.rebuild_model(
        [
            {"round_number": 1, "seeds": strat.seed_history[1], "gradients": strat.gradient_history[1]},
            {"round_number": 2, "seeds": strat.seed_history[2], "gradients": strat.gradient_history[2]},
        ],
        learning_rate=eta,
    )
    ref = ref_client.x_current.detach().cpu().clone()

    assert torch.allclose(got, ref, atol=1e-6), (
        f"dropped-then-rejoined client double-applied a round: got {got.tolist()} "
        f"!= single-application reference {ref.tolist()}"
    )


def test_restart_does_not_double_apply_already_downloaded_rounds():
    """FR-16: a restarted client that re-downloads the CURRENT global must not replay history.

    On restart the client reuses its deterministic client_id and re-downloads x_{r-1}, but the
    server still remembers client_last_round from before the crash and hands every round since —
    which the old client replayed ON TOP of the already-current model, double-applying. Adopting
    the global that corresponds to round r means the client is synced through r-1, so the
    server's (stale) re-handed rounds q..r-1 are a no-op.
    """
    torch.manual_seed(0)
    model = TinyNet()
    strat = _make_strategy(model, K=1, P=1, eta=0.05)

    # The server ran rounds 3 and 4 with other clients after this client crashed following round 3.
    for r in (3, 4):
        strat.seed_history[r] = strat.generate_seeds(r)
        strat.gradient_history[r] = [[0.1 * r]]
    strat.client_last_round["C"] = 2  # last recorded baseline before the crash (synced through 2)

    # Restart: a fresh client object re-downloads the current global (server current_round == 5, so
    # the global is x_4 -> synced through 4). We stand in the model's state_dict for that download.
    downloaded_global = OrderedDict((k, v.detach().clone()) for k, v in model.state_dict().items())
    restarted = DeComFLClient(model=TinyNet(), train_loader=None, device="cpu")
    restarted.load_global_model(downloaded_global, synced_through_round=4)
    x_after_download = restarted.x_current.detach().clone()

    # First config fetch at round 5: the server still thinks C is synced through 2, so it re-hands
    # rounds 3 and 4 — which the client already holds via the download.
    hist = strat.get_rebuild_history("C", current_round=5)
    assert [h["round_number"] for h in hist] == [3, 4]
    restarted.rebuild_model(hist, learning_rate=0.05)

    assert torch.allclose(restarted.x_current, x_after_download, atol=1e-6), (
        "restarted client double-applied rounds it already downloaded: "
        f"{restarted.x_current.tolist()} != {x_after_download.tolist()}"
    )
