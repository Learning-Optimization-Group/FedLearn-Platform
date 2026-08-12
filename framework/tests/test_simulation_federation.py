"""P0-1a / P0-1c / P0-1d: tests for the in-process federated simulator.

Written before the implementation (TDD).

``SimulatedFederation`` drives the *production* ``FLCoordinator`` and the *production*
strategies directly, with no gRPC, no port reservation and no OS processes. That is the whole
point: a simulated result must exercise the same aggregation code a deployed run does, or it
proves nothing about the platform.

What each group of tests pins down:

* **P0-1a — it is a real federation.** Rounds advance, the global model changes, clients are
  sampled by the server, and the loss actually falls. A simulator that runs 1000 clients but
  silently averages nothing is worse than no simulator.
* **P0-1c — no wall-clock dependence.** The coordinator resolves a full round inline on the
  last submit, and an *incomplete* round (dropout) is resolved deterministically rather than
  by waiting out a 120-second timeout. A simulator that inherits the timeout path takes
  33 hours to run 1000 dropout rounds.
* **P0-1d — the wire stays testable.** Routing clients through the real safetensors
  encode/decode must not change the result, which is what licenses running most experiments
  without it.
"""

import hashlib
import json
from collections import OrderedDict

import pytest
import torch
import torch.nn as nn

from fedlearn.server.strategy import FedAvg
from fedlearn.simulation.federation import SimulatedFederation
from fedlearn.simulation.partition import iid_partition


# --------------------------------------------------------------------------------------
# A deliberately tiny, fully deterministic learning problem
# --------------------------------------------------------------------------------------

FEATURES, CLASSES, N_SAMPLES = 4, 3, 240


def _make_dataset(seed: int = 0):
    """Linearly separable synthetic classification data, fixed across the whole module."""
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(FEATURES, CLASSES, generator=g)
    x = torch.randn(N_SAMPLES, FEATURES, generator=g)
    y = (x @ w).argmax(dim=1)
    return x, y


X, Y = _make_dataset()


class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(FEATURES, CLASSES)

    def forward(self, x):
        return self.fc(x)


def _initial_params() -> "OrderedDict[str, torch.Tensor]":
    torch.manual_seed(1234)
    return OrderedDict(
        (k, v.detach().clone()) for k, v in TinyNet().state_dict().items()
    )


class _ListLoader:
    """Minimal DataLoader stand-in: finite iterable of batches, exposing ``.dataset``.

    Kept explicit rather than using ``torch.utils.data.DataLoader`` so that batch order is a
    pure function of the index list and contributes no hidden randomness to the determinism
    tests.
    """

    def __init__(self, indices, batch_size=16):
        self.indices = list(indices)
        self.batch_size = batch_size
        self.dataset = self.indices

    def __iter__(self):
        for i in range(0, len(self.indices), self.batch_size):
            sel = self.indices[i:i + self.batch_size]
            yield X[sel], Y[sel]


def _make_factory(num_clients: int, seed: int = 0):
    """Build a client_factory over an IID partition of the shared dataset."""
    parts = iid_partition(N_SAMPLES, num_clients, seed=seed)

    def factory(client_id: int, rng):
        from fedlearn.client.local_trainer import LocalTrainer
        with_torch = rng.torch_seed()
        torch.manual_seed(with_torch)  # deterministic per-client init; scoped by the runner
        return LocalTrainer(model=TinyNet(), train_loader=_ListLoader(parts[client_id]))

    return factory


def _evaluate(_round, params) -> tuple:
    """Server-side evaluation on the full dataset — the loss curve the tests assert on."""
    net = TinyNet()
    net.load_state_dict(params)
    net.eval()
    with torch.no_grad():
        logits = net(X)
        loss = nn.functional.cross_entropy(logits, Y).item()
        acc = (logits.argmax(dim=1) == Y).float().mean().item()
    return loss, {"accuracy": acc}


def _federation(num_clients=8, clients_per_round=4, seed=0, **kw) -> SimulatedFederation:
    init = _initial_params()
    strategy = FedAvg(
        initial_parameters=init,
        evaluate_fn=_evaluate,
        min_fit_clients=1,
        clients_per_round=clients_per_round,
    )
    return SimulatedFederation(
        strategy=strategy,
        client_factory=_make_factory(num_clients),
        num_clients=num_clients,
        clients_per_round=clients_per_round,
        seed=seed,
        initial_parameters=init,
        client_config={"learning_rate": 0.5, "local_epochs": 2},
        **kw,
    )


# --------------------------------------------------------------------------------------
# P0-1a — it is a real federation
# --------------------------------------------------------------------------------------

class TestFederationRuns:
    def test_runs_the_requested_number_of_rounds(self):
        result = _federation().run(num_rounds=5)
        assert len(result.rounds) == 5
        assert [r.round for r in result.rounds] == [1, 2, 3, 4, 5]

    def test_server_samples_the_configured_cohort_each_round(self):
        result = _federation(num_clients=8, clients_per_round=3).run(num_rounds=4)
        for r in result.rounds:
            assert len(r.selected) == 3
            assert len(set(r.selected)) == 3, "a client must not be sampled twice in one round"
            assert all(0 <= c < 8 for c in r.selected)

    def test_the_global_model_actually_changes(self):
        init = _initial_params()
        result = _federation().run(num_rounds=3)
        changed = any(
            not torch.allclose(init[k], result.final_params[k])
            for k in init
        )
        assert changed, "the global model is unchanged — aggregation is a no-op"

    def test_loss_decreases(self):
        """The federation must learn. Guards against a simulator that runs but averages nothing."""
        result = _federation(num_clients=8, clients_per_round=8).run(num_rounds=15)
        losses = [r.loss for r in result.rounds if r.loss is not None]
        assert len(losses) == 15
        assert losses[-1] < losses[0], f"loss did not fall: {losses[0]:.4f} -> {losses[-1]:.4f}"

    def test_no_ports_or_grpc_channels_are_opened(self):
        """The simulator must not touch the network — that is what lifts the 11-port ceiling."""
        import socket

        created = []
        real_socket = socket.socket

        def spy(*a, **kw):
            created.append(a)
            return real_socket(*a, **kw)

        socket.socket = spy
        try:
            _federation().run(num_rounds=2)
        finally:
            socket.socket = real_socket
        assert created == [], f"simulator opened {len(created)} socket(s)"


# --------------------------------------------------------------------------------------
# Determinism — the property that makes results comparable
# --------------------------------------------------------------------------------------

class TestDeterminism:
    def test_same_seed_gives_a_bitwise_identical_model(self):
        a = _federation(seed=7).run(num_rounds=5)
        b = _federation(seed=7).run(num_rounds=5)
        assert a.final_digest == b.final_digest
        for k in a.final_params:
            assert torch.equal(a.final_params[k], b.final_params[k]), f"key {k} diverged"

    def test_different_seed_gives_a_different_run(self):
        a = _federation(seed=1).run(num_rounds=5)
        b = _federation(seed=2).run(num_rounds=5)
        # Client *selection* is seed-driven, so the trajectories must differ.
        assert [r.selected for r in a.rounds] != [r.selected for r in b.rounds]

    def test_client_selection_is_reproducible(self):
        a = _federation(seed=3).run(num_rounds=6)
        b = _federation(seed=3).run(num_rounds=6)
        assert [r.selected for r in a.rounds] == [r.selected for r in b.rounds]

    def test_per_round_loss_curve_is_reproducible(self):
        a = _federation(seed=11).run(num_rounds=4)
        b = _federation(seed=11).run(num_rounds=4)
        assert [r.loss for r in a.rounds] == [r.loss for r in b.rounds]

    def test_global_torch_rng_is_not_leaked(self):
        """``run()`` must not perturb the caller's global torch stream.

        The federation is built *before* the snapshot, because building it seeds torch
        globally on purpose (``_initial_params``) — folding construction into the measurement
        would test the fixture rather than the simulator.

        This is stricter than "client training is scoped". The strategy's ``evaluate_fn``
        is server-side user code, and this module's one constructs a ``TinyNet`` to load
        parameters into — an entirely ordinary thing to do that draws from global torch RNG
        on every round. Only a run-level scope closes that.
        """
        fed = _federation()

        torch.manual_seed(999)
        before = torch.rand(3)

        torch.manual_seed(999)
        fed.run(num_rounds=2)
        after = torch.rand(3)

        assert torch.equal(before, after), "run() leaked global torch RNG state"

    def test_a_run_is_unaffected_by_the_callers_global_torch_state(self):
        """The other direction of hermeticity: ambient global state must not steer a run.

        If it could, a result would depend on whatever the caller happened to do beforehand —
        so the same seed would not reproduce the same number in a different script.
        """
        fed_a = _federation(seed=31)
        fed_b = _federation(seed=31)

        torch.manual_seed(1)
        a = fed_a.run(num_rounds=3)
        torch.manual_seed(20260812)
        b = fed_b.run(num_rounds=3)

        assert a.final_digest == b.final_digest
        assert [r.loss for r in a.rounds] == [r.loss for r in b.rounds]


# --------------------------------------------------------------------------------------
# P0-1c — no wall-clock dependence
# --------------------------------------------------------------------------------------

class TestNoWallClockDependence:
    def test_a_full_round_never_enters_the_timeout_path(self):
        """The Nth submit aggregates inline, so the polling wait must never be reached.

        Asserted by giving the coordinator an absurdly small round timeout: if the simulator
        depended on the timeout path at all, this would force-aggregate and flag the round.
        """
        fed = _federation(round_timeout_s=0.001)
        result = fed.run(num_rounds=3)
        assert all(not r.forced for r in result.rounds)
        assert all(len(r.reported) == 4 for r in result.rounds)

    def test_dropout_resolves_deterministically_and_fast(self):
        """An incomplete round must resolve immediately, not wait out the dropout deadline."""
        import time

        fed = _federation(num_clients=10, clients_per_round=5, seed=4,
                          dropout_rate=0.4, round_timeout_s=120.0)
        t0 = time.monotonic()
        result = fed.run(num_rounds=4)
        elapsed = time.monotonic() - t0

        assert elapsed < 20.0, (
            f"dropout rounds took {elapsed:.1f}s — the simulator is waiting on the "
            f"120s wall-clock timeout instead of resolving inline"
        )
        assert any(r.dropped for r in result.rounds), "dropout_rate=0.4 produced no drops"
        for r in result.rounds:
            assert len(r.reported) + len(r.dropped) == 5
            if r.dropped:
                assert r.forced, "an incomplete round must be marked as force-resolved"

    def test_dropout_is_reproducible(self):
        a = _federation(num_clients=10, clients_per_round=5, seed=4, dropout_rate=0.4).run(3)
        b = _federation(num_clients=10, clients_per_round=5, seed=4, dropout_rate=0.4).run(3)
        assert [r.dropped for r in a.rounds] == [r.dropped for r in b.rounds]
        assert a.final_digest == b.final_digest

    def test_total_dropout_stops_the_run_and_records_why(self):
        """Every selected client dropping is a legitimate (if bad) outcome, not a deadlock.

        The coordinator's own policy is to stop rather than aggregate an empty cohort (which
        would produce a zero-key aggregate and wipe the global model). The simulator honours
        that policy rather than overriding it, so simulated dropout behaves as deployed
        dropout would — and says so in the record instead of silently truncating.
        """
        fed = _federation(num_clients=4, clients_per_round=2, seed=1, dropout_rate=1.0)
        result = fed.run(num_rounds=3)
        assert result.stopped_early is True
        assert result.stop_reason, "an early stop must record its reason"
        assert len(result.rounds) == 1, "the run must not continue past a dead round"
        assert result.rounds[0].reported == []
        assert result.rounds[0].forced is True


# --------------------------------------------------------------------------------------
# P0-1d — wire in the loop
# --------------------------------------------------------------------------------------

class TestWireInTheLoop:
    def test_full_wire_matches_no_wire_bitwise(self):
        """float32 safetensors round-trip is lossless, so the wire must not move the result.

        This is the test that licenses running the bulk of experiments with the wire off: if
        it ever fails, the codec has developed a precision or ordering bug and every
        wire-free result is suspect.
        """
        off = _federation(seed=21, wire_in_the_loop=0.0).run(num_rounds=4)
        on = _federation(seed=21, wire_in_the_loop=1.0).run(num_rounds=4)
        assert off.final_digest == on.final_digest

    def test_partial_wire_fraction_is_recorded_and_deterministic(self):
        a = _federation(seed=22, wire_in_the_loop=0.5).run(num_rounds=3)
        b = _federation(seed=22, wire_in_the_loop=0.5).run(num_rounds=3)
        assert a.final_digest == b.final_digest
        assert a.meta["wire_in_the_loop"] == 0.5
        assert sum(r.wire_clients for r in a.rounds) > 0, "no client was routed through the wire"

    def test_wire_bytes_are_accounted(self):
        result = _federation(seed=23, wire_in_the_loop=1.0).run(num_rounds=2)
        assert all(r.wire_bytes > 0 for r in result.rounds)


# --------------------------------------------------------------------------------------
# The record — a run that cannot be described cannot be published
# --------------------------------------------------------------------------------------

class TestResultRecord:
    def test_result_is_json_serializable_with_full_provenance(self):
        result = _federation(seed=5).run(num_rounds=3)
        blob = result.to_json()
        json.dumps(blob)  # must land in research/results/ unmodified

        meta = blob["meta"]
        for key in (
            "seed", "num_clients", "clients_per_round", "num_rounds", "strategy",
            "wire_in_the_loop", "dropout_rate", "torch_version", "platform",
            "wall_seconds", "final_digest",
        ):
            assert key in meta, f"meta is missing {key!r} — the run would not be reproducible"

        assert len(blob["per_round"]) == 3
        assert blob["per_round"][0]["round"] == 1

    def test_final_digest_is_the_canonical_safetensors_hash(self):
        """The digest must be the same function the wire uses, so it is cross-language checkable."""
        from fedlearn.communication.safetensors_codec import save_safetensors

        result = _federation(seed=6).run(num_rounds=2)
        blob = save_safetensors(
            [(k, v.detach().cpu().numpy()) for k, v in result.final_params.items()]
        )
        assert result.final_digest == hashlib.sha256(blob).hexdigest()


# --------------------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------------------

class TestValidation:
    def test_cohort_larger_than_the_federation_is_rejected(self):
        with pytest.raises(ValueError, match="clients_per_round"):
            _federation(num_clients=3, clients_per_round=5)

    @pytest.mark.parametrize("bad", [-0.1, 1.1])
    def test_out_of_range_fractions_are_rejected(self, bad):
        with pytest.raises(ValueError):
            _federation(wire_in_the_loop=bad)
        with pytest.raises(ValueError):
            _federation(dropout_rate=bad)


# --------------------------------------------------------------------------------------
# P0-1e — scale
# --------------------------------------------------------------------------------------

@pytest.mark.slow
def test_thousand_clients_completes():
    """The headline acceptance criterion: 1000 clients is expressible at all.

    Deselected by default (`-m "not slow"`); the full scale sweep lives in
    ``research/benchmarks/simulation_scale.py``.
    """
    fed = _federation(num_clients=1000, clients_per_round=50, seed=0)
    result = fed.run(num_rounds=5)
    assert len(result.rounds) == 5
    assert all(len(r.selected) == 50 for r in result.rounds)
