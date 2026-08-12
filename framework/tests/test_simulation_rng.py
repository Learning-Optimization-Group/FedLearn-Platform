"""P0-1b: tests for isolated per-client RNG streams.

Written before the implementation (TDD).

Determinism in a federated simulation is harder than "call ``manual_seed`` once", because a
single global stream couples every client to every other one: client 7's minibatch order then
depends on how many clients ran before it, so changing ``clients_per_round`` — or dropping one
client — silently changes the trajectory of all the rest. A comparison across cohort sizes is
then not a controlled experiment.

The property these tests enforce, and the one the whole design turns on:

    **A client's stream is a pure function of (run_seed, client_id).**

It does not depend on the cohort size, on how many clients were constructed before it, on the
round, or on whether any other client was sampled. That is what makes "1000 clients" and
"10 clients" comparable, and what makes a dropped client a genuinely local perturbation.
"""

import numpy as np
import pytest
import torch

from fedlearn.simulation.rng import ClientRng, RunRng, torch_rng_scope


class TestClientStreamIsolation:
    def test_same_identity_gives_the_same_stream(self):
        a = ClientRng(run_seed=1234, client_id=7).numpy.random(5)
        b = ClientRng(run_seed=1234, client_id=7).numpy.random(5)
        assert np.array_equal(a, b)

    def test_different_client_ids_give_different_streams(self):
        a = ClientRng(run_seed=1234, client_id=7).numpy.random(5)
        b = ClientRng(run_seed=1234, client_id=8).numpy.random(5)
        assert not np.array_equal(a, b)

    def test_different_run_seeds_give_different_streams(self):
        a = ClientRng(run_seed=1, client_id=7).numpy.random(5)
        b = ClientRng(run_seed=2, client_id=7).numpy.random(5)
        assert not np.array_equal(a, b)

    def test_stream_is_independent_of_cohort_size(self):
        """The load-bearing property: client 5 is unaffected by how many peers exist.

        A ``SeedSequence.spawn()``-based design fails this test, because spawning N children
        makes each child's entropy depend on N. That failure mode is invisible until someone
        compares a 10-client run to a 1000-client run and the 10-client numbers no longer
        reproduce.
        """
        small = RunRng(seed=99).client(5).numpy.random(4)
        large = RunRng(seed=99).client(5).numpy.random(4)

        # Construct many peers first — order and count must not matter.
        run = RunRng(seed=99)
        for cid in range(1000):
            if cid != 5:
                run.client(cid).numpy.random(3)
        after_peers = RunRng(seed=99).client(5).numpy.random(4)

        assert np.array_equal(small, large)
        assert np.array_equal(small, after_peers)

    def test_stream_is_independent_of_construction_order(self):
        forward = [RunRng(seed=7).client(c).numpy.random(2) for c in range(5)]
        run = RunRng(seed=7)
        backward = {c: run.client(c).numpy.random(2) for c in reversed(range(5))}
        for c in range(5):
            assert np.array_equal(forward[c], backward[c])


class TestRoundScopedStreams:
    def test_rounds_are_distinguishable(self):
        c = ClientRng(run_seed=5, client_id=3)
        r1 = c.for_round(1).numpy.random(4)
        r2 = c.for_round(2).numpy.random(4)
        assert not np.array_equal(r1, r2)

    def test_round_streams_are_reproducible_out_of_order(self):
        """Round 5 must be reproducible without having run rounds 1-4 first.

        This is what lets a failed run be resumed, or a single anomalous round be re-examined,
        without re-running everything before it.
        """
        direct = ClientRng(run_seed=5, client_id=3).for_round(5).numpy.random(4)
        c = ClientRng(run_seed=5, client_id=3)
        for r in range(1, 5):
            c.for_round(r).numpy.random(9)
        after = c.for_round(5).numpy.random(4)
        assert np.array_equal(direct, after)


class TestTorchIntegration:
    def test_torch_generator_is_deterministic_and_isolated(self):
        g1 = ClientRng(run_seed=11, client_id=2).torch_generator()
        g2 = ClientRng(run_seed=11, client_id=2).torch_generator()
        g3 = ClientRng(run_seed=11, client_id=3).torch_generator()

        a = torch.rand(4, generator=g1)
        b = torch.rand(4, generator=g2)
        c = torch.rand(4, generator=g3)

        assert torch.equal(a, b)
        assert not torch.equal(a, c)

    def test_torch_rng_scope_restores_global_state(self):
        """Seeding torch globally inside a client must not leak out of that client.

        Without restoration, running client A changes what client B draws, which re-couples
        exactly the streams this module exists to separate.
        """
        torch.manual_seed(1000)
        before = torch.rand(3)

        torch.manual_seed(1000)
        with torch_rng_scope(seed=4242):
            torch.rand(3)  # perturb global state inside the scope
        after = torch.rand(3)

        assert torch.equal(before, after), "global torch RNG state was not restored"

    def test_torch_rng_scope_is_deterministic(self):
        with torch_rng_scope(seed=77):
            a = torch.rand(4)
        with torch_rng_scope(seed=77):
            b = torch.rand(4)
        assert torch.equal(a, b)


class TestServerStream:
    """The server draws too — cohort selection, dropout, wire routing.

    Its stream was originally keyed on a ``client_id`` of -1, which numpy's ``SeedSequence``
    rejects outright ("expected non-negative integer") from deep inside Cython. These tests
    cover the server stream directly so that gap cannot reopen silently.
    """

    def test_server_stream_works_and_is_deterministic(self):
        a = RunRng(seed=3).server_rng().random(4)
        b = RunRng(seed=3).server_rng().random(4)
        assert np.array_equal(a, b)

    def test_server_stream_is_round_scoped_and_order_free(self):
        r5 = RunRng(seed=3).server_rng(round_num=5).random(4)
        run = RunRng(seed=3)
        for r in range(1, 5):
            run.server_rng(round_num=r).random(7)
        again = run.server_rng(round_num=5).random(4)
        assert np.array_equal(r5, again)
        assert not np.array_equal(r5, RunRng(seed=3).server_rng(round_num=6).random(4))

    def test_server_stream_does_not_collide_with_any_client(self):
        server = RunRng(seed=3).server_rng().random(4)
        for cid in range(200):
            assert not np.array_equal(server, RunRng(seed=3).client(cid).numpy.random(4))

    def test_negative_identity_is_rejected_with_a_useful_message(self):
        with pytest.raises(ValueError, match="non-negative"):
            ClientRng(run_seed=1, client_id=-1).numpy.random(1)
        with pytest.raises(ValueError, match="non-negative"):
            ClientRng(run_seed=-1, client_id=1).numpy.random(1)


class TestProvenance:
    def test_client_rng_reports_its_identity_for_the_meta_block(self):
        import json

        c = ClientRng(run_seed=1234, client_id=7)
        prov = c.provenance()
        json.dumps(prov)
        assert prov["run_seed"] == 1234
        assert prov["client_id"] == 7
        # A stable digest lets a note assert "this is the same stream" without dumping draws.
        assert isinstance(prov["stream_digest"], str) and len(prov["stream_digest"]) == 16
        assert ClientRng(run_seed=1234, client_id=7).provenance() == prov
        assert ClientRng(run_seed=1234, client_id=8).provenance()["stream_digest"] != prov["stream_digest"]
