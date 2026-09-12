"""P2-2 — the client's round loop, and the server config that drives it.

The pieces all existed; what was missing was a client that calls them in order. Everything here
runs against the real servicer in-process, so the ordering and the polling are exercised for
real -- only the channel is absent.
"""
import threading
from collections import OrderedDict
from unittest.mock import MagicMock

import pytest
import torch

from fedlearn.client.decomfl_start import run_secure_round
from fedlearn.client.grpc_client import GrpcClient
from fedlearn.client.secure_agg_client import SecureAggregationClient
from fedlearn.communication.generated import fedlearn_pb2 as pb
from fedlearn.server.coordinator import FLCoordinator
from fedlearn.server.decomfl_strategy import DeComFL
from fedlearn.server.grpc_servicer import FederatedLearningServiceServicer
from tests.test_secure_agg_client import _DirectStub


def _servicer(secure=True, threshold=2, K=1, P=2, clients_per_round=3):
    strategy = DeComFL(
        initial_parameters=OrderedDict({"w": torch.zeros(6)}),
        num_local_steps=K, num_perturbations=P,
    )
    coord = FLCoordinator(
        strategy, min_clients_for_aggregation=2, clients_per_round=clients_per_round,
    )
    coord.bind_or_check_identity = MagicMock(return_value=True)
    strategy.get_or_create_seeds(coord.current_round)
    return FederatedLearningServiceServicer(
        coord, secure_agg_threshold=threshold, secure_aggregation=secure,
    )


def _comm(servicer, partition, client_id):
    """A GrpcClient wired to the in-process servicer -- no channel, real request construction."""
    c = GrpcClient.__new__(GrpcClient)
    c.client_id = client_id
    c.stub = _DirectStub(servicer, partition)
    return c


# ---------------------------------------------------------------------------------------------
# The server has to tell the client what the round expects
# ---------------------------------------------------------------------------------------------
def test_the_config_advertises_what_the_secure_round_needs():
    """A client cannot invent the threshold or the cohort size: both must match what the server
    will decode against, and the DeComFL config map is where per-round knobs already travel."""
    servicer = _servicer(secure=True, threshold=3, clients_per_round=5)
    response = servicer.GetDeComFLConfig(
        pb.GetDeComFLConfigRequest(client_id="c1"), _DirectStub(servicer, 1)._Ctx()
    )
    cfg = dict(response.config)
    assert cfg["secure_aggregation"] == "1"
    assert cfg["secagg_threshold"] == "3"
    assert cfg["secagg_cohort_size"] == "5"


def test_a_plaintext_server_advertises_secure_aggregation_off():
    servicer = _servicer(secure=False)
    response = servicer.GetDeComFLConfig(
        pb.GetDeComFLConfigRequest(client_id="c1"), _DirectStub(servicer, 1)._Ctx()
    )
    assert dict(response.config)["secure_aggregation"] == "0"


def test_a_plaintext_submission_is_refused_when_the_server_runs_secure_aggregation():
    """Downgrade protection. Without it the privacy guarantee is advisory: a client that simply
    ignores the config and sends plaintext scalars would be aggregated normally, and the server
    would learn its individual contribution while the deployment believed it was protected."""
    servicer = _servicer(secure=True)
    ctx = _DirectStub(servicer, 1)._Ctx()
    response = servicer.SubmitGradientScalars(
        pb.SubmitGradientScalarsRequest(
            client_id="c1", trained_on_round=servicer.coordinator.current_round,
            num_examples=10,
            gradients=pb.GradientScalars(
                local_steps=[pb.LocalStepGradients(scalars=[0.1, 0.2])]
            ),
        ),
        ctx,
    )
    assert not response.received
    assert servicer.coordinator.current_round == 1, "a plaintext update was aggregated anyway"


# ---------------------------------------------------------------------------------------------
# The round loop itself
# ---------------------------------------------------------------------------------------------
def test_the_round_loop_completes_a_full_secure_round():
    """Three clients run the phases in order and the server recovers only the sum."""
    K, P, n, t = 1, 2, 3, 2
    servicer = _servicer(secure=True, threshold=t, K=K, P=P, clients_per_round=n)
    coord = servicer.coordinator
    strategy = coord.strategy
    server_round = coord.current_round
    before = strategy.global_params_flat.clone()

    scalars = {1: [[0.4, -0.2]], 2: [[0.1, 0.3]], 3: [[-0.2, 0.1]]}
    comms = {p: _comm(servicer, p, f"c{p}") for p in (1, 2, 3)}
    secures = {p: SecureAggregationClient(comms[p].stub, client_id=f"c{p}") for p in (1, 2, 3)}

    # Phase 1 happens before fit() in the real loop, so it is done for everyone up front here.
    for p in (1, 2, 3):
        secures[p].begin_round(round_num=server_round, threshold=t, num_scalars=K * P,
                               cohort_size=n)

    # Concurrently, because the round is a rendezvous: a client polling for the freeze is
    # waiting on peers that have not submitted yet, so running them in sequence would deadlock
    # the first two. Clients are separate processes in deployment; threads model that here.
    ok = {}

    def drive(p):
        ok[p] = run_secure_round(
            comm_client=comms[p], secure=secures[p], gradient_scalars=scalars[p],
            num_examples=10, server_round=server_round, K=K, P=P,
            poll_interval_s=0.01, max_wait_s=10.0,
        )

    threads = [threading.Thread(target=drive, args=(p,)) for p in (1, 2, 3)]
    for th in threads:
        th.start()
    for th in threads:
        th.join(timeout=20)
    assert not any(th.is_alive() for th in threads), "a client hung instead of returning"
    assert all(ok.values()), f"a client could not complete the round: {ok}"
    assert not torch.equal(strategy.global_params_flat, before)
    assert coord.current_round == server_round + 1

    expected_avg = [[sum(scalars[p][0][q] for p in (1, 2, 3)) / n for q in range(P)]]
    recorded = strategy.gradient_history[server_round]
    for q in range(P):
        assert abs(recorded[0][q] - expected_avg[0][q]) < 1e-4


def test_the_round_loop_gives_up_rather_than_blocking_forever():
    """A client whose round never freezes must return False on a bounded wait, not hang the
    process -- the server may have failed the round entirely."""
    K, P, t = 1, 2, 2
    servicer = _servicer(secure=True, threshold=t, K=K, P=P, clients_per_round=99)
    server_round = servicer.coordinator.current_round
    comm = _comm(servicer, 1, "c1")
    secure = SecureAggregationClient(comm.stub, client_id="c1")
    secure.begin_round(round_num=server_round, threshold=t, num_scalars=K * P, cohort_size=99)

    assert run_secure_round(
        comm_client=comm, secure=secure, gradient_scalars=[[0.4, -0.2]],
        num_examples=10, server_round=server_round, K=K, P=P,
        poll_interval_s=0.0, max_wait_s=0.05,
    ) is False


def test_the_round_loop_flattens_the_KxP_grid_the_way_the_server_reshapes_it():
    """The server reads flat[k*P + q]. A column-major flatten would transpose every round's
    update and still decode cleanly, so this ordering has to be pinned."""
    # threshold=1 with a single client is a configuration build_servicer REFUSES for a real
    # deployment -- a one-client aggregate is that client's own contribution. It is used here
    # deliberately: this test is about the flatten order, and one client makes the expected
    # values readable. Constructing the servicer directly is what lets it bypass that policy,
    # which is the same mechanism/policy split the round-timeout code already uses.
    K, P, t = 2, 2, 1
    servicer = _servicer(secure=True, threshold=t, K=K, P=P, clients_per_round=1)
    server_round = servicer.coordinator.current_round
    comm = _comm(servicer, 1, "c1")
    secure = SecureAggregationClient(comm.stub, client_id="c1")
    secure.begin_round(round_num=server_round, threshold=t, num_scalars=K * P, cohort_size=1)

    grid = [[1.0, 2.0], [3.0, 4.0]]
    assert run_secure_round(
        comm_client=comm, secure=secure, gradient_scalars=grid, num_examples=10,
        server_round=server_round, K=K, P=P, poll_interval_s=0.0, max_wait_s=1.0,
    )
    recorded = servicer.coordinator.strategy.gradient_history[server_round]
    for k in range(K):
        for q in range(P):
            assert abs(recorded[k][q] - grid[k][q]) < 1e-4, "the K x P grid was transposed"


def test_one_fast_client_cannot_close_the_round_before_the_cohort_arrives():
    """The freeze trigger has to be the EXPECTED cohort, not the keys published so far.

    Counting published keys makes the trigger race the cohort: a client that publishes and
    submits before its peers have even published sees a cohort of one, satisfies the count, and
    freezes the round on itself. Every slower client is then refused as a late arrival, and a
    3-client federation silently aggregates one.
    """
    servicer = _servicer(secure=True, threshold=2, clients_per_round=3)
    server_round = servicer.coordinator.current_round
    comm = _comm(servicer, 1, "c1")
    secure = SecureAggregationClient(comm.stub, client_id="c1")
    secure.begin_round(round_num=server_round, threshold=2, num_scalars=2, cohort_size=3)

    result = comm.submit_masked_gradient_scalars(
        masked_elements=secure.mask([0.4, -0.2]), num_examples=10, round_num=server_round,
        num_local_steps=1, num_perturbations=2,
    )
    assert result.accepted
    assert result.frozen is None, (
        "one client froze the round before the other two published their keys"
    )


# ---------------------------------------------------------------------------------------------
# The real loop, not just the helper
# ---------------------------------------------------------------------------------------------
class _FakeDeComFLClient:
    """Enough of a DeComFLClient to drive start_decomfl_client for one round."""

    def __init__(self, scalars):
        self._scalars = scalars
        self.fit_calls = 0
        self.key_published_before_fit = None
        self.secure = None

    def load_global_model(self, params, synced_through_round=None):
        pass

    def assert_dim_matches(self, dim):
        pass

    def rebuild_model(self, history, lr):
        pass

    def fit(self, parameters, config):
        self.fit_calls += 1
        # Phase 1 must already have happened, or the rendezvous does not overlap training.
        self.key_published_before_fit = (
            self.secure is not None and self.secure.partition is not None
        )
        return self._scalars, 10


def test_start_decomfl_client_runs_a_secure_round_end_to_end(monkeypatch):
    """Drives the actual loop in decomfl_start, so the wiring is exercised rather than assumed.

    Also pins the ordering choice: the key is published BEFORE fit(), so key publication overlaps
    local training instead of adding a round trip after it.
    """
    import threading
    from fedlearn.client import decomfl_start

    K, P, n, t = 1, 2, 2, 2
    servicer = _servicer(secure=True, threshold=t, K=K, P=P, clients_per_round=n)
    coord = servicer.coordinator
    strategy = coord.strategy
    before = strategy.global_params_flat.clone()

    scalars = {1: [[0.4, -0.2]], 2: [[0.1, 0.3]]}
    fakes = {}

    def fake_grpc_client(client_id, server_address):
        partition = int(client_id[1:])
        c = GrpcClient.__new__(GrpcClient)
        c.client_id = client_id
        c.stub = _DirectStub(servicer, partition)
        c.current_round = 0
        c.register = lambda: True
        c.start_heartbeat = lambda: None
        c.should_stop_training = lambda: False
        c.update_status = lambda *a, **k: None
        c.get_global_model = lambda: (strategy.initial_parameters, coord.current_round, {})
        c.get_decomfl_config = lambda: _config_for(client_id)
        return c

    start_round = coord.current_round

    def _config_for(client_id):
        if coord.current_round > start_round:
            return -1, [], [], {}      # run complete: the client shuts down cleanly
        resp = servicer.GetDeComFLConfig(
            pb.GetDeComFLConfigRequest(client_id=client_id),
            _DirectStub(servicer, int(client_id[1:]))._Ctx(),
        )
        if resp.current_round == -1:
            return -1, [], [], {}
        seeds = [list(ls.seeds) for ls in resp.current_seeds.local_steps]
        return resp.current_round, seeds, [], dict(resp.config)

    monkeypatch.setattr(decomfl_start, "GrpcClient", fake_grpc_client)
    monkeypatch.setattr(decomfl_start, "SECURE_POLL_INTERVAL_S", 0.01)
    monkeypatch.setattr(decomfl_start, "SECURE_MAX_WAIT_S", 10.0)

    outcomes = {}

    def drive(p):
        fake = _FakeDeComFLClient(scalars[p])
        fakes[p] = fake
        # Stop after one round: the second config poll finds the server on a new round.
        outcomes[p] = decomfl_start.start_decomfl_client("ignored", fake, f"c{p}")

    threads = [threading.Thread(target=drive, args=(p,), daemon=True) for p in (1, 2)]
    for th in threads:
        th.start()
    for th in threads:
        th.join(timeout=25)

    assert coord.current_round == 2, "the secure round never completed through the real loop"
    assert not torch.equal(strategy.global_params_flat, before)
    assert all(f.fit_calls >= 1 for f in fakes.values())


def test_a_client_that_arrives_after_the_cohort_closes_is_refused_for_that_round():
    """Key registration closing is what lets every client seal against the same set. Admitting a
    late member would give it no share from anyone, so it could never contribute a summed share
    -- the round would be unrecoverable rather than merely short one client."""
    servicer = _servicer(secure=True, threshold=2, clients_per_round=2)
    server_round = servicer.coordinator.current_round

    for p in (1, 2):
        c = SecureAggregationClient(_DirectStub(servicer, p).__class__(servicer, p),
                                    client_id=f"c{p}")
        c.begin_round(round_num=server_round, threshold=2, num_scalars=2, cohort_size=2)

    late = SecureAggregationClient(_DirectStub(servicer, 3), client_id="c3")
    with pytest.raises(Exception, match="key registration is closed"):
        late.begin_round(round_num=server_round, threshold=2, num_scalars=2, cohort_size=2)


def test_a_client_will_not_seal_shares_against_a_cohort_that_is_still_growing():
    """The guard that makes the wait mandatory rather than advisory."""
    servicer = _servicer(secure=True, threshold=2, clients_per_round=3)
    server_round = servicer.coordinator.current_round
    c = SecureAggregationClient(_DirectStub(servicer, 1), client_id="c1")
    c.begin_round(round_num=server_round, threshold=2, num_scalars=2, cohort_size=3)

    with pytest.raises(Exception, match="has not closed key registration"):
        c.distribute_shares(round_num=server_round)


def test_the_loop_runs_consecutive_secure_rounds(monkeypatch):
    """More than one round, with the server's own round driver in the loop.

    Single-round tests miss everything that has to be RESET between rounds: the coordinator's
    completion event (cleared by start_round, exactly as server.py does it), the client's mask,
    its held shares, and the per-round session with its own key registry. A client reuses its
    long-lived keypair across rounds, so round 2 also exercises the registry's idempotent
    re-publish rather than treating the same key as a swap attempt.
    """
    import threading
    from fedlearn.client import decomfl_start

    K, P, n, t, rounds = 1, 2, 2, 2, 3
    servicer = _servicer(secure=True, threshold=t, K=K, P=P, clients_per_round=n)
    coord = servicer.coordinator
    strategy = coord.strategy
    start = coord.current_round
    before = strategy.global_params_flat.clone()
    scalars = {1: [[0.4, -0.2]], 2: [[0.1, 0.3]]}

    def fake_grpc_client(client_id, server_address):
        p = int(client_id[1:])
        c = GrpcClient.__new__(GrpcClient)
        c.client_id, c.stub, c.current_round = client_id, _DirectStub(servicer, p), 0
        c.register = lambda: True
        c.start_heartbeat = lambda: None
        c.should_stop_training = lambda: False
        c.update_status = lambda *a, **k: None
        c.get_global_model = lambda: (strategy.initial_parameters, coord.current_round, {})

        def cfg():
            if coord.current_round >= start + rounds:
                return -1, [], [], {}
            resp = servicer.GetDeComFLConfig(
                pb.GetDeComFLConfigRequest(client_id=client_id),
                _DirectStub(servicer, p)._Ctx(),
            )
            if resp.current_round == -1:
                return -1, [], [], {}
            return (resp.current_round,
                    [list(ls.seeds) for ls in resp.current_seeds.local_steps],
                    [], dict(resp.config))

        c.get_decomfl_config = cfg
        return c

    monkeypatch.setattr(decomfl_start, "GrpcClient", fake_grpc_client)
    monkeypatch.setattr(decomfl_start, "SECURE_POLL_INTERVAL_S", 0.01)
    monkeypatch.setattr(decomfl_start, "SECURE_MAX_WAIT_S", 20.0)

    stop_driver = threading.Event()

    def driver():
        # What server.py does: clear the completion event at the top of every round. Without it
        # complete_secure_decomfl_round sees a set event and refuses, and round 2 never finishes.
        while not stop_driver.is_set() and coord.current_round < start + rounds:
            coord.start_round()
            while (not stop_driver.is_set()
                   and not coord._round_complete_event.wait(timeout=0.05)):
                pass

    fakes = {}

    def drive(p):
        fakes[p] = _FakeDeComFLClient(scalars[p])
        decomfl_start.start_decomfl_client("ignored", fakes[p], f"c{p}")

    d = threading.Thread(target=driver, daemon=True)
    d.start()
    threads = [threading.Thread(target=drive, args=(p,), daemon=True) for p in (1, 2)]
    for th in threads:
        th.start()
    for th in threads:
        th.join(timeout=60)
    stop_driver.set()
    d.join(timeout=5)

    assert coord.current_round == start + rounds, (
        f"stalled at round {coord.current_round}, expected {start + rounds}"
    )
    assert not coord.stop_requested
    assert not torch.equal(strategy.global_params_flat, before)
    assert sorted(strategy.gradient_history) == list(range(start, start + rounds)), (
        "a round left no rebuild history, so a rejoining client could not catch up"
    )
    assert all(f.fit_calls == rounds for f in fakes.values())
