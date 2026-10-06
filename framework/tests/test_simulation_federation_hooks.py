"""Collect-then-submit hook: with an identity hook the run is identical to the default path."""
from collections import OrderedDict

import torch

from fedlearn.server.strategy import FedAvg
from fedlearn.simulation.federation import SimulatedFederation


class _Client:
    def __init__(self, cid):
        self.cid = cid

    def fit(self, global_params, config):
        return OrderedDict((k, v + 0.01 * (self.cid + 1)) for k, v in global_params.items()), 10 + self.cid


def _fed(**kw):
    init = OrderedDict(w=torch.zeros(3))
    return SimulatedFederation(FedAvg(init, clients_per_round=4), lambda cid, rng: _Client(cid),
                               num_clients=6, clients_per_round=4, seed=3, initial_parameters=init, **kw)


def _identity(r, col, g, sel, drop):
    return col, {"seen": len(col)}


def test_identity_hook_reproduces_the_default_path():
    a = _fed().run(3)
    b = _fed(round_hook=_identity).run(3)
    assert a.final_digest == b.final_digest
    assert [r.round for r in b.rounds] == [1, 2, 3]
    assert [r.extra["seen"] for r in b.rounds] == [4, 4, 4]
    assert all(r.extra == {} for r in a.rounds)


def test_identity_hook_reproduces_the_default_path_with_dropout():
    a = _fed(dropout_rate=0.25).run(3)
    b = _fed(dropout_rate=0.25, round_hook=_identity).run(3)
    assert a.final_digest == b.final_digest
    assert [r.reported for r in a.rounds] == [r.reported for r in b.rounds]


def test_hook_can_drop_clients_and_the_round_is_force_resolved():
    out = _fed(round_hook=lambda r, col, g, sel, drop: (col[:3], {})).run(2)
    assert all(r.forced and len(r.reported) == 3 for r in out.rounds)


def test_submissions_are_made_in_ascending_client_order():
    out = _fed(round_hook=lambda r, col, g, sel, drop: (list(reversed(col)), {})).run(1)
    assert out.rounds[0].reported == sorted(out.rounds[0].reported)


def test_on_round_and_should_stop():
    calls = []
    out = _fed().run(5, on_round=lambda rec: calls.append(rec.round), should_stop=lambda: len(calls) >= 2)
    assert calls == [1, 2] and out.stopped_early and out.stop_reason == "stopped by request"
