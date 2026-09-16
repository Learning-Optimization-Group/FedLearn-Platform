"""FR-14: the DeComFL server's flat layout must equal each participating client's TRAINABLE layout.

The client flattens `named_parameters()` filtered by `requires_grad`; the server flattens whatever
OrderedDict it is handed. When a full `model.state_dict()` is passed as `initial_parameters` for a model
with frozen params (LoRA base, partial fine-tune — e.g. the golden TinyNet's frozen fc2) the server's
flat vector is LONGER than the client's, so the shared-seed perturbation `z` misaligns and the model
diverges silently. These tests pin the two guardrails: `params.trainable_state` (the correct
`initial_parameters` builder) and `DeComFL.validate_participant_dim` (fail loud instead of diverging).
"""
from collections import OrderedDict

import pytest
import torch
import torch.nn as nn

from fedlearn.estimators import params
from fedlearn.server.decomfl_strategy import DeComFL
from fedlearn.client.decomfl_client import DeComFLClient
from fedlearn.server.coordinator import FLCoordinator
from fedlearn.server.grpc_servicer import FederatedLearningServiceServicer
from fedlearn.communication.generated import fedlearn_pb2


class FrozenNet(nn.Module):
    """fc1 trainable (25) + fc2 FROZEN (18) — the golden-TinyNet shape. state_dict() carries the frozen
    fc2 (all float32, so it cats fine) → 43 dims vs the client's 25 trainable: the silent-divergence setup."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 5)   # 5*4 + 5 = 25 trainable
        self.fc2 = nn.Linear(5, 3)   # 5*3 + 3 = 18, frozen
        for p in self.fc2.parameters():
            p.requires_grad_(False)


def _decomfl(initial_parameters):
    return DeComFL(initial_parameters=initial_parameters, min_fit_clients=1,
                   num_local_steps=1, num_perturbations=1, seed=1)


def test_trainable_state_excludes_frozen_params():
    m = FrozenNet()
    ts = params.trainable_state(m)
    assert set(ts.keys()) == {"fc1.weight", "fc1.bias"}                 # no frozen fc2
    assert sum(t.numel() for t in ts.values()) == params.num_trainable(m) == 25
    assert len(OrderedDict(m.state_dict())) == 4                        # full state_dict still has fc2
    # detached snapshot: mutating the source model does not bleed into the captured state
    before = ts["fc1.bias"].clone()
    with torch.no_grad():
        m.fc1.bias.add_(1.0)
    assert torch.equal(ts["fc1.bias"], before)


def test_server_built_from_trainable_state_matches_client_dim():
    m = FrozenNet()
    server = _decomfl(params.trainable_state(m))
    assert server.model_dim == params.num_trainable(m) == 25           # d_server == d_client
    server.validate_participant_dim(25)                                # no raise
    server.validate_participant_dim(25, client_id="phone-1")           # no raise


def test_server_built_from_full_state_dict_is_detected_not_silent():
    m = FrozenNet()
    # The BUG shape: a full state_dict (incl. frozen fc2) inflates the server's flat vector past the client's.
    server = _decomfl(OrderedDict(m.state_dict()))
    assert server.model_dim == 43
    assert server.model_dim > params.num_trainable(m)                  # would silently misalign z
    with pytest.raises(ValueError, match="dimension mismatch"):        # now FAIL-LOUD
        server.validate_participant_dim(params.num_trainable(m), client_id="phone-1")


# ---- MO-19: the CLIENT-side half of the same guard (server advertises model_dim; client checks) ----

class _CfgCtx:
    """Minimal unary context for GetDeComFLConfig (no identity gate on this read RPC)."""

    def invocation_metadata(self):
        return []

    def set_code(self, code):
        pass

    def set_details(self, details):
        pass


def test_client_asserts_its_trainable_dim_matches_the_server_model_dim():
    m = FrozenNet()  # 25 trainable (fc1); fc2 frozen
    client = DeComFLClient(model=m, train_loader=None, device="cpu")
    client.assert_dim_matches(25)  # d_server == d_client -> no raise
    with pytest.raises(ValueError, match="dimension mismatch"):
        client.assert_dim_matches(43)  # the full-state_dict d_server=43 vs d_client=25 bug -> fail loud


def test_server_advertises_its_model_dim_in_the_decomfl_config():
    # The server publishes model_dim so ANY client (python or mobile) can self-check at the handshake.
    m = FrozenNet()
    strategy = _decomfl(params.trainable_state(m))  # model_dim == 25
    coordinator = FLCoordinator(strategy, 1, 1)
    servicer = FederatedLearningServiceServicer(coordinator)
    resp = servicer.GetDeComFLConfig(fedlearn_pb2.GetDeComFLConfigRequest(client_id="c0"), _CfgCtx())
    assert resp.config["model_dim"] == "25"
