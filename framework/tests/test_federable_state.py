"""Non-float32 buffers are excluded from the federated set, so BatchNorm models can federate.

The safetensors wire is float32-only, deliberately: it has to decode in the libtorch-free mobile
C++ core, so other dtypes raise rather than being silently coerced. Every BatchNorm module carries
an int64 ``num_batches_tracked``, so a FULL-arm run on ANY BatchNorm model died on the first
``GetGlobalModel``::

    ValueError: Tensor 'bn1.num_batches_tracked' has dtype torch.int64

That blocked ResNets — the most common architecture in the FL literature — from the FULL arm
entirely. It surfaced only when CIFAR_RESNET18 became the first catalog recipe with BatchNorm;
PNEUMONIA_CNN, CNN, MLP and TINYNET_GOLDEN are all float32-clean.

WHAT THIS IS, AND WHAT IT IS NOT
--------------------------------
Excluded: tensors whose dtype is not float32. In practice that is ``num_batches_tracked``, a batch
COUNTER. Averaging a counter across clients is meaningless, so nothing of value is lost, and each
client simply keeps its own.

NOT excluded: ``running_mean`` and ``running_var``, which are float32 and continue to be averaged.
Dropping those too would be FedBN, a different algorithm with different convergence behaviour, and
a far larger change than unblocking the wire. The line is drawn at what the wire can carry.

The exclusion must be applied by ONE function used on both sides. Client/server divergence over
which tensors are federated has been this codebase's recurring failure, and two independent filters
would drift.
"""

import sys
from collections import OrderedDict

import pytest
import torch
import torch.nn as nn

from fedlearn.estimators.params import federable_state, non_federable_names


class _WithBN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3)
        self.bn = nn.BatchNorm2d(4)
        self.fc = nn.Linear(4, 2)

    def forward(self, x):
        return self.fc(self.bn(self.conv(x)).mean((2, 3)))


class _FloatOnly(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 3)
        self.fc2 = nn.Linear(3, 2)

    def forward(self, x):
        return self.fc2(self.fc1(x))


class TestWhatIsExcluded:
    def test_the_int64_counter_is_excluded(self):
        """THE fix. num_batches_tracked is the tensor that blocked every BatchNorm model."""
        sd = _WithBN().state_dict()
        assert "bn.num_batches_tracked" in sd, "fixture does not exercise the case"
        assert "bn.num_batches_tracked" not in federable_state(sd)

    def test_the_result_is_entirely_float32(self):
        """The property the wire actually requires — stated as the postcondition, so a new
        non-float buffer type is covered without naming it."""
        for tensor in federable_state(_WithBN().state_dict()).values():
            assert tensor.dtype == torch.float32

    def test_running_statistics_are_KEPT(self):
        """running_mean/var are float32 and keep being averaged. Dropping them would be FedBN — a
        different algorithm — not a wire fix. This test is what stops that drift."""
        fed = federable_state(_WithBN().state_dict())
        assert "bn.running_mean" in fed
        assert "bn.running_var" in fed

    def test_weights_and_biases_are_kept(self):
        fed = federable_state(_WithBN().state_dict())
        for k in ("conv.weight", "conv.bias", "bn.weight", "bn.bias", "fc.weight", "fc.bias"):
            assert k in fed, f"{k} was dropped"

    def test_a_float_only_model_is_untouched(self):
        """No existing recipe may change behaviour: for a float32-only model this is the identity,
        preserving keys, order and tensor identity."""
        sd = _FloatOnly().state_dict()
        fed = federable_state(sd)
        assert list(fed) == list(sd)
        for k in sd:
            assert fed[k] is sd[k], f"{k} was copied or altered"

    def test_key_order_is_preserved(self):
        """Ordering is load-bearing for the deterministic wire and for the sha256 an artifact is
        addressed by."""
        sd = _WithBN().state_dict()
        expected = [k for k in sd if sd[k].dtype == torch.float32]
        assert list(federable_state(sd)) == expected


class TestTheExclusionIsVisible:
    def test_excluded_names_are_reportable(self):
        """Silent dropping is what makes this dangerous. The caller must be able to say WHAT it
        withheld, so a run can log it and a reader can audit it."""
        assert non_federable_names(_WithBN().state_dict()) == ["bn.num_batches_tracked"]

    def test_nothing_to_report_for_a_float_only_model(self):
        assert non_federable_names(_FloatOnly().state_dict()) == []


class TestBothSidesAgreeByConstruction:
    def test_the_same_function_produces_the_same_set(self):
        """Client and server must federate an identical key set. The guarantee here is that there
        is only ONE filter — two implementations would drift, which is precisely how the frozen arm
        broke twice already."""
        model = _WithBN()
        client_side = set(federable_state(model.state_dict()))
        server_side = set(federable_state(OrderedDict(model.state_dict())))
        assert client_side == server_side

    def test_the_filtered_payload_serialises(self):
        """End-to-end justification: the whole point is that the result crosses the wire."""
        from fedlearn.communication.serializer import state_dict_to_safetensors

        sd = _WithBN().state_dict()
        with pytest.raises(ValueError, match="int64"):
            state_dict_to_safetensors(sd, num_examples=0)      # unfiltered still fails, loudly
        blob = state_dict_to_safetensors(federable_state(sd), num_examples=0)
        assert blob, "the filtered payload did not serialise"
