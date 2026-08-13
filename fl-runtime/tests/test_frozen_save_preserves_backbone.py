"""A frozen run must not destroy the backbone its own output depends on.

Found by the live FROZEN_HEAD federation on 2026-08-13. After a successful 3-round run, the saved
model contained **two** keys::

    PRE  keys (10): conv1.weight ... fc3.bias
    POST keys  (2): fc3.bias, fc3.weight

The server writes the final *global* model to ``--model-path``, and under a subset arm the global
model IS the head. So the run overwrote the only copy of the backbone with a 2-key file:

* the saved artifact is **not a usable model** — you cannot run inference without the backbone;
* the backbone is **unrecoverable**, because ``--model-path`` held the only copy and was the file
  overwritten in place;
* a second frozen run then starts from a file with no backbone at all.

This is the same class as commit ``21699bc`` — silent destruction of something that cannot be
regenerated — and P1-2's own comment already stated the intended contract:

    "The .npz deliberately keeps the FULL model -- the frozen backbone has to stay recoverable --
     so the arm is applied here rather than at save time."

That intent was written and never implemented on the save side. The fix merges the non-federated
parameters back in, so what is saved is a complete model: aggregated head over the frozen backbone
the run actually used.
"""

import os
import sys
from collections import OrderedDict

import pytest
import torch

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, ".."))

import fl_server  # noqa: E402


def _full():
    return OrderedDict((k, torch.full((2, 2), float(i)))
                       for i, k in enumerate(["conv1.weight", "conv1.bias",
                                              "fc3.weight", "fc3.bias"]))


def _head_after_training():
    return OrderedDict([("fc3.weight", torch.full((2, 2), 99.0)),
                        ("fc3.bias", torch.full((2, 2), 98.0))])


class TestMergeBackNonFederatedParameters:
    def test_the_saved_model_is_complete(self):
        """THE regression: every key of the original model must survive the run."""
        merged = fl_server.merge_non_federated(_head_after_training(), _full())
        assert set(merged) == set(_full()), \
            f"saved model lost keys: {sorted(set(_full()) - set(merged))}"

    def test_the_trained_head_wins(self):
        """The federated result must not be overwritten by the stale initial values."""
        merged = fl_server.merge_non_federated(_head_after_training(), _full())
        assert torch.equal(merged["fc3.weight"], torch.full((2, 2), 99.0))
        assert torch.equal(merged["fc3.bias"], torch.full((2, 2), 98.0))

    def test_the_backbone_is_the_one_the_run_actually_used(self):
        """Not a fresh init: the head was trained against THESE frozen weights, so pairing it with
        any other backbone produces a model that never existed during training."""
        full = _full()
        merged = fl_server.merge_non_federated(_head_after_training(), full)
        for k in ("conv1.weight", "conv1.bias"):
            assert torch.equal(merged[k], full[k])

    def test_key_order_follows_the_original_model(self):
        """state_dict ordering is load-bearing for the safetensors wire and for sha256 provenance;
        a merge that reordered keys would change the digest of an otherwise identical model."""
        merged = fl_server.merge_non_federated(_head_after_training(), _full())
        assert list(merged) == list(_full())

    def test_a_full_arm_run_is_untouched(self):
        """When nothing was withheld, the merge must be the identity — no new behaviour on the
        path that every existing project uses."""
        full = _full()
        final = OrderedDict((k, v * 2) for k, v in full.items())
        merged = fl_server.merge_non_federated(final, full)
        assert list(merged) == list(final)
        for k in final:
            assert torch.equal(merged[k], final[k])

    def test_missing_initial_parameters_is_not_fatal(self):
        """A save must never fail because the merge could not run — losing the trained head to
        protect the backbone would be a worse outcome than the bug."""
        merged = fl_server.merge_non_federated(_head_after_training(), None)
        assert set(merged) == {"fc3.weight", "fc3.bias"}

    def test_extra_federated_keys_are_kept(self):
        """Defensive: a key present in the result but not in the original must not be dropped."""
        final = OrderedDict(_head_after_training())
        final["new.key"] = torch.zeros(1)
        merged = fl_server.merge_non_federated(final, _full())
        assert "new.key" in merged
