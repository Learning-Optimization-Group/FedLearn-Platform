"""A frozen backbone must not adapt its normalisation statistics to local data.

Measured on 2026-08-13, and it is the difference between the frozen arm working and appearing to.

`apply_arm` sets `requires_grad=False` on every backbone parameter, which stops gradients. It does
**not** stop BatchNorm: `running_mean` / `running_var` are *buffers*, not parameters, and a module
in `train()` mode updates them on every forward pass regardless of `requires_grad`. So the
"frozen" backbone kept re-estimating its normalisation from each client's shard.

What that cost, measured on CIFAR-10 with a linear probe on frozen features:

    BN held fixed:   pretrained 80.37% federated  (80.35% offline probe -- they agree)
    BN adapting:     pretrained 72.85% federated
    random backbone:            25.10% offline probe, BN fixed, same arch/resolution/data

BN adaptation costs ~7.5 points on a pretrained backbone: it re-estimates from one client's shard
the statistics ImageNet training had already fitted well.

CORRECTION worth keeping: this was first read as "BN adaptation lifts a RANDOM backbone to 72%",
from a federated random-backbone control that turned out to be invalid. Under a subset arm the
backbone is NEVER transmitted, so each client builds it locally from the recipe -- a random .npz
cannot change what the client runs. The valid random comparison is the offline probe.

Three reasons this is a defect and not a tuning choice:

1. **The arm's premise is violated.** The frozen arm exists so the backbone can be delivered once
   and then stay fixed; a backbone whose statistics track local data is not fixed.
2. **Clients silently diverge.** BN statistics are data-dependent and are never federated, so each
   client ends up with a different effective backbone while the server evaluates against the
   original one.
3. **It hides the value of pretraining**, which is the entire reason to freeze a backbone.

The research campaign never hit this because it fed the frozen arm PRE-EXTRACTED features, so the
backbone ran in eval mode by construction. The product path trains through the backbone and does not.
"""

import os
import sys

import pytest
import torch
import torch.nn as nn

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, ".."))

import recipes  # noqa: E402


class _NetWithBN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.BatchNorm2d(4))
        self.fc = nn.Linear(4, 2)

    def forward(self, x):
        return self.fc(self.features(x).mean((2, 3)))


class TestFrozenModulesDoNotTrackStatistics:
    def test_frozen_batchnorm_stays_in_eval_mode(self):
        """THE fix: after applying a subset arm and calling train(), the frozen BN must be eval."""
        m = _NetWithBN()
        recipes.apply_arm(m, "FROZEN_HEAD", ["fc."])
        m.train()
        recipes.freeze_untrained_modules(m, ["fc."])
        bn = m.features[1]
        assert bn.training is False, "frozen BatchNorm is still in train mode; it will adapt"

    def test_the_trainable_head_stays_in_train_mode(self):
        """Only the frozen part is pinned — the head must keep normal training behaviour."""
        m = _NetWithBN()
        recipes.apply_arm(m, "FROZEN_HEAD", ["fc."])
        m.train()
        recipes.freeze_untrained_modules(m, ["fc."])
        assert m.fc.training is True, "the trainable head was frozen too"

    def test_running_statistics_do_not_move(self):
        """The behavioural assertion, not just the flag: buffers must be unchanged after forwards."""
        m = _NetWithBN()
        recipes.apply_arm(m, "FROZEN_HEAD", ["fc."])
        m.train()
        recipes.freeze_untrained_modules(m, ["fc."])
        bn = m.features[1]
        before = (bn.running_mean.clone(), bn.running_var.clone())
        for _ in range(5):
            m(torch.randn(8, 3, 6, 6) * 10 + 5)      # deliberately off-distribution
        assert torch.equal(bn.running_mean, before[0]), "running_mean drifted"
        assert torch.equal(bn.running_var, before[1]), "running_var drifted"

    def test_without_the_fix_statistics_would_move(self):
        """Proves the test above is actually testing something: the same model WITHOUT the fix
        does drift, so this is a real behaviour change and not a vacuous assertion."""
        m = _NetWithBN()
        recipes.apply_arm(m, "FROZEN_HEAD", ["fc."])
        m.train()                                     # no freeze_untrained_modules call
        bn = m.features[1]
        before = bn.running_mean.clone()
        for _ in range(5):
            m(torch.randn(8, 3, 6, 6) * 10 + 5)
        assert not torch.equal(bn.running_mean, before), \
            "control failed: BN did not drift even without the fix, so the fix proves nothing"

    def test_the_full_arm_is_untouched(self):
        """A FULL run must keep BatchNorm adapting — that is correct there, and this must not
        quietly change the behaviour of every existing project."""
        m = _NetWithBN()
        recipes.apply_arm(m, "FULL", None)
        m.train()
        recipes.freeze_untrained_modules(m, None)
        assert m.features[1].training is True, "the FULL arm's BatchNorm was frozen"

    def test_it_is_idempotent_and_survives_a_train_call(self):
        """train() re-enables every child, so the client must re-apply this each epoch. Calling it
        twice, or after train(), must behave identically."""
        m = _NetWithBN()
        recipes.apply_arm(m, "FROZEN_HEAD", ["fc."])
        for _ in range(2):
            m.train()
            recipes.freeze_untrained_modules(m, ["fc."])
        assert m.features[1].training is False
        assert m.fc.training is True


@pytest.mark.slow
class TestOnTheRealPretrainedRecipe:
    def test_resnet18_backbone_batchnorms_are_pinned(self):
        m = recipes.get_recipe("CIFAR_RESNET18").build_model("cpu")
        pre = recipes.trainable_prefixes("CIFAR_RESNET18", "FROZEN_HEAD")
        recipes.apply_arm(m, "FROZEN_HEAD", pre)
        m.train()
        recipes.freeze_untrained_modules(m, pre)
        bns = [mod for mod in m.modules() if isinstance(mod, nn.BatchNorm2d)]
        assert bns, "no BatchNorm found in resnet18"
        assert not any(b.training for b in bns), \
            f"{sum(b.training for b in bns)}/{len(bns)} BatchNorm layers still adapting"
