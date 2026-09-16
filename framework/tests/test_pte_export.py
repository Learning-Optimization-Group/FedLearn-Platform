# framework/tests/test_pte_export.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "mobile_client", "scripts"))

import torch
import torch.nn as nn
import pytest
from pte_export import export_functional_pte, trainable_flat, trainable_names


class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 5)
        self.fc2 = nn.Linear(5, 3)
        for p in self.fc2.parameters():
            p.requires_grad_(False)
    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


def _eager_loss(model, flat, x, y):
    # mirror the wrapper: write flat into trainable params, frozen stay as-is, then forward.
    names = trainable_names(model)
    off = 0
    sd = dict(model.named_parameters())
    params = {}
    for n in names:
        k = sd[n].numel()
        params[n] = flat[off:off + k].reshape(sd[n].shape); off += k
    for n, p in model.named_parameters():
        if not p.requires_grad:
            params[n] = p.detach()
    from torch.func import functional_call
    logits = functional_call(model, params, (x,))
    return float(torch.nn.functional.cross_entropy(logits, y))


def test_flat_param_ordering_matches_named_parameters():
    torch.manual_seed(0)
    m = TinyNet().eval()
    assert trainable_names(m) == ["fc1.weight", "fc1.bias"]
    assert trainable_flat(m).numel() == 25


def test_pte_forward_matches_eager(tmp_path):
    pytest.importorskip("executorch")
    torch.manual_seed(0)
    m = TinyNet().eval()
    gin = torch.Generator().manual_seed(123)
    x = torch.randn(8, 4, generator=gin)
    y = torch.randint(0, 3, (8,), generator=gin)
    flat = trainable_flat(m)

    pte = export_functional_pte(m, (x, y))
    from executorch.runtime import Runtime
    pte_path = tmp_path / "tiny.pte"           # pytest tmp_path is auto-cleaned (no leak)
    pte_path.write_bytes(pte)
    method = Runtime.get().load_program(str(pte_path)).load_method("forward")
    et_loss = float(method.execute([flat, x, y])[0])
    assert abs(et_loss - _eager_loss(m, flat, x, y)) < 1e-4
