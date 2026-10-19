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


# Stage 3: a device trains its own dataset, whose example count need not be the example batch the programs were
# exported with. A static export refuses another count; max_batch makes it dynamic up to that bound.

def _load(tmp_path, name, pte):
    from executorch.runtime import Runtime
    path = tmp_path / name
    path.write_bytes(pte)
    return Runtime.get().load_program(str(path)).load_method("forward")


def _batch(n):
    gin = torch.Generator().manual_seed(123)
    return torch.randn(n, 4, generator=gin), torch.randint(0, 3, (n,), generator=gin)


def test_a_static_loss_program_refuses_another_example_count(tmp_path):
    pytest.importorskip("executorch")
    torch.manual_seed(0)
    m = TinyNet().eval()
    method = _load(tmp_path, "static.pte", export_functional_pte(m, _batch(8)))
    x6, y6 = _batch(8)[0][:6], _batch(8)[1][:6]
    with pytest.raises(Exception):
        method.execute([trainable_flat(m), x6, y6])


@pytest.mark.parametrize("n", [1, 6, 8])
def test_a_dynamic_batch_loss_program_matches_eager_at_any_count_up_to_its_bound(tmp_path, n):
    pytest.importorskip("executorch")
    torch.manual_seed(0)
    m = TinyNet().eval()
    method = _load(tmp_path, "dyn.pte", export_functional_pte(m, _batch(8), max_batch=8))
    x, y = _batch(8)
    flat = trainable_flat(m)
    got = float(method.execute([flat, x[:n], y[:n]])[0])
    assert abs(got - _eager_loss(m, flat, x[:n], y[:n])) < 1e-6


@pytest.mark.parametrize("n", [1, 6, 8])
def test_a_dynamic_batch_infer_program_returns_one_row_per_example(tmp_path, n):
    pytest.importorskip("executorch")
    from pte_export import export_functional_infer_pte
    torch.manual_seed(0)
    m = TinyNet().eval()
    method = _load(tmp_path, "infer.pte", export_functional_infer_pte(m, _batch(8)[0], max_batch=8))
    x = _batch(8)[0][:n]
    logits = method.execute([trainable_flat(m), x])[0]
    assert tuple(logits.shape) == (n, 3)
    assert torch.allclose(logits, m(x), atol=1e-6)


def test_a_batch_bound_below_one_is_refused():
    from pte_export import _batch_dim
    with pytest.raises(ValueError):
        _batch_dim(0)
