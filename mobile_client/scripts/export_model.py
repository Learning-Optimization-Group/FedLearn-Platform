"""Export TorchScript models for the mobile FL client — ONE script, all mobile tiers (15-LLD §13.18).

torch.jit.script each tier and save model_<tier>.pt + print param count + sha256 (the mobile core
verifies the hash before jit::load). Input dim 64 matches the Model Testing 8x8 grid; output dim 10.
100M is NOT a mobile tier (A6 §M-H2: ~2 GB working set OOMs phones) — it is rejected here.

Usage:
    cd mobile_client && python scripts/export_model.py --out assets/models --tiers 1M 10M
"""
from __future__ import annotations

import argparse
import hashlib
import os

import torch
import torch.nn as nn

IN_DIM = 64
OUT_DIM = 10

# Hidden layer sizes tuned to approximate each tier's parameter count (exact count printed on export).
TIER_HIDDEN = {
    "1M": [1000, 1000],
    "10M": [2200, 2200, 2200],
}


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: list[int], out_dim: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers += [nn.Linear(prev, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def param_count(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="assets/models")
    ap.add_argument("--tiers", nargs="+", default=["1M", "10M"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # pte_export.py is a sibling in this scripts/ dir.
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from pte_export import export_functional_pte, trainable_flat

    for tier in args.tiers:
        if tier not in TIER_HIDDEN:
            raise SystemExit(
                f"unknown/unsupported mobile tier '{tier}'. Mobile tiers: {list(TIER_HIDDEN)}. "
                "100M is not a mobile tier (A6 M-H2)."
            )
        torch.manual_seed(args.seed)
        model = MLP(IN_DIM, TIER_HIDDEN[tier], OUT_DIM).eval()
        n = param_count(model)
        scripted = torch.jit.script(model)
        path = os.path.join(args.out, f"model_{tier}.pt")
        scripted.save(path)
        with open(path, "rb") as fh:
            sha = hashlib.sha256(fh.read()).hexdigest()
        print(f"{tier}: params={n:,} file={path} sha256={sha}")

        # Functional .pte for ExecuTorch (all tier params are trainable -> flat input).
        gex = torch.Generator().manual_seed(args.seed)
        x = torch.randn(2, IN_DIM, generator=gex)
        yb = torch.randint(0, OUT_DIM, (2,), generator=gex)
        pte = export_functional_pte(model, (x, yb))
        pte_path = os.path.join(args.out, f"model_{tier}.pte")
        with open(pte_path, "wb") as fh:
            fh.write(pte)
        pte_sha = hashlib.sha256(pte).hexdigest()
        print(f"{tier}: pte_file={pte_path} pte_sha256={pte_sha} flat_dim={trainable_flat(model).numel()}")


if __name__ == "__main__":
    main()
