"""DA-14 Ph3.0/Ph3.3 — frozen-backbone head-only federated learning: communication + utility.

Transfer-learning-style FL where a frozen backbone is shared across all peers and only a small
trainable head is federated (the DA-11 subset-federation contract). This measures the two axes that
make it attractive, honestly:

* COMMUNICATION — the per-round wire payload is the HEAD, never the full model. Measured with the
  SAME production safetensors codec used on the socket (benchmarks.wire_bytes.first_order_model_bytes),
  so the head-vs-full-model byte ratio is a REAL wire measurement, not an estimate. The win grows
  with backbone size (freezing a big backbone is exactly when it pays off).
* UTILITY — a real multi-round head-only FedAvg run converges: the federated head learns a fixed
  target defined over the frozen backbone's features (local SGD -> the real FedAvgAggregator ->
  apply_trainable_subset each round), while the frozen backbone is byte-identical throughout and
  never rides the wire.

Honest framing (for the paper): the utility task is a SEEDED SYNTHETIC separable target (a linear
rule over the frozen features), chosen so a linear head CAN fit it — it demonstrates the mechanism
and the comms/utility trade-off, not a production accuracy on real data. The COMMUNICATION ratios
are real production wire bytes and stand on their own. Reproduce:
    cd framework && PYTHONPATH=src python benchmarks/frozen_backbone_fl.py
"""
import argparse
import json
import os
import sys
from collections import OrderedDict

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
import torch.nn as nn

from fedlearn.estimators.params import trainable_state
from fedlearn.server.strategy import FedAvgAggregator
from fedlearn.server.subset_federation import apply_trainable_subset, guard_client_updates
from benchmarks.wire_bytes import first_order_model_bytes

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


class _Derived(nn.Module):
    """A frozen Linear backbone + a trainable Linear head (the derived-model shape)."""

    def __init__(self, d_in, d_hidden, n_classes):
        super().__init__()
        self.backbone = nn.Linear(d_in, d_hidden)
        self.head = nn.Linear(d_hidden, n_classes)
        for p in self.backbone.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        return self.head(torch.relu(self.backbone(x)))


def _build(d_in, d_hidden, n_classes, seed=0):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return _Derived(d_in, d_hidden, n_classes)


def head_wire_win(sizes):
    """For each (d_in, d_hidden, n_classes): the head-only vs full-model per-round wire bytes, using
    the production safetensors codec. Returns [{size, full_bytes, head_bytes, ratio}]."""
    out = []
    for (d_in, d_hidden, n_classes) in sizes:
        m = _build(d_in, d_hidden, n_classes)
        full = first_order_model_bytes(OrderedDict(m.state_dict()))
        head = first_order_model_bytes(trainable_state(m))
        out.append({"size": [d_in, d_hidden, n_classes], "full_bytes": full,
                    "head_bytes": head, "ratio": round(full / head, 2)})
    return out


def run_head_federation(rounds=15, clients=3, d_in=256, d_hidden=128, n_classes=3, seed=0):
    """A real head-only FedAvg run on a seeded synthetic target. Returns convergence + invariants."""
    torch.manual_seed(seed)
    base = _build(d_in, d_hidden, n_classes, seed=seed)          # the shared frozen backbone
    gt = nn.Linear(d_hidden, n_classes)                          # fixed target over frozen features
    n = 120 * clients
    X = torch.randn(n, d_in)
    with torch.no_grad():
        y = gt(torch.relu(base.backbone(X))).argmax(1)
    split = 100 * clients
    Xtr, ytr, Xte, yte = X[:split], y[:split], X[split:], y[split:]
    parts = [(Xtr[i::clients], ytr[i::clients]) for i in range(clients)]

    def peer():
        m = _Derived(d_in, d_hidden, n_classes)
        m.load_state_dict(base.state_dict())                    # every peer shares the exact backbone
        for p in m.backbone.parameters():
            p.requires_grad_(False)
        return m

    server = peer()
    nets = [peer() for _ in range(clients)]
    backbone0 = server.backbone.weight.detach().clone()

    def accuracy(m):
        m.eval()
        with torch.no_grad():
            return (m(Xte).argmax(1) == yte).float().mean().item()

    def train_head(m, cx, cy, epochs=5, lr=0.5):
        opt = torch.optim.SGD([p for p in m.parameters() if p.requires_grad], lr=lr)
        loss_fn = nn.CrossEntropyLoss()
        m.train()
        for _ in range(epochs):
            opt.zero_grad()
            loss_fn(m(cx), cy).backward()
            opt.step()

    initial_acc = accuracy(server)
    wire_head_only = True
    per_round = []
    for _ in range(rounds):
        global_head = trainable_state(server)
        updates = []
        for i, (m, (cx, cy)) in enumerate(zip(nets, parts)):
            apply_trainable_subset(m, global_head)
            train_head(m, cx, cy)
            u = trainable_state(m)
            if set(u.keys()) != {"head.weight", "head.bias"}:
                wire_head_only = False
            updates.append((f"c{i}", u, cx.shape[0]))
        guard_client_updates([u for _, u, _ in updates], server)
        apply_trainable_subset(server, FedAvgAggregator().aggregate(updates))
        per_round.append(round(accuracy(server), 4))

    return {
        "rounds": rounds, "clients": clients,
        "initial_acc": round(initial_acc, 4), "final_acc": round(accuracy(server), 4),
        "per_round_acc": per_round,
        "backbone_unchanged": bool(torch.equal(server.backbone.weight.detach(), backbone0)),
        "wire_is_head_only": wire_head_only,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rounds", type=int, default=15)
    ap.add_argument("--clients", type=int, default=3)
    ap.add_argument("--out", default=os.path.join(RESULTS_DIR, "frozen_backbone_fl"))
    args = ap.parse_args()

    sizes = [(64, 32, 3), (256, 128, 3), (1024, 512, 10), (4096, 1024, 100)]
    comms = head_wire_win(sizes)
    util = run_head_federation(rounds=args.rounds, clients=args.clients)
    payload = {"communication": comms, "utility": util}

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(args.out + ".json", "w") as f:
        json.dump(payload, f, indent=2)
    with open(args.out + ".md", "w") as f:
        f.write("# Frozen-backbone head-only FL — communication + utility\n\n")
        f.write("## Communication (real safetensors wire bytes, per round)\n\n")
        f.write("| backbone d_in->d_hidden->classes | full model | head only | win |\n|---|---|---|---|\n")
        for c in comms:
            s = c["size"]
            f.write(f"| {s[0]}->{s[1]}->{s[2]} | {c['full_bytes']:,} B | {c['head_bytes']:,} B | {c['ratio']}x |\n")
        f.write(f"\n## Utility (seeded synthetic target)\n\n"
                f"{util['clients']} clients x {util['rounds']} rounds of head-only FedAvg: "
                f"accuracy {util['initial_acc']} -> {util['final_acc']}; "
                f"frozen backbone unchanged: {util['backbone_unchanged']}; "
                f"wire head-only every round: {util['wire_is_head_only']}.\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
