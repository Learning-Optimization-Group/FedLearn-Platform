"""TE-15 — fair, apples-to-apples algorithm comparison on one task (the professor's ask).

Runs the same task, same fixed non-IID partition, same seed through each algorithm's REAL
`aggregate_fit`, and records per round: test accuracy, loss, wall-clock, and TRUTHFUL cumulative
bytes on the wire (via `benchmarks.wire_bytes`, measuring the actual serialized payloads — the one
axis the platform never measured). Emits the professor's table plus JSON/MD artifacts.

Scope of this file: the first-order family (FedAvg / FedProx / FedOpt), which share the local-SGD
client loop — FedProx adds the proximal term `mu*(w - w_global)`, FedOpt does its adaptive step
server-side. DeComFL is zeroth-order (a different client loop) — its per-round BYTE cost is included
analytically here (seeds+scalars are deterministic), and its convergence curve is produced by the
existing gRPC harness (`run_full_test_suite.py` Test 3) over thousands of rounds; see `--help`.

Fairness policy (declared, not fit to outcome): task/model/partition/seed/batch/local-epochs are
shared; only the learning rate is tuned per algorithm (published in the record). Bytes are protobuf
payload bytes before HTTP/2 framing (identical ~1% across algorithms); DeComFL's one-shot O(d)
initial model download is reported separately, mirroring the DeComFL paper's own accounting.
"""
import argparse
import json
import os
import sys
import time
from collections import OrderedDict
from typing import Dict, List, Optional

# Make the harness runnable as a script (`python benchmarks/algo_comparison.py`) without env setup:
# put the framework root (for `benchmarks.*`) and src (for `fedlearn.*`) on the path.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from fedlearn.server.strategy import FedAvg, FedProx, FedOpt
from benchmarks.wire_bytes import first_order_model_bytes, decomfl_upload_bytes, decomfl_download_config_bytes

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

FIRST_ORDER = {"fedavg", "fedprox", "fedopt"}


# --------------------------------------------------------------------------------------------------
# Models / tasks
# --------------------------------------------------------------------------------------------------
class SmallCNN(nn.Module):
    """A compact conv net (~a few 10k params) for the image task — small enough for ZO to have a
    fair shot (ZO variance scales with d), matching the DeComFL paper's small-CNN MNIST setup."""

    def __init__(self, in_ch: int = 1, num_classes: int = 10, side: int = 28):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.head = nn.Linear(16 * (side // 4) * (side // 4), num_classes)

    def forward(self, x):
        return self.head(torch.flatten(self.conv(x), 1))


class TinyMLP(nn.Module):
    """A tiny MLP for the offline synthetic smoke task (no BatchNorm/buffers → clean state_dict)."""

    def __init__(self, dim: int, hidden: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, num_classes))

    def forward(self, x):
        return self.net(x)


def make_synthetic(num_classes=4, dim=20, per_class=120, sep=2.0, seed=0):
    """Offline separable-Gaussian classification — lets the harness be smoke-tested with no network
    or dataset download. The real task is MNIST (--task mnist)."""
    gen = torch.Generator().manual_seed(seed)
    xs, ys = [], []
    centers = torch.randn(num_classes, dim, generator=gen) * sep
    for c in range(num_classes):
        xs.append(centers[c] + torch.randn(per_class, dim, generator=gen))
        ys.append(torch.full((per_class,), c, dtype=torch.long))
    x = torch.cat(xs); y = torch.cat(ys)
    perm = torch.randperm(len(x), generator=gen)
    x, y = x[perm], y[perm]
    n_test = len(x) // 5
    return x[n_test:], y[n_test:], x[:n_test], y[:n_test], {"kind": "mlp", "dim": dim, "num_classes": num_classes}


def load_mnist(root: Optional[str] = None):
    """Full MNIST (downloads on first use). The real task for the professor's table."""
    from torchvision import datasets, transforms
    root = root or os.path.join(RESULTS_DIR, "_data")
    tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    tr = datasets.MNIST(root, train=True, download=True, transform=tfm)
    te = datasets.MNIST(root, train=False, download=True, transform=tfm)
    train_x = torch.stack([tr[i][0] for i in range(len(tr))])
    train_y = torch.tensor([tr[i][1] for i in range(len(tr))])
    test_x = torch.stack([te[i][0] for i in range(len(te))])
    test_y = torch.tensor([te[i][1] for i in range(len(te))])
    return train_x, train_y, test_x, test_y, {"kind": "cnn", "side": 28, "num_classes": 10}


def build_model(meta) -> nn.Module:
    if meta["kind"] == "cnn":
        return SmallCNN(num_classes=meta["num_classes"], side=meta["side"])
    return TinyMLP(meta["dim"], 32, meta["num_classes"])


# --------------------------------------------------------------------------------------------------
# Partition / eval
# --------------------------------------------------------------------------------------------------
def dirichlet_partition(labels: torch.Tensor, num_clients: int, alpha: float, seed: int) -> List[List[int]]:
    """Fixed non-IID label partition, identical across algorithms (seeded generator, not global RNG)."""
    gen = torch.Generator().manual_seed(seed)
    num_classes = int(labels.max().item()) + 1
    idx_by_class = [torch.where(labels == c)[0] for c in range(num_classes)]
    client_idx: List[List[int]] = [[] for _ in range(num_clients)]
    for c in range(num_classes):
        idx = idx_by_class[c][torch.randperm(len(idx_by_class[c]), generator=gen)]
        props = torch.distributions.Dirichlet(torch.full((num_clients,), alpha)).sample()
        cuts = (torch.cumsum(props, 0) * len(idx)).long()[:-1]
        for cid, part in enumerate(torch.tensor_split(idx, cuts)):
            client_idx[cid].extend(part.tolist())
    return client_idx


@torch.no_grad()
def evaluate(model: nn.Module, params: OrderedDict, test_x, test_y, batch=512) -> tuple:
    model.load_state_dict(OrderedDict((k, v.clone()) for k, v in params.items()))
    model.eval()
    crit = nn.CrossEntropyLoss(reduction="sum")
    correct, loss_sum, n = 0, 0.0, len(test_x)
    for i in range(0, n, batch):
        xb, yb = test_x[i:i + batch], test_y[i:i + batch]
        out = model(xb)
        loss_sum += float(crit(out, yb))
        correct += int((out.argmax(1) == yb).sum())
    return correct / n, loss_sum / n


# --------------------------------------------------------------------------------------------------
# One algorithm run
# --------------------------------------------------------------------------------------------------
def _build_strategy(algo: str, initial: OrderedDict, num_clients: int):
    if algo == "fedavg":
        return FedAvg(initial_parameters=initial, min_fit_clients=num_clients)
    if algo == "fedprox":
        return FedProx(initial_parameters=initial, min_fit_clients=num_clients)
    if algo == "fedopt":
        return FedOpt(initial_parameters=initial, min_fit_clients=num_clients)
    raise ValueError(f"algo {algo!r} is not a first-order strategy handled by this harness")


def run_algorithm(algo: str, data, *, num_clients=8, rounds=20, local_epochs=1, lr=0.05,
                  batch_size=32, alpha=1.0, proximal_mu=0.1, seed=0) -> Dict:
    """Run `algo` to `rounds` on the shared task; return the per-round metric records."""
    train_x, train_y, test_x, test_y, meta = data
    torch.manual_seed(seed)
    model = build_model(meta)
    initial = OrderedDict((k, v.detach().clone()) for k, v in model.state_dict().items())
    d = sum(v.numel() for v in initial.values())

    strategy = _build_strategy(algo, OrderedDict((k, v.clone()) for k, v in initial.items()), num_clients)
    global_params = strategy.initialize_parameters()

    parts = dirichlet_partition(train_y, num_clients, alpha, seed)
    loaders = []
    for cid in range(num_clients):
        idx = torch.tensor(parts[cid], dtype=torch.long)
        ds = TensorDataset(train_x[idx], train_y[idx])
        loaders.append((len(idx), DataLoader(ds, batch_size=batch_size, shuffle=True)))

    records: List[dict] = []
    cum_bytes = 0
    net = build_model(meta)
    crit = nn.CrossEntropyLoss()
    t0 = time.monotonic()

    for rnd in range(rounds):
        updates = []
        for cid, (n, loader) in enumerate(loaders):
            if n == 0:
                continue
            net.load_state_dict(OrderedDict((k, v.clone()) for k, v in global_params.items()))
            net.train()
            opt = torch.optim.SGD(net.parameters(), lr=lr)
            anchor = {k: v.detach().clone() for k, v in net.state_dict().items()} if algo == "fedprox" else None
            for _ in range(local_epochs):
                for xb, yb in loader:
                    opt.zero_grad()
                    loss = crit(net(xb), yb)
                    loss.backward()
                    if algo == "fedprox":  # proximal term: grad += mu*(w - w_global)
                        for name, p in net.named_parameters():
                            if p.grad is not None:
                                p.grad.add_(proximal_mu * (p.detach() - anchor[name]))
                    opt.step()
            state = OrderedDict((k, v.detach().clone().float()) for k, v in net.state_dict().items())
            updates.append((str(cid), state, n))
            # Truthful wire bytes: this client uploaded its model AND downloaded the global this round.
            cum_bytes += first_order_model_bytes(state, n) + first_order_model_bytes(global_params)

        aggregated = strategy.aggregate_fit(rnd, updates)
        if aggregated is not None:
            global_params = aggregated
        acc, loss = evaluate(net, global_params, test_x, test_y)
        records.append({
            "round": rnd + 1, "accuracy": round(acc, 4), "loss": round(loss, 4),
            "wall_s": round(time.monotonic() - t0, 2), "cum_bytes": cum_bytes,
        })
    return {"algo": algo, "lr": lr, "d": d, "num_clients": num_clients, "records": records}


def decomfl_byte_projection(d: int, rounds: int, num_clients: int, K=1, P=10) -> Dict:
    """Analytic per-round byte cost of DeComFL on the same model (seeds+scalars only, plus the
    one-shot O(d) initial model download reported separately). Convergence is a separate overnight
    run — this quantifies only the communication axis, which is DeComFL's entire claim."""
    up = decomfl_upload_bytes(K, P)
    down = decomfl_download_config_bytes(K, P)
    per_round = num_clients * (up + down)
    one_shot = num_clients * first_order_model_bytes(OrderedDict([("w", torch.zeros(d, dtype=torch.float32))]))
    return {"algo": "decomfl", "K": K, "P": P, "per_round_bytes": per_round,
            "cum_bytes_at_rounds": per_round * rounds, "one_shot_download_bytes": one_shot}


# --------------------------------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------------------------------
def rounds_to_target(records: List[dict], target: float) -> Optional[int]:
    for r in records:
        if r["accuracy"] >= target:
            return r["round"]
    return None


def render_markdown(runs: List[Dict], decomfl: Dict, target: float) -> str:
    lines = ["# Algorithm comparison — fair, same-task", ""]
    lines.append(f"| algorithm | d | lr | rounds→{int(target*100)}% | final acc | bytes→{int(target*100)}% | bytes @ final |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for run in runs:
        recs = run["records"]
        r2t = rounds_to_target(recs, target)
        b2t = next((r["cum_bytes"] for r in recs if r["accuracy"] >= target), None)
        fin = recs[-1]
        lines.append(f"| {run['algo']} | {run['d']:,} | {run['lr']} | "
                     f"{r2t if r2t else '—'} | {fin['accuracy']:.3f} | "
                     f"{b2t if b2t else '—':,} | {fin['cum_bytes']:,} |".replace("—:,", "—"))
    lines.append("")
    lines.append(f"**DeComFL byte projection (same model, K={decomfl['K']}, P={decomfl['P']}):** "
                 f"{decomfl['per_round_bytes']:,} bytes/round (all clients), "
                 f"{decomfl['cum_bytes_at_rounds']:,} over {len(runs[0]['records'])} rounds, "
                 f"plus a one-shot {decomfl['one_shot_download_bytes']:,}-byte model download. "
                 f"Convergence curve: run the gRPC DeComFL harness overnight (see module docstring).")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=["synthetic", "mnist"], default="mnist")
    ap.add_argument("--algos", default="fedavg,fedprox,fedopt")
    ap.add_argument("--rounds", type=int, default=100)
    ap.add_argument("--clients", type=int, default=8)
    ap.add_argument("--local-epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--target", type=float, default=0.9)
    ap.add_argument("--out", default=os.path.join(RESULTS_DIR, "algo_comparison"))
    args = ap.parse_args()

    data = make_synthetic(seed=args.seed) if args.task == "synthetic" else load_mnist()
    runs = [run_algorithm(a.strip(), data, num_clients=args.clients, rounds=args.rounds,
                          local_epochs=args.local_epochs, lr=args.lr, alpha=args.alpha, seed=args.seed)
            for a in args.algos.split(",") if a.strip() in FIRST_ORDER]
    decomfl = decomfl_byte_projection(runs[0]["d"], args.rounds, args.clients)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    payload = {"task": args.task, "target": args.target, "runs": runs, "decomfl_bytes": decomfl}
    with open(args.out + ".json", "w") as f:
        json.dump(payload, f, indent=2)
    md = render_markdown(runs, decomfl, args.target)
    with open(args.out + ".md", "w") as f:
        f.write(md)
    print(md)


if __name__ == "__main__":
    main()
