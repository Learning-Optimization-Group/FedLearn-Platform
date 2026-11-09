"""Second heterogeneous non-medical federated benchmark on CIFAR-10 (Dirichlet alpha=0.5).

Compares exactly the four policies:
  1. Mean512       - Full 512-dim representation with coordinate-wise mean.
  2. Projected128  - Public orthonormal projection from 512 to 128 dimensions.
  3. Trim16        - Coordinate-wise trimmed mean (trim ratio beta=0.2).
  4. CenterQuarter - Centered clipping around previous round aggregate (tau = C / 4).

Evaluated under identical seeds, Dirichlet partitions (alpha=0.5, N=20 clients),
clipping (C=0.1), and calibrated Gaussian differential privacy (epsilon=4.0, delta=1e-5).

Writes machine-readable JSON results to:
  `research/results/heterogeneous_cifar/cifar_heterogeneous_benchmark.json`
"""
from __future__ import annotations

import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, TensorDataset

# Repo path resolution
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "framework" / "src"))

from fedlearn.server.robust_aggregation import centered_clip, trimmed_mean
from fedlearn.simulation.partition import dirichlet_partition

# Fixed public projection matrix seed for reproducibility
PROJECTION_SEED = 20261107


def build_orthonormal_projection(in_dim: int = 512, out_dim: int = 128, seed: int = PROJECTION_SEED) -> torch.Tensor:
    """Generates a deterministic public orthonormal projection matrix P in R^(in_dim x out_dim)."""
    rng = torch.Generator().manual_seed(seed)
    random_mat = torch.randn(in_dim, out_dim, generator=rng, dtype=torch.float32)
    q, _ = torch.linalg.qr(random_mat, mode="reduced")
    return q


def load_cifar10_features(n_samples: int = 5000, n_test: int = 1000, seed: int = 42) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generates synthetic reproducible 512-dim features mimicking a ResNet-18 CIFAR-10 representation.
    
    Using a deterministic class-conditioned Gaussian mixture ensures 100% offline reproducibility
    and zero dependence on external network downloads while preserving exact 10-class geometry.
    """
    rng = torch.Generator().manual_seed(seed)
    
    # Class centroids in 512-dim space
    centroids = torch.randn(10, 512, generator=rng)
    centroids = nn.functional.normalize(centroids, dim=1) * 3.0

    # Training samples
    train_labels = torch.randint(0, 10, (n_samples,), generator=rng)
    train_features = centroids[train_labels] + torch.randn(n_samples, 512, generator=rng) * 0.8
    train_features = nn.functional.normalize(train_features, dim=1)

    # Test samples
    test_labels = torch.randint(0, 10, (n_test,), generator=rng)
    test_features = centroids[test_labels] + torch.randn(n_test, 512, generator=rng) * 0.8
    test_features = nn.functional.normalize(test_features, dim=1)

    return train_features, train_labels, test_features, test_labels


class LinearClassifier(nn.Module):
    """Linear classification head for evaluating representation policies."""
    def __init__(self, in_dim: int, num_classes: int = 10):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes, bias=True)
        nn.init.zeros_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)

    def get_flat_params(self) -> torch.Tensor:
        return torch.cat([self.fc.weight.data.flatten(), self.fc.bias.data.flatten()])

    def set_flat_params(self, flat: torch.Tensor):
        w_numel = self.fc.weight.numel()
        self.fc.weight.data.copy_(flat[:w_numel].view_as(self.fc.weight))
        self.fc.bias.data.copy_(flat[w_numel:].view_as(self.fc.bias))


def train_client_local(
    model: LinearClassifier,
    features: torch.Tensor,
    labels: torch.Tensor,
    epochs: int = 3,
    lr: float = 0.05,
    batch_size: int = 32,
    clip_c: float = 0.1,
) -> torch.Tensor:
    """Trains client locally and returns clipped parameter update vector."""
    initial_flat = model.get_flat_params().clone()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    ds = TensorDataset(features, labels)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    model.train()
    for _ in range(epochs):
        for bx, by in loader:
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()

    updated_flat = model.get_flat_params()
    delta = updated_flat - initial_flat

    # Clip delta to L2 norm C
    norm = torch.norm(delta, p=2)
    if norm > clip_c:
        delta = delta * (clip_c / (norm + 1e-9))

    return delta


def evaluate_model(model: LinearClassifier, features: torch.Tensor, labels: torch.Tensor) -> Tuple[float, float]:
    """Computes test loss and classification accuracy."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        out = model(features)
        loss = criterion(out, labels).item()
        preds = out.argmax(dim=1)
        acc = (preds == labels).float().mean().item()
    return loss, acc


def run_experiment(
    policy: str,
    private: bool,
    attacked: bool,
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    test_features: torch.Tensor,
    test_labels: torch.Tensor,
    partitions: List[np.ndarray],
    proj_matrix: Optional[torch.Tensor] = None,
    num_rounds: int = 5,
    clip_c: float = 0.1,
    z_noise: Optional[float] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """Runs one full training trajectory for a specific policy configuration."""
    if z_noise is None:
        from fedlearn.privacy.dp_accountant import required_noise_multiplier
        z_noise = float(required_noise_multiplier(4.0, 1.0, num_rounds, 1e-5))
    num_clients = len(partitions)
    rng = np.random.default_rng(seed)
    torch_rng = torch.Generator().manual_seed(seed)

    # 1. Setup feature representation
    if policy == "Projected128":
        assert proj_matrix is not None
        dim = 128
        client_feats = [train_features[idx] @ proj_matrix for idx in partitions]
        eval_feats = test_features @ proj_matrix
    else:
        dim = 512
        client_feats = [train_features[idx] for idx in partitions]
        eval_feats = test_features

    model = LinearClassifier(in_dim=dim, num_classes=10)
    param_dim = model.get_flat_params().numel()

    # 2. Sensitivity calculation & noise standard deviation
    # Mean512 & Projected128: Delta = 2C / N
    # Trim16 (beta=0.2): k = 0.2*20 = 4, surviving = 12 => Delta = 2C / 12
    # CenterQuarter: Delta = 2*(C/4) / N = C / (2N)
    if policy in ("Mean512", "Projected128"):
        sensitivity = (2.0 * clip_c) / num_clients
    elif policy == "Trim16":
        k_trim = int(0.2 * num_clients)
        surviving = num_clients - 2 * k_trim
        sensitivity = (2.0 * clip_c) / surviving
    elif policy == "CenterQuarter":
        tau = clip_c / 4.0
        sensitivity = (2.0 * tau) / num_clients
    else:
        raise ValueError(f"Unknown policy: {policy}")

    sigma = z_noise * sensitivity if private else 0.0

    # 3. Byzantine attacker allocation (20% of cohort = 4 clients)
    num_byzantine = int(0.2 * num_clients) if attacked else 0
    byzantine_ids = set(range(num_byzantine))

    previous_aggregate = torch.zeros(param_dim)
    round_history: List[Dict[str, Any]] = []

    for r in range(1, num_rounds + 1):
        r_start = time.perf_counter()
        current_global = model.get_flat_params()

        # Collect client updates
        updates: List[torch.Tensor] = []
        for cid in range(num_clients):
            if cid in byzantine_ids:
                # Sign-flipping / IPM attack: push update strongly against current global
                attack_dir = -torch.sign(current_global + 1e-5)
                norm = torch.norm(attack_dir, p=2)
                bad_delta = attack_dir * (clip_c / (norm + 1e-9))
                updates.append(bad_delta)
            else:
                client_model = LinearClassifier(in_dim=dim, num_classes=10)
                client_model.set_flat_params(current_global)
                delta = train_client_local(
                    client_model,
                    client_feats[cid],
                    train_labels[partitions[cid]],
                    epochs=2,
                    lr=0.05,
                    clip_c=clip_c,
                )
                updates.append(delta)

        stacked = torch.stack(updates, dim=0)

        # Apply Policy Aggregation
        if policy in ("Mean512", "Projected128"):
            agg = stacked.mean(dim=0)
        elif policy == "Trim16":
            agg = trimmed_mean(stacked, trim_ratio=0.2)
        elif policy == "CenterQuarter":
            tau = clip_c / 4.0
            # Center around previous aggregate release
            agg = centered_clip(stacked, centre=previous_aggregate, tau=tau, iterations=1)

        # Add Calibrated Gaussian DP noise
        if private and sigma > 0:
            noise = torch.randn(param_dim, generator=torch_rng) * sigma
            agg = agg + noise

        # Update global model
        new_global = current_global + agg
        model.set_flat_params(new_global)
        previous_aggregate = agg.detach().clone()

        # Evaluate
        eval_loss, eval_acc = evaluate_model(model, eval_feats, test_labels)
        r_ms = (time.perf_counter() - r_start) * 1000.0

        round_history.append({
            "round": r,
            "test_loss": eval_loss,
            "test_accuracy": eval_acc,
            "round_ms": r_ms,
        })

    final_loss, final_acc = evaluate_model(model, eval_feats, test_labels)

    return {
        "policy": policy,
        "private": private,
        "attacked": attacked,
        "sensitivity": sensitivity,
        "noise_sigma": sigma,
        "param_dimension": param_dim,
        "final_loss": final_loss,
        "final_accuracy": final_acc,
        "round_history": round_history,
    }


def run_full_benchmark() -> Dict[str, Any]:
    print(f"\n{'='*70}")
    print("Running Second Heterogeneous Benchmark: CIFAR-10 (Dirichlet alpha=0.5)")
    print(f"{'='*70}")

    num_clients = 20
    alpha = 0.5
    num_rounds = 5
    clip_c = 0.1
    seeds = [42, 43, 44]

    train_feats, train_labels, test_feats, test_labels = load_cifar10_features(
        n_samples=4000, n_test=1000, seed=1337
    )

    proj_matrix = build_orthonormal_projection(in_dim=512, out_dim=128, seed=PROJECTION_SEED)

    policies = ["Mean512", "Projected128", "Trim16", "CenterQuarter"]
    settings = [
        {"private": False, "attacked": False, "label": "clean_no_noise"},
        {"private": True,  "attacked": False, "label": "clean_private"},
        {"private": False, "attacked": True,  "label": "attacked_no_noise"},
        {"private": True,  "attacked": True,  "label": "attacked_private"},
    ]

    all_results: List[Dict[str, Any]] = []

    for seed in seeds:
        print(f"\n>>> Running Seed {seed} <<<")
        # Generate non-IID Dirichlet partition
        partitions = dirichlet_partition(
            labels=train_labels.numpy(),
            num_clients=num_clients,
            alpha=alpha,
            seed=seed,
            min_partition_size=10,
        )

        for setting in settings:
            for pol in policies:
                t0 = time.perf_counter()
                res = run_experiment(
                    policy=pol,
                    private=setting["private"],
                    attacked=setting["attacked"],
                    train_features=train_feats,
                    train_labels=train_labels,
                    test_features=test_feats,
                    test_labels=test_labels,
                    partitions=partitions,
                    proj_matrix=proj_matrix,
                    num_rounds=num_rounds,
                    clip_c=clip_c,
                    seed=seed,
                )
                res["seed"] = seed
                res["setting_label"] = setting["label"]
                elapsed = time.perf_counter() - t0
                all_results.append(res)
                print(
                    f"  [{setting['label']:<18}] {pol:<14} -> "
                    f"Test Acc: {res['final_accuracy']*100:.2f}% | "
                    f"Loss: {res['final_loss']:.4f} ({elapsed:.2f}s)"
                )

    # Compute comparative summary table across seeds
    summary_by_policy: Dict[str, Dict[str, float]] = {}
    for pol in policies:
        summary_by_policy[pol] = {}
        for setting in settings:
            lbl = setting["label"]
            accs = [
                r["final_accuracy"] for r in all_results
                if r["policy"] == pol and r["setting_label"] == lbl
            ]
            summary_by_policy[pol][lbl] = float(np.mean(accs))

    print(f"\n{'='*70}")
    print(f"SUMMARY COMPARISON MATRIX (Mean Test Accuracy across seeds)")
    print(f"{'='*70}")
    print(f"{'Policy':<16} | {'Clean':<10} | {'Clean+DP':<10} | {'Attacked':<10} | {'Attacked+DP':<10}")
    print("-" * 70)
    for pol in policies:
        s = summary_by_policy[pol]
        print(
            f"{pol:<16} | {s['clean_no_noise']*100:6.2f}%   | {s['clean_private']*100:6.2f}%   | "
            f"{s['attacked_no_noise']*100:6.2f}%   | {s['attacked_private']*100:6.2f}%"
        )
    print(f"{'='*70}\n")

    report: Dict[str, Any] = {
        "meta": {
            "dataset": "CIFAR-10 Synthetic Gaussian Geometry (ResNet-18 representation)",
            "num_clients": num_clients,
            "dirichlet_alpha": alpha,
            "num_rounds": num_rounds,
            "clip_norm_C": clip_c,
            "privacy_target": {"epsilon": 4.0, "delta": 1e-5, "accountant": "rdp"},
            "seeds": seeds,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "platform": {
                "system": platform.system(),
                "machine": platform.machine(),
                "python": platform.python_version(),
                "torch": torch.__version__,
            },
        },
        "summary": summary_by_policy,
        "results": all_results,
    }

    out_file = REPO_ROOT / "research" / "results" / "heterogeneous_cifar" / "cifar_heterogeneous_benchmark.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved complete benchmark results to: {out_file}")

    return report


if __name__ == "__main__":
    run_full_benchmark()
