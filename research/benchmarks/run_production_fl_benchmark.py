"""Production Federated Learning Benchmark on CIFAR-10 with Pre-trained ResNet-18 Backbone.

Demonstrates high-accuracy, production-grade federated learning on real vision data:
  1. Pre-trained ImageNet-1k ResNet-18 frozen feature representation (512-dim).
  2. Dirichlet non-IID client heterogeneity (alpha=0.5, N=20 clients).
  3. Realistic convergence horizon (T=25 communication rounds).
  4. Server-side FedAdam optimizer with momentum (beta1=0.9, beta2=0.99, eta_g=0.1).
  5. Calibrated gradient clipping (C=2.0) and differential privacy.
  6. Compares the 4 policies: Mean512, Projected128, Trim16, and CenterQuarter.

Writes machine-readable JSON results to:
  `research/results/production_fl/production_cifar_benchmark.json`
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

from fedlearn.server.robust_aggregation import adaptive_centered_clip, centered_clip, trimmed_mean
from fedlearn.simulation.partition import dirichlet_partition

PROJECTION_SEED = 20261107


def build_orthonormal_projection(in_dim: int = 512, out_dim: int = 128, seed: int = PROJECTION_SEED) -> torch.Tensor:
    """Generates a deterministic public orthonormal projection matrix P in R^(in_dim x out_dim)."""
    rng = torch.Generator().manual_seed(seed)
    random_mat = torch.randn(in_dim, out_dim, generator=rng, dtype=torch.float32)
    q, _ = torch.linalg.qr(random_mat, mode="reduced")
    return q


def extract_real_cifar10_resnet_features(
    n_train: int = 4000,
    n_test: int = 1000,
    data_root: Optional[str] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extracts 512-dim ImageNet ResNet-18 features from real CIFAR-10 images.
    
    Uses standard ImageNet normalization and 112x112 resizing matching `fl-runtime/recipes.py`.
    """
    import torchvision
    from torchvision import datasets, transforms

    if data_root is None:
        candidates = [
            Path("/Users/anurag/code/FedLearn-Platform/data"),
            REPO_ROOT / "data",
            REPO_ROOT / "fl-runtime" / "data",
            REPO_ROOT / "framework" / "data",
            Path("./data"),
        ]
        for c in candidates:
            if (c / "cifar-10-batches-py").exists():
                data_root = str(c)
                break
        if data_root is None:
            data_root = "/Users/anurag/code/FedLearn-Platform/data"

    tf = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    ds_train = datasets.CIFAR10(root=data_root, train=True, download=False, transform=tf)
    ds_test = datasets.CIFAR10(root=data_root, train=False, download=False, transform=tf)

    sub_train = Subset(ds_train, range(min(n_train, len(ds_train))))
    sub_test = Subset(ds_test, range(min(n_test, len(ds_test))))

    train_loader = DataLoader(sub_train, batch_size=128, shuffle=False)
    test_loader = DataLoader(sub_test, batch_size=128, shuffle=False)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
    resnet = torchvision.models.resnet18(weights=weights)
    resnet.fc = nn.Identity()
    resnet.eval().to(device)

    def _extract(loader):
        feats, labels = [], []
        with torch.no_grad():
            for x, y in loader:
                f = resnet(x.to(device))
                feats.append(f.cpu())
                labels.append(y)
        return torch.cat(feats, dim=0), torch.cat(labels, dim=0)

    train_f, train_y = _extract(train_loader)
    test_f, test_y = _extract(test_loader)
    return train_f, train_y, test_f, test_y


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
    epochs: int = 2,
    lr: float = 0.1,
    momentum: float = 0.9,
    batch_size: int = 32,
    clip_c: float = 2.0,
    seed: int = 42,
) -> torch.Tensor:
    """Trains client locally with SGD momentum and returns clipped parameter update vector."""
    initial_flat = model.get_flat_params().clone()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    criterion = nn.CrossEntropyLoss()

    ds = TensorDataset(features, labels)
    loader_rng = torch.Generator().manual_seed(seed)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, generator=loader_rng)

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


def run_production_experiment(
    policy: str,
    private: bool,
    attacked: bool,
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    test_features: torch.Tensor,
    test_labels: torch.Tensor,
    partitions: List[np.ndarray],
    proj_matrix: Optional[torch.Tensor] = None,
    num_rounds: int = 25,
    clip_c: float = 2.0,
    target_epsilon: Optional[float] = 4.0,
    target_delta: float = 1e-5,
    z_noise: Optional[float] = None,
    center_mode: Optional[str] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """Runs one full production FL training trajectory for a specific policy configuration."""
    num_clients = len(partitions)
    torch_rng = torch.Generator().manual_seed(seed)

    # 1. Feature Representation Setup
    num_classes = int(torch.max(torch.cat([train_labels, test_labels])).item()) + 1
    if policy == "Projected128":
        assert proj_matrix is not None
        dim = proj_matrix.shape[1]
        client_feats = [train_features[idx] @ proj_matrix for idx in partitions]
        eval_feats = test_features @ proj_matrix
    else:
        dim = train_features.shape[1]
        client_feats = [train_features[idx] for idx in partitions]
        eval_feats = test_features

    model = LinearClassifier(in_dim=dim, num_classes=num_classes)
    param_dim = model.get_flat_params().numel()

    # 2. DP Sensitivity & Noise Standard Deviation
    tau = clip_c / 4.0
    effective_center_mode = "none"
    if policy in ("Mean512", "Projected128"):
        sensitivity = (2.0 * clip_c) / num_clients
    elif policy == "Trim16":
        k_trim = int(0.2 * num_clients)
        surviving = num_clients - 2 * k_trim
        sensitivity = (2.0 * clip_c) / max(surviving, 1)
    elif policy in ("CenterQuarter", "CenterQuarter_Origin"):
        effective_center_mode = center_mode or "origin"
        sensitivity = (2.0 * tau) / num_clients
    elif policy == "CenterQuarter_History":
        effective_center_mode = "history"
        sensitivity = (2.0 * tau) / num_clients
    elif policy == "EqualRadiusMean":
        # Equal-radius control: algebraically identical to origin-centered clipping at tau
        effective_center_mode = "origin"
        sensitivity = (2.0 * tau) / num_clients
    else:
        raise ValueError(f"Unknown policy: {policy}")

    accounted_eps = None
    best_order = None
    if private:
        from fedlearn.privacy.dp_accountant import compute_rdp, get_epsilon, required_noise_multiplier
        if z_noise is None:
            assert target_epsilon is not None
            z_eff = float(required_noise_multiplier(target_epsilon, 1.0, num_rounds, target_delta))
        else:
            z_eff = float(z_noise)
        rdp_vals = compute_rdp(1.0, z_eff, num_rounds)
        accounted_eps, best_order = get_epsilon(rdp_vals, target_delta)
        accounted_eps = float(accounted_eps)
        best_order = float(best_order)
        sigma = float(z_eff * sensitivity)
    else:
        z_eff = 0.0
        sigma = 0.0

    # 3. Byzantine Adversaries (20% = 4 clients)
    num_byzantine = int(0.2 * num_clients) if attacked else 0
    byzantine_ids = set(range(num_byzantine))

    # 4. Server FedAdam State
    server_m = torch.zeros(param_dim)
    server_v = torch.zeros(param_dim)
    beta1, beta2, eta_g, eps = 0.9, 0.99, 0.1, 1e-4

    previous_aggregate = torch.zeros(param_dim)
    round_history: List[Dict[str, Any]] = []
    center_norms: List[float] = []

    for r in range(1, num_rounds + 1):
        r_start = time.perf_counter()
        current_global = model.get_flat_params()

        updates: List[torch.Tensor] = []
        for cid in range(num_clients):
            if cid in byzantine_ids:
                # Malicious sign-flipping attack
                attack_dir = -torch.sign(current_global + 1e-5)
                norm = torch.norm(attack_dir, p=2)
                bad_delta = attack_dir * (clip_c / (norm + 1e-9))
                updates.append(bad_delta)
            else:
                client_model = LinearClassifier(in_dim=dim, num_classes=num_classes)
                client_model.set_flat_params(current_global)
                client_delta = train_client_local(
                    client_model,
                    client_feats[cid],
                    train_labels[partitions[cid]],
                    epochs=2,
                    lr=0.1,
                    momentum=0.9,
                    clip_c=clip_c,
                    seed=seed + (cid + 1) * 1000 + r * 17,
                )
                updates.append(client_delta)

        stacked = torch.stack(updates, dim=0)

        # Finite value safety check
        if not torch.isfinite(stacked).all():
            raise ValueError(f"Round {r}: Non-finite client updates encountered in {policy}")

        # Policy Aggregation
        if policy in ("Mean512", "Projected128"):
            agg = stacked.mean(dim=0)
        elif policy == "Trim16":
            agg = trimmed_mean(stacked, trim_ratio=0.2)
        elif policy in ("CenterQuarter", "CenterQuarter_Origin"):
            if effective_center_mode == "origin":
                centre = torch.zeros(param_dim)
            else:
                centre = previous_aggregate
            center_norms.append(float(torch.norm(centre).item()))
            agg = centered_clip(stacked, centre=centre, tau=tau, iterations=1)
        elif policy == "CenterQuarter_History":
            centre = previous_aggregate
            center_norms.append(float(torch.norm(centre).item()))
            agg = centered_clip(stacked, centre=centre, tau=tau, iterations=1)
        elif policy == "EqualRadiusMean":
            # Direct norm-clipped average to radius tau
            norms = torch.clamp(torch.norm(stacked, dim=1, keepdim=True), min=1e-12)
            scales = torch.clamp(tau / norms, max=1.0)
            agg = (stacked * scales).mean(dim=0)
            center_norms.append(0.0)

        # DP Gaussian Noise injection
        if private and sigma > 0:
            noise = torch.randn(param_dim, generator=torch_rng) * sigma
            agg = agg + noise

        # Server-side FedAdam update step
        server_m = beta1 * server_m + (1.0 - beta1) * agg
        server_v = beta2 * server_v + (1.0 - beta2) * (agg ** 2)
        m_hat = server_m / (1.0 - beta1 ** r)
        v_hat = server_v / (1.0 - beta2 ** r)
        server_step = eta_g * m_hat / (torch.sqrt(v_hat) + eps)

        new_global = current_global + server_step
        model.set_flat_params(new_global)
        previous_aggregate = agg.detach().clone()

        r_ms = (time.perf_counter() - r_start) * 1000.0
        if r % 5 == 0 or r == num_rounds:
            eval_loss, eval_acc = evaluate_model(model, eval_feats, test_labels)
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
        "center_mode": effective_center_mode,
        "center_norms_per_round": center_norms,
        "sensitivity": sensitivity,
        "noise_multiplier_z": z_eff if private else None,
        "accounted_epsilon": accounted_eps if private else None,
        "best_order": best_order if private else None,
        "delta": target_delta if private else None,
        "noise_sigma": sigma,
        "param_dimension": param_dim,
        "final_loss": final_loss,
        "final_accuracy": final_acc,
        "round_history": round_history,
    }


def get_git_provenance() -> Dict[str, Any]:
    """Captures git commit hash and working tree status."""
    import subprocess
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=REPO_ROOT).strip()
        status = subprocess.check_output(["git", "status", "--porcelain"], text=True, cwd=REPO_ROOT).strip()
        return {"commit": commit, "dirty": len(status) > 0}
    except Exception as e:
        return {"commit": "unknown", "dirty": True, "error": str(e)}


def compute_partitions_hash(partitions: List[np.ndarray]) -> str:
    """Computes a SHA-256 fingerprint over client partition arrays."""
    import hashlib
    h = hashlib.sha256()
    for p in partitions:
        h.update(p.tobytes())
    return h.hexdigest()


def run_full_production_benchmark() -> Dict[str, Any]:
    print(f"\n{'='*75}")
    print("PRODUCTION FEDERATED LEARNING DUAL-REGIME BENCHMARK (ResNet-18 Backbone)")
    print(f"{'='*75}")

    num_clients = 20
    alpha = 0.5
    num_rounds = 25
    clip_c = 2.0
    seeds = [42, 43, 44]

    print("Extracting ImageNet-1k ResNet-18 features from real CIFAR-10 data...")
    t_feat = time.perf_counter()
    train_feats, train_labels, test_feats, test_labels = extract_real_cifar10_resnet_features(
        n_train=4000, n_test=1000
    )
    print(f"Feature extraction done in {time.perf_counter() - t_feat:.2f}s "
          f"(train={train_feats.shape}, test={test_feats.shape})")

    proj_matrix = build_orthonormal_projection(in_dim=512, out_dim=128, seed=PROJECTION_SEED)

    policies = [
        "Mean512",
        "Projected128",
        "Trim16",
        "CenterQuarter_Origin",
        "CenterQuarter_History",
        "EqualRadiusMean",
    ]

    settings = [
        {"private": False, "attacked": False, "label": "clean_no_noise",       "regime": "control",        "z_noise": 0.0},
        {"private": False, "attacked": True,  "label": "attacked_no_noise",    "regime": "control",        "z_noise": 0.0},
        {"private": True,  "attacked": False, "label": "clean_private_z05",    "regime": "exploratory",    "z_noise": 0.5},
        {"private": True,  "attacked": True,  "label": "attacked_private_z05", "regime": "exploratory",    "z_noise": 0.5},
        {"private": True,  "attacked": False, "label": "clean_private_eps4",   "regime": "certified_dp",   "z_noise": None, "target_epsilon": 4.0},
        {"private": True,  "attacked": True,  "label": "attacked_private_eps4",  "regime": "certified_dp",   "z_noise": None, "target_epsilon": 4.0},
    ]

    all_results: List[Dict[str, Any]] = []
    out_file = REPO_ROOT / "research" / "results" / "production_fl" / "production_cifar_dual_regime_benchmark.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)

    git_meta = get_git_provenance()

    for seed in seeds:
        print(f"\n>>> Running Seed {seed} <<<")
        partitions = dirichlet_partition(
            labels=train_labels.numpy(),
            num_clients=num_clients,
            alpha=alpha,
            seed=seed,
            min_partition_size=10,
        )
        part_hash = compute_partitions_hash(partitions)

        for setting in settings:
            for pol in policies:
                t0 = time.perf_counter()
                res = run_production_experiment(
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
                    target_epsilon=setting.get("target_epsilon", 4.0),
                    target_delta=1e-5,
                    z_noise=setting["z_noise"],
                    seed=seed,
                )
                res["seed"] = seed
                res["setting_label"] = setting["label"]
                res["regime"] = setting["regime"]
                res["partition_hash"] = part_hash
                elapsed = time.perf_counter() - t0
                all_results.append(res)
                print(
                    f"  [{setting['label']:<20}] {pol:<22} -> "
                    f"Acc: {res['final_accuracy']*100:.2f}% | "
                    f"Loss: {res['final_loss']:.4f} ({elapsed:.2f}s)"
                )

                # Incremental checkpointing
                partial_report = {
                    "meta": {
                        "benchmark": "Production Federated Learning (ResNet-18 ImageNet Backbone)",
                        "dataset": "CIFAR-10 real images (112x112, ImageNet normalized)",
                        "num_clients": num_clients,
                        "dirichlet_alpha": alpha,
                        "num_rounds": num_rounds,
                        "clip_norm_C": clip_c,
                        "git_provenance": git_meta,
                        "num_results_so_far": len(all_results),
                    },
                    "results": all_results,
                }
                out_file.write_text(json.dumps(partial_report, indent=2, allow_nan=False) + "\n")

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

    print(f"\n{'='*95}")
    print("PRODUCTION FL SUMMARY MATRIX (Mean Test Accuracy across 3 seeds)")
    print(f"{'='*95}")
    print(f"{'Policy':<22} | {'Clean':<8} | {'Attacked':<8} | {'Clean(z=0.5)':<12} | {'Att(z=0.5)':<12} | {'Clean(eps=4)':<12} | {'Att(eps=4)':<12}")
    print("-" * 95)
    for pol in policies:
        s = summary_by_policy[pol]
        print(
            f"{pol:<22} | "
            f"{s['clean_no_noise']*100:6.2f}% | "
            f"{s['attacked_no_noise']*100:6.2f}% | "
            f"{s['clean_private_z05']*100:10.2f}% | "
            f"{s['attacked_private_z05']*100:10.2f}% | "
            f"{s['clean_private_eps4']*100:10.2f}% | "
            f"{s['attacked_private_eps4']*100:10.2f}%"
        )
    print(f"{'='*95}\n")

    report: Dict[str, Any] = {
        "meta": {
            "benchmark": "Production Federated Learning (ResNet-18 ImageNet Backbone)",
            "dataset": "CIFAR-10 real images (112x112, ImageNet normalized, first 4000 train / 1000 test)",
            "num_clients": num_clients,
            "dirichlet_alpha": alpha,
            "num_rounds": num_rounds,
            "clip_norm_C": clip_c,
            "server_optimizer": "FedAdam (beta1=0.9, beta2=0.99, eta_g=0.1)",
            "regimes": {
                "exploratory": {"z_noise": 0.5, "accountant_nominal_eps": 98.02585, "delta": 1e-5},
                "certified_dp": {"target_epsilon": 4.0, "delta": 1e-5, "accountant_z": 6.48041},
            },
            "seeds": seeds,
            "git_provenance": git_meta,
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

    out_file.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Saved complete production dual-regime benchmark results to: {out_file}")

    return report


if __name__ == "__main__":
    run_full_production_benchmark()
