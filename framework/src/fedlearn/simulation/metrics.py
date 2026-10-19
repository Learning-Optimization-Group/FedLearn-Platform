"""Pure metric functions for simulation records (no state, no I/O)."""
from __future__ import annotations

import hashlib
import math
from typing import Dict, Iterable, List, Optional, Sequence, Set

import torch
import torch.nn.functional as F


def binary_auc(scores: torch.Tensor, labels: torch.Tensor) -> Optional[float]:
    """Exact rank (Mann-Whitney) AUC for the positive class, tie-aware."""
    n_pos, n_neg = int((labels == 1).sum()), int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    order = torch.argsort(scores)
    s, lab = scores[order], labels[order]
    ranks = torch.empty(s.numel(), dtype=torch.double)
    i, n = 0, s.numel()
    while i < n:
        j = i
        while j + 1 < n and s[j + 1] == s[i]:
            j += 1
        ranks[i:j + 1] = (i + 1 + j + 1) / 2.0
        i = j + 1
    rank_pos = float(ranks[lab == 1].sum())
    return (rank_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def classification_metrics(logits: torch.Tensor, labels: torch.Tensor, bins: int = 10,
                           *, auc_score: str = "probability_v1") -> Dict[str, Optional[float]]:
    if auc_score not in ("probability_v1", "margin_v2"):
        raise ValueError("unknown AUROC score version")
    if auc_score == "margin_v2" and logits.shape[1] != 2:
        raise ValueError("margin_v2 requires binary logits")
    probs = torch.softmax(logits, dim=1)
    preds = logits.argmax(1)
    out: Dict[str, Optional[float]] = {
        "accuracy": (preds == labels).float().mean().item(),
        "loss": F.cross_entropy(logits, labels).item(),
        "auc": None, "sensitivity": None, "specificity": None, "balanced_accuracy": None,
    }
    if logits.shape[1] == 2:
        if auc_score == "margin_v2":
            # Convert before subtraction; float32 softmax can collapse distinct ranks.
            cpu_logits = logits.detach().to(device="cpu", dtype=torch.float64)
            cpu_labels = labels.detach().cpu()
            out["auc"] = binary_auc(cpu_logits[:, 1] - cpu_logits[:, 0], cpu_labels)
            out["auc_probability"] = binary_auc(probs[:, 1].detach().cpu(), cpu_labels)
        else:
            out["auc"] = binary_auc(probs[:, 1], labels)
        pos, neg = labels == 1, labels == 0
        sens = (preds[pos] == 1).float().mean().item() if pos.any() else None
        spec = (preds[neg] == 0).float().mean().item() if neg.any() else None
        out["sensitivity"], out["specificity"] = sens, spec
        if sens is not None and spec is not None:
            out["balanced_accuracy"] = (sens + spec) / 2.0
    conf, correct = probs.max(1).values, (preds == labels).float()
    idx = torch.clamp((conf * bins).long(), max=bins - 1)
    ece = 0.0
    for b in range(bins):
        sel = idx == b
        if sel.any():
            ece += abs(correct[sel].mean().item() - conf[sel].mean().item()) * sel.float().mean().item()
    out["ece"] = ece
    return out


def backdoor_success(logits: torch.Tensor, labels: torch.Tensor, target: int) -> Optional[float]:
    """Share of non-target-class samples the model assigns to the backdoor target."""
    mask = labels != target
    if not mask.any():
        return None
    return (logits[mask].argmax(1) == target).float().mean().item()


def flat(params, keys: Sequence[str]) -> torch.Tensor:
    return torch.cat([params[k].detach().float().reshape(-1) for k in keys])


def deviation_ratio(agg_all, agg_honest, global_params) -> Optional[float]:
    """||agg_all - agg_honest|| / ||agg_honest - global||, as in the breakdown harness."""
    keys = list(global_params.keys())
    step = (flat(agg_honest, keys) - flat(global_params, keys)).norm().item()
    if step <= 1e-12:
        return None
    return (flat(agg_all, keys) - flat(agg_honest, keys)).norm().item() / step


def mean_pairwise_cosine(vectors: List[torch.Tensor]) -> Optional[float]:
    if len(vectors) < 2:
        return None
    m = torch.stack([v / (v.norm() + 1e-12) for v in vectors])
    gram = m @ m.T
    n = m.shape[0]
    return ((gram.sum() - gram.diagonal().sum()) / (n * (n - 1))).item()


def exclusion_rates(kept_share: Optional[Dict[str, float]], attacker_ids: Set[str]) -> Optional[Dict[str, Optional[float]]]:
    if kept_share is None:
        return None
    honest = [1.0 - s for c, s in kept_share.items() if c not in attacker_ids]
    attack = [1.0 - s for c, s in kept_share.items() if c in attacker_ids]
    return {"honest": sum(honest) / len(honest) if honest else None,
            "attacker": sum(attack) / len(attack) if attack else None}


def client_fairness(accuracies: Sequence[Optional[float]]) -> Optional[Dict[str, float]]:
    """Per-client accuracy summary: mean, mean of the worst ceil(10%) of clients, population std.
    ``None`` entries (clients with no data) are skipped."""
    vals = sorted(a for a in accuracies if a is not None)
    if not vals:
        return None
    k = max(1, math.ceil(0.1 * len(vals)))
    mean = sum(vals) / len(vals)
    return {"mean": mean, "worst_10pct": sum(vals[:k]) / k,
            "std": math.sqrt(sum((v - mean) ** 2 for v in vals) / len(vals))}


def state_digest(states: Iterable) -> str:
    """Same bytes as ``benchmarks/capstone_matched_study._digest``: sorted keys, float32 values."""
    h = hashlib.sha256()
    for state in states:
        for k in sorted(state):
            h.update(k.encode())
            h.update(state[k].detach().float().contiguous().numpy().tobytes())
    return h.hexdigest()[:16]
