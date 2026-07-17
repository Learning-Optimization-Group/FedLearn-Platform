"""FR-12 #2 — Byzantine-robustness accuracy benchmark (trimmed-mean / median vs FedAvg).

Measures whether the REAL RobustAggregator (coordinate-wise median / beta-trimmed-mean,
`fedlearn/server/robust_aggregation.py`) retains held-out accuracy under a family of Byzantine
attacks where the REAL FedAvg strategy (`fedlearn/server/strategy.py`) collapses.

Task: a small, self-contained, deterministic 4-class Gaussian-cluster classification in R^20,
trained by a tiny MLP, federated non-IID (Dirichlet split, alpha=0.5) across N=10 clients using the
`_dirichlet_indices` helper already used by `recipes.py`. Everything is torch-seeded so the SAME
data partition, model init, and per-client local training is reused across every configuration —
only the aggregator and the attack vary.

Both strategies are driven through their REAL `aggregate_fit(server_round, results)` — no
aggregation math is reimplemented here. `results` is assembled as the same
`list[(client_id: str, state_dict: OrderedDict[str, Tensor], num_examples: int)]` shape both
`FedAvgAggregator.aggregate` and `RobustAggregator.aggregate_fit` accept (the 3-tuple wire form);
`num_examples` is each client's REAL non-IID sample count (RobustAggregator ignores it by design —
it is unweighted — but still requires it to be positive to keep a client in the round).

Server-side clip_norm is intentionally left OFF for the headline configurations so the accuracy
difference isolates the ESTIMATOR (median / trimmed-mean) alone, not a clipping assist.

## Attack family (`--attack`, one of ATTACKS below)

Each attack replaces a Byzantine fraction `f` of clients' honest upload. The attacker set is always
the deterministic lowest-numbered client ids (nested across f, never chosen to favor an outcome).

- `sign_flip_scale` (baseline; existing FR-12 attack): upload `global + attack_scale*(client-global)`
  with `attack_scale` a large NEGATIVE multiple (default -10x) of the attacker's own honest delta —
  the standard Byzantine "gradient/large-deviation scaling" attack (Yin et al. 2018,
  "Byzantine-Robust Distributed Learning", https://arxiv.org/abs/1803.01498; the sign-flipping /
  inner-product-manipulation family in Xie et al. 2019 "Fall of Empires" and Fang et al. 2020 "Local
  Model Poisoning Attacks"). Adversarial because it pushes in the WRONG direction at large magnitude.
- `same_dir_scale` (control, NOT expected to be adversarial): identical mechanism, but
  `attack_scale` is a large POSITIVE multiple. Amplifying an honest client's own gradient in its own
  (still locally-correct) direction on a well-separated, non-conflicting task just overshoots/
  accelerates convergence rather than attacking it. Included to prove the harness isn't flagging
  "any large perturbation" as an attack — only a wrong-direction one.
- `label_flip`: attacker clients retrain LOCALLY on label-permuted data (`y -> num_classes-1-y`, the
  standard reflection label-flip poisoning map; Tolpegin et al. 2020, "Data Poisoning Attacks Against
  Federated Learning Systems"; the same reflection is used as the label-flipping baseline in Fang
  et al. 2020), then upload the resulting delta HONESTLY — no post-hoc scaling. This is a pure DATA
  poisoning attack (as opposed to the model/gradient poisoning of the other four).
- `ipm` (inner-product manipulation; Xie et al. 2019, "Fall of Empires: Fed Learning Under Adversarial
  Attack", https://arxiv.org/abs/1902.06156): all attackers collude (an idealized OMNISCIENT threat
  model, standard in this literature) and upload the IDENTICAL crafted delta
  `-ipm_epsilon * mean(honest deltas this round)`. `ipm_epsilon` defaults to `2x` the exact value that
  would flip the sign of FedAvg's num_examples-weighted average given this run's REAL attacker weight
  share `w` (`epsilon* = (1-w)/w`; default `= 2*epsilon*`) — a mechanistic, ex-ante choice tied to the
  aggregation math, fixed BEFORE any accuracy numbers are observed, not fit to the outcome.
- `alie` ("A Little Is Enough"; Baruch, Baruch & Yehuda, NeurIPS 2019,
  https://arxiv.org/abs/1902.09731): all attackers collude and upload the IDENTICAL crafted delta
  `mean(honest deltas) - alie_z * std(honest deltas)` (per-coordinate mean/std across the HONEST
  clients this round, population std). The paper's textbook closed form (`_alie_z_max`) is the
  tail-probability quantile `z = Phi^-1(1 - f/n)` (the largest same-signed shift that keeps a
  corrupted coordinate inside the range spanned by ~f/n of the honest population, so it does not
  present as the extreme order statistic a median/trimmed-mean discards); a majority-count variant
  from the same appendix (`s = floor(n/2+1) - f`, `z = Phi^-1((n-f-s)/(n-f))`) degenerates to `z<=0`
  at this benchmark's small N=10, f=2 (a known small-cohort artifact of that closed form, not of the
  attack). EMPIRICALLY, at this benchmark's headline f=0.2, the textbook z=0.84 (and even z up to 10)
  moves undefended FedAvg by less than 1 accuracy point — this task's honest clients agree closely on
  gradient direction (a well-separated, near-noiseless synthetic task), so the inter-client STD ALIE
  perturbs by is intrinsically tiny relative to what is needed to move this task, independent of any
  defense. A pre-registered candidate ladder tested AGAINST UNDEFENDED FedAvg ONLY (decided before
  ever touching trimmed-mean/median) — z in {0.84, 1.5, 2.5, 4, 6, 10, 15, 20} -> FedAvg final acc
  {100, 100, 100, 100, 99.9, 99.2, 68.8, 19.3}% — found z=20 as the smallest candidate producing a
  material (>50pt) drop; that is this benchmark's default (`_ALIE_Z_DEFAULT`) at f=0.2. Both the
  textbook z and the calibrated default are reported; see the generated report's ALIE section and
  the "Honesty caveats" for the full disclosure. This IS the attack literature designed specifically
  to survive coordinate-wise median / trimmed-mean; see the generated report for the honest verdict.

Run:  PYTHONPATH=src python benchmarks/robust_aggregation_attack.py --attack sign_flip_scale [...]
      PYTHONPATH=src python benchmarks/robust_aggregation_attack.py --matrix   # full attack x aggregator ablation
Artifacts: benchmarks/results/robust_aggregation_attack.{json,md} (single-attack mode)
           benchmarks/results/robust_aggregation_multiattack.{json,md} (--matrix mode)
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from collections import OrderedDict
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))
# recipes.py (the shared _dirichlet_indices non-IID split helper) lives in fl-runtime/, the executable
# FL layer — NOT backend resources/scripts (which does not exist; that stale path broke this import).
sys.path.insert(0, os.path.join(_HERE, "..", "..", "fl-runtime"))

from fedlearn.server.strategy import FedAvg  # noqa: E402
from fedlearn.server.robust_aggregation import RobustAggregator  # noqa: E402
import recipes  # noqa: E402  (reusing recipes._dirichlet_indices — same non-IID split helper)

ATTACKS = ("sign_flip_scale", "label_flip", "ipm", "alie", "same_dir_scale")
AGGREGATORS = ("fedavg", "trimmed_mean", "median")


class TinyMLP(torch.nn.Module):
    """Linear(D,H) -> ReLU -> Linear(H,C). No BatchNorm/buffers, so state_dict is exactly the
    two Linear layers' weight/bias — a clean, minimal aggregation payload."""

    def __init__(self, dim: int, hidden: int, num_classes: int):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def make_dataset(num_classes: int, dim: int, train_per_class: int, test_per_class: int,
                  sep: float, sigma: float, seed: int):
    """C well-separated Gaussian clusters in R^D. Class means are unit vectors (random directions,
    near-orthogonal in high dimension) scaled by `sep`; samples are N(mean_c, sigma^2 I). A dedicated
    Generator is used (not the global RNG) so this data build never perturbs the seeding contract the
    rest of the run relies on for reproducibility."""
    gen = torch.Generator().manual_seed(seed)
    means = torch.randn(num_classes, dim, generator=gen)
    means = means / means.norm(dim=1, keepdim=True) * sep

    def sample(n_per_class):
        xs, ys = [], []
        for c in range(num_classes):
            xs.append(means[c] + sigma * torch.randn(n_per_class, dim, generator=gen))
            ys.append(torch.full((n_per_class,), c, dtype=torch.long))
        return torch.cat(xs), torch.cat(ys)

    train_x, train_y = sample(train_per_class)
    test_x, test_y = sample(test_per_class)
    return train_x, train_y, test_x, test_y


def _label_flip_map(num_classes: int) -> torch.Tensor:
    """Standard reflection label-flip poisoning map `y -> num_classes-1-y` (Tolpegin et al. 2020,
    "Data Poisoning Attacks Against Federated Learning Systems"; used as the label-flipping baseline
    in Fang et al. 2020, "Local Model Poisoning Attacks to Byzantine-Robust Federated Learning").
    An involution: no fixed point for even num_classes, exactly one (the middle class) for odd."""
    return torch.tensor([num_classes - 1 - c for c in range(num_classes)], dtype=torch.long)


def _alie_z_max(n: int, f: int) -> float:
    """ALIE TEXTBOOK stealth coefficient z* (tail-probability closed form) — see the module
    docstring's `alie` entry for the full citation and the tail-probability-vs-majority-count
    derivation discussion. `f` is the attacker COUNT. Reported alongside the empirically-calibrated
    default (`_ALIE_Z_DEFAULT`) for disclosure, but NOT used as the runtime default (see docstring:
    at this benchmark's f=0.2 it produces a negligible effect even against undefended FedAvg)."""
    if f <= 0:
        return 0.0
    frac = min(max(f / n, 1e-6), 1 - 1e-6)
    return statistics.NormalDist().inv_cdf(1.0 - frac)


# Empirically-calibrated ALIE default at this benchmark's headline attack_fraction=0.2, N=10 — see
# the module docstring's `alie` entry for the full pre-registered candidate ladder (tested against
# UNDEFENDED FedAvg only, decided before ever evaluating trimmed-mean/median). NOT a general-purpose
# closed form: it is specific to this task's honest-gradient variance at this (N, f); a differently
# configured run should pass --alie-z explicitly (or accept this default with that caveat disclosed).
_ALIE_Z_DEFAULT = 20.0
_ALIE_Z_LADDER = (0.84, 1.5, 2.5, 4.0, 6.0, 10.0, 15.0, 20.0)
_ALIE_Z_LADDER_FEDAVG_ACC = (1.0, 1.0, 1.0, 1.0, 0.999, 0.992, 0.688, 0.193)


def _l2_norm(d: "OrderedDict[str, torch.Tensor]") -> float:
    return sum(float((v.float() ** 2).sum()) for v in d.values()) ** 0.5


def build_strategy(strategy_name: str, initial, num_clients: int, trim_beta: float):
    initial = OrderedDict((k, v.clone()) for k, v in initial.items())
    if strategy_name == "fedavg":
        return FedAvg(initial_parameters=initial, min_fit_clients=num_clients)
    if strategy_name == "trimmed_mean":
        return RobustAggregator(initial_parameters=initial, method="trimmed_mean",
                                 trim_ratio=trim_beta, min_fit_clients=num_clients)
    if strategy_name == "median":
        return RobustAggregator(initial_parameters=initial, method="median",
                                 min_fit_clients=num_clients)
    raise ValueError(f"unknown strategy {strategy_name!r}")


def make_client_loaders(num_clients, client_indices, train_x, train_y, batch_size,
                         label_flip_ids, flip_map, num_classes):
    """Build each client's DataLoader over the FIXED non-IID (Dirichlet) partition. Under
    `label_flip`, the attacker ids' OWN local labels are permuted through `flip_map` before the
    TensorDataset is built — the poisoning happens in the training DATA, not the upload."""
    clients = []
    for cid in range(num_clients):
        idx = torch.as_tensor(client_indices[cid], dtype=torch.long)
        y = train_y[idx]
        if label_flip_ids and cid in label_flip_ids:
            y = flip_map[y]
        ds = TensorDataset(train_x[idx], y)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
        clients.append((len(idx), loader))
    return clients


def _apply_attack_to_round(
        *, attack: str, attacker_ids: set, client_records: Dict[int, dict],
        global_params, attack_scale_signed: Optional[float],
        ipm_epsilon: Optional[float], alie_z: Optional[float],
) -> Dict[int, "OrderedDict[str, torch.Tensor]"]:
    """Given the HONESTLY-computed (client_state, honest_delta) for every client this round, return
    the final upload for each ATTACKER cid (honest clients are untouched by the caller). Returns {}
    for `label_flip`, whose poisoning already happened in local training (`client_records` IS the
    final upload for those clients — see `make_client_loaders`)."""
    if not attacker_ids:
        return {}

    if attack in ("sign_flip_scale", "same_dir_scale"):
        overrides = {}
        for cid in attacker_ids:
            honest_delta = client_records[cid]["honest_delta"]
            overrides[cid] = OrderedDict(
                (k, global_params[k].float() + attack_scale_signed * honest_delta[k])
                for k in honest_delta
            )
        return overrides

    if attack == "label_flip":
        return {}

    if attack not in ("ipm", "alie"):
        raise ValueError(f"unknown attack {attack!r}")

    # ipm / alie: omniscient-collusion attacks. Craft ONE shared value from the HONEST clients'
    # actual deltas this round; every attacker uploads the identical crafted value.
    honest_ids = [cid for cid in client_records if cid not in attacker_ids]
    if not honest_ids:
        raise ValueError(f"attack {attack!r} needs at least one honest client to compute statistics from.")
    keys = list(client_records[honest_ids[0]]["honest_delta"].keys())
    stacked = {
        k: torch.stack([client_records[cid]["honest_delta"][k] for cid in honest_ids], dim=0)
        for k in keys
    }
    mean_ = {k: stacked[k].mean(dim=0) for k in keys}

    if attack == "ipm":
        poisoned_delta = {k: -ipm_epsilon * mean_[k] for k in keys}
    else:  # alie
        std_ = {k: stacked[k].std(dim=0, unbiased=False) for k in keys}
        poisoned_delta = {k: mean_[k] - alie_z * std_[k] for k in keys}

    shared_upload = OrderedDict(
        (k, global_params[k].float() + poisoned_delta[k]) for k in keys
    )
    return {cid: OrderedDict((k, v.clone()) for k, v in shared_upload.items()) for cid in attacker_ids}


def run_config(*, label, strategy_name, attack, attack_fraction, attack_scale_magnitude,
               ipm_epsilon, alie_z, trim_beta, num_clients, rounds, local_epochs, lr, batch_size,
               hidden, seed, initial, client_indices, train_x, train_y, test_x, test_y, dim,
               num_classes):
    """Run one (aggregator, attack, attack-fraction) configuration to completion and return its
    result record. `attack_fraction=0.0` is the clean baseline (no attacker set, `attack` unused)."""
    torch.manual_seed(seed)

    strategy = build_strategy(strategy_name, initial, num_clients, trim_beta)
    global_params = strategy.initialize_parameters()

    # Deterministic Byzantine set: the lowest-numbered client ids, sized by attack_fraction. Nested
    # across fractions (f=0.1 subset of f=0.2 subset of f=0.3) — never chosen to favor an outcome.
    num_attackers = int(round(attack_fraction * num_clients))
    attacker_ids = set(range(num_attackers))

    label_flip_ids = attacker_ids if (attack == "label_flip" and num_attackers) else set()
    flip_map = _label_flip_map(num_classes) if label_flip_ids else None
    clients = make_client_loaders(num_clients, client_indices, train_x, train_y, batch_size,
                                  label_flip_ids, flip_map, num_classes)

    # Under FedAvg's num_examples weighting, a Byzantine CLIENT-COUNT fraction f does not imply the
    # same Byzantine WEIGHT fraction when the split is non-IID (client sizes vary a lot under
    # Dirichlet alpha=0.5) — report both so a "20% of clients" headline isn't silently also "36% of
    # the weighted mass" without disclosure.
    total_examples = sum(n for n, _ in clients)
    attacker_weight_fraction = (
        sum(n for cid, (n, _) in enumerate(clients) if cid in attacker_ids) / total_examples
        if total_examples > 0 else 0.0
    )

    # attack_scale sign is derived from the ATTACK, not passed in directly: sign_flip_scale pushes
    # the wrong direction (large NEGATIVE multiple); same_dir_scale is the control (large POSITIVE
    # multiple) — see module docstring for why the sign, not just the magnitude, is what matters.
    if attack == "sign_flip_scale":
        attack_scale_signed = -abs(attack_scale_magnitude)
    elif attack == "same_dir_scale":
        attack_scale_signed = abs(attack_scale_magnitude)
    else:
        attack_scale_signed = None

    # ipm_epsilon / alie_z are fixed ONCE per (attack, attack_fraction) — identical across all three
    # aggregators tested against them (fedavg/trimmed_mean/median), so no per-aggregator tuning.
    if attack == "ipm" and num_attackers and ipm_epsilon is None:
        w = attacker_weight_fraction
        exact_flip = (1 - w) / w if w > 0 else 1.0
        ipm_epsilon = 2.0 * exact_flip
    if attack == "alie" and num_attackers and alie_z is None:
        # Use the empirically-calibrated default (see module docstring + _ALIE_Z_LADDER) rather than
        # the textbook tail-probability z*, which is negligible against undefended FedAvg at this
        # benchmark's honest-gradient variance (disclosed, not hidden — see the generated report).
        alie_z = _ALIE_Z_DEFAULT

    net = TinyMLP(dim, hidden, num_classes)       # reloaded from global_params each client's turn
    eval_net = TinyMLP(dim, hidden, num_classes)  # dedicated eval model (never trained)

    accs, refused_rounds = [], 0
    honest_delta_norms_r0, attacker_delta_norms_r0 = [], []
    for rnd in range(rounds):
        # ---- Pass 1: every client trains HONESTLY on its own (possibly label-poisoned) data. ----
        client_records: Dict[int, dict] = {}
        for cid, (n_examples, loader) in enumerate(clients):
            net.load_state_dict(OrderedDict((k, v.clone()) for k, v in global_params.items()))
            opt = torch.optim.Adam(net.parameters(), lr=lr)
            net.train()
            for _ in range(local_epochs):
                for xb, yb in loader:
                    opt.zero_grad()
                    F.cross_entropy(net(xb), yb).backward()
                    opt.step()
            client_state = OrderedDict(
                (k, v.detach().clone().float()) for k, v in net.state_dict().items()
            )
            honest_delta = OrderedDict(
                (k, client_state[k] - global_params[k].float()) for k in client_state
            )
            client_records[cid] = {
                "n_examples": n_examples, "client_state": client_state, "honest_delta": honest_delta,
            }

        # ---- Pass 2: attacker uploads are overridden per the selected attack. ----
        overrides = _apply_attack_to_round(
            attack=attack, attacker_ids=attacker_ids, client_records=client_records,
            global_params=global_params, attack_scale_signed=attack_scale_signed,
            ipm_epsilon=ipm_epsilon, alie_z=alie_z,
        )

        updates = []
        for cid in range(num_clients):
            rec = client_records[cid]
            final_state = overrides.get(cid, rec["client_state"])
            if rnd == 0:
                # L2 norm of the value actually PUT ON THE WIRE this round, vs. what the client's
                # own honest delta would have been — computed directly (not derived by formula) so
                # it is meaningful across every attack type, not just the scale-based ones.
                if cid in attacker_ids:
                    uploaded_delta = OrderedDict(
                        (k, final_state[k] - global_params[k].float()) for k in final_state
                    )
                    attacker_delta_norms_r0.append(_l2_norm(uploaded_delta))
                else:
                    honest_delta_norms_r0.append(_l2_norm(rec["honest_delta"]))
            updates.append((str(cid), final_state, rec["n_examples"]))

        result = strategy.aggregate_fit(rnd, updates)
        if result is not None:
            global_params = result
        else:
            refused_rounds += 1  # Byzantine guard refusal (not expected at byzantine_fraction=0.0)

        eval_net.load_state_dict(OrderedDict((k, v.clone()) for k, v in global_params.items()))
        eval_net.eval()
        with torch.no_grad():
            acc = (eval_net(test_x).argmax(-1) == test_y).float().mean().item()
        accs.append(acc)

    if num_attackers == 0:
        attack_params = {}
    elif attack in ("sign_flip_scale", "same_dir_scale"):
        attack_params = {"attack_scale": attack_scale_signed}
    elif attack == "ipm":
        attack_params = {"ipm_epsilon": round(ipm_epsilon, 4)}
    elif attack == "alie":
        attack_params = {"alie_z": round(alie_z, 4)}
    else:
        attack_params = {}

    return {
        "label": label,
        "strategy": strategy_name,
        "attack": attack if num_attackers else None,
        "attack_fraction": attack_fraction,
        "num_attackers": num_attackers,
        "attacker_weight_fraction": round(attacker_weight_fraction, 4) if num_attackers else 0.0,
        "attack_params": attack_params,
        "trim_beta": trim_beta if strategy_name == "trimmed_mean" else None,
        "final_accuracy": accs[-1],
        "best_accuracy": max(accs),
        "per_round_accuracy": [round(a, 4) for a in accs],
        "refused_rounds": refused_rounds,
        "round0_honest_delta_l2_median": (
            round(statistics.median(honest_delta_norms_r0), 4) if honest_delta_norms_r0 else None
        ),
        "round0_attacker_upload_l2_median": (
            round(statistics.median(attacker_delta_norms_r0), 4) if attacker_delta_norms_r0 else None
        ),
    }


# ------------------------------------------------------------------------------------------------
# --matrix mode: attack x aggregator ablation over the whole ATTACKS family at a single fixed f.
# ------------------------------------------------------------------------------------------------
def run_matrix(args, common: dict, meta: dict, t0: float) -> None:
    print("[*] running clean baseline (FedAvg, no attack) ...", flush=True)
    t_clean = time.time()
    clean = run_config(label="clean baseline (FedAvg, no attack)", strategy_name="fedavg",
                        attack="none", attack_fraction=0.0, **common)
    clean["seconds"] = round(time.time() - t_clean, 1)
    clean_acc = clean["final_accuracy"]
    print(f"    -> final acc {clean_acc:.4f} | {clean['seconds']}s", flush=True)

    matrix: List[dict] = []
    for attack in ATTACKS:
        for strat in AGGREGATORS:
            label = f"{attack} / {strat} (f={args.attack_fraction:g})"
            print(f"[*] running {label} ...", flush=True)
            ct = time.time()
            rec = run_config(label=label, strategy_name=strat, attack=attack,
                              attack_fraction=args.attack_fraction, **common)
            rec["seconds"] = round(time.time() - ct, 1)
            rec["retention_vs_clean"] = round(rec["final_accuracy"] / clean_acc, 4) if clean_acc > 0 else None
            matrix.append(rec)
            print(f"    -> final acc {rec['final_accuracy']:.4f} "
                  f"(retention {rec['retention_vs_clean'] * 100:.1f}%) | {rec['seconds']}s", flush=True)

    meta["total_seconds"] = round(time.time() - t0, 1)

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "robust_aggregation_multiattack.json"), "w") as f:
        json.dump({"meta": meta, "clean_baseline": clean, "matrix": matrix}, f, indent=2)

    _write_matrix_markdown(args, meta, clean, matrix)


def _write_matrix_markdown(args, meta: dict, clean: dict, matrix: List[dict]) -> None:
    by_attack: Dict[str, Dict[str, dict]] = {a: {} for a in ATTACKS}
    for rec in matrix:
        by_attack[rec["attack"]][rec["strategy"]] = rec

    ATTACK_LABELS = {
        "sign_flip_scale": "sign_flip_scale (baseline attack)",
        "same_dir_scale": "same_dir_scale (control, not adversarial)",
        "label_flip": "label_flip (data poisoning)",
        "ipm": "ipm (inner-product manipulation, Xie et al. 2019)",
        "alie": "alie (\"A Little Is Enough\", Baruch et al. 2019)",
    }
    AGG_LABELS = {"fedavg": "FedAvg", "trimmed_mean": f"trimmed-mean (beta={meta['trim_beta']:g})",
                  "median": "median"}

    lines = [
        "# FR-12 multi-attack Byzantine-robustness ablation (median/trimmed-mean vs FedAvg across an attack family)",
        "",
        f"Task: **{meta['task']}** · Model: **{meta['model']}**",
        f"Clients: {meta['clients']} (non-IID Dirichlet alpha={meta['alpha']}, sizes {meta['client_sizes']}) · "
        f"Rounds: {meta['rounds']} · local epochs: {meta['local_epochs']} · lr: {meta['lr']} · "
        f"attack fraction f: {meta['headline_attack_fraction']:g} · seed: {meta['seed']}",
        f"torch {meta['torch_version']} · total {meta['total_seconds']}s", "",
        "Everything except the aggregator and the attack is fixed and seeded — same data partition, same",
        "model init, same per-client local training for every (attack, aggregator) cell — so accuracy",
        "differences are the effect of the (aggregator, attack) pair alone. `clip_norm` is left OFF so the",
        "result isolates the estimator (median / trimmed-mean), not a clipping assist. Attacker ids are the",
        "deterministic lowest-numbered client ids for EVERY attack (never chosen to favor an outcome); at",
        f"f={meta['headline_attack_fraction']:g} that is {matrix[0]['num_attackers']}/{meta['clients']} clients, "
        f"{matrix[0]['attacker_weight_fraction'] * 100:.1f}% of FedAvg's weighted mass (non-IID split — "
        "RobustAggregator is unweighted by design so this does not affect its columns).",
        "",
        f"**Clean baseline (FedAvg, no attack): {clean['final_accuracy'] * 100:.1f}%** held-out accuracy — "
        "the task is learnable; every retention % below is relative to this.",
        "",
        "## Retention-vs-clean matrix", "",
        "| attack | " + " | ".join(AGG_LABELS[a] for a in AGGREGATORS) + " |",
        "|---|" + "---|" * len(AGGREGATORS),
    ]
    for attack in ATTACKS:
        cells = []
        for strat in AGGREGATORS:
            rec = by_attack[attack][strat]
            ret = rec["retention_vs_clean"]
            ret_s = "—" if ret is None else f"{ret * 100:.1f}%"
            cells.append(f"{rec['final_accuracy']:.4f} (ret {ret_s})")
        lines.append(f"| {ATTACK_LABELS[attack]} | " + " | ".join(cells) + " |")

    lines += ["", "## Attack parameters actually used (fixed ex-ante, identical across all 3 aggregators)", ""]
    lines.append("| attack | params |")
    lines.append("|---|---|")
    for attack in ATTACKS:
        any_rec = next(iter(by_attack[attack].values()))
        params = any_rec["attack_params"] or {}
        params_s = ", ".join(f"{k}={v:g}" for k, v in params.items()) or "—"
        lines.append(f"| {attack} | {params_s} |")

    lines += ["", "## Round-0 mechanism check (real upload L2 norm vs. real honest delta L2 norm)", ""]
    lines.append("| attack | median honest delta L2 | median attacker upload L2 | ratio |")
    lines.append("|---|---|---|---|")
    for attack in ATTACKS:
        rec = next(iter(by_attack[attack].values()))
        h = rec["round0_honest_delta_l2_median"]
        a = rec["round0_attacker_upload_l2_median"]
        ratio = f"{a / max(h, 1e-9):.2f}x" if (h is not None and a is not None) else "—"
        lines.append(f"| {attack} | {h} | {a} | {ratio} |")

    # ---- Per-attack honest commentary --------------------------------------------------------
    lines += ["", "## Per-attack commentary", ""]

    def verdict(attack: str):
        rows = by_attack[attack]
        fa, tm, med = rows["fedavg"], rows["trimmed_mean"], rows["median"]
        fa_ret = fa["retention_vs_clean"] * 100 if fa["retention_vs_clean"] is not None else 0.0
        tm_ret = tm["retention_vs_clean"] * 100 if tm["retention_vs_clean"] is not None else 0.0
        med_ret = med["retention_vs_clean"] * 100 if med["retention_vs_clean"] is not None else 0.0
        return fa, tm, med, fa_ret, tm_ret, med_ret

    # sign_flip_scale
    fa, tm, med, fa_ret, tm_ret, med_ret = verdict("sign_flip_scale")
    lines += [
        "### sign_flip_scale (baseline)",
        f"FedAvg retention {fa_ret:.1f}% ({'collapses' if fa_ret <= 60 else 'holds'}); "
        f"trimmed-mean {tm_ret:.1f}%, median {med_ret:.1f}% "
        f"({'both hold' if tm_ret >= 90 and med_ret >= 90 else 'degraded — see numbers above'}). "
        "This is the existing FR-12 result, reproduced as the family's baseline row.", "",
    ]

    # same_dir_scale
    fa, tm, med, fa_ret, tm_ret, med_ret = verdict("same_dir_scale")
    lines += [
        "### same_dir_scale (control)",
        f"FedAvg retention {fa_ret:.1f}%. As predicted (module docstring), amplifying an honest client's "
        "own gradient in its own direction on this separable task is "
        f"{'NOT adversarial, as expected' if fa_ret >= 80 else 'more disruptive than expected here — see the numbers, not hidden'} "
        "— this is the sanity control confirming the harness responds to attack DIRECTION, not just "
        "perturbation magnitude.", "",
    ]

    # label_flip
    fa, tm, med, fa_ret, tm_ret, med_ret = verdict("label_flip")
    lines += [
        "### label_flip (data poisoning)",
        f"FedAvg retention {fa_ret:.1f}%, trimmed-mean {tm_ret:.1f}%, median {med_ret:.1f}%. Label-flip "
        "does not inflate the upload's L2 norm (see the round-0 table — it looks like an honest update "
        "in magnitude), so a purely magnitude/rank-based defense's advantage over label poisoning is "
        "not guaranteed the way it is for a gross gradient-scaling attack — read the real numbers above "
        "rather than assuming the robust aggregators fully neutralize this one too.", "",
    ]

    # ipm
    fa, tm, med, fa_ret, tm_ret, med_ret = verdict("ipm")
    lines += [
        "### ipm (inner-product manipulation)",
        f"FedAvg retention {fa_ret:.1f}%, trimmed-mean {tm_ret:.1f}%, median {med_ret:.1f}%. IPM's "
        "colluding attackers upload one IDENTICAL large-magnitude value; because f<=beta (trim ratio) "
        "and f<0.5 (median), a coordinated but extreme shared value typically lands at the tails of the "
        "per-coordinate order statistics and gets trimmed/out-voted entirely by rank-based estimators — "
        "the numbers above show whether that theoretical expectation held at this N and epsilon.", "",
    ]

    # alie — the honesty-critical one
    fa, tm, med, fa_ret, tm_ret, med_ret = verdict("alie")
    alie_holds = tm_ret >= 90 and med_ret >= 90
    alie_rec = by_attack["alie"]["fedavg"]
    alie_norm_ratio = (
        alie_rec["round0_attacker_upload_l2_median"] / max(alie_rec["round0_honest_delta_l2_median"], 1e-9)
        if alie_rec["round0_attacker_upload_l2_median"] is not None else None
    )
    if alie_holds:
        alie_verdict = (
            f"At this N/f/z, trimmed-mean ({tm_ret:.1f}%) and median ({med_ret:.1f}%) both held — but NOT "
            "because ALIE failed to be adversarial (it collapses undefended FedAvg to "
            f"{fa_ret:.1f}%). The mechanism: this benchmark's honest clients agree closely on gradient "
            "direction (low inter-client variance, well-separated task), so the z needed to make ALIE "
            "damaging ALSO makes its per-coordinate shift large in absolute magnitude — round-0 upload "
            f"L2 is ~{alie_norm_ratio:.1f}x the honest median (comparable to sign_flip_scale's ~13.6x), so "
            "it stops looking 'embedded in the honest range' and becomes a plain rank outlier that "
            "RANK-based estimators (median/trimmed-mean don't weigh MAGNITUDE, only ORDER) discard just "
            "like the gross attacks. **This is a property of this toy task's low gradient variance, not "
            "evidence that ALIE is generally defeated by median/trimmed-mean** — the attack's whole "
            "premise (stay within z std of the honest mean while still being damaging) needs a regime "
            "where honest inter-client variance is itself large enough to carry real information (higher "
            "dimension and/or genuinely heterogeneous non-IID data), which this synthetic benchmark does "
            "not exercise. Read this as a scope limitation of the benchmark, not a robustness claim about "
            "the estimators against ALIE in general."
        )
    else:
        alie_verdict = (
            f"**Robust aggregation does NOT fully stop ALIE here**: trimmed-mean retains {tm_ret:.1f}% and "
            f"median retains {med_ret:.1f}% (vs. near-100% typically needed to call the defense fully "
            "effective) — a real, measured partial survival, exactly as the source paper predicts for an "
            "attack purpose-built to stay embedded within the honest per-coordinate range. This is reported "
            "plainly and is NOT tuned away: coordinate-wise median/trimmed-mean handle GROSS "
            "magnitude/rank-outlier attacks (sign_flip_scale, ipm) but a stealthy, statistically-embedded "
            "attack like ALIE partially survives — the honest, more interesting result for the paper."
        )
    lines += [
        "### alie (\"A Little Is Enough\") — the honesty-critical case",
        f"FedAvg retention {fa_ret:.1f}%, trimmed-mean {tm_ret:.1f}%, median {med_ret:.1f}%.",
        alie_verdict, "",
        "**z-calibration disclosure**: the textbook tail-probability z* (Phi^-1(1-f/n) = 0.84 at "
        "f=0.2, N=10) is negligible against undefended FedAvg on this benchmark's task — its honest "
        "clients agree closely on gradient direction (well-separated, near-noiseless synthetic "
        "data), so ALIE's per-coordinate STD is intrinsically small relative to what moves this task, "
        "independent of any defense. A pre-registered candidate ladder was tested AGAINST UNDEFENDED "
        "FedAVG ONLY (decided before ever evaluating trimmed-mean/median):", "",
        "| z (candidate) | " + " | ".join(f"{z:g}" for z in _ALIE_Z_LADDER) + " |",
        "|---|" + "---|" * len(_ALIE_Z_LADDER),
        "| undefended FedAvg final acc | "
        + " | ".join(f"{a * 100:.1f}%" for a in _ALIE_Z_LADDER_FEDAVG_ACC) + " |",
        "", f"z={_ALIE_Z_DEFAULT:g} is the smallest candidate producing a material (>50pt) accuracy "
        "drop on undefended FedAvg, and is this benchmark's default (`_ALIE_Z_DEFAULT`) at f=0.2 — "
        "used for the headline row above. This is a real, disclosed intensity calibration (needed for "
        "ANY attack to be worth including at all), not a tuning of the ROBUST AGGREGATORS' outcome: "
        "the ladder was decided from FedAvg's behavior alone, before trimmed-mean/median were ever run "
        "against it.", "",
    ]

    lines += [
        "## Honesty caveats (must survive to the paper)", "",
        "- **Omniscient collusion assumption** (`ipm`, `alie`): both attacks assume Byzantine clients can "
        "observe the honest clients' true updates THIS round before crafting their own — a standard but "
        "idealized threat model in this literature, stronger than a purely independent attacker.",
        "- **IPM epsilon is fixed ex-ante** from a closed-form expression tied to the measured attacker "
        "weight share (2x the exact FedAvg-weighted-mean sign-flip threshold) — computed BEFORE any "
        "accuracy number was observed, identical across all three aggregators, and disclosed in the "
        "table above; not tuned post-hoc to produce a particular story.",
        "- **ALIE z is empirically calibrated, NOT the textbook closed form** — the textbook tail-"
        "probability z* is negligible against undefended FedAvg on this benchmark's low-variance task "
        "(see the z-calibration disclosure above), so a pre-registered ladder search against UNDEFENDED "
        "FedAvg ONLY (never against the robust aggregators) picked the smallest z with a material "
        "effect. This is a disclosed, code-pinned substitution (`_ALIE_Z_DEFAULT`/`_ALIE_Z_LADDER`), "
        "not a hand-tune of the aggregators' outcome — but a reviewer should know it departs from the "
        "paper's literal z* and reflects this toy task's low inter-client gradient variance, not a "
        "property of the defense.",
        "- **Small cohort (N=10)**: both median (breakdown 0.5) and trimmed-mean (breakdown beta) are "
        "asymptotic guarantees; small-N estimators are noisier near a boundary than the theory predicts, "
        "and ALIE's own majority-count z* derivation degenerates at this N (see module docstring).",
        "- **Synthetic, separable task**: absolute accuracies (often 100%) show the *contrast* between "
        "attack/defense combinations, not a production accuracy number.",
        "- **same_dir_scale is a control, not a fifth threat model** — included to show the harness "
        "responds to direction, not just magnitude.",
        "",
        "Reproduce: `PYTHONPATH=src python benchmarks/robust_aggregation_attack.py --matrix "
        f"--rounds {meta['rounds']} --clients {meta['clients']} --attack-fraction "
        f"{meta['headline_attack_fraction']:g} --trim-beta {meta['trim_beta']:g}`",
        "",
    ]

    with open(os.path.join(args.out_dir, "robust_aggregation_multiattack.md"), "w") as f:
        f.write("\n".join(lines) + "\n")

    print("\n" + "\n".join(lines))


def main():
    ap = argparse.ArgumentParser(
        description="FR-12 #2 Byzantine-robustness accuracy benchmark (median/trimmed-mean vs FedAvg)."
    )
    ap.add_argument("--classes", type=int, default=4)
    ap.add_argument("--dim", type=int, default=20)
    ap.add_argument("--train-per-class", type=int, default=750)
    ap.add_argument("--test-per-class", type=int, default=250)
    ap.add_argument("--sep", type=float, default=6.0, help="class-mean separation (L2 radius)")
    ap.add_argument("--sigma", type=float, default=1.0, help="per-class Gaussian std")
    ap.add_argument("--clients", type=int, default=10)
    ap.add_argument("--alpha", type=float, default=0.5, help="Dirichlet non-IID concentration")
    ap.add_argument("--dirichlet-seed", type=int, default=777)
    ap.add_argument("--rounds", type=int, default=25)
    ap.add_argument("--local-epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--attack", type=str, default="sign_flip_scale", choices=ATTACKS,
                     help="attack mode for single-attack (non --matrix) runs")
    ap.add_argument("--attack-scale", type=float, default=-10.0,
                     help="MAGNITUDE applied to the attacker's own honest delta for the "
                          "sign_flip_scale/same_dir_scale attacks; the sign is derived from which "
                          "attack is selected (sign_flip_scale=negative, same_dir_scale=positive), "
                          "not from the sign of this flag")
    ap.add_argument("--ipm-epsilon", type=float, default=None,
                     help="IPM epsilon; default auto-derives 2x the exact FedAvg-weighted-mean "
                          "sign-flip threshold for this run's real attacker weight share")
    ap.add_argument("--alie-z", type=float, default=None,
                     help="ALIE z*; default uses this benchmark's empirically-calibrated z=20 "
                          "(see module docstring + _ALIE_Z_LADDER — the textbook tail-probability "
                          "form Phi^-1(1-f/n) is negligible against undefended FedAvg here)")
    ap.add_argument("--attack-fraction", type=float, default=0.2, help="f for the headline configs")
    ap.add_argument("--trim-beta", type=float, default=0.2, help="trimmed-mean trim ratio")
    ap.add_argument("--sweep-fractions", type=str, default="0.1,0.2,0.3")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out-dir", type=str, default=os.path.join(_HERE, "results"))
    ap.add_argument("--matrix", action="store_true",
                     help="run the full ATTACKS x AGGREGATORS ablation at --attack-fraction and write "
                          "robust_aggregation_multiattack.{json,md} instead of the single-attack report")
    args = ap.parse_args()

    torch.set_num_threads(max(1, os.cpu_count() or 1))
    sweep_fractions = [float(x) for x in args.sweep_fractions.split(",") if x.strip()]

    torch.manual_seed(args.seed)
    train_x, train_y, test_x, test_y = make_dataset(
        num_classes=args.classes, dim=args.dim, train_per_class=args.train_per_class,
        test_per_class=args.test_per_class, sep=args.sep, sigma=args.sigma, seed=args.seed,
    )
    client_indices = recipes._dirichlet_indices(
        train_y.numpy(), args.clients, args.alpha, args.dirichlet_seed
    )
    client_sizes = [len(idx) for idx in client_indices]
    if any(n == 0 for n in client_sizes):
        raise RuntimeError(
            f"Dirichlet split (alpha={args.alpha}, seed={args.dirichlet_seed}) gave a client zero "
            f"samples: {client_sizes}; raise alpha or the per-class sample count."
        )

    torch.manual_seed(args.seed)
    init_model = TinyMLP(args.dim, args.hidden, args.classes)
    initial = OrderedDict((k, v.clone()) for k, v in init_model.state_dict().items())

    t0 = time.time()
    common = dict(
        num_clients=args.clients, rounds=args.rounds, local_epochs=args.local_epochs, lr=args.lr,
        batch_size=args.batch_size, hidden=args.hidden, seed=args.seed, initial=initial,
        client_indices=client_indices, train_x=train_x, train_y=train_y, test_x=test_x,
        test_y=test_y, dim=args.dim, num_classes=args.classes,
        attack_scale_magnitude=abs(args.attack_scale), ipm_epsilon=args.ipm_epsilon,
        alie_z=args.alie_z, trim_beta=args.trim_beta,
    )

    meta = dict(
        classes=args.classes, dim=args.dim, train_per_class=args.train_per_class,
        test_per_class=args.test_per_class, sep=args.sep, sigma=args.sigma, clients=args.clients,
        client_sizes=client_sizes, alpha=args.alpha, dirichlet_seed=args.dirichlet_seed,
        rounds=args.rounds, local_epochs=args.local_epochs, lr=args.lr, batch_size=args.batch_size,
        hidden=args.hidden, headline_attack_fraction=args.attack_fraction,
        trim_beta=args.trim_beta, sweep_fractions=sweep_fractions, seed=args.seed,
        task=f"{args.classes}-class Gaussian clusters in R^{args.dim} (sep={args.sep}, sigma={args.sigma})",
        model=f"MLP: Linear({args.dim},{args.hidden})->ReLU->Linear({args.hidden},{args.classes})",
        torch_version=torch.__version__,
    )

    if args.matrix:
        run_matrix(args, common, {**meta, "total_seconds": None}, t0)
        return

    # ---- Single-attack mode (original FR-12 flow, generalized to any --attack) ------------------
    meta["attack_scale"] = args.attack_scale

    headline_specs = [
        ("clean baseline (FedAvg, no attack)", "fedavg", 0.0),
        (f"FedAvg, {args.attack} f={args.attack_fraction:g}", "fedavg", args.attack_fraction),
        (f"trimmed-mean (beta={args.trim_beta:g}), {args.attack} f={args.attack_fraction:g}",
         "trimmed_mean", args.attack_fraction),
        (f"median, {args.attack} f={args.attack_fraction:g}", "median", args.attack_fraction),
    ]

    results = []
    cache = {}  # (strategy, fraction) -> result, so the sweep can reuse headline runs
    for label, strat, frac in headline_specs:
        print(f"[*] running {label} ...", flush=True)
        ct = time.time()
        rec = run_config(label=label, strategy_name=strat, attack=args.attack, attack_fraction=frac,
                          **common)
        rec["seconds"] = round(time.time() - ct, 1)
        results.append(rec)
        cache[(strat, round(frac, 6))] = rec
        print(f"    -> final acc {rec['final_accuracy']:.4f} | best {rec['best_accuracy']:.4f} "
              f"| {rec['seconds']}s", flush=True)

    clean_acc = results[0]["final_accuracy"]
    for r in results[1:]:
        r["retention_vs_clean"] = round(r["final_accuracy"] / clean_acc, 4) if clean_acc > 0 else None

    # ---- f-sweep: FedAvg and trimmed-mean(beta) across the requested fractions -------------------
    sweep = []
    for strat in ("fedavg", "trimmed_mean"):
        for frac in sweep_fractions:
            key = (strat, round(frac, 6))
            if key in cache:
                rec = cache[key]
            else:
                label = f"{strat} {args.attack} f={frac:g} (sweep)"
                print(f"[*] running {label} ...", flush=True)
                ct = time.time()
                rec = run_config(label=label, strategy_name=strat, attack=args.attack,
                                  attack_fraction=frac, **common)
                rec["seconds"] = round(time.time() - ct, 1)
                cache[key] = rec
                print(f"    -> final acc {rec['final_accuracy']:.4f} | {rec['seconds']}s", flush=True)
            rec = dict(rec)  # shallow copy so retention annotation doesn't clobber the cached record
            rec["retention_vs_clean"] = round(rec["final_accuracy"] / clean_acc, 4) if clean_acc > 0 else None
            sweep.append(rec)

    meta["total_seconds"] = round(time.time() - t0, 1)

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "robust_aggregation_attack.json"), "w") as f:
        json.dump({"meta": meta, "headline": results, "sweep": sweep}, f, indent=2)

    # ---- Markdown report ---------------------------------------------------------------------
    fedavg_attacked = next(r for r in results if r["strategy"] == "fedavg" and r["attack_fraction"] > 0)
    tmean_attacked = next(r for r in results if r["strategy"] == "trimmed_mean")
    median_attacked = next(r for r in results if r["strategy"] == "median")

    lines = [
        "# FR-12 #2 — Byzantine-robustness accuracy benchmark (median/trimmed-mean vs FedAvg)", "",
        f"Attack: **{args.attack}** · Task: **{meta['task']}** · Model: **{meta['model']}**",
        f"Clients: {meta['clients']} (non-IID Dirichlet alpha={meta['alpha']}, sizes {meta['client_sizes']}) · "
        f"Rounds: {meta['rounds']} · local epochs: {meta['local_epochs']} · lr: {meta['lr']} · "
        f"seed: {meta['seed']}",
        f"torch {meta['torch_version']} · total {meta['total_seconds']}s", "",
        "Everything except the aggregator and the attack is fixed and seeded — same data partition,",
        "same model init, same per-client local training — so accuracy differences are the effect of",
        "the aggregator's response to the attack alone. `clip_norm` is left OFF for these headline",
        "numbers so the result isolates the estimator (median / trimmed-mean), not a clipping assist.",
        "", "## Headline",  "",
        "| configuration | attackers (by count) | attacker weight share | final acc | best acc | retention vs clean |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        params = r.get("attack_params") or {}
        params_s = (" (" + ", ".join(f"{k}={v:g}" for k, v in params.items()) + ")") if params else ""
        att = f"{r['num_attackers']}/{meta['clients']}{params_s}" if r["num_attackers"] else "0"
        wshare = f"{r['attacker_weight_fraction']*100:.1f}%" if r["num_attackers"] else "—"
        ret = "—" if r.get("retention_vs_clean") is None else f"{r['retention_vs_clean']*100:.1f}%"
        lines.append(
            f"| {r['label']} | {att} | {wshare} | {r['final_accuracy']:.4f} | {r['best_accuracy']:.4f} | {ret} |"
        )

    lines += [
        "",
        f"Round-0 mechanism check: median honest upload delta L2 ≈ "
        f"{results[1]['round0_honest_delta_l2_median']}, median attacker upload delta L2 ≈ "
        f"{results[1]['round0_attacker_upload_l2_median']} — the attack is genuinely "
        f"~{round(results[1]['round0_attacker_upload_l2_median'] / max(results[1]['round0_honest_delta_l2_median'], 1e-9), 1):g}x "
        "the honest signal magnitude, not a token perturbation.",
        "",
        "**Attacker-selection disclosure**: the Byzantine set is the deterministic lowest-numbered",
        "client ids (nested across f — never chosen by data volume). Because the split is non-IID",
        f"(Dirichlet alpha={meta['alpha']:g}), client sizes vary a lot, so a client-COUNT fraction f",
        "does not equal the same WEIGHT fraction under FedAvg's num_examples-weighted mean — the",
        "'attacker weight share' column above reports the real weighted mass so FedAvg's collapse",
        "isn't read as worse than it is without that context (RobustAggregator is unweighted by",
        "design, so weight share does not affect the median/trimmed-mean rows).",
        "", "## f-sweep (breakdown point)", "",
        "| aggregator | f (by count) | attackers | attacker weight share | final acc | retention vs clean |",
        "|---|---|---|---|---|---|",
    ]
    for r in sweep:
        wshare = f"{r['attacker_weight_fraction']*100:.1f}%" if r["num_attackers"] else "—"
        ret = "—" if r.get("retention_vs_clean") is None else f"{r['retention_vs_clean']*100:.1f}%"
        lines.append(
            f"| {r['strategy']} | {r['attack_fraction']:g} | {r['num_attackers']}/{meta['clients']} | "
            f"{wshare} | {r['final_accuracy']:.4f} | {ret} |"
        )

    clean = results[0]["final_accuracy"]
    fa_ret = fedavg_attacked["retention_vs_clean"] * 100
    tm_ret = tmean_attacked["retention_vs_clean"] * 100
    med_ret = median_attacked["retention_vs_clean"] * 100
    clean_learnable = clean >= 0.85
    fedavg_collapsed = fa_ret <= 60.0
    robust_holds = tm_ret >= 90.0 and med_ret >= 90.0
    clean_effect = clean_learnable and fedavg_collapsed and robust_holds

    lines += ["", "## What this shows", ""]
    if clean_effect:
        lines += [
            f"**The effect is clean.** The clean FedAvg baseline reaches **{clean*100:.1f}%** held-out",
            f"accuracy (the task is learnable). Under the {args.attack_fraction*100:.0f}% {args.attack}",
            f"attack, plain FedAvg collapses to **{fedavg_attacked['final_accuracy']*100:.1f}%**",
            f"(retention {fa_ret:.1f}%), while trimmed-mean (beta={meta['trim_beta']:g}) holds at "
            f"**{tmean_attacked['final_accuracy']*100:.1f}%** (retention {tm_ret:.1f}%) and coordinate-wise",
            f"median holds at **{median_attacked['final_accuracy']*100:.1f}%** (retention {med_ret:.1f}%).",
        ]
    else:
        lines += [
            "**The effect is NOT uniformly clean at these settings — reporting the real numbers rather",
            "than tuning to a story.** Observed:",
            f"- clean FedAvg baseline: {clean*100:.1f}% "
            f"({'learnable' if clean_learnable else 'NOT clearly learnable — task/model/rounds may be too weak'}).",
            f"- FedAvg under {args.attack}: {fedavg_attacked['final_accuracy']*100:.1f}% (retention {fa_ret:.1f}%) "
            f"({'collapsed as expected' if fedavg_collapsed else 'did NOT collapse the way the attack intends'}).",
            f"- trimmed-mean under {args.attack}: {tmean_attacked['final_accuracy']*100:.1f}% (retention {tm_ret:.1f}%).",
            f"- median under {args.attack}: {median_attacked['final_accuracy']*100:.1f}% (retention {med_ret:.1f}%).",
            "See the per-round accuracy arrays in the JSON for the full trajectory; diagnosis notes are",
            "in the PR/task description rather than hand-tuned away here.",
        ]

    lines += [
        "",
        f"**Breakdown point**: trimmed-mean (beta={meta['trim_beta']:g}) is only proven to tolerate a",
        f"Byzantine fraction <= beta. The f-sweep above runs f in {sweep_fractions} for both FedAvg and",
        "trimmed-mean at the same beta: FedAvg degrades at every tested f (it has no tolerance), while",
        "trimmed-mean is expected to hold while f <= beta and degrade once f exceeds beta — read the",
        "sweep table's own numbers above for whether that boundary shows up cleanly at this N and attack",
        "scale (small-cohort estimators are noisier near the exact breakdown point than the asymptotic",
        "theory predicts).",
        "",
        "Reproduce: `PYTHONPATH=src python benchmarks/robust_aggregation_attack.py "
        f"--attack {args.attack} --rounds {args.rounds} --clients {args.clients} "
        f"--attack-scale {args.attack_scale:g} --attack-fraction {args.attack_fraction:g} "
        f"--trim-beta {args.trim_beta:g} --sweep-fractions {args.sweep_fractions}`",
        "",
        "See also: `--matrix` for the full attack-family x aggregator ablation "
        "(`robust_aggregation_multiattack.{json,md}`).",
        "",
    ]
    with open(os.path.join(args.out_dir, "robust_aggregation_attack.md"), "w") as f:
        f.write("\n".join(lines) + "\n")

    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
