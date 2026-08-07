"""Phase 2 — the non-DP 2x2 that separates "pretraining helped" from "freezing helped".

Four arms over the same real chest-X-ray data and the same federation:

                     | frozen backbone + federated head | full federated fine-tune
    pretrained init  | B  (the proposal)                | C  (accuracy ceiling)
    random init      | A  (control)                     | D  (true from-scratch)

Phase 1 (`research/results/frozen-backbone/phase1_pretraining_data_efficiency.json`) established the
two constraints this harness is built around:

* The pretrained-vs-random gap is a **data-efficiency** effect — +0.164 AUC at n=20 decaying to
  +0.018 at n=1400. Per-client shard size is therefore a first-class swept factor here, not a
  by-product of the pool size.
* A single shared learning rate is **not a fair control** across arms: lr=0.5 is tuned for pretrained
  features and drives the random-feature head to exactly chance, while a properly tuned probe on the
  same features reaches AUC 0.969. Each arm selects its own LR.
"""
from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch


def dirichlet_partition(labels, num_clients, alpha, seed, per_client=None):
    """Partition example indices across clients by a per-class Dirichlet(alpha) draw.

    Two modes, because shard size is the Phase-1 lever and must vary independently of heterogeneity:

    * ``per_client=None`` — split the whole pool. ``alpha`` governs how each class spreads across
      clients; shard size falls out of the draw and every index is used exactly once.
    * ``per_client=n`` — every client gets exactly ``n`` examples and ``alpha`` instead governs the
      LABEL MIX inside each shard. Shards stay disjoint but the pool is not exhausted.
    """
    labels = torch.as_tensor(labels)
    g = torch.Generator().manual_seed(int(seed))
    # NB: torch.distributions.Dirichlet.sample() draws from the GLOBAL RNG and cannot be handed a
    # generator, so seeding it would silently leak run-to-run state. numpy's RandomState is scoped.
    rng = np.random.RandomState(int(seed))
    classes = labels.unique().tolist()
    pools = {}
    for cls in classes:
        idx = (labels == cls).nonzero(as_tuple=True)[0]
        pools[cls] = idx[torch.randperm(len(idx), generator=g)].tolist()

    if per_client is None:
        # Split the WHOLE pool: alpha governs how each class spreads across clients, and shard size
        # falls out of the draw.
        parts = [[] for _ in range(num_clients)]
        for cls in classes:
            idx = pools[cls]
            props = torch.from_numpy(rng.dirichlet([float(alpha)] * num_clients))
            cuts = (torch.cumsum(props, 0) * len(idx)).long().tolist()
            prev = 0
            for c in range(num_clients):
                end = cuts[c] if c < num_clients - 1 else len(idx)
                parts[c].extend(idx[prev:end])
                prev = end
        return parts

    # Fixed shard size: alpha governs the LABEL MIX *within* each client's shard, while every client
    # gets exactly `per_client` examples. Phase 1 made shard size the experiment's key factor, so it
    # must vary independently of alpha and of the client count rather than being a truncation of a
    # whole-pool split (which starves clients at realistic N and aborts the sweep).
    total = per_client * num_clients
    available = sum(len(p) for p in pools.values())
    if total > available:
        raise ValueError(
            f"need {total} examples ({num_clients} clients x {per_client}) but the pool holds "
            f"{available}; lower per_client or the client count"
        )

    parts = []
    for _c in range(num_clients):
        mix = rng.dirichlet([float(alpha)] * len(classes))
        counts = [int(round(m * per_client)) for m in mix]
        shard = []
        for cls, want in zip(classes, counts):
            take = min(want, len(pools[cls]))
            shard.extend(pools[cls][:take])
            del pools[cls][:take]
        # Top up (or trim) to land on exactly per_client after rounding and pool exhaustion.
        for cls in sorted(classes, key=lambda k: -len(pools[k])):
            while len(shard) < per_client and pools[cls]:
                shard.append(pools[cls].pop(0))
        parts.append(shard[:per_client])

    return parts


def label_skew(parts, labels):
    """Mean total-variation distance between each client's label distribution and the global one.

    0 = every client mirrors the global mix (IID); higher = more label skew. Used to verify the
    alpha knob actually changes heterogeneity rather than just reshuffling.
    """
    labels = torch.as_tensor(labels)
    classes = labels.unique().tolist()
    glob = torch.tensor([(labels == c).sum().item() for c in classes], dtype=torch.float)
    glob = glob / glob.sum()

    tvs = []
    for p in parts:
        if not len(p):
            continue
        y = labels[torch.as_tensor(p)]
        d = torch.tensor([(y == c).sum().item() for c in classes], dtype=torch.float)
        d = d / d.sum()
        tvs.append(0.5 * (d - glob).abs().sum().item())
    return sum(tvs) / len(tvs)


# The 2x2. Both factors vary independently, which is what lets A/B isolate the value of pretrained
# features while B/C isolates the cost of freezing. Dropping either level collapses the design.
ARMS = {
    "A": {"pretrained": False, "mode": "frozen", "label": "random frozen backbone + federated head"},
    "B": {"pretrained": True,  "mode": "frozen", "label": "pretrained frozen backbone + federated head"},
    "C": {"pretrained": True,  "mode": "full",   "label": "pretrained backbone, full federated fine-tune"},
    "D": {"pretrained": False, "mode": "full",   "label": "random init, full federated fine-tune"},
}


def arm_spec(arm):
    """Factor levels for one arm of the 2x2."""
    if arm not in ARMS:
        raise KeyError(f"unknown arm {arm!r}; expected one of {sorted(ARMS)}")
    return ARMS[arm]


NORMS = ("batch", "group")

# GroupNorm's paper default. Reduced per site when it does not divide the channel count, which never
# happens for ResNet's 64/128/256/512 stages but does for narrower backbones.
GN_MAX_GROUPS = 32


def convert_bn_to_gn(module, max_groups=GN_MAX_GROUPS):
    """Replace every ``BatchNorm2d`` with a ``GroupNorm`` over the same channels, in place.

    Two independent reasons, either of which would justify it:

    * **Portability.** ExecuTorch's trainable export rejects BatchNorm —
      ``_native_batch_norm_legit_functional`` is not in the Core ATen opset — so a BatchNorm arm
      cannot run on the mobile client at all. GroupNorm exports cleanly and needs only the two
      backward kernels portable ships.
    * **Correctness under non-IID.** BatchNorm estimates running statistics per client and then
      averages them across clients, which is a documented federated failure mode (Hsieh et al. 2020).
      GroupNorm carries no running statistics.

    Conv weights are untouched, so a pretrained backbone stays pretrained; only the (cheap) norm
    affine parameters are re-initialised.
    """
    for name, child in module.named_children():
        if isinstance(child, torch.nn.BatchNorm2d):
            c = child.num_features
            groups = min(max_groups, c)
            while c % groups:
                groups -= 1
            setattr(module, name, torch.nn.GroupNorm(groups, c))
        else:
            convert_bn_to_gn(child, max_groups=max_groups)
    return module


def frozen_backbone_bytes(backbone_name="resnet18"):
    """One-time server->client delivery of the FROZEN backbone, in production-codec bytes.

    The frozen arms upload only a head (a few KB per round), but a client cannot produce features
    without the backbone, and the backbone has to reach the device once. Quoting the per-round figure
    alone flatters the design by hiding that delivery. The DeComFL accounting already separates its
    one-shot download (`decomfl_oneshot_download_bytes`); this is the same term for this experiment.

    Measured with the same safetensors path the socket uses, so it is directly comparable to the
    per-round numbers rather than a parameter-count estimate.
    """
    from collections import OrderedDict

    from torchvision import models

    from benchmarks.wire_bytes import first_order_model_bytes

    net = getattr(models, backbone_name)(weights=None)
    net.fc = torch.nn.Identity()
    return first_order_model_bytes(OrderedDict((n, p.detach()) for n, p in net.named_parameters()))


def _peak_rss_mb():
    """Peak resident set size for this process, in MB. The memory axis that decides whether an arm
    fits on a client at all — the backward-pass activation spike is what OOMs low-RAM devices."""
    import resource
    import sys

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports KB, macOS reports bytes.
    return round(peak / (1024 * 1024 if sys.platform == "darwin" else 1024), 2)


def build_model(arm, *, feat_dim, n_classes, backbone_name="resnet18", seed=0, norm="batch"):
    """The trainable surface for one arm.

    ``frozen`` arms consume PRE-EXTRACTED features, so the model is just the linear head — the frozen
    backbone is never instantiated here because it never trains and never rides the wire. ``full``
    arms need the real backbone with its classifier resized, every parameter trainable.

    ``norm`` selects the normalisation layer for the ``full`` arms. ``"batch"`` is the default so the
    committed B-vs-C record stays reproducible; ``"group"`` is the configuration that can actually be
    exported for on-device training (see :func:`convert_bn_to_gn`). Frozen arms ignore it — a linear
    head has no norm layer.
    """
    if norm not in NORMS:
        raise ValueError(f"unknown norm {norm!r}; expected one of {NORMS}")

    spec = arm_spec(arm)
    torch.manual_seed(int(seed))

    if spec["mode"] == "frozen":
        return torch.nn.Linear(feat_dim, n_classes)

    from torchvision import models

    net = getattr(models, backbone_name)(weights="DEFAULT" if spec["pretrained"] else None)
    net.fc = torch.nn.Linear(net.fc.in_features, n_classes)
    if norm == "group":
        convert_bn_to_gn(net)
    for p in net.parameters():
        p.requires_grad_(True)
    return net


def round_wire_bytes(model):
    """Per-round upload bytes for this arm, via the SAME safetensors codec the socket uses.

    Only trainable tensors ride the wire (the frozen-backbone subset-federation contract), so this is
    a real measurement of the communication axis rather than a parameter-count estimate.
    """
    from collections import OrderedDict

    from benchmarks.wire_bytes import first_order_model_bytes

    trainable = OrderedDict(
        (n, p.detach()) for n, p in model.named_parameters() if p.requires_grad
    )
    return first_order_model_bytes(trainable)


def fit_head(x, y, *, lr, epochs=40, seed=0, weight_decay=0.0):
    """Train a linear head on features with plain full-batch SGD. Deterministic given ``seed``."""
    torch.manual_seed(int(seed))
    model = torch.nn.Linear(x.shape[1], int(y.max().item()) + 1)
    opt = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = torch.nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        if not torch.isfinite(loss):
            break  # diverged: stop rather than propagate NaNs into the reported metric
        loss.backward()
        opt.step()
    return model


def auc_from_logits(logits, y):
    """Binary ROC-AUC from raw logits via the rank identity (no sklearn in the hot path).

    Non-finite logits mean the run diverged; that is reported as chance rather than crashing, so a
    bad candidate scores badly instead of taking the whole sweep down.
    """
    if not torch.isfinite(logits).all():
        return 0.5

    def _binary_auc(scores, pos_mask):
        n_pos, n_neg = int(pos_mask.sum()), int((~pos_mask).sum())
        if n_pos == 0 or n_neg == 0:
            return None  # a class absent from this split contributes nothing
        ranks = scores.argsort().argsort().double() + 1.0
        return float((ranks[pos_mask].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))

    k = logits.shape[1]
    if k == 2:
        # Binary: unchanged from the original definition, so every existing number reproduces.
        v = _binary_auc((logits[:, 1] - logits[:, 0]).double(), y == 1)
        return float("nan") if v is None else v

    # Multi-class: macro one-vs-rest. Needed because the campaign's binary-only AUC silently
    # returns values outside [0,1] on >2 columns (measured: 1.25) rather than failing, which would
    # corrupt any many-class result. Classes absent from the split are skipped, not counted as 0.
    vals = []
    for c in range(k):
        rest = torch.cat([logits[:, :c], logits[:, c + 1:]], dim=1).max(dim=1).values
        v = _binary_auc((logits[:, c] - rest).double(), y == c)
        if v is not None:
            vals.append(v)
    return float(sum(vals) / len(vals)) if vals else float("nan")


def head_auc(model, x, y):
    """Binary ROC-AUC via the rank identity (no sklearn dependency in the hot path).

    A diverged model produces non-finite logits; that is reported as chance rather than crashing,
    so a bad LR candidate scores badly instead of taking the whole sweep down.
    """
    model.eval()
    with torch.no_grad():
        return auc_from_logits(model(x), y)


def select_lr(x, y, *, candidates, seed=0, val_frac=0.3, epochs=40):
    """Pick the LR with the best HELD-OUT AUC.

    Phase 1 showed a single shared LR is not a fair control across arms — lr=0.5 is tuned for
    pretrained features and drives the larger-scale random-feature head to exactly chance. Selecting
    per arm is what makes the A-vs-B comparison honest rather than an artefact of one arm's tuning.
    Ties break toward the SMALLER LR, which is the more stable choice at equal measured utility.
    """
    g = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(len(y), generator=g)
    n_val = max(1, int(len(y) * val_frac))
    val, tr = perm[:n_val], perm[n_val:]

    best_lr, best_auc = None, -1.0
    for lr in sorted(candidates):
        auc = head_auc(fit_head(x[tr], y[tr], lr=lr, epochs=epochs, seed=seed), x[val], y[val])
        if auc == auc and auc > best_auc:  # NaN-safe
            best_lr, best_auc = lr, auc
    return best_lr


def should_stop_early(history, *, patience=3, min_delta=1e-3):
    """True when the metric has not improved on its BEST value for ``patience`` rounds.

    The first 2x2 matrix ran a fixed 20-round budget and 79% of its cells finished unconverged, so
    most numbers reported where the budget stopped the arm rather than where the arm tops out —
    which makes cross-arm comparison a comparison of convergence rates, not of quality. Training to
    a plateau removes that confound.

    Compared against the best value so far (not the last), so one bad round cannot be mistaken for
    convergence.
    """
    if len(history) < patience + 1:
        return False
    best_before = max(history[:-patience])
    return all(v - best_before < min_delta for v in history[-patience:])


DEFAULT_LR_CANDIDATES = (1.0, 0.5, 0.1, 0.05, 0.01, 0.005)


def run_arm(arm, *, train_x, train_y, test_x, test_y, clients, clients_per_round, alpha,
            rounds, local_epochs, per_client=None, seed=0, lr_candidates=DEFAULT_LR_CANDIDATES,
            weight_decay=0.0, patience=None, min_delta=1e-3, backbone_name="resnet18"):
    """One federated arm over pre-extracted frozen-backbone features.

    Uses the production ``FedAvgAggregator`` and the subset-federation contract, so only the head
    rides the wire and the byte count is a real codec measurement. The LR is selected per arm on a
    held-out split (Phase 1: a shared LR is not a fair control).

    Returns per-round curves plus a ``meta`` provenance block, per the repo's benchmark-recording rule.
    """
    import time

    from collections import OrderedDict

    from fedlearn.server.strategy import FedAvgAggregator

    spec = arm_spec(arm)
    if spec["mode"] == "full":
        raise ValueError(
            f"arm {arm} ({spec['label']}) is a full fine-tune and needs IMAGES, not pre-extracted "
            f"features — features come from a frozen backbone and cannot train one. Use run_full_arm()."
        )

    parts = dirichlet_partition(train_y, clients, alpha, seed, per_client=per_client)
    lr = select_lr(train_x, train_y, candidates=list(lr_candidates), seed=seed)

    n_classes = int(train_y.max().item()) + 1
    torch.manual_seed(int(seed))
    server = torch.nn.Linear(train_x.shape[1], n_classes)
    wire_bytes = round_wire_bytes(server)

    g = torch.Generator().manual_seed(int(seed))
    per_round = []
    cum_up = cum_down = 0
    t0 = time.time()
    for rnd in range(1, rounds + 1):
        r0 = time.time()
        chosen = torch.randperm(clients, generator=g)[:clients_per_round].tolist()
        updates = []
        for cid in chosen:
            idx = torch.as_tensor(parts[cid])
            if not len(idx):
                continue
            local = torch.nn.Linear(train_x.shape[1], n_classes)
            local.load_state_dict(server.state_dict())
            opt = torch.optim.SGD(local.parameters(), lr=lr, weight_decay=weight_decay)
            loss_fn = torch.nn.CrossEntropyLoss()
            local.train()
            for _ in range(local_epochs):
                opt.zero_grad()
                loss = loss_fn(local(train_x[idx]), train_y[idx])
                if not torch.isfinite(loss):
                    break
                loss.backward()
                opt.step()
            updates.append((f"c{cid}", OrderedDict(
                (k, v.detach().clone()) for k, v in local.state_dict().items()), len(idx)))

        if updates:
            server.load_state_dict(FedAvgAggregator().aggregate(updates))

        auc = head_auc(server, test_x, test_y)
        with torch.no_grad():
            acc = float((server(test_x).argmax(1) == test_y).float().mean())
        # Bidirectional and ACCUMULATED (not participants x round_index, which is only correct when
        # participation never varies — clients with empty shards are skipped, so it does).
        up_r = wire_bytes * len(updates)
        down_r = wire_bytes * len(updates)   # the server broadcasts the same head it receives back
        cum_up += up_r
        cum_down += down_r
        per_round.append({"round": rnd, "auc": round(auc, 4), "accuracy": round(acc, 4),
                          "participants": len(updates),
                          "round_sec": round(time.time() - r0, 2),
                          "bytes_up_round": up_r, "bytes_down_round": down_r,
                          "cum_bytes_up": cum_up, "cum_bytes_down": cum_down,
                          "cum_bytes_total": cum_up + cum_down,
                          "cum_wire_bytes_up": cum_up})
        # Train to a PLATEAU rather than a fixed count: `rounds` becomes a cap, not the budget.
        if patience and should_stop_early([r["auc"] for r in per_round],
                                          patience=patience, min_delta=min_delta):
            break

    return _summarize(arm, per_round, {
        "patience": patience, "min_delta": min_delta,
        "rounds_run": len(per_round), "rounds_cap": rounds,
        "stopped_early": bool(patience) and len(per_round) < rounds,
        # --- communication, both directions + the one-shot term (see frozen_backbone_bytes) ---
        "wire_bytes_up_per_client_round": wire_bytes,
        "wire_bytes_down_per_client_round": wire_bytes,
        "oneshot_backbone_download_bytes": frozen_backbone_bytes(backbone_name),
        "cum_bytes_up": cum_up, "cum_bytes_down": cum_down,
        "cum_bytes_total": cum_up + cum_down,
        # --- compute ---
        "peak_rss_mb": _peak_rss_mb(),
        "trainable_params": sum(p.numel() for p in server.parameters() if p.requires_grad),
        "total_sec": round(time.time() - t0, 1),
        "pretrained": spec["pretrained"], "mode": spec["mode"],
        "clients": clients, "clients_per_round": clients_per_round, "alpha": alpha,
        "rounds": rounds, "local_epochs": local_epochs, "per_client": per_client,
        "shard_sizes": [len(p) for p in parts],
        "label_skew": round(label_skew(parts, train_y), 4),
        "seed": seed, "selected_lr": lr, "lr_candidates": list(lr_candidates),
        "weight_decay": weight_decay,
        "feat_dim": int(train_x.shape[1]), "n_classes": n_classes,
        "n_train": int(len(train_y)), "n_test": int(len(test_y)),
        "wire_bytes_per_client_round": wire_bytes,
        "wire_codec": "safetensors (production first_order_model_bytes)",
        "total_local_steps": rounds * local_epochs,
    })


def _summarize(arm, per_round, meta):
    """Assemble the shared result schema. Frozen and full arms MUST report identically or the
    B-vs-C comparison cannot be made."""
    aucs = [r["auc"] for r in per_round]
    half = max(1, len(aucs) // 2)
    improvement = (sum(aucs[-half:]) / half) - (sum(aucs[:half]) / half) if aucs else float("nan")
    meta = dict(meta)
    # A run that early-stopped DID converge by this harness's own definition (the metric failed to
    # beat its best for `patience` rounds). The half-vs-half test alone is the wrong instrument for
    # that case: a curve that rises steeply then flattens still shows a large half-to-half delta,
    # so it would report converged=False on a run that plainly plateaued. Trust the stop criterion
    # when it fired; fall back to the half test only for fixed-budget runs.
    meta["converged"] = bool(meta.get("stopped_early")) or bool(len(aucs) >= 4 and improvement < 0.01)
    meta["converged_basis"] = "early-stop plateau" if meta.get("stopped_early") else "half-vs-half delta"
    meta["auc_improvement_last_half"] = round(improvement, 4)
    return {
        "arm": arm,
        "label": arm_spec(arm)["label"],
        "final_auc": per_round[-1]["auc"] if per_round else float("nan"),
        "final_accuracy": per_round[-1]["accuracy"] if per_round else float("nan"),
        "best_auc": max(aucs, default=float("nan")),
        "per_round": per_round,
        "meta": meta,
    }


def run_full_arm(arm, *, data_dir, clients, clients_per_round, alpha, rounds, local_epochs,
                 img_size=224, batch_size=32, lr=0.01, momentum=0.9, weight_decay=1e-4,
                 per_client=None, seed=0, device="cpu", backbone_name="resnet18", num_workers=0,
                 patience=None, min_delta=1e-3, norm="batch"):
    """Arms C and D — full federated fine-tune over raw images.

    Every parameter trains and the whole model rides the wire each round, which is precisely the cost
    the frozen arms avoid. Kept schema-identical to :func:`run_arm` so B-vs-C is a like-for-like read.
    """
    import time
    from collections import OrderedDict

    from torch.utils.data import DataLoader, Subset
    from torchvision import transforms
    from torchvision.datasets import ImageFolder

    from fedlearn.server.strategy import FedAvgAggregator

    spec = arm_spec(arm)
    if spec["mode"] != "full":
        raise ValueError(f"arm {arm} is a frozen-head arm; use run_arm() with extracted features")

    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    train_ds = ImageFolder(os.path.join(data_dir, "train"), transform=tf)
    test_ds = ImageFolder(os.path.join(data_dir, "test"), transform=tf)
    train_y = torch.tensor(train_ds.targets)

    parts = dirichlet_partition(train_y, clients, alpha, seed, per_client=per_client)

    torch.manual_seed(int(seed))
    server = build_model(arm, feat_dim=0, n_classes=len(train_ds.classes),
                         backbone_name=backbone_name, seed=seed, norm=norm).to(device)
    wire_bytes = round_wire_bytes(server)
    backbone0 = server.conv1.weight.detach().clone().cpu()

    test_loader = DataLoader(test_ds, batch_size=batch_size, num_workers=num_workers)
    loss_fn = torch.nn.CrossEntropyLoss()
    g = torch.Generator().manual_seed(int(seed))
    per_round, t0 = [], time.time()
    cum_up = cum_down = 0

    for rnd in range(1, rounds + 1):
        r0 = time.time()
        chosen = torch.randperm(clients, generator=g)[:clients_per_round].tolist()
        updates = []
        for cid in chosen:
            if not len(parts[cid]):
                continue
            # `norm` MUST match the server's — the load below is strict, and a BatchNorm client
            # cannot accept a GroupNorm server's state_dict (no running_mean/running_var keys).
            local = build_model(arm, feat_dim=0, n_classes=len(train_ds.classes),
                                backbone_name=backbone_name, seed=seed, norm=norm).to(device)
            local.load_state_dict(server.state_dict())
            opt = torch.optim.SGD(local.parameters(), lr=lr, momentum=momentum,
                                  weight_decay=weight_decay)
            loader = DataLoader(Subset(train_ds, parts[cid]), batch_size=batch_size,
                                shuffle=True, num_workers=num_workers)
            local.train()
            for _ in range(local_epochs):
                for xb, yb in loader:
                    opt.zero_grad()
                    loss = loss_fn(local(xb.to(device)), yb.to(device))
                    if not torch.isfinite(loss):
                        break
                    loss.backward()
                    opt.step()
            updates.append((f"c{cid}", OrderedDict(
                (k, v.detach().cpu().clone()) for k, v in local.state_dict().items()), len(parts[cid])))

        if updates:
            server.load_state_dict(FedAvgAggregator().aggregate(updates))
            server.to(device)

        server.eval()
        logits, ys = [], []
        with torch.no_grad():
            for xb, yb in test_loader:
                logits.append(server(xb.to(device)).cpu())
                ys.append(yb)
        logits, ys = torch.cat(logits), torch.cat(ys)
        auc = auc_from_logits(logits, ys)
        acc = float((logits.argmax(1) == ys).float().mean())
        up_r = wire_bytes * len(updates)
        down_r = wire_bytes * len(updates)   # the server broadcasts the full model it gets back
        cum_up += up_r
        cum_down += down_r
        per_round.append({"round": rnd, "auc": round(auc, 4), "accuracy": round(acc, 4),
                          "participants": len(updates),
                          "round_sec": round(time.time() - r0, 2),
                          "bytes_up_round": up_r, "bytes_down_round": down_r,
                          "cum_bytes_up": cum_up, "cum_bytes_down": cum_down,
                          "cum_bytes_total": cum_up + cum_down,
                          "cum_wire_bytes_up": cum_up})
        if patience and should_stop_early([r["auc"] for r in per_round],
                                          patience=patience, min_delta=min_delta):
            break

    meta = {
        "patience": patience, "min_delta": min_delta,
        "rounds_run": len(per_round), "rounds_cap": rounds,
        "stopped_early": bool(patience) and len(per_round) < rounds,
        "pretrained": spec["pretrained"], "mode": spec["mode"],
        "clients": clients, "clients_per_round": clients_per_round, "alpha": alpha,
        "rounds": rounds, "local_epochs": local_epochs, "per_client": per_client,
        "shard_sizes": [len(p) for p in parts],
        "label_skew": round(label_skew(parts, train_y), 4),
        "seed": seed, "selected_lr": lr, "lr_candidates": [lr],
        "weight_decay": weight_decay, "momentum": momentum,
        "feat_dim": 0, "n_classes": len(train_ds.classes),
        "n_train": len(train_ds), "n_test": len(test_ds),
        "wire_bytes_per_client_round": wire_bytes,
        # --- communication, both directions ---
        "wire_bytes_up_per_client_round": wire_bytes,
        "wire_bytes_down_per_client_round": wire_bytes,
        # The full arm ships the whole model every round, so there is no SEPARATE one-shot delivery
        # to declare — unlike the frozen arms, whose per-round figure omits the backbone.
        "oneshot_backbone_download_bytes": 0,
        "cum_bytes_up": cum_up, "cum_bytes_down": cum_down,
        "cum_bytes_total": cum_up + cum_down,
        # --- compute ---
        "peak_rss_mb": _peak_rss_mb(),
        "trainable_params": sum(p.numel() for p in server.parameters() if p.requires_grad),
        "wire_codec": "safetensors (production first_order_model_bytes)",
        "total_local_steps": rounds * local_epochs,
        "backbone_name": backbone_name, "norm": norm,
        "img_size": img_size, "batch_size": batch_size,
        "device": device, "total_sec": round(time.time() - t0, 1),
        "backbone_changed": not torch.equal(server.conv1.weight.detach().cpu(), backbone0),
    }
    return _summarize(arm, per_round, meta)


def _accepted_kwargs(fn, kw):
    """Subset of ``kw`` that ``fn`` actually declares.

    The CLI builds one kwarg bag for both runners, but they take different options — image-space
    settings (num_workers, img_size, batch_size) are meaningless to the feature-space runner.
    Filtering against the callee's real signature rather than a hardcoded exclusion list means
    adding an option to one runner can never crash the other mid-sweep.
    """
    import inspect

    allowed = set(inspect.signature(fn).parameters)
    return {k: v for k, v in kw.items() if k in allowed}


def _emit_run(out_dir, run):
    """Persist ONE completed cell immediately and return its path.

    A 24-cell sweep runs for hours. Writing the combined payload only at the end means any
    interruption discards every finished cell — which is what happened on the first attempt and
    forced the results to be rebuilt from stdout. One file per cell makes partial sweeps durable by
    construction; the combined payload is still written at the end for convenience.
    """
    import json

    os.makedirs(out_dir, exist_ok=True)
    m = run.get("meta", {})
    # Every factor that VARIES across a sweep must appear in the name. alpha was missing, so a
    # multi-alpha sweep silently overwrote its own per-cell files (the combined payload kept the
    # data, but the per-cell copies were lost). Include alpha and backbone.
    name = (f"{run['arm']}_shard{m.get('per_client', 'all')}"
            f"_a{m.get('alpha', 'na')}_{m.get('backbone_name', 'feat')}"
            f"_{m.get('norm', 'batch')}"
            f"_seed{m.get('seed', 0)}.json")
    path = os.path.join(out_dir, name)
    with open(path, "w") as fh:
        json.dump(run, fh, indent=2)
    return path


def _run_one(arm, *, data_dir, backbone_name, device, feature_cache, **kw):
    """Dispatch an arm to the right runner: frozen arms consume cached features, full arms images."""
    if arm_spec(arm)["mode"] == "frozen":
        from benchmarks.dp_on_head_xray import extract_features

        key = (backbone_name, arm_spec(arm)["pretrained"])
        if key not in feature_cache:
            feature_cache[key] = extract_features(
                data_dir, backbone=backbone_name, pretrained=key[1],
                img_size=kw.get("img_size", 224), device=device, backbone_seed=kw.get("seed", 0))
        f = feature_cache[key]
        return run_arm(arm,
                       train_x=f["train_x"].cpu(), train_y=f["train_y"].cpu(),
                       test_x=f["test_x"].cpu(), test_y=f["test_y"].cpu(),
                       **_accepted_kwargs(run_arm, kw))
    return run_full_arm(arm, data_dir=data_dir, backbone_name=backbone_name, device=device,
                        **_accepted_kwargs(run_full_arm, kw))


def main():
    import argparse
    import json
    import platform
    import time

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=os.environ.get(
        "FEDLEARN_PNEUMONIA_DIR", os.path.expanduser("~/fedlearn-demo/chest_xray")))
    ap.add_argument("--arms", default="A,B,C,D")
    ap.add_argument("--backbone", default="resnet18")
    ap.add_argument("--norm", default="batch", choices=list(NORMS),
                    help="normalisation for the FULL arms (C/D). 'group' is the only variant that can "
                         "be exported for on-device training (ExecuTorch rejects BatchNorm) and is "
                         "also the federated-correct choice under non-IID data. Frozen arms ignore it.")
    ap.add_argument("--per-client", default="10,25,70", help="shard-size sweep (the Phase-1 lever)")
    ap.add_argument("--seeds", default="0", help="comma-separated; multi-seed closes the campaign's biggest gap")
    ap.add_argument("--clients", type=int, default=20)
    ap.add_argument("--clients-per-round", type=int, default=10)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--local-epochs", type=int, default=3)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--patience", type=int, default=None,
                    help="early-stop patience in rounds; --rounds becomes a CAP. Omit for a fixed budget.")
    ap.add_argument("--min-delta", type=float, default=1e-3, help="AUC improvement that counts as progress")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(_ROOT), "research", "results",
                                                  "frozen-backbone", "frozen_vs_finetune_xray.json"))
    args = ap.parse_args()

    device = args.device
    if device == "auto":
        device = ("cuda" if torch.cuda.is_available() else
                  "mps" if torch.backends.mps.is_available() else "cpu")

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    shards = [int(x) for x in args.per_client.split(",") if x.strip()]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]

    cache, runs, t0 = {}, [], time.time()
    for seed in seeds:
        for per_client in shards:
            for arm in arms:
                tag = f" · norm={args.norm}" if arm_spec(arm)["mode"] == "full" else ""
                print(f"[*] arm {arm} · shard {per_client} · seed {seed}{tag} on {device}", flush=True)
                r = _run_one(arm, data_dir=args.data_dir, backbone_name=args.backbone,
                             device=device, feature_cache=cache,
                             clients=args.clients, clients_per_round=args.clients_per_round,
                             alpha=args.alpha, rounds=args.rounds, local_epochs=args.local_epochs,
                             per_client=per_client, seed=seed,
                             img_size=args.img_size, batch_size=args.batch_size,
                             num_workers=args.num_workers, norm=args.norm,
                             patience=args.patience, min_delta=args.min_delta)
                # Persist THIS cell before starting the next one — a multi-hour sweep must not lose
                # finished work to an interruption (see _emit_run).
                cell_path = _emit_run(os.path.join(os.path.dirname(args.out), "cells"), r)
                print(f"    auc={r['final_auc']:.4f} acc={r['final_accuracy']:.4f} "
                      f"converged={r['meta']['converged']} rounds={r['meta'].get('rounds_run')} "
                      f"wire={r['meta']['wire_bytes_per_client_round']}B -> {os.path.basename(cell_path)}",
                      flush=True)
                runs.append(r)

    payload = {
        "experiment": "frozen-backbone vs full fine-tune 2x2 on real chest X-ray (non-DP)",
        "meta": {
            "platform": platform.platform(), "device": device,
            "torch": torch.__version__, "backbone": args.backbone,
            "data_dir": args.data_dir, "clients": args.clients,
            "clients_per_round": args.clients_per_round, "alpha": args.alpha,
            "rounds": args.rounds, "local_epochs": args.local_epochs,
            "shard_sweep": shards, "seeds": seeds, "arms": arms,
            "total_sec": round(time.time() - t0, 1),
        },
        "arms_legend": ARMS,
        "runs": runs,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(payload, fh, indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()


def extract_medmnist_features(name, *, backbone="resnet18", pretrained=True, img_size=224,
                              device="cpu", batch_size=128, cache_dir=None, subset=None):
    """Frozen-backbone features for a MedMNIST dataset, cached to disk.

    The chest-X-ray task this experiment was built on is binary and only 1,400 training images, which
    has been the standing limitation on every conclusion drawn from it. MedMNIST supplies genuinely
    out-of-domain medical datasets that are both far larger and MULTI-CLASS (e.g. pathmnist: 107,180
    images, 9 classes), without the ImageNet contamination that rules out ImageNet-100 here.

    Returns the same dict shape as dp_on_head_xray.extract_features so the federated runners consume
    it unchanged.
    """
    import hashlib

    import numpy as np
    import torch.nn as nn
    from medmnist import INFO
    import medmnist as mm
    from torchvision import models, transforms

    info = INFO[name]
    n_classes = len(info["label"])
    DataClass = getattr(mm, info["python_class"])

    cache_dir = cache_dir or os.path.join(_ROOT, "benchmarks", "results", "feature_cache")
    os.makedirs(cache_dir, exist_ok=True)
    key = hashlib.sha256(
        f"{name}|{backbone}|{pretrained}|{img_size}|{subset}".encode()).hexdigest()[:16]
    cache_path = os.path.join(cache_dir, f"medmnist_{name}_{backbone}_"
                                         f"{'pt' if pretrained else 'rand'}_{key}.pt")
    if os.path.exists(cache_path):
        return torch.load(cache_path, weights_only=True)

    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: t.repeat(3, 1, 1) if t.shape[0] == 1 else t),  # grayscale -> RGB
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    torch.manual_seed(0)
    net = getattr(models, backbone)(weights="DEFAULT" if pretrained else None)
    feat_dim = net.fc.in_features
    net.fc = nn.Identity()
    net = net.to(device).eval()

    out = {}
    for split in ("train", "test"):
        ds = DataClass(split=split, transform=tf, download=True, size=64)
        idx = list(range(len(ds)))
        if subset:
            idx = idx[:subset]
        loader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(ds, idx), batch_size=batch_size, num_workers=4)
        feats, labels = [], []
        with torch.no_grad():
            for xb, yb in loader:
                feats.append(net(xb.to(device)).cpu())
                labels.append(yb.squeeze(-1).long())
        out[f"{split}_x"] = torch.cat(feats)
        out[f"{split}_y"] = torch.cat(labels)

    out.update({"feat_dim": feat_dim, "backbone": backbone, "pretrained": pretrained,
                "img_size": img_size, "dataset": name, "n_classes": n_classes})
    torch.save(out, cache_path)
    return out
