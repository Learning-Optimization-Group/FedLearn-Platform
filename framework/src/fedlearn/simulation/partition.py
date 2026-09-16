"""Seeded, native dataset partitioners for federated simulation (P0-2a).

Replaces ``flwr_datasets.FederatedDataset`` partitioning. That dependency is the reason
``backend/fl-platform-api/requirements.txt`` pins ``cryptography<45.0.0``, which makes the
framework's ``>=46.0.6`` security floor unreachable (the SE-22 residual). Dropping it also
removes the situation where a competitor's package governs this platform's security posture.

Every partitioner here returns ``List[np.ndarray]`` of *index arrays into the dataset* — not
data — so a partitioner is independent of how the dataset is stored, is cheap to hold for
thousands of clients, and can be recorded verbatim in a result's ``meta`` block.

All four satisfy the same three-part contract, enforced by ``tests/test_simulation_partition.py``:

* **complete** — the union of client index sets is exactly ``range(n_samples)``;
* **disjoint** — no index is held by two clients;
* **deterministic in ``seed``** — and genuinely dependent on it.

The heterogeneity knobs are deliberately different shapes, because the literature uses all
of them and a result is only comparable to a paper that used the same one:

============================  =========================================================
partitioner                   how heterogeneity is controlled
============================  =========================================================
``iid_partition``             none — the IID control arm
``dirichlet_partition``       ``alpha``: smaller is more skewed (the common modern default)
``shard_partition``           sort-and-shard; ``shards_per_client`` bounds classes per client
``pathological_partition``    exactly ``classes_per_client`` classes per client
============================  =========================================================
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

__all__ = [
    "iid_partition",
    "dirichlet_partition",
    "shard_partition",
    "pathological_partition",
    "partition_report",
]


def _check_clients(num_clients: int) -> None:
    if num_clients < 1:
        raise ValueError(f"num_clients must be >= 1, got {num_clients}")


def _as_labels(labels: Sequence) -> np.ndarray:
    arr = np.asarray(labels)
    if arr.ndim != 1:
        raise ValueError(f"labels must be 1-D, got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError("labels must be non-empty")
    return arr


# --------------------------------------------------------------------------------------
# IID
# --------------------------------------------------------------------------------------

def iid_partition(n_samples: int, num_clients: int, seed: int) -> List[np.ndarray]:
    """Shuffle and deal into ``num_clients`` near-equal parts (sizes differ by at most 1).

    The control arm: any non-IID effect must be measured against this, not against a
    centralized baseline, or the comparison confounds federation with heterogeneity.
    """
    _check_clients(num_clients)
    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples}")

    rng = np.random.default_rng(seed)
    idx = rng.permutation(n_samples)
    return [np.sort(part) for part in np.array_split(idx, num_clients)]


# --------------------------------------------------------------------------------------
# Dirichlet
# --------------------------------------------------------------------------------------

def dirichlet_partition(
    labels: Sequence,
    num_clients: int,
    alpha: float,
    seed: int,
    min_partition_size: int = 0,
) -> List[np.ndarray]:
    """Label-skewed split: for each class, draw client proportions from ``Dir(alpha)``.

    This is the standard modern non-IID protocol. ``alpha -> 0`` drives each class onto a
    few clients (near-pathological); ``alpha -> inf`` reproduces the global label
    distribution on every client (IID).

    Args:
        labels: per-sample class labels, length ``n_samples``.
        num_clients: number of partitions to produce.
        alpha: Dirichlet concentration. Must be > 0.
        seed: RNG seed; the same seed reproduces the split exactly.
        min_partition_size: if > 0, guarantee every client holds at least this many samples.
            Low ``alpha`` naturally produces empty clients, and an empty client is not a
            benign edge case — ``num_examples == 0`` is rejected at coordinator ingress
            (``coordinator.py``) and would silently shrink the effective cohort.

    Raises:
        ValueError: if ``alpha <= 0``, or if ``min_partition_size`` is unsatisfiable
            (``num_clients * min_partition_size > n_samples``).
    """
    _check_clients(num_clients)
    y = _as_labels(labels)
    n = y.size

    if alpha <= 0:
        raise ValueError(f"alpha must be > 0, got {alpha}")
    if min_partition_size < 0:
        raise ValueError(f"min_partition_size must be >= 0, got {min_partition_size}")
    if num_clients * min_partition_size > n:
        raise ValueError(
            f"min_partition_size={min_partition_size} is unsatisfiable: "
            f"{num_clients} clients x {min_partition_size} > {n} samples"
        )

    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    buckets: List[List[np.ndarray]] = [[] for _ in range(num_clients)]

    for c in classes:
        c_idx = np.where(y == c)[0]
        rng.shuffle(c_idx)
        proportions = rng.dirichlet(np.repeat(alpha, num_clients))
        # Cut points from the cumulative proportions. Using a single cumulative cut (rather
        # than per-client rounding) is what guarantees completeness and disjointness: every
        # element of c_idx lands in exactly one slice, whatever the proportions are.
        cuts = (np.cumsum(proportions) * len(c_idx)).astype(int)[:-1]
        for client_id, chunk in enumerate(np.split(c_idx, cuts)):
            if chunk.size:
                buckets[client_id].append(chunk)

    parts = [
        np.sort(np.concatenate(b)) if b else np.array([], dtype=np.int64)
        for b in buckets
    ]

    if min_partition_size > 0:
        parts = _repair_min_size(parts, min_partition_size)

    return parts


def _repair_min_size(parts: List[np.ndarray], min_size: int) -> List[np.ndarray]:
    """Deterministically move samples from the largest clients to under-filled ones.

    Preferred over the usual "redraw until it fits" loop for two reasons: it always
    terminates (feasibility is checked by the caller), and it does not consume a variable
    number of RNG draws — which would make the *rest* of a run's randomness depend on how
    many redraws happened, quietly breaking reproducibility across alpha values.

    Tie-breaks are by lowest client index throughout, so the repair is a pure function of
    its input.
    """
    work = [list(p) for p in parts]
    guard = sum(len(p) for p in work) + 1  # strictly decreasing potential; cannot loop forever

    while guard > 0:
        deficits = [i for i, p in enumerate(work) if len(p) < min_size]
        if not deficits:
            break
        receiver = deficits[0]
        # Largest donor that stays at or above the floor after giving one away.
        donors = [i for i, p in enumerate(work) if i != receiver and len(p) > min_size]
        if not donors:
            # Feasibility was checked by the caller, so this is unreachable; fail loudly
            # rather than return a partition that silently violates the requested floor.
            raise ValueError(
                f"cannot satisfy min_partition_size={min_size}: no donor above the floor"
            )
        donor = max(donors, key=lambda i: (len(work[i]), -i))
        work[donor].sort()
        work[receiver].append(work[donor].pop())
        guard -= 1

    return [np.sort(np.asarray(p, dtype=np.int64)) for p in work]


# --------------------------------------------------------------------------------------
# Shard (McMahan et al., sort-and-shard)
# --------------------------------------------------------------------------------------

def shard_partition(
    labels: Sequence,
    num_clients: int,
    shards_per_client: int,
    seed: int,
) -> List[np.ndarray]:
    """Sort by label, cut into ``num_clients * shards_per_client`` contiguous shards, deal them.

    The original FedAvg non-IID construction. Because shards are label-contiguous, a client
    holding ``s`` shards sees at most ``s + 1`` distinct classes — one more than ``s`` only
    when a shard straddles a class boundary. ``shards_per_client=2`` is the canonical setting.
    """
    _check_clients(num_clients)
    if shards_per_client < 1:
        raise ValueError(f"shards_per_client must be >= 1, got {shards_per_client}")

    y = _as_labels(labels)
    n = y.size
    n_shards = num_clients * shards_per_client
    if n_shards > n:
        raise ValueError(
            f"{n_shards} shards requested ({num_clients} x {shards_per_client}) "
            f"but only {n} samples available"
        )

    # Stable sort so ties within a class are ordered by index — the sort itself contributes
    # no randomness, keeping the seed the only source of variation.
    order = np.argsort(y, kind="stable")
    shards = np.array_split(order, n_shards)

    rng = np.random.default_rng(seed)
    shard_order = rng.permutation(n_shards)

    parts: List[np.ndarray] = []
    for i in range(num_clients):
        picked = shard_order[i * shards_per_client:(i + 1) * shards_per_client]
        parts.append(np.sort(np.concatenate([shards[s] for s in picked])))
    return parts


# --------------------------------------------------------------------------------------
# Pathological (exactly k classes per client)
# --------------------------------------------------------------------------------------

def pathological_partition(
    labels: Sequence,
    num_clients: int,
    classes_per_client: int,
    seed: int,
) -> List[np.ndarray]:
    """Give every client exactly ``classes_per_client`` distinct classes.

    The worst-case heterogeneity arm, and the one that most cleanly separates methods that
    depend on local label coverage (linear probing on a frozen encoder) from methods that do
    not. Unlike ``shard_partition`` the class count per client is exact, not bounded.

    Classes are dealt from a seeded permutation using a rotating stride, which guarantees
    distinctness within a client and even coverage across classes.

    Raises:
        ValueError: if ``classes_per_client`` exceeds the number of distinct classes, or if
            ``num_clients * classes_per_client`` is too small to cover every class (which
            would drop samples and violate completeness).
    """
    _check_clients(num_clients)
    y = _as_labels(labels)
    classes = np.unique(y)
    n_classes = classes.size

    if classes_per_client < 1:
        raise ValueError(f"classes_per_client must be >= 1, got {classes_per_client}")
    if classes_per_client > n_classes:
        raise ValueError(
            f"classes_per_client={classes_per_client} exceeds the {n_classes} distinct "
            f"classes present"
        )
    if num_clients * classes_per_client < n_classes:
        raise ValueError(
            f"classes_per_client={classes_per_client} x {num_clients} clients cannot cover "
            f"{n_classes} classes; some class would have no holder and its samples would be "
            f"dropped"
        )

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_classes)

    # Rotating stride: client i takes perm[(i*k + j) % n_classes]. Distinct within a client
    # because k <= n_classes, and every class is taken at least once because
    # num_clients * k >= n_classes.
    assignment: List[np.ndarray] = [
        perm[[(i * classes_per_client + j) % n_classes for j in range(classes_per_client)]]
        for i in range(num_clients)
    ]

    holders: Dict[int, List[int]] = {int(c): [] for c in range(n_classes)}
    for client_id, cls_slots in enumerate(assignment):
        for c in cls_slots:
            holders[int(c)].append(client_id)

    buckets: List[List[np.ndarray]] = [[] for _ in range(num_clients)]
    for c_pos, c in enumerate(classes):
        owners = holders[c_pos]
        c_idx = np.where(y == c)[0]
        rng.shuffle(c_idx)
        for client_id, chunk in zip(owners, np.array_split(c_idx, len(owners))):
            if chunk.size:
                buckets[client_id].append(chunk)

    return [
        np.sort(np.concatenate(b)) if b else np.array([], dtype=np.int64)
        for b in buckets
    ]


# --------------------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------------------

def partition_report(parts: Sequence[np.ndarray], labels: Sequence) -> dict:
    """Summarise a partition into a JSON-serializable block for a result's ``meta``.

    A result that records only "alpha=0.5" cannot be checked later; a result that records the
    realised client sizes and label skew can. ``mean_max_class_share`` is the headline
    statistic: 1.0 means every client holds a single class, ``1/n_classes`` means every client
    mirrors the global distribution.
    """
    y = _as_labels(labels)
    n_classes = int(np.unique(y).size)

    sizes = [int(len(p)) for p in parts]
    shares: List[float] = []
    for p in parts:
        if len(p) == 0:
            continue
        counts = np.bincount(y[np.asarray(p, dtype=np.int64)], minlength=int(y.max()) + 1)
        shares.append(float(counts.max() / counts.sum()))

    return {
        "num_clients": len(parts),
        "total_samples": int(sum(sizes)),
        "num_classes": n_classes,
        "client_sizes": sizes,
        "min_client_size": int(min(sizes)) if sizes else 0,
        "max_client_size": int(max(sizes)) if sizes else 0,
        "empty_clients": int(sum(1 for s in sizes if s == 0)),
        "mean_max_class_share": float(np.mean(shares)) if shares else 0.0,
        "iid_reference_share": 1.0 / n_classes if n_classes else 0.0,
    }
