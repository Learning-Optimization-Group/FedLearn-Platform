"""P0-2a: tests for the native, seeded dataset partitioners.

Written before the implementation (TDD). These partitioners replace ``flwr_datasets``'s
``FederatedDataset`` partitioning, which is the dependency that pins ``cryptography<45.0.0``
and makes the framework's ``>=46.0.6`` security floor unreachable (SE-22 residual).

The contract every partitioner must satisfy, and which these tests enforce:

  1. **Completeness** — the union of all client index sets is exactly the input index set.
     A partitioner that silently drops samples produces experiments whose effective dataset
     size is unknown, which is unrecoverable after the fact.
  2. **Disjointness** — no index is assigned to two clients. Duplicated samples inflate the
     effective dataset and break the weighted-average semantics of FedAvg.
  3. **Determinism** — the same seed yields byte-identical assignments, so a run is
     reproducible from its ``meta`` block alone.
  4. **Seed sensitivity** — a different seed yields a different assignment, so "deterministic"
     cannot be satisfied by ignoring the seed and returning a constant.
  5. **Heterogeneity is monotone in the knob** — smaller Dirichlet alpha means more label skew.
     Without this, a partitioner can be "correct" (complete + disjoint) while producing IID
     data under every alpha, which would silently invalidate every non-IID claim.
"""

import numpy as np
import pytest

from fedlearn.simulation.partition import (
    dirichlet_partition,
    iid_partition,
    pathological_partition,
    shard_partition,
    partition_report,
)


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------

def _labels(n_per_class: int, n_classes: int) -> np.ndarray:
    """Balanced label vector: n_per_class samples of each of n_classes classes."""
    return np.repeat(np.arange(n_classes), n_per_class)


def _assert_partition_valid(parts, n_samples):
    """Completeness + disjointness, the two properties no partitioner may violate."""
    seen = np.concatenate([np.asarray(p, dtype=np.int64) for p in parts]) if parts else np.array([], dtype=np.int64)
    assert seen.size == n_samples, (
        f"completeness violated: {seen.size} indices assigned, expected {n_samples}"
    )
    assert np.unique(seen).size == n_samples, "disjointness violated: an index was assigned twice"
    assert seen.min() >= 0 and seen.max() < n_samples, "index out of range"


def _mean_max_class_share(parts, labels, n_classes):
    """Mean over clients of (largest class count / client size).

    1.0 means every client holds exactly one class (maximally skewed); 1/n_classes means
    every client mirrors the global distribution (IID). This is the heterogeneity statistic
    the alpha-monotonicity test is written against.
    """
    shares = []
    for p in parts:
        if len(p) == 0:
            continue
        counts = np.bincount(labels[np.asarray(p, dtype=np.int64)], minlength=n_classes)
        shares.append(counts.max() / counts.sum())
    return float(np.mean(shares))


# --------------------------------------------------------------------------------------
# IID
# --------------------------------------------------------------------------------------

class TestIidPartition:
    def test_is_complete_and_disjoint(self):
        parts = iid_partition(n_samples=1000, num_clients=10, seed=0)
        assert len(parts) == 10
        _assert_partition_valid(parts, 1000)

    def test_sizes_are_balanced_within_one(self):
        parts = iid_partition(n_samples=1003, num_clients=10, seed=0)
        sizes = sorted(len(p) for p in parts)
        assert sizes[-1] - sizes[0] <= 1, f"IID split should be balanced within 1, got {sizes}"

    def test_is_deterministic(self):
        a = iid_partition(n_samples=500, num_clients=7, seed=42)
        b = iid_partition(n_samples=500, num_clients=7, seed=42)
        assert [list(x) for x in a] == [list(x) for x in b]

    def test_is_seed_sensitive(self):
        a = iid_partition(n_samples=500, num_clients=7, seed=1)
        b = iid_partition(n_samples=500, num_clients=7, seed=2)
        assert [list(x) for x in a] != [list(x) for x in b]

    def test_single_client_gets_everything(self):
        parts = iid_partition(n_samples=100, num_clients=1, seed=0)
        assert len(parts) == 1
        assert sorted(parts[0]) == list(range(100))


# --------------------------------------------------------------------------------------
# Dirichlet
# --------------------------------------------------------------------------------------

class TestDirichletPartition:
    def test_is_complete_and_disjoint(self):
        labels = _labels(100, 10)          # 1000 samples, 10 classes
        parts = dirichlet_partition(labels, num_clients=10, alpha=0.5, seed=0)
        assert len(parts) == 10
        _assert_partition_valid(parts, labels.size)

    def test_is_deterministic(self):
        labels = _labels(100, 10)
        a = dirichlet_partition(labels, num_clients=10, alpha=0.5, seed=7)
        b = dirichlet_partition(labels, num_clients=10, alpha=0.5, seed=7)
        assert [list(x) for x in a] == [list(x) for x in b]

    def test_is_seed_sensitive(self):
        labels = _labels(100, 10)
        a = dirichlet_partition(labels, num_clients=10, alpha=0.5, seed=1)
        b = dirichlet_partition(labels, num_clients=10, alpha=0.5, seed=2)
        assert [list(x) for x in a] != [list(x) for x in b]

    def test_alpha_controls_heterogeneity_monotonically(self):
        """Smaller alpha => more label skew. This is the property the knob exists for.

        Averaged over seeds because a single Dirichlet draw is itself noisy; the claim is
        about the distribution the knob induces, not about one sample from it.
        """
        labels = _labels(200, 10)          # 2000 samples, 10 classes
        skew = {}
        for alpha in (0.05, 0.5, 100.0):
            vals = [
                _mean_max_class_share(
                    dirichlet_partition(labels, num_clients=20, alpha=alpha, seed=s), labels, 10
                )
                for s in range(5)
            ]
            skew[alpha] = float(np.mean(vals))

        assert skew[0.05] > skew[0.5] > skew[100.0], (
            f"alpha must control skew monotonically, got {skew}"
        )
        # Sanity anchors: alpha=0.05 is near-pathological, alpha=100 is near-IID (1/10 = 0.1).
        assert skew[0.05] > 0.6, f"alpha=0.05 should be strongly skewed, got {skew[0.05]:.3f}"
        assert skew[100.0] < 0.25, f"alpha=100 should be near-IID, got {skew[100.0]:.3f}"

    def test_min_partition_size_is_respected(self):
        """A client with zero samples crashes the FedAvg weighted average (num_examples=0).

        Low alpha naturally produces empty clients, so the partitioner must be able to
        guarantee a floor rather than emitting them.
        """
        labels = _labels(100, 10)
        parts = dirichlet_partition(
            labels, num_clients=20, alpha=0.05, seed=3, min_partition_size=5
        )
        _assert_partition_valid(parts, labels.size)
        assert min(len(p) for p in parts) >= 5

    def test_impossible_min_partition_size_raises(self):
        labels = _labels(1, 10)            # 10 samples
        with pytest.raises(ValueError, match="min_partition_size"):
            dirichlet_partition(labels, num_clients=10, alpha=0.5, seed=0, min_partition_size=5)


# --------------------------------------------------------------------------------------
# Shard (McMahan et al. sort-and-shard)
# --------------------------------------------------------------------------------------

class TestShardPartition:
    def test_is_complete_and_disjoint(self):
        labels = _labels(100, 10)
        parts = shard_partition(labels, num_clients=10, shards_per_client=2, seed=0)
        assert len(parts) == 10
        _assert_partition_valid(parts, labels.size)

    def test_two_shards_per_client_bounds_classes_per_client(self):
        """The canonical FedAvg non-IID setup: sort by label, cut into 2*N shards, 2 per client.

        Each shard is label-contiguous, so a client holding s shards sees at most s+1 distinct
        classes (a shard can straddle one class boundary).
        """
        labels = _labels(100, 10)
        parts = shard_partition(labels, num_clients=10, shards_per_client=2, seed=0)
        for p in parts:
            n_classes = np.unique(labels[np.asarray(p, dtype=np.int64)]).size
            assert n_classes <= 3, f"2 shards should span <=3 classes, saw {n_classes}"

    def test_is_deterministic_and_seed_sensitive(self):
        labels = _labels(100, 10)
        a = shard_partition(labels, num_clients=10, shards_per_client=2, seed=5)
        b = shard_partition(labels, num_clients=10, shards_per_client=2, seed=5)
        c = shard_partition(labels, num_clients=10, shards_per_client=2, seed=6)
        assert [list(x) for x in a] == [list(x) for x in b]
        assert [list(x) for x in a] != [list(x) for x in c]


# --------------------------------------------------------------------------------------
# Pathological (exactly k classes per client)
# --------------------------------------------------------------------------------------

class TestPathologicalPartition:
    def test_each_client_holds_exactly_k_classes(self):
        labels = _labels(100, 10)
        parts = pathological_partition(labels, num_clients=10, classes_per_client=2, seed=0)
        _assert_partition_valid(parts, labels.size)
        for p in parts:
            n_classes = np.unique(labels[np.asarray(p, dtype=np.int64)]).size
            assert n_classes == 2, f"expected exactly 2 classes per client, saw {n_classes}"

    def test_is_deterministic(self):
        labels = _labels(100, 10)
        a = pathological_partition(labels, num_clients=10, classes_per_client=2, seed=11)
        b = pathological_partition(labels, num_clients=10, classes_per_client=2, seed=11)
        assert [list(x) for x in a] == [list(x) for x in b]

    def test_more_classes_than_available_raises(self):
        labels = _labels(100, 3)
        with pytest.raises(ValueError, match="classes_per_client"):
            pathological_partition(labels, num_clients=5, classes_per_client=5, seed=0)


# --------------------------------------------------------------------------------------
# Reporting — the record must be able to describe what it ran on
# --------------------------------------------------------------------------------------

class TestPartitionReport:
    def test_report_is_json_serializable_and_describes_the_split(self):
        import json

        labels = _labels(100, 10)
        parts = dirichlet_partition(labels, num_clients=10, alpha=0.5, seed=0)
        rep = partition_report(parts, labels)

        json.dumps(rep)  # must round-trip into a result JSON's meta block

        assert rep["num_clients"] == 10
        assert rep["total_samples"] == labels.size
        assert len(rep["client_sizes"]) == 10
        assert sum(rep["client_sizes"]) == labels.size
        # The heterogeneity statistic is what makes two runs comparable after the fact.
        assert 0.0 <= rep["mean_max_class_share"] <= 1.0
        assert rep["min_client_size"] == min(rep["client_sizes"])


# --------------------------------------------------------------------------------------
# Cross-partitioner invariants
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("num_clients", [1, 2, 7, 50])
def test_all_partitioners_hold_the_contract_across_client_counts(num_clients):
    labels = _labels(100, 10)  # 1000 samples
    for parts in (
        iid_partition(labels.size, num_clients, seed=0),
        dirichlet_partition(labels, num_clients, alpha=1.0, seed=0),
        shard_partition(labels, num_clients, shards_per_client=2, seed=0),
    ):
        assert len(parts) == num_clients
        _assert_partition_valid(parts, labels.size)
