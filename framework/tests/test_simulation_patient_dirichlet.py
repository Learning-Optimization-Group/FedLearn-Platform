"""patient_dirichlet_partition: whole patients, every client non-empty, label skew grows as alpha shrinks."""
import numpy as np
import pytest

from fedlearn.simulation.partition import patient_dirichlet_partition


def _cohort(patients=600, seed=0):
    rng = np.random.default_rng(seed)
    ids, labels = [], []
    for p in range(1, patients + 1):
        positive = rng.random() < 0.2
        for _ in range(int(rng.integers(1, 4))):
            ids.append(str(p))
            labels.append(int(positive and rng.random() < 0.8))
    return ids, np.array(labels)


def _positive_share(parts, labels):
    return np.array([labels[p].mean() for p in parts])


@pytest.mark.parametrize("alpha", [0.1, 1.0, 100.0])
def test_every_row_once_patients_whole_and_no_empty_client(alpha):
    ids, labels = _cohort()
    parts = patient_dirichlet_partition(ids, labels, 40, alpha, 7)
    rows = np.concatenate(parts)
    assert len(parts) == 40 and all(len(p) > 0 for p in parts)
    assert sorted(rows.tolist()) == list(range(len(ids)))
    owner = {}
    for c, p in enumerate(parts):
        for i in p:
            assert owner.setdefault(ids[i], c) == c


def test_deterministic_in_seed():
    ids, labels = _cohort()
    a = patient_dirichlet_partition(ids, labels, 40, 0.5, 3)
    assert all(np.array_equal(x, y) for x, y in zip(a, patient_dirichlet_partition(ids, labels, 40, 0.5, 3)))
    assert not all(np.array_equal(x, y) for x, y in zip(a, patient_dirichlet_partition(ids, labels, 40, 0.5, 4)))


def test_smaller_alpha_gives_more_label_skew():
    ids, labels = _cohort()
    spread = {a: float(np.std(_positive_share(patient_dirichlet_partition(ids, labels, 40, a, 1), labels)))
              for a in (0.1, 1.0, 100.0)}
    assert spread[0.1] > spread[1.0] > spread[100.0]


@pytest.mark.parametrize("kwargs, message", [
    (dict(alpha=0.0), "alpha"), (dict(num_clients=0), "num_clients"), (dict(num_clients=10_000), "exceed"),
])
def test_invalid_arguments(kwargs, message):
    ids, labels = _cohort(50)
    args = dict(num_clients=10, alpha=1.0, seed=0) | kwargs
    with pytest.raises(ValueError, match=message):
        patient_dirichlet_partition(ids, labels, args["num_clients"], args["alpha"], args["seed"])
