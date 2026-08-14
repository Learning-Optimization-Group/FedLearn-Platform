# 07 — Data Partitioning & Non-IID Scenarios

## Table of Contents
- [Why Non-IID Data Matters](#why-non-iid-data-matters)
- [IID vs. Non-IID Illustrated](#iid-vs-non-iid-illustrated)
- [The Four Shipped Partitioners](#the-four-shipped-partitioners)
  - [The Shared Contract](#the-shared-contract)
  - [iid_partition](#iid_partition)
  - [dirichlet_partition](#dirichlet_partition)
  - [shard_partition](#shard_partition)
  - [pathological_partition](#pathological_partition)
  - [partition_report](#partition_report)
- [Dirichlet Distribution for Non-IID Partitioning](#dirichlet-distribution-for-non-iid-partitioning)
  - [Mathematical Intuition](#mathematical-intuition)
  - [Visualising Partitions](#visualising-partitions)
- [Practical Data Setup Patterns](#practical-data-setup-patterns)
  - [MNIST with Non-IID Split](#mnist-with-non-iid-split)
  - [ECG Classification Data](#ecg-classification-data)
  - [LLM Dataset Partitioning](#llm-dataset-partitioning)
- [Simulating Real-World Heterogeneity](#simulating-real-world-heterogeneity)
- [Impact on Convergence](#impact-on-convergence)

---

## Why Non-IID Data Matters

In real federated learning deployments, the data on each device is **not independently and identically distributed (non-IID)**:
- A hospital has only the patient types it serves (e.g., only paediatric cases)
- A mobile keyboard model sees only the user's writing style
- An ECG monitor on a cardiac ward sees mostly abnormal readings

When data is non-IID, the local gradient directions on each client point in different directions. FedAvg's weighted average of these gradients can diverge from the true global gradient — a phenomenon known as **client drift**. Extreme non-IID distributions can cause FedAvg to converge to suboptimal solutions or fail to converge at all.

Understanding and simulating non-IID scenarios is critical for meaningful research and accurate performance benchmarks.

---

## IID vs. Non-IID Illustrated

**IID (α → ∞):** Each client has a balanced, uniform sample of all classes.

```
Client 0:  ████░░░░████░░░░████░░░░████░░░░████░░░░  (uniform)
Client 1:  ████░░░░████░░░░████░░░░████░░░░████░░░░
Client 2:  ████░░░░████░░░░████░░░░████░░░░████░░░░
           Class 0  Class 1  Class 2  Class 3  Class 4
```

**Moderate Non-IID (α = 0.5):** Some clients are biased toward certain classes.

```
Client 0:  ████████████░░░░░░░░░░░░██░░░░░░░░░░░░░░░░
Client 1:  ░░░░░░░░████████░░░░░░░░████░░░░░░░░░░░░░░
Client 2:  ░░░░░░░░░░░░████████████░░░░░░░░████░░░░░░
```

**Extreme Non-IID (α → 0):** Each client has only one or two classes.

```
Client 0:  ████████████████████████░░░░░░░░░░░░░░░░░░  (mostly class 0)
Client 1:  ░░░░░░░░░░░░░░░░████████████████░░░░░░░░░░  (mostly class 2)
Client 2:  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░███████████  (mostly class 4)
```

---

## The Four Shipped Partitioners

**Do not hand-roll a partitioner.** `fedlearn.simulation.partition` ships four, all seeded and all
tested against the same contract. They are the native replacement for `flwr_datasets.FederatedDataset`
partitioning, and they land the simulator's side of that removal.

> **Credit the unblocking correctly.** Dropping `flwr` is what freed the platform's `cryptography`
> and `protobuf` floors (see
> [01 — Technology Stack](01_architecture_overview.md#technology-stack)), but the change that
> actually did it was the native CIFAR-10 IID shard in `fl-runtime/recipes.py` — `flwr-datasets`'
> only remaining consumer. These four partitioners are used by the simulator and its tests, not by
> `fl-runtime`; `simulation/partition.py`'s own module docstring still describes the cap as a live
> residual, which is stale. Both pieces belong to the same effort; only the `recipes.py` one was
> load-bearing for the dependency floors.

```python
from fedlearn.simulation.partition import (
    iid_partition, dirichlet_partition, shard_partition, pathological_partition,
    partition_report,
)
```

The heterogeneity knobs are deliberately **different shapes**, because the literature uses all of
them and a result is only comparable to a paper that used the same one:

| Partitioner | How heterogeneity is controlled | Use it when |
|---|---|---|
| `iid_partition` | none — the IID **control arm** | you need a baseline that isolates federation from heterogeneity |
| `dirichlet_partition` | `alpha` — smaller is more skewed | the common modern default |
| `shard_partition` | sort-and-shard; `shards_per_client` **bounds** classes per client | reproducing the original FedAvg (McMahan et al.) construction |
| `pathological_partition` | **exactly** `classes_per_client` classes per client | worst-case heterogeneity; cleanly separates methods that depend on local label coverage (e.g. linear probing on a frozen encoder) from methods that do not |

### The Shared Contract

Every partitioner returns `List[np.ndarray]` of **index arrays into the dataset** — not data. That
makes a partitioner independent of how the dataset is stored, cheap to hold for thousands of
clients, and recordable verbatim in a result's `meta` block.

All four satisfy the same three-part contract, enforced by `tests/test_simulation_partition.py`:

- **complete** — the union of client index sets is exactly `range(n_samples)`
- **disjoint** — no index is held by two clients
- **deterministic in `seed`** — and *genuinely dependent on it*

### iid_partition

```python
iid_partition(n_samples: int, num_clients: int, seed: int) -> List[np.ndarray]
```

Shuffle and deal into near-equal parts (sizes differ by at most 1). Note it takes `n_samples`, not
labels — it does not need them.

> Any non-IID effect must be measured against **this**, not against a centralized baseline, or the
> comparison confounds federation with heterogeneity.

### dirichlet_partition

```python
dirichlet_partition(labels, num_clients, alpha, seed, min_partition_size=0) -> List[np.ndarray]
```

For each class, draw client proportions from `Dir(alpha)` and cut the class's shuffled indices at
the cumulative proportions. Using a **single cumulative cut** (rather than per-client rounding) is
what guarantees completeness and disjointness: every index lands in exactly one slice, whatever the
proportions are.

`min_partition_size` is the parameter that matters operationally. Low `alpha` naturally produces
**empty clients**, and an empty client is not a benign edge case — `num_examples == 0` is rejected at
coordinator ingress, so it silently shrinks the effective cohort.

The repair is deterministic — move samples from the largest donor to the most-deficient client,
tie-breaking by lowest index — rather than the usual "redraw until it fits" loop. Two reasons, and
the second is the important one:

1. It always terminates (feasibility, `num_clients * min_partition_size <= n_samples`, is checked up
   front and raises otherwise).
2. It consumes a **fixed** number of RNG draws. A redraw loop would make the *rest* of a run's
   randomness depend on how many redraws happened, quietly breaking reproducibility across `alpha`
   values — the kind of bug that is invisible in a passing test suite and fatal in an experiments
   table.

### shard_partition

```python
shard_partition(labels, num_clients, shards_per_client, seed) -> List[np.ndarray]
```

Sort by label, cut into `num_clients * shards_per_client` contiguous shards, deal them from a seeded
permutation. The original FedAvg non-IID construction; `shards_per_client=2` is the canonical
setting.

Because shards are label-contiguous, a client holding `s` shards sees **at most `s + 1`** distinct
classes — one more than `s` only when a shard straddles a class boundary. The sort is `kind="stable"`
so ties within a class are ordered by index and the sort itself contributes no randomness, keeping
the seed the only source of variation.

### pathological_partition

```python
pathological_partition(labels, num_clients, classes_per_client, seed) -> List[np.ndarray]
```

Every client gets **exactly** `classes_per_client` distinct classes — unlike `shard_partition`, where
the count is only bounded. Classes are dealt from a seeded permutation using a rotating stride
(`perm[(i*k + j) % n_classes]`), which guarantees distinctness within a client and even coverage
across classes.

Two `ValueError`s guard the degenerate cases: `classes_per_client` exceeding the number of distinct
classes, and `num_clients * classes_per_client < n_classes` — the latter because some class would
then have no holder and its samples would be **dropped**, violating completeness.

### partition_report

```python
partition_report(parts, labels) -> dict
```

Summarises a partition into a JSON-serializable block for a result's `meta`. A result that records
only `alpha=0.5` cannot be checked later; one that records the *realised* sizes and skew can.

| Field | Meaning |
|---|---|
| `num_clients`, `total_samples`, `num_classes` | shape of the split |
| `client_sizes`, `min_client_size`, `max_client_size` | realised per-client sizes |
| `empty_clients` | count of zero-sample clients — should be 0 for a usable run |
| **`mean_max_class_share`** | the headline statistic: **1.0** = every client holds a single class |
| `iid_reference_share` | `1 / num_classes` — the value `mean_max_class_share` takes under a perfect IID split |

Comparing those last two is the honest way to state "how non-IID was this run" — an `alpha` value
alone is a request, not a measurement.

---

## Dirichlet Distribution for Non-IID Partitioning

The Dirichlet distribution `Dir(α)` over a k-simplex is the standard way to generate heterogeneous data partitions in federated learning research.

### Mathematical Intuition

Given a dataset with classes `C = {0, 1, ..., K-1}` and `N` clients:

For each class `c`, sample a proportion vector from the Dirichlet distribution:

```
p_c ~ Dir(α, N)
```

where `p_c = [p_{c,0}, p_{c,1}, ..., p_{c,N-1}]` and `Σ p_{c,i} = 1`.

Then assign a fraction `p_{c,i}` of class `c`'s samples to client `i`.

**The α parameter controls heterogeneity:**

| α value | Behaviour | Use case |
|---------|-----------|---------|
| α = 100.0 | Nearly IID | Baseline comparison |
| α = 1.0 | Mildly non-IID | Moderate heterogeneity |
| α = 0.5 | Moderately non-IID | Standard non-IID benchmark |
| α = 0.1 | Highly non-IID | Severe heterogeneity |
| α → 0 | Pathological | 1 class per client |

### Visualising Partitions

`partition_report` gives you the numbers; this gives you the picture. It takes **index arrays**
straight from any of the four partitioners.

```python
import matplotlib.pyplot as plt
import numpy as np

def plot_data_distribution(parts, labels, out="data_distribution.png"):
    """
    Visualise the class distribution across clients.
    Each row = one client, each column = one class.

    parts:  List[np.ndarray] from any fedlearn.simulation.partition function
    labels: the same 1-D label array passed to the partitioner
    """
    labels = np.asarray(labels)
    num_classes = int(np.unique(labels).size)
    distribution = np.zeros((len(parts), num_classes))

    for i, idx in enumerate(parts):
        counts = np.bincount(labels[np.asarray(idx, dtype=np.int64)], minlength=num_classes)
        distribution[i] = counts

    row_sums = distribution.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1          # an empty client would divide by zero
    distribution_pct = distribution / row_sums * 100

    fig, ax = plt.subplots(figsize=(12, len(parts) * 0.4 + 2))
    im = ax.imshow(distribution_pct, aspect='auto', cmap='YlOrRd')
    ax.set_xlabel('Class'); ax.set_ylabel('Client')
    ax.set_title('Data Distribution Across Clients (%)')
    plt.colorbar(im, ax=ax, label='% of client data')
    plt.tight_layout(); plt.savefig(out, dpi=150)
    return distribution
```

---

## Practical Data Setup Patterns

### MNIST with Non-IID Split

```python
import argparse
import fedlearn as fl
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from fedlearn.simulation.partition import dirichlet_partition, partition_report

parser = argparse.ArgumentParser()
parser.add_argument("--id", type=int, default=0)
parser.add_argument("--num_clients", type=int, default=3)
parser.add_argument("--alpha", type=float, default=0.5)
parser.add_argument("--server_address", default="localhost:50051")
args = parser.parse_args()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

full_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
labels = full_dataset.targets.numpy()

# Every client MUST pass the same (labels, num_clients, alpha, seed) to get the same split.
parts = dirichlet_partition(
    labels=labels,
    num_clients=args.num_clients,
    alpha=args.alpha,          # 0.5 = moderately non-IID
    seed=42,
    min_partition_size=16,     # never hand a client 0 samples — ingress rejects it
)
print(partition_report(parts, labels))     # record this in the run's meta block

my_dataset = Subset(full_dataset, parts[args.id])
train_loader = DataLoader(my_dataset, batch_size=32, shuffle=True, num_workers=2)
print(f"Client {args.id}: {len(my_dataset)} samples")

client = fl.LocalTrainer(model=CNN(), train_loader=train_loader)
fl.client.start_client(args.server_address, client, f"client_{args.id}")
```

> **The partition is reproduced independently on each client, not distributed by the server.** Every
> client must be given the identical `(labels, num_clients, alpha, seed)` tuple, or the split is not
> a partition at all — clients will overlap and some samples will be trained on twice while others
> are never seen. The seed is the contract.

### ECG Classification Data

The ECG dataset requires custom loading (CSV format) with binary labels (Normal=0, Abnormal=1):

```python
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split

class ECGDataset(Dataset):
    def __init__(self, csv_path, client_id=None, num_clients=None, alpha=0.5):
        df = pd.read_csv(csv_path, header=None)

        # Last column is label (0=Normal, 1=Abnormal)
        self.X = torch.FloatTensor(df.iloc[:, :-1].values)
        self.y = torch.LongTensor(df.iloc[:, -1].values)

        if client_id is not None and num_clients is not None:
            # Apply Dirichlet partition
            indices = self._get_client_indices(client_id, num_clients, alpha)
            self.X = self.X[indices]
            self.y = self.y[indices]

    def _get_client_indices(self, client_id, num_clients, alpha):
        np.random.seed(42)
        n = len(self.y)
        labels = self.y.numpy()

        client_indices = [[] for _ in range(num_clients)]
        for c in [0, 1]:   # binary classification
            class_idx = np.where(labels == c)[0]
            np.random.shuffle(class_idx)

            proportions = np.random.dirichlet(np.full(num_clients, alpha))
            splits = (np.cumsum(proportions) * len(class_idx)).astype(int)
            splits[-1] = len(class_idx)

            start = 0
            for i, end in enumerate(splits):
                client_indices[i].extend(class_idx[start:end].tolist())
                start = end

        return client_indices[client_id]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# Usage
dataset = ECGDataset(
    csv_path="ecg_data/ecg.csv",
    client_id=0,       # this client's ID
    num_clients=5,     # total number of clients
    alpha=0.5,         # Dirichlet α
)
train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
```

### LLM Dataset Partitioning

For LLMs fine-tuned on text classification (e.g., SuperGLUE CommitmentBank):

```python
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import Dataset

class CBDataset(Dataset):
    """SuperGLUE CommitmentBank dataset, tokenized for OPT-125M."""

    CB_LABEL_MAP = {"entailment": 0, "contradiction": 1, "neutral": 2}

    def __init__(self, split="train", max_length=128, client_id=None, num_clients=None):
        dataset = load_dataset("super_glue", "cb", split=split)
        tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")

        self.encodings = []
        self.labels = []

        for item in dataset:
            text = f"Premise: {item['premise']} Hypothesis: {item['hypothesis']}"
            encoding = tokenizer(
                text,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            self.encodings.append({
                "input_ids": encoding["input_ids"].squeeze(0),
                "attention_mask": encoding["attention_mask"].squeeze(0),
            })
            self.labels.append(self.CB_LABEL_MAP[item["label"]])

        # Partition for federated learning
        if client_id is not None and num_clients is not None:
            indices = self._partition(client_id, num_clients)
            self.encodings = [self.encodings[i] for i in indices]
            self.labels = [self.labels[i] for i in indices]

    def _partition(self, client_id, num_clients):
        """Simple round-robin for small LLM datasets (CB has only 250 train examples)."""
        return list(range(client_id, len(self.labels), num_clients))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.encodings[idx], torch.tensor(self.labels[idx])
```

> **Note for LLM datasets:** Text datasets for fine-tuning are often small (CB has 250 train examples). With 3 clients, each client gets ~83 examples — fine-tuning at this scale is still meaningful because the model starts from strong pre-trained representations.

---

## Simulating Real-World Heterogeneity

Beyond class distribution, real federated settings have additional sources of heterogeneity:

> The patterns below are **illustrative sketches**, not shipped API — unlike the partitioners above,
> which are. Two of them have first-class support you should prefer:
>
> - **Dropout / stragglers** — `SimulatedFederation(dropout_rate=…)` models dropout deterministically
>   from the run seed, and resolves the round *immediately* instead of sleeping out the deployed
>   120-second deadline. Each `RoundRecord` carries `selected` / `reported` / `dropped` / `forced`,
>   so a forced (partial-cohort) round can never be mistaken for a clean one. See
>   [01 — The In-Process Simulator](01_architecture_overview.md#the-in-process-simulator).
> - **Per-client randomness** — `ClientRng` / `RunRng` give each client an isolated stream keyed on
>   `(seed, client_id[, round])`, so a straggler or a dropped client cannot perturb anyone else's
>   draws.

### System Heterogeneity (Stragglers)

Some clients are faster than others. Simulate with throttled training:

```python
class SlowClient(MNISTClient):
    def __init__(self, *args, slowdown_factor=3.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.slowdown_factor = slowdown_factor

    def fit(self, parameters, config):
        params, n = super().fit(parameters, config)
        # Simulate slow device
        time.sleep(random.uniform(0, self.slowdown_factor))
        return params, n
```

### Sample Size Heterogeneity

Clients with vastly different dataset sizes:

```python
# Simulate power-law distribution of client dataset sizes
def power_law_partition(dataset, num_clients, exponent=1.5, seed=42):
    np.random.seed(seed)
    n = len(dataset)

    # Power-law weights: client 0 has most data, last client has least
    weights = np.array([1 / (i + 1)**exponent for i in range(num_clients)])
    weights /= weights.sum()
    sizes = (weights * n).astype(int)
    sizes[-1] = n - sizes[:-1].sum()  # give remainder to last client

    return random_split(dataset, sizes.tolist())
```

### Label Noise Injection

Simulate labelling errors on specific clients:

```python
class NoisyLabelClient(MNISTClient):
    def __init__(self, *args, noise_rate=0.2, **kwargs):
        super().__init__(*args, **kwargs)
        self.noise_rate = noise_rate  # fraction of labels to randomly flip

    def fit(self, parameters, config):
        self.model.load_state_dict(parameters)
        self.model.train()

        for inputs, targets in self.train_loader:
            # Inject label noise
            noise_mask = torch.rand(targets.size(0)) < self.noise_rate
            random_labels = torch.randint(0, 10, targets.shape)
            noisy_targets = torch.where(noise_mask, random_labels, targets)

            # Train with noisy labels
            self.optimizer.zero_grad()
            loss = self.criterion(self.model(inputs), noisy_targets)
            loss.backward()
            self.optimizer.step()

        return self.model.state_dict(), len(self.train_loader.dataset)
```

---

## Impact on Convergence

> **No numbers are quoted here, deliberately.** Earlier revisions of this page carried a
> FedAvg-vs-FedProx accuracy table across α. **Those figures are not reproducible from anything in
> this repository** — no committed harness produces them and no result file backs them — so they
> have been removed rather than restated. Treat any α-vs-accuracy figure you find in an old copy of
> this page as unverified.

What *is* committed, and re-runnable, is the harness that would produce such a table honestly:

| Harness (`framework/benchmarks/`) | Measures |
|---|---|
| `algo_comparison.py` | one task, one **fixed** non-IID partition, one seed, run through each algorithm's **real** `aggregate_fit`; records per-round accuracy, loss, wall-clock and truthful cumulative wire bytes |
| `comms_regimes.py` | per-round communication cost across three regimes: full-model FedAvg, head-only (frozen-backbone) FedAvg, and DeComFL |
| `zeroth_vs_first_order.py` | zeroth-order vs first-order convergence **and** bytes side by side |

Run one and read its own `results/*.md`; those files are generated from the JSON record, not
hand-written. See [08 — The Committed Benchmark Harnesses](08_examples.md#the-committed-benchmark-harnesses).

The qualitative statements below are standard results from the FL literature (McMahan et al. 2017;
Li et al. 2020), not measurements from this codebase:

1. The more non-IID the data (lower α), the worse FedAvg performs relative to centralised training.
2. FedProx's proximal term limits how far each client's model can drift from the global model,
   mitigating the effect — **within its stability envelope**; past `lr·mu ≥ 2` the same term
   amplifies drift instead. See [05 — FedProx](05_strategies.md#fedprox--proximal-regularisation-shipped).
3. Whether DeComFL is *less* affected by non-IID data than first-order methods is an open question
   here, not an established result. The plausible argument — that zeroth-order estimation variance
   dominates the heterogeneity noise — is untested on this platform. Do not cite it as a finding.

### Choosing α for Experiments

| Research Goal | Recommended α |
|--------------|--------------|
| Baseline (mimic IID) | α = 10.0 — or better, use `iid_partition` as the true control arm |
| Standard FL benchmark | α = 0.5 |
| Stress test for robustness | α = 0.1 |
| Worst-case | prefer `pathological_partition(classes_per_client=1)` — **exact**, where a small α is only *probably* near-degenerate |
| Exact replication of prior work | Check the paper's α — **and which partitioner it used**; a Dirichlet result is not comparable to a sort-and-shard result at any α |

Whatever you choose, record `partition_report(parts, labels)` alongside it. `mean_max_class_share`
versus `iid_reference_share` is the measured skew; α alone is only the request.
