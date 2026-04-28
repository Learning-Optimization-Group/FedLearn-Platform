# 07 — Data Partitioning & Non-IID Scenarios

## Table of Contents
- [Why Non-IID Data Matters](#why-non-iid-data-matters)
- [IID vs. Non-IID Illustrated](#iid-vs-non-iid-illustrated)
- [Dirichlet Distribution for Non-IID Partitioning](#dirichlet-distribution-for-non-iid-partitioning)
  - [Mathematical Intuition](#mathematical-intuition)
  - [Implementation](#implementation)
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

### Implementation

```python
import numpy as np
from torch.utils.data import Dataset, Subset
from typing import List, Optional

def dirichlet_partition(
    dataset: Dataset,
    num_clients: int,
    alpha: float = 0.5,
    seed: int = 42,
    min_samples_per_client: int = 10,
) -> List[Subset]:
    """
    Partition a dataset using Dirichlet distribution for non-IID splits.

    Args:
        dataset:                 Full training dataset
        num_clients:             Number of federated clients
        alpha:                   Dirichlet concentration parameter
                                 (lower = more heterogeneous)
        seed:                    Random seed for reproducibility
        min_samples_per_client:  Minimum samples to guarantee per client
                                 (prevents degenerate partitions)

    Returns:
        List of Subset objects, one per client
    """
    np.random.seed(seed)

    # Extract all labels efficiently
    if hasattr(dataset, 'targets'):
        # torchvision datasets expose targets directly
        labels = np.array(dataset.targets)
    else:
        labels = np.array([dataset[i][1] for i in range(len(dataset))])

    num_classes = len(np.unique(labels))
    num_samples = len(labels)

    # Build index list per class
    class_indices = {c: np.where(labels == c)[0].tolist() for c in range(num_classes)}
    for c in class_indices:
        np.random.shuffle(class_indices[c])

    client_indices = [[] for _ in range(num_clients)]

    for c in range(num_classes):
        # Sample proportions from Dirichlet
        proportions = np.random.dirichlet(np.full(num_clients, alpha))

        # Convert proportions to integer counts (ensure no client gets 0)
        indices = class_indices[c]
        n_class = len(indices)

        # Compute cumulative split points
        splits = (np.cumsum(proportions) * n_class).astype(int)
        splits[-1] = n_class  # ensure last split covers all

        start = 0
        for i, end in enumerate(splits):
            client_indices[i].extend(indices[start:end])
            start = end

    # Validate: ensure minimum samples per client
    for i, indices in enumerate(client_indices):
        if len(indices) < min_samples_per_client:
            raise ValueError(
                f"Client {i} has only {len(indices)} samples with alpha={alpha}. "
                f"Increase alpha or reduce num_clients."
            )
        np.random.shuffle(indices)  # shuffle within each client's partition

    return [Subset(dataset, indices) for indices in client_indices]


def stratified_partition(
    dataset: Dataset,
    num_clients: int,
    seed: int = 42,
) -> List[Subset]:
    """
    Perfect IID partition: each client gets an equal, stratified sample of all classes.
    Use as a baseline to isolate the effect of non-IID data distribution.
    """
    np.random.seed(seed)

    if hasattr(dataset, 'targets'):
        labels = np.array(dataset.targets)
    else:
        labels = np.array([dataset[i][1] for i in range(len(dataset))])

    num_classes = len(np.unique(labels))
    client_indices = [[] for _ in range(num_clients)]

    for c in range(num_classes):
        class_idx = np.where(labels == c)[0]
        np.random.shuffle(class_idx)
        # Round-robin assignment
        for i, idx in enumerate(class_idx):
            client_indices[i % num_clients].append(idx)

    return [Subset(dataset, indices) for indices in client_indices]
```

### Visualising Partitions

```python
import matplotlib.pyplot as plt
import numpy as np

def plot_data_distribution(client_subsets, dataset, num_classes=10):
    """
    Visualise the class distribution across clients.
    Each row = one client, each column = one class.
    """
    if hasattr(dataset, 'targets'):
        all_labels = np.array(dataset.targets)
    else:
        all_labels = np.array([dataset[i][1] for i in range(len(dataset))])

    num_clients = len(client_subsets)
    distribution = np.zeros((num_clients, num_classes))

    for i, subset in enumerate(client_subsets):
        client_labels = all_labels[subset.indices]
        for c in range(num_classes):
            distribution[i, c] = np.sum(client_labels == c)

    # Normalise rows
    row_sums = distribution.sum(axis=1, keepdims=True)
    distribution_pct = distribution / row_sums * 100

    fig, ax = plt.subplots(figsize=(12, num_clients * 0.4 + 2))
    im = ax.imshow(distribution_pct, aspect='auto', cmap='YlOrRd')
    ax.set_xlabel('Class')
    ax.set_ylabel('Client')
    ax.set_title('Data Distribution Across Clients (%)')
    plt.colorbar(im, ax=ax, label='% of client data')
    plt.tight_layout()
    plt.savefig('data_distribution.png', dpi=150)
    return distribution
```

---

## Practical Data Setup Patterns

### MNIST with Non-IID Split

```python
# examples/simple_federation/run_client.py pattern
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import argparse

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

full_dataset = datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)

# Partition according to Dirichlet distribution
client_subsets = dirichlet_partition(
    full_dataset,
    num_clients=args.num_clients,
    alpha=args.alpha,  # 0.5 = moderately non-IID
    seed=42,
)

# This client gets its assigned partition
my_dataset = client_subsets[args.id]
train_loader = DataLoader(my_dataset, batch_size=32, shuffle=True, num_workers=2)

print(f"Client {args.id}: {len(my_dataset)} samples")

client = MNISTClient(model=CNN(), train_loader=train_loader)
fl.client.start_client(args.server_address, client, f"client_{args.id}")
```

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

Experimental observations from the FedLearn examples:

| Setting | FedAvg Accuracy (10 rounds) | FedProx Accuracy (10 rounds) | Notes |
|---------|----------------------------|------------------------------|-------|
| IID (α=100) | 95.2% | 95.1% | FedProx overhead not needed |
| α = 0.5 | 91.3% | 93.7% | FedProx noticeably better |
| α = 0.1 | 84.1% | 89.2% | Significant drift with FedAvg |
| α = 0.01 | 71.5% | 81.8% | Extreme drift; FedProx helps a lot |

**Key takeaways:**
1. The more non-IID the data (lower α), the worse FedAvg performs relative to centralised training.
2. FedProx's proximal term limits how far each client's model can drift from the global model, mitigating the effect.
3. DeComFL is less affected by non-IID data because zeroth-order estimates are inherently noisy — the noise from non-IID is less dominant compared to the ZO estimation variance.

### Choosing α for Experiments

| Research Goal | Recommended α |
|--------------|--------------|
| Baseline (mimic IID) | α = 10.0 |
| Standard FL benchmark | α = 0.5 |
| Stress test for robustness | α = 0.1 |
| Worst-case (1 class per client) | α = 0.01 |
| Exact replication of prior work | Check the paper's α value |
