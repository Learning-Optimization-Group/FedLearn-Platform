"""
recipes.py — single source of truth for FedLearn model "recipes".

A *recipe* bundles {architecture + dataset loader + input transform + class
labels + input kind + UI metadata} under a stable key (CNN, MLP, TRANSFORMER,
PNEUMONIA_CNN). The FL scripts (client.py, fl_server.py, init_model.py,
infer.py, data.py) dispatch the PNEUMONIA_CNN path through here, and the backend
serves the catalog (`python recipes.py --describe`) at GET /api/model-recipes so
the project-creation picker is data-driven — adding a model type is one entry
here, no Java/TS edits.

Design note: metadata for ALL recipes lives in RECIPE_METADATA (cheap, no torch
import needed for --describe). The NEW PNEUMONIA_CNN recipe is *fully functional*
here (model + dataset + transform + inference spec). The legacy CNN/MLP/
TRANSFORMER training paths still live in their existing scripts; only their
metadata is mirrored here for the catalog.

Pneumonia dataset resolution (first match wins):
  1. FEDLEARN_PNEUMONIA_DIR — a local folder laid out as ImageFolder:
        <dir>/train/{NORMAL,PNEUMONIA}/*.{jpg,png}   (client training data)
        <dir>/test/{NORMAL,PNEUMONIA}/*               (server eval data)
     This is the zero-network, guaranteed path (e.g. the Kaggle chest_xray set).
  2. HuggingFace `datasets` — FEDLEARN_PNEUMONIA_DATASET (default
        keremberke/chest-xray-classification), config FEDLEARN_PNEUMONIA_CONFIG
        (default "full").
Set FEDLEARN_PNEUMONIA_SUBSET=<N> to cap samples per split for fast demo rounds.
"""

import argparse
import json
import os
import sys

# ---------------------------------------------------------------------------
# Catalog metadata — the ONLY thing --describe needs (no torch import).
# ---------------------------------------------------------------------------
RECIPE_METADATA = [
    {
        "key": "PNEUMONIA_CNN",
        "display_name": "Pneumonia Chest X-ray",
        "input_kind": "image",
        "classes": ["NORMAL", "PNEUMONIA"],
        "base_models": ["pneumonia_cnn"],
        "optimizers": ["Adam", "SGD", "AdamW", "RMSprop"],
    },
    {
        "key": "CNN",
        "display_name": "Image classifier (CIFAR-10)",
        "input_kind": "image",
        "classes": ["airplane", "automobile", "bird", "cat", "deer",
                    "dog", "frog", "horse", "ship", "truck"],
        "base_models": ["net", "ResNet", "VGGNet", "AlexNet"],
        "optimizers": ["Adam", "SGD", "RMSprop", "AdamW"],
    },
    {
        "key": "MLP",
        "display_name": "ECG heartbeat (Normal/Abnormal)",
        "input_kind": "vector",
        "classes": ["Normal", "Abnormal"],
        "base_models": ["ecg_mlp"],
        "optimizers": ["Adam", "AdamW", "SGD"],
    },
    {
        "key": "TRANSFORMER",
        "display_name": "Text classifier (OPT-125M)",
        "input_kind": "text",
        "classes": [],
        "base_models": ["opt-125m", "bert-tiny"],
        "optimizers": ["AdamW", "Adam"],
    },
]

_METADATA_BY_KEY = {r["key"]: r for r in RECIPE_METADATA}


def describe():
    """Return the catalog metadata (list of dicts). Used by --describe."""
    return RECIPE_METADATA


# ---------------------------------------------------------------------------
# Pneumonia recipe specifics.
# ---------------------------------------------------------------------------
PNEUMONIA_CLASSES = ["NORMAL", "PNEUMONIA"]
PNEUMONIA_IMG_SIZE = 224


def _subset_cap():
    """Optional per-split sample cap (env FEDLEARN_PNEUMONIA_SUBSET) for fast demos."""
    raw = os.environ.get("FEDLEARN_PNEUMONIA_SUBSET", "").strip()
    if not raw:
        return None
    try:
        n = int(raw)
        return n if n > 0 else None
    except ValueError:
        return None


def build_pneumonia_cnn():
    """PneumoniaCNN — 1x224x224 grayscale -> 2 logits (NORMAL, PNEUMONIA).

    Faithful to docs/guides/pneumonia_demo_plan.md.
    """
    import torch.nn as nn

    class PneumoniaCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(128 * 28 * 28, 256),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(256, 2),
            )

        def forward(self, x):
            return self.classifier(self.features(x))

    return PneumoniaCNN()


def pneumonia_transform():
    """Grayscale -> 224x224 -> tensor -> Normalize([-1,1]). Used for train AND inference."""
    import torchvision.transforms as T
    return T.Compose([
        T.Grayscale(num_output_channels=1),
        T.Resize((PNEUMONIA_IMG_SIZE, PNEUMONIA_IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5]),
    ])


def _dirichlet_indices(labels, num_clients, alpha, seed):
    """Non-IID partition of sample indices over classes via a Dirichlet draw.

    Mirrors the ECG/LLM dirichlet_split already used in this repo. Returns a list
    of `num_clients` index lists. Same (num_clients, alpha, seed) => same split,
    so every device computes an identical, non-overlapping partition.
    """
    import numpy as np
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    classes = sorted(set(int(x) for x in labels.tolist()))
    client_indices = [[] for _ in range(num_clients)]
    distribution = rng.dirichlet([alpha] * num_clients, len(classes))
    for ci, k in enumerate(classes):
        idx_k = np.where(labels == k)[0]
        rng.shuffle(idx_k)
        splits = (np.cumsum(distribution[ci]) * len(idx_k)).astype(int)[:-1]
        for client_i, part in enumerate(np.split(idx_k, splits)):
            client_indices[client_i].extend(int(x) for x in part)
    for i in range(num_clients):
        rng.shuffle(client_indices[i])
    return client_indices


class _ImageFolderDataset:
    """torch Dataset over a local ImageFolder split, applying pneumonia_transform.

    Yields (tensor[1,224,224], int_label) with NORMAL=0, PNEUMONIA=1.
    """

    def __init__(self, root_split, indices=None):
        from torchvision.datasets import ImageFolder
        self._folder = ImageFolder(root_split, transform=pneumonia_transform())
        # ImageFolder sorts classes alphabetically -> NORMAL=0, PNEUMONIA=1.
        self.targets = list(self._folder.targets)
        self.indices = list(range(len(self._folder))) if indices is None else list(indices)

    def labels(self):
        return [self.targets[i] for i in self.indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        return self._folder[self.indices[i]]


class _HFXrayDataset:
    """torch Dataset over a HuggingFace image-classification split.

    Auto-detects the image + label columns and remaps to NORMAL=0, PNEUMONIA=1.
    """

    def __init__(self, hf_split, indices=None):
        self._ds = hf_split
        cols = hf_split.column_names
        self._img_col = next((c for c in ("image", "img", "pixel_values") if c in cols), cols[0])
        self._lbl_col = next((c for c in ("labels", "label", "target") if c in cols), None)
        if self._lbl_col is None:
            raise ValueError(f"Could not find a label column in {cols}")
        # Build name->canonical-index remap (NORMAL=0, PNEUMONIA=1) when names exist.
        self._remap = None
        try:
            names = hf_split.features[self._lbl_col].names  # ClassLabel
            remap = {}
            for raw_idx, name in enumerate(names):
                up = str(name).upper()
                if "PNEU" in up:
                    remap[raw_idx] = 1
                elif "NORMAL" in up or "HEALTH" in up:
                    remap[raw_idx] = 0
            if len(remap) == len(names):
                self._remap = remap
        except (AttributeError, KeyError, TypeError):
            self._remap = None
        raw_labels = list(hf_split[self._lbl_col])
        self._labels = [self._remap.get(int(v), int(v)) if self._remap else int(v) for v in raw_labels]
        self.indices = list(range(len(hf_split))) if indices is None else list(indices)
        self._transform = pneumonia_transform()

    def labels(self):
        return [self._labels[i] for i in self.indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        row = self._ds[idx]
        img = row[self._img_col]
        if not hasattr(img, "mode"):  # not a PIL image (e.g. path/array)
            from PIL import Image
            img = Image.open(img) if isinstance(img, str) else Image.fromarray(img)
        return self._transform(img), self._labels[idx]


def _full_dataset(split):
    """Return a dataset wrapper for the requested split ('train' or 'test')."""
    local_dir = os.environ.get("FEDLEARN_PNEUMONIA_DIR", "").strip()
    if local_dir:
        split_dir = os.path.join(local_dir, split)
        if not os.path.isdir(split_dir):
            # Some layouts use 'val' instead of a 'test' split.
            alt = os.path.join(local_dir, "val") if split == "test" else None
            if alt and os.path.isdir(alt):
                split_dir = alt
            else:
                raise FileNotFoundError(
                    f"FEDLEARN_PNEUMONIA_DIR set but '{split_dir}' not found. "
                    f"Expected <dir>/train and <dir>/test (or /val) with NORMAL/ and PNEUMONIA/ subfolders."
                )
        return _ImageFolderDataset(split_dir)

    from datasets import load_dataset
    name = os.environ.get("FEDLEARN_PNEUMONIA_DATASET", "keremberke/chest-xray-classification")
    cfg = os.environ.get("FEDLEARN_PNEUMONIA_CONFIG", "full")
    hf_split = "test" if split == "test" else "train"
    kwargs = {"split": hf_split, "trust_remote_code": True}
    ds = load_dataset(name, cfg, **kwargs) if cfg else load_dataset(name, **kwargs)
    return _HFXrayDataset(ds)


def load_pneumonia_client_data(partition_id, num_clients, alpha=0.5, seed=42,
                               batch_size=16, val_fraction=0.1):
    """Return (train_loader, val_loader) for one client's Dirichlet shard."""
    import numpy as np
    from torch.utils.data import DataLoader, Subset

    base = _full_dataset("train")
    labels = base.labels()
    cap = _subset_cap()
    if cap is not None and cap < len(labels):
        # Stratified-ish cap: keep first `cap` indices after a label-shuffle.
        rng = np.random.default_rng(seed)
        keep = rng.permutation(len(labels))[:cap]
        base = Subset(base, keep.tolist())
        labels = [labels[i] for i in keep.tolist()]

    client_indices = _dirichlet_indices(labels, num_clients, alpha, seed)
    if not (0 <= partition_id < num_clients):
        raise ValueError(f"partition_id {partition_id} out of range for num_clients {num_clients}")
    my = client_indices[partition_id]
    if len(my) == 0:
        raise ValueError(f"Dirichlet split gave client {partition_id} zero samples; raise alpha or data size.")

    # Hold out a small per-client validation slice.
    n_val = max(1, int(len(my) * val_fraction)) if len(my) > 1 else 0
    val_idx, train_idx = my[:n_val], my[n_val:]
    train_loader = DataLoader(Subset(base, train_idx), batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(Subset(base, val_idx), batch_size=batch_size, shuffle=False, num_workers=0) if val_idx else None
    return train_loader, val_loader


def load_pneumonia_server_test_data(batch_size=32):
    """Return a DataLoader over the held-out chest X-ray test split (server-only)."""
    import numpy as np
    from torch.utils.data import DataLoader, Subset

    base = _full_dataset("test")
    cap = _subset_cap()
    if cap is not None and cap < len(base):
        rng = np.random.default_rng(123)
        keep = rng.permutation(len(base))[:cap].tolist()
        base = Subset(base, keep)
    return DataLoader(base, batch_size=batch_size, shuffle=False, num_workers=0)


# ---------------------------------------------------------------------------
# Recipe dispatch object (used by the FL scripts).
# ---------------------------------------------------------------------------
class Recipe:
    def __init__(self, meta):
        self.key = meta["key"]
        self.display_name = meta["display_name"]
        self.input_kind = meta["input_kind"]
        self.classes = list(meta["classes"])
        self.base_models = list(meta["base_models"])
        self.optimizers = list(meta["optimizers"])

    @property
    def is_functional(self):
        """Whether this recipe's model/data live in recipes.py (vs legacy scripts)."""
        return self.key == "PNEUMONIA_CNN"

    def build_model(self, device="cpu"):
        if self.key == "PNEUMONIA_CNN":
            return build_pneumonia_cnn().to(device)
        raise NotImplementedError(f"build_model not implemented in recipes.py for {self.key}")

    def input_transform(self):
        if self.key == "PNEUMONIA_CNN":
            return pneumonia_transform()
        raise NotImplementedError(f"input_transform not implemented in recipes.py for {self.key}")

    def load_client_data(self, partition_id, num_clients, **kw):
        if self.key == "PNEUMONIA_CNN":
            return load_pneumonia_client_data(partition_id, num_clients, **kw)
        raise NotImplementedError(f"load_client_data not implemented in recipes.py for {self.key}")

    def load_server_test_data(self, **kw):
        if self.key == "PNEUMONIA_CNN":
            return load_pneumonia_server_test_data(**kw)
        raise NotImplementedError(f"load_server_test_data not implemented in recipes.py for {self.key}")


def get_recipe(key):
    """Return the Recipe for `key` (case-insensitive). Raises on unknown key."""
    if key is None:
        raise ValueError("recipe key is None")
    meta = _METADATA_BY_KEY.get(str(key).upper())
    if meta is None:
        raise ValueError(f"Unknown recipe key: {key}")
    return Recipe(meta)


def is_recipe(key):
    return key is not None and str(key).upper() in _METADATA_BY_KEY


def main():
    parser = argparse.ArgumentParser(description="FedLearn model recipe catalog.")
    parser.add_argument("--describe", action="store_true",
                        help="Print the recipe catalog as JSON to stdout and exit.")
    args = parser.parse_args()
    if args.describe:
        json.dump(describe(), sys.stdout)
        sys.stdout.write("\n")
        return 0
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
