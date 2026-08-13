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
# ---------------------------------------------------------------------------
# Training arm (P1) — frozen-head vs full fine-tune as a DECLARED property.
#
# Before this, the arm was inferred from the recipe KEY: client.py held a module-global
# `USE_DERIVED = False` set by `USE_DERIVED = (mt == "FROZEN_DEMO")`. That gave every recipe
# exactly one hard-coded arm, so PNEUMONIA_CNN could not be run frozen and full as two arms of
# one comparison through the product path — which is the comparison research/results/
# frozen-backbone/ (177 result files) is built on. It also meant a result could not say which
# arm produced it, which is the bug class behind commit 21699bc ("frozen arm silently
# mislabelled its backbone, risking cell overwrites").
#
# A recipe now declares `supported_arms` and, per arm, `trainable_spec[arm]`: the module-name
# prefixes that stay trainable (None = everything). The runtime asks the recipe what to freeze
# rather than comparing keys, which generalises the frozen path to every recipe.
TRAINING_ARMS = ("FULL", "FROZEN_HEAD")
DEFAULT_ARM = "FULL"        # an omitted arm means FULL, so existing projects are unchanged


def apply_arm(model, arm, trainable_prefixes):
    """Set ``requires_grad`` across ``model`` according to the arm, in place.

    ``trainable_prefixes=None`` (the FULL arm) makes everything trainable. Otherwise only
    parameters whose name starts with one of the prefixes stay trainable.

    Always writes EVERY parameter's flag rather than only clearing some, so applying an arm is
    idempotent and a process that ran a frozen arm cannot leak frozen state into a later FULL
    run in the same interpreter.

    Raises if the prefixes match nothing: a typo'd prefix would otherwise freeze the entire model
    and train nothing, which looks like a converged-but-terrible run rather than an error.
    """
    if arm not in TRAINING_ARMS:
        raise ValueError(f"unknown arm {arm!r}; expected one of {TRAINING_ARMS}")
    if trainable_prefixes is None:
        for _n, p in model.named_parameters():
            p.requires_grad_(True)
        return model
    prefixes = tuple(trainable_prefixes)
    hit = 0
    for name, p in model.named_parameters():
        on = name.startswith(prefixes)
        p.requires_grad_(on)
        hit += int(on)
    if hit == 0:
        raise ValueError(
            f"arm {arm!r}: no parameter matches prefixes {list(prefixes)} — the whole model would "
            f"be frozen and nothing would train")
    return model


def freeze_untrained_modules(model, trainable_prefixes):
    """Put every module OUTSIDE the trainable set into eval mode, in place.

    ``apply_arm`` stops gradients; this stops the other way a "frozen" backbone changes.
    BatchNorm's ``running_mean``/``running_var`` are BUFFERS, not parameters, so ``requires_grad``
    does not touch them: a module left in train mode re-estimates them from local data on every
    forward pass. Measured cost of that on CIFAR-10 (linear probe on frozen features):

        BN held fixed:   pretrained 80.37% federated / 80.35% offline probe
    BN adapting:     pretrained 72.85% federated
    random backbone:            25.10% offline probe (BN fixed, same arch/resolution/data)

    So BN adaptation DEGRADES good features by ~7.5 points -- it re-estimates, from one client's
    shard, statistics that ImageNet training had already fitted well. (An earlier reading of this
    said BN adaptation *lifted a random backbone* to 72%; that came from a federated "random
    control" that was invalid. Under a subset arm the backbone is never transmitted, so the client
    always builds it locally from the recipe and a random .npz cannot change it. The valid
    random comparison is the offline probe: 25.10%.)

    Three things are wrong with letting it adapt: the arm's premise is that the
    backbone is delivered once and stays fixed; BN statistics are data-dependent and never
    federated, so clients silently diverge from each other and from the server's copy; and it hides
    the value of pretraining, which is the only reason to freeze a backbone at all.

    ``trainable_prefixes=None`` (the FULL arm) is a no-op — BatchNorm SHOULD adapt when the whole
    model is being trained, and no existing project's behaviour changes.

    ``nn.Module.train()`` re-enables every child, so this must be re-applied after each call to it.
    """
    if not trainable_prefixes:
        return model
    prefixes = tuple(trainable_prefixes)
    for name, module in model.named_modules():
        if not name:
            continue                     # the root stays in train mode
        # Prefixes are PARAMETER-name prefixes ("fc."), while these are MODULE names ("fc"), so the
        # separator is appended before comparing. Matching the raw module name would put the
        # trainable head itself into eval mode -- silently disabling its dropout and, for a head
        # that has one, its own BatchNorm.
        if not (name + ".").startswith(prefixes):
            module.eval()
    return model


def validate_arm(recipe_key, arm):
    """Resolve + validate an arm for a recipe. ``None`` -> DEFAULT_ARM. Raises on unsupported.

    Called at project creation so a bad arm fails there rather than at FL-server spawn.
    """
    meta = _METADATA_BY_KEY.get(recipe_key)
    if meta is None:
        raise ValueError(f"unknown recipe {recipe_key!r}")
    supported = meta.get("supported_arms", (DEFAULT_ARM,))
    if arm in (None, ""):
        # FULL when the recipe supports it, so every existing project is unchanged. A recipe that
        # cannot run FULL (CIFAR_RESNET18: its BatchNorm buffers are int64 and the wire is
        # float32-only) would otherwise fail EVERY creation that omits the arm, which is a
        # capability limit turning into an outage. With exactly one supported arm the choice is
        # unambiguous, so resolve to it rather than raising.
        if DEFAULT_ARM in supported:
            resolved = DEFAULT_ARM
        elif len(supported) == 1:
            resolved = supported[0]
        else:
            raise ValueError(
                f"recipe {recipe_key!r} does not support {DEFAULT_ARM} and offers several arms "
                f"{list(supported)}; the arm must be stated explicitly.")
    else:
        resolved = str(arm)
    if resolved not in supported:
        raise ValueError(
            f"recipe {recipe_key!r} does not support arm {resolved!r}; supported: {list(supported)}")
    return resolved


def trainable_prefixes(recipe_key, arm):
    """The module-name prefixes that stay trainable for this (recipe, arm). None = all."""
    arm = validate_arm(recipe_key, arm)
    spec = _METADATA_BY_KEY[recipe_key].get("trainable_spec", {})
    return spec.get(arm, None)


def arm_stamp(recipe_key, arm):
    """JSON-serializable provenance for a result's ``meta`` block.

    Carries the PREFIXES as well as the arm name deliberately: two runs can share an arm name
    while freezing different modules, and a stamp that recorded only the name would let those be
    compared as if identical — the failure commit 21699bc describes.
    """
    arm = validate_arm(recipe_key, arm)
    pre = trainable_prefixes(recipe_key, arm)
    return {"recipe": recipe_key, "arm": arm,
            "trainable_prefixes": list(pre) if pre is not None else None}


RECIPE_METADATA = [
    {
        "key": "PNEUMONIA_CNN",
        "supported_arms": ["FULL", "FROZEN_HEAD"],
        "trainable_spec": {"FULL": None, "FROZEN_HEAD": ["classifier."]},
        "display_name": "Pneumonia Chest X-ray",
        "input_kind": "image",
        "classes": ["NORMAL", "PNEUMONIA"],
        "base_models": ["pneumonia_cnn"],
        "optimizers": ["Adam", "SGD", "AdamW", "RMSprop"],
        "requirements": {"min_ram_gb": 4, "min_storage_gb": 0.2, "mobile_safe": True,
                         "max_trainable_params": 5000000, "min_os_android": 27, "min_os_ios": "13.0"},
    },
    {
        "key": "CNN",
        "supported_arms": ["FULL", "FROZEN_HEAD"],
        # CnnNet is conv1/conv2/fc1/fc2/fc3 — it has NO "classifier." module, so the prefix this
        # entry originally declared matched nothing and FROZEN_HEAD raised at model build. fc3 is
        # the final classification layer, i.e. the actual head.
        "trainable_spec": {"FULL": None, "FROZEN_HEAD": ["fc3."]},
        "display_name": "Image classifier (CIFAR-10)",
        "input_kind": "image",
        "classes": ["airplane", "automobile", "bird", "cat", "deer",
                    "dog", "frog", "horse", "ship", "truck"],
        "base_models": ["net"],
        "optimizers": ["Adam", "SGD", "RMSprop", "AdamW"],
        "requirements": {"min_ram_gb": 2, "min_storage_gb": 0.1, "mobile_safe": True,
                         "max_trainable_params": 1000000},
    },
    {
        "key": "CIFAR_RESNET18",
        # Both arms. FULL was blocked until 2026-08-13: every BatchNorm module carries an int64
        # num_batches_tracked and the safetensors wire is float32-only, so a FULL run died on the
        # first GetGlobalModel. Non-float32 tensors are now excluded from the federated set
        # (framework federable_state) -- num_batches_tracked is a batch COUNTER, meaningless to
        # average, and each client keeps its own. running_mean/running_var are float32 and are
        # still federated, so this is a wire fix and not FedBN.
        "supported_arms": ["FULL", "FROZEN_HEAD"],
        # torchvision's ResNet head is `fc`; everything else is the ImageNet-pretrained backbone.
        "trainable_spec": {"FULL": None, "FROZEN_HEAD": ["fc."]},
        # Roadmap item (2): a recipe that STARTS from pretrained weights instead of from scratch.
        # Declared rather than implicit so a result can say which weights produced it — the frozen
        # arm's whole premise is that the backbone is worth keeping, and "which backbone" is then
        # the first question anyone asks of the number.
        "pretrained": {
            "source": "torchvision",
            "weights": "ResNet18_Weights.IMAGENET1K_V1",
            "note": "ImageNet-1k classification weights; the 1000-class head is discarded and "
                    "replaced by a freshly initialised 10-class CIFAR head.",
        },
        "display_name": "Image classifier (CIFAR-10, pretrained ResNet-18)",
        "input_kind": "image",
        "classes": ["airplane", "automobile", "bird", "cat", "deer",
                    "dog", "frog", "horse", "ship", "truck"],
        "base_models": ["resnet18"],
        "optimizers": ["Adam", "SGD", "AdamW"],
        "requirements": {"min_ram_gb": 4, "min_storage_gb": 0.2, "mobile_safe": False,
                         "max_trainable_params": 12000000},
    },
    {
        "key": "MLP",
        "supported_arms": ["FULL"],
        "trainable_spec": {"FULL": None},
        "display_name": "ECG heartbeat (Normal/Abnormal)",
        "input_kind": "vector",
        "classes": ["Normal", "Abnormal"],
        "base_models": ["ecg_mlp"],
        "optimizers": ["Adam", "AdamW", "SGD"],
        "requirements": {"min_ram_gb": 2, "min_storage_gb": 0.05, "mobile_safe": True,
                         "max_trainable_params": 200000},
    },
    {
        "key": "TRANSFORMER",
        "supported_arms": ["FULL"],
        "trainable_spec": {"FULL": None},
        "display_name": "Text classifier (OPT-125M)",
        "input_kind": "text",
        "classes": ["entailment", "contradiction", "neutral"],
        "base_models": ["opt-125m"],
        "optimizers": ["AdamW", "Adam"],
        "requirements": {"min_ram_gb": 8, "min_storage_gb": 1.5, "mobile_safe": False,
                         "max_trainable_params": 125000000},
    },
    {
        "key": "LLM_LORA",
        "supported_arms": ["FULL"],
        "trainable_spec": {"FULL": None},
        "display_name": "Text LLM (LoRA fine-tune)",
        "input_kind": "text",
        "task_type": "SEQ_CLASSIFICATION",
        "classes": ["negative", "positive"],
        "base_models": ["qwen2.5-0.5b", "tinyllama-1.1b"],
        "optimizers": ["AdamW", "Adam"],
        "lora": {"r": 8, "alpha": 16, "dropout": 0.05,
                 "target_modules": ["q_proj", "v_proj"]},
        "aggregation": "FFA_LORA",
        "requirements": {"min_ram_gb": 8, "min_storage_gb": 2, "mobile_safe": False,
                         "max_trainable_params": 2000000, "min_os_android": 0, "min_os_ios": "0"},
    },
    {
        # DEMO/TEST recipe: the golden DeComFL TinyNet (Linear(4,5)->ReLU->Linear(5,3), fc2 FROZEN;
        # 25 trainable). Exists so fl_server can build + eval the SAME model the mobile ExecuTorch
        # golden .pte encodes, enabling an on-device DeComFL round-trip end to end. See MO-15.
        "key": "TINYNET_GOLDEN",
        # fc2 is frozen BY CONSTRUCTION in this recipe (25 trainable = fc1), so its frozen
        # behaviour is baked into build_model rather than selectable. FULL-only keeps the golden
        # .pte parity contract intact: an arm switch would change the trainable layout the
        # on-device golden encodes.
        "supported_arms": ["FULL"],
        "trainable_spec": {"FULL": None},
        "display_name": "On-device DeComFL demo (TinyNet)",
        "input_kind": "vector",
        "classes": ["c0", "c1", "c2"],
        "base_models": ["tinynet_golden"],
        "optimizers": ["SGD"],
        "requirements": {"min_ram_gb": 1, "min_storage_gb": 0.01, "mobile_safe": True,
                         "max_trainable_params": 25, "min_os_android": 27, "min_os_ios": "13.0"},
    },
]

_METADATA_BY_KEY = {r["key"]: r for r in RECIPE_METADATA}


_ARM_TRADEOFF_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "arm_tradeoff.json")


def arm_tradeoff(recipe_key=None):
    """The measured frozen-vs-full trade-off FOR ONE RECIPE, or None.

    Generated by ``scripts/build_arm_tradeoff.py`` from that recipe's OWN measurement — never
    hand-written, and never borrowed from another recipe. Returns None rather than raising: a
    missing trade-off must cost the user an explanation, not the ability to create a project.
    Kept torch-free so ``--describe`` stays cheap.

    Passing ``recipe_key=None`` returns the whole map, for tooling.
    """
    try:
        with open(_ARM_TRADEOFF_PATH) as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return None
    by_recipe = data.get("by_recipe", {})
    if recipe_key is None:
        return by_recipe
    # Keyed lookup, never a fallback to "some other recipe's measurement". Returning a default
    # here is exactly the bug this replaced: one chest X-ray result was attached to every dual-arm
    # recipe, so a CIFAR-10 recipe advertised a pneumonia figure.
    return by_recipe.get(str(recipe_key).upper())


def describe():
    """Return the catalog metadata (list of dicts). Used by --describe.

    Recipes that offer a CHOICE of arms are annotated with the measured trade-off, so the picker
    can show what the choice costs at the point it is made. Recipes with a single arm are not:
    there is nothing to trade off, and attaching it would imply the un-offered arm had been
    evaluated for that recipe.
    """
    out = []
    for meta in RECIPE_METADATA:
        if len(meta.get("supported_arms", ())) > 1:
            tradeoff = arm_tradeoff(meta["key"])
            if tradeoff:
                meta = dict(meta, arm_tradeoff=tradeoff)
        out.append(meta)
    return out


def catalog_keys():
    """The advertised (selectable) recipe keys, in catalog order — the valid ``--model-type``
    values. Data-driven so a new catalog recipe is automatically an accepted model type (no
    argparse enum edit in fl_server.py / client.py). Excludes non-catalog keys (e.g. BLOOD_CNN),
    which stay dispatchable-but-not-selectable, exactly like the old hardcoded lists."""
    return [r["key"] for r in RECIPE_METADATA]


def build_tinynet_golden():
    """DEMO: the golden DeComFL TinyNet — Linear(4,5)->ReLU->Linear(5,3), fc2 FROZEN.
    Byte-matches framework/tests/fixtures/decomfl_golden/generate_zo.py so fl_server can build +
    server-side-eval the SAME model the mobile ExecuTorch golden .pte encodes (25 trainable = fc1).

    DETERMINISTIC init (seed 0, exactly as generate_zo.py freezes the golden): the frozen fc2 is
    NEVER synced over the wire (only the 25 trainable fc1 params are), so every federation peer —
    the phone's golden .pte AND every desktop client that builds via this recipe — must materialise
    the *identical* frozen backbone, or their zeroth-order gradient scalars would be derivatives of
    different functions and the aggregate would be meaningless. Seeding here makes the desktop a
    genuine peer of the phone. Uses fork_rng so the caller's global RNG stream is untouched.
    """
    import torch
    import torch.nn as nn

    class TinyNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(4, 5)
            self.fc2 = nn.Linear(5, 3)
            for p in self.fc2.parameters():
                p.requires_grad_(False)

        def forward(self, x):
            return self.fc2(torch.relu(self.fc1(x)))

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        return TinyNet()


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


def _hf_load_kwargs(hf_split, env=None):
    """Build the ``datasets.load_dataset`` kwargs for the HuggingFace pneumonia fallback.

    SE-19: remote-code execution is OFF by default. ``load_dataset(repo, ..., trust_remote_code=True)``
    downloads and RUNS the dataset repo's loader script on THIS (backend) host; the default repo is
    unpinned, so a supply-chain compromise of it would be arbitrary code execution the moment a
    PNEUMONIA_CNN run starts. We therefore:
      * never enable ``trust_remote_code`` unless the operator explicitly sets
        ``FEDLEARN_PNEUMONIA_TRUST_REMOTE_CODE=1`` (a deliberate, auditable choice), and
      * pin the dataset to a commit when ``FEDLEARN_PNEUMONIA_REVISION`` is set, so even an opted-in
        run executes a known, immutable revision rather than whatever the repo's HEAD becomes.
    """
    env = os.environ if env is None else env
    kwargs = {"split": hf_split}
    revision = env.get("FEDLEARN_PNEUMONIA_REVISION", "").strip()
    if revision:
        kwargs["revision"] = revision
    if env.get("FEDLEARN_PNEUMONIA_TRUST_REMOTE_CODE", "").strip() == "1":
        kwargs["trust_remote_code"] = True
    return kwargs


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
    kwargs = _hf_load_kwargs(hf_split)  # SE-19: no remote-code exec unless explicitly opted in
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
# Blood Cell recipe specifics (MedMNIST BloodMNIST — 8-class microscopy).
# Peripheral-blood-cell classification: a clean, balanced, demo-friendly medical
# imaging recipe. Trains to ~88% in a few CPU epochs.
#
# Data resolution: MedMNIST auto-downloads BloodMNIST (.npz, ~30MB, cached under
# ~/.medmnist) on first use. Same recipe-backed contract as PNEUMONIA_CNN.
#
# NOT in RECIPE_METADATA / the project-creation catalog (SE-10): the recipe
# below is fully functional (build_blood_cnn() and load_blood_*_data() have
# been verified to build the model and pull a real BloodMNIST batch), but
# `medmnist` — and its transitive `scikit-image`/`fire` deps — is not declared
# in any of this repo's requirement files (framework/requirements.txt drives
# the actual fl_server.py spawn env; backend/fl-platform-api/requirements.txt,
# client-docker/requirements.txt and client-docker/packaging/requirements-
# client.txt cover the rest). Advertising this key would let the SE-10 catalog
# gate pass and then crash the spawn on ModuleNotFoundError the moment
# load_blood_server_test_data() runs — the same failure class SE-10 exists to
# prevent, just one import deeper. Re-enable by adding those dependencies
# everywhere fl_server.py/client.py run (verify aarch64/Jetson wheel
# availability for scikit-image first), re-adding the RECIPE_METADATA entry,
# and wiring is_blood dispatch branches in fl_server.py/client.py mirroring
# the PNEUMONIA_CNN branches.
# ---------------------------------------------------------------------------
BLOOD_CLASSES = ["Basophil", "Eosinophil", "Erythroblast",
                 "Immature granulocyte", "Lymphocyte", "Monocyte",
                 "Neutrophil", "Platelet"]
BLOOD_IMG_SIZE = 28

# DA-14 Phase 0: register BLOOD_CNN as a NON-CATALOG recipe. init_model.py dispatches BLOOD_CNN
# (recipes.get_recipe('BLOOD_CNN')), but it was absent from the registry so that call raised a
# ValueError — a latent crash. Registering it here makes get_recipe/is_recipe resolve it, while
# describe() still serves only RECIPE_METADATA, so it stays OUT of --describe / the project-creation
# picker (SE-10: its `medmnist` dep is in no requirements file; advertising it would pass the catalog
# gate and then crash the spawn on ModuleNotFoundError). Re-promote to RECIPE_METADATA once medmnist
# (+ scikit-image/fire, aarch64 wheels verified) ships everywhere fl_server.py/client.py run.
# DA-14 Ph3.0 (recipe side): a frozen-backbone + trainable-head derived model. Activates the DA-11
# partial-load path (reconstruct_frozen_backbone) at the recipe layer — a fresh model loads a
# content-addressed frozen backbone and federates ONLY its head (the trainable subset). Registered
# NON-CATALOG (a demo, superseded by the real derivation-record recipe in Ph3.3).
def build_frozen_backbone_model(num_classes, backbone_bytes=None, d_in=256, d_hidden=128, device="cpu"):
    """A small derived model: a frozen ``Linear`` backbone + a trainable ``Linear`` head. If
    ``backbone_bytes`` is given, the backbone is loaded (non-strict) from that content-addressed blob
    and re-frozen via the DA-11 ``reconstruct_frozen_backbone``; otherwise the backbone is frozen in
    place. The head is the only trainable (federated) subset."""
    import torch
    import torch.nn as nn

    class _FrozenBackboneNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Linear(d_in, d_hidden)
            self.head = nn.Linear(d_hidden, num_classes)
            # Declare the frozen layout at construction — reconstruct_frozen_backbone validates the
            # blob against frozen_state(model).keys() BEFORE loading, so the backbone must already be
            # frozen (this is also the no-blob steady state: frozen backbone + trainable head).
            for p in self.backbone.parameters():
                p.requires_grad_(False)

        def forward(self, x):
            return self.head(torch.relu(self.backbone(x)))

    if backbone_bytes is not None:
        model = _FrozenBackboneNet()
        from fedlearn.backbone.distribution import reconstruct_frozen_backbone
        reconstruct_frozen_backbone(model, backbone_bytes)  # validate blob==frozen layout, load, re-freeze
    else:
        # Deterministic frozen backbone (seed 0, like build_tinynet_golden): EVERY federation peer
        # (server + all clients) must rebuild the IDENTICAL frozen backbone from the recipe alone —
        # the head is broadcast/aggregated over the wire, the backbone is not. A random backbone per
        # peer would make the aggregated head meaningless. fork_rng leaves the caller's RNG untouched.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            model = _FrozenBackboneNet()
    return model.to(device)


FROZEN_DEMO_INPUT_DIM = 256
FROZEN_DEMO_NUM_CLASSES = 3


def _frozen_demo_dataset(seed, n):
    """A deterministic, self-contained synthetic (features, labels) vector dataset for FROZEN_DEMO —
    no external data. Labels follow a fixed linear rule so the head has a real (learnable) target."""
    import torch
    from torch.utils.data import TensorDataset
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, FROZEN_DEMO_INPUT_DIM, generator=g)
    rule = torch.randn(FROZEN_DEMO_INPUT_DIM, FROZEN_DEMO_NUM_CLASSES,
                       generator=torch.Generator().manual_seed(0))  # fixed across peers/splits
    y = (X @ rule).argmax(dim=1)
    return TensorDataset(X, y)


def load_frozen_demo_client_data(partition_id, num_clients, batch_size=16, **kw):
    """(train_loader, val_loader) of self-contained synthetic vector batches for one FROZEN_DEMO
    client shard. Deterministic per partition; the derived recipe is spawnable without any dataset."""
    from torch.utils.data import DataLoader
    ds = _frozen_demo_dataset(seed=1000 + partition_id, n=128)
    return (DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0),
            DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0))


def load_frozen_demo_server_test_data(batch_size=64, **kw):
    """Server-side synthetic test loader for FROZEN_DEMO (held-out seed)."""
    from torch.utils.data import DataLoader
    return DataLoader(_frozen_demo_dataset(seed=999, n=64), batch_size=batch_size, shuffle=False)


_NONCATALOG_METADATA = [
    {
        "key": "BLOOD_CNN",
        "supported_arms": ["FULL", "FROZEN_HEAD"],
        "trainable_spec": {"FULL": None, "FROZEN_HEAD": ["classifier."]},
        "display_name": "Blood cell classifier (BloodMNIST)",
        "input_kind": "image",
        "classes": BLOOD_CLASSES,
        "base_models": ["blood_cnn"],
        "optimizers": ["Adam", "SGD", "AdamW"],
        "requirements": {"min_ram_gb": 2, "min_storage_gb": 0.1, "mobile_safe": True,
                         "max_trainable_params": 1000000},
    },
    {
        "key": "FROZEN_DEMO",
        "supported_arms": ["FULL", "FROZEN_HEAD"],
        "trainable_spec": {"FULL": None, "FROZEN_HEAD": ["head."]},
        "display_name": "Frozen-backbone demo (head-only federation)",
        "input_kind": "vector",
        "classes": ["c0", "c1", "c2"],
        "base_models": ["frozen_demo"],
        "optimizers": ["SGD"],
        "requirements": {"min_ram_gb": 1, "min_storage_gb": 0.01, "mobile_safe": True,
                         "max_trainable_params": 100000},
    },
]
_METADATA_BY_KEY.update({r["key"]: r for r in _NONCATALOG_METADATA})


def build_blood_cnn():
    """BloodCNN — 3x28x28 RGB -> 8 logits (peripheral blood cell types)."""
    import torch.nn as nn

    class BloodCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 14
                nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 7
                nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 3
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(128 * 3 * 3, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 8),
            )

        def forward(self, x):
            return self.classifier(self.features(x))

    return BloodCNN()


def blood_transform():
    """RGB -> 28x28 -> tensor -> Normalize([-1,1]). Used for train AND inference."""
    import torchvision.transforms as T
    return T.Compose([
        T.Lambda(lambda im: im.convert("RGB")),
        T.Resize((BLOOD_IMG_SIZE, BLOOD_IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


class _MedMNISTDataset:
    """torch Dataset over a MedMNIST split, applying a given transform.

    Yields (tensor[3,28,28], int_label). Labels come from the MedMNIST ``.labels``
    array (shape Nx1) flattened to ints, so the Dirichlet partitioner sees plain
    integer class labels.
    """

    def __init__(self, medmnist_cls, split, transform):
        import numpy as np
        self._ds = medmnist_cls(split=split, download=True, size=BLOOD_IMG_SIZE,
                                transform=transform)
        self._labels = np.asarray(self._ds.labels).flatten().astype(int).tolist()

    def labels(self):
        return list(self._labels)

    def __len__(self):
        return len(self._labels)

    def __getitem__(self, i):
        img, _ = self._ds[i]
        return img, self._labels[i]


def _blood_full_dataset(split):
    """Return a dataset wrapper for the requested BloodMNIST split ('train'/'test')."""
    from medmnist import BloodMNIST
    hf_split = "test" if split == "test" else "train"
    return _MedMNISTDataset(BloodMNIST, hf_split, blood_transform())


def load_blood_client_data(partition_id, num_clients, alpha=0.5, seed=42,
                           batch_size=64, val_fraction=0.1):
    """Return (train_loader, val_loader) for one client's Dirichlet shard."""
    from torch.utils.data import DataLoader, Subset

    base = _blood_full_dataset("train")
    labels = base.labels()

    client_indices = _dirichlet_indices(labels, num_clients, alpha, seed)
    if not (0 <= partition_id < num_clients):
        raise ValueError(f"partition_id {partition_id} out of range for num_clients {num_clients}")
    my = client_indices[partition_id]
    if len(my) == 0:
        raise ValueError(f"Dirichlet split gave client {partition_id} zero samples; raise alpha or data size.")

    n_val = max(1, int(len(my) * val_fraction)) if len(my) > 1 else 0
    val_idx, train_idx = my[:n_val], my[n_val:]
    train_loader = DataLoader(Subset(base, train_idx), batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(Subset(base, val_idx), batch_size=batch_size, shuffle=False, num_workers=0) if val_idx else None
    return train_loader, val_loader


def load_blood_server_test_data(batch_size=128):
    """Return a DataLoader over the held-out BloodMNIST test split (server-only)."""
    from torch.utils.data import DataLoader
    return DataLoader(_blood_full_dataset("test"), batch_size=batch_size, shuffle=False, num_workers=0)


# ---------------------------------------------------------------------------
# MLP — ECG heartbeat classification (140-float vector -> Normal/Abnormal).
# Thin wrappers over data_loaders.ecg_loader; every hyperparameter comes from
# config.get_dataset_config("ecg") (NOT recipe-style defaults) so the Dirichlet partition and
# train/test split are byte-identical to the legacy client.py / fl_server.py call sites.
# num_clients is caller-supplied (each site passes its own value; do not hardcode it).
# ---------------------------------------------------------------------------
def _ecg_default_csv_path():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "ecg_data", "ecg.csv")


def _read_ecg_csv(dataset_path):
    import numpy as np
    import pandas as pd
    df = pd.read_csv(dataset_path or _ecg_default_csv_path(), header=None)
    X = df.iloc[:, :-1].values.astype(np.float32)
    y = df.iloc[:, -1].values.astype(np.int64)
    return X, y


def load_ecg_client_data(partition_id, num_clients, dataset_path=None, **kw):
    """(train_loader, val_loader) for one ECG client shard — byte-identical to the legacy
    client.py DeComFL/MLP block. Hyperparameters sourced from the ecg dataset config."""
    from config import get_dataset_config
    from data_loaders.ecg_loader import get_ecg_loaders
    c = get_dataset_config("ecg")
    X, y = _read_ecg_csv(dataset_path)
    train_loader, val_loader, _info = get_ecg_loaders(
        X=X, y=y, client_id=partition_id, num_clients=num_clients,
        batch_size_train=c.batch_size_train, batch_size_test=c.batch_size_test,
        data_fraction=c.data_fraction, alpha=c.alpha, test_size=c.test_size,
        num_workers=0, seed=c.seed)
    return train_loader, val_loader


def load_ecg_server_test_data(num_clients, dataset_path=None, **kw):
    """Server-side ECG test DataLoader — byte-identical to the legacy fl_server.py block.
    num_clients is required for split-cache consistency (the caller passes config.num_clients)."""
    from config import get_dataset_config
    from data_loaders.ecg_loader import get_test_loader
    c = get_dataset_config("ecg")
    X, y = _read_ecg_csv(dataset_path)
    test_loader, _info = get_test_loader(
        X=X, y=y, num_clients=num_clients, batch_size=c.batch_size_test,
        alpha=c.alpha, data_fraction=c.data_fraction, test_size=c.test_size,
        num_workers=0, seed=c.seed)
    return test_loader


# ---------------------------------------------------------------------------
# CNN — CIFAR-10 image classification. Source asymmetry is INTENTIONAL and preserved:
# the client shard comes from HuggingFace 'cifar10' (IID shard of the seed-42-shuffled
# train split); the server test set comes from torchvision CIFAR10.
# num_clients is IGNORED for the partitioner — the legacy path shards into a FIXED
# CNN_NUM_PARTITIONS(=10) regardless of client count. Do NOT route through the Dirichlet
# partitioner (_dirichlet_indices) — that would change every shard.
#
# P0-2b: this used to go through flwr_datasets.FederatedDataset. That dependency pulled in
# flwr 1.20.0, which caps cryptography<45.0.0 and made the framework's own >=46.0.6 security
# floor unreachable (the SE-22 residual) — a competitor's package governing this platform's
# security posture, for one function call. The shard is now taken directly and is
# BYTE-IDENTICAL to what flwr produced, so every CIFAR-10 result recorded before the swap
# stays comparable to everything after it. That equivalence is verified, not assumed:
# research/benchmarks/verify_flwr_shard_equivalence.py, recorded under
# research/results/reproducibility/flwr_shard_equivalence.json.
# ---------------------------------------------------------------------------
CNN_NUM_PARTITIONS = 10       # == client.py NUM_PARTITIONS; fixed, NOT num_clients
CNN_BATCH_SIZE = 32           # == client.py BATCH_SIZE
CNN_SERVER_TEST_BATCH = 128   # == data.py CNN server test batch
CNN_SHUFFLE_SEED = 42         # == the flwr FederatedDataset default this replaced


def _cnn_transform():
    """CIFAR-10 tensor transform: ToTensor -> Normalize to [-1,1] per channel.
    == client.py CNN branch and data.py CNN branch (kept identical on both sides)."""
    import torchvision.transforms as T
    return T.Compose([T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def _cnn_iid_shard(partition_id, num_shards=CNN_NUM_PARTITIONS, seed=CNN_SHUFFLE_SEED):
    """One IID shard of the shuffled CIFAR-10 train split.

    Reproduces exactly what flwr_datasets did, which was itself only a thin wrapper over
    `datasets`: FederatedDataset defaults to shuffle=True/seed=42 and shuffles each split
    BEFORE partitioning, and IidPartitioner.load_partition(i) is precisely
    `shard(num_shards=N, index=i, contiguous=True)`. Both facts were read off the installed
    flwr-datasets 0.5.0 source and then verified empirically per-partition.
    """
    import datasets as hf_datasets
    train = hf_datasets.load_dataset("cifar10")["train"].shuffle(seed=seed)
    return train.shard(num_shards=num_shards, index=partition_id, contiguous=True)


def load_cnn_client_data(partition_id, num_clients, batch_size=CNN_BATCH_SIZE, **kw):
    """(train_loader, val_loader) for one CIFAR-10 client shard — byte-identical to the legacy
    client.py CNN branch. num_clients is accepted but IGNORED: the shard is a fixed
    CNN_NUM_PARTITIONS(=10) IID shard, independent of client count."""
    from torch.utils.data import DataLoader
    partition = _cnn_iid_shard(partition_id)
    parts = partition.train_test_split(test_size=0.2, seed=42)
    tf = _cnn_transform()

    def _apply(batch):
        batch["img"] = [tf(img) for img in batch["img"]]
        return batch

    parts = parts.with_transform(_apply)
    return (
        DataLoader(parts["train"], batch_size=batch_size, shuffle=True, num_workers=0),
        DataLoader(parts["test"], batch_size=batch_size, num_workers=0),
    )


def load_cnn_server_test_data(batch_size=CNN_SERVER_TEST_BATCH, **kw):
    """Server-side CIFAR-10 test DataLoader — byte-identical to the legacy data.py CNN branch.
    Uses TORCHVISION CIFAR10 (not the flwr/HF source the client uses) — asymmetry preserved."""
    from torchvision import datasets as tv_datasets
    from torch.utils.data import DataLoader
    test_dataset = tv_datasets.CIFAR10(root="./data", train=False, download=True,
                                       transform=_cnn_transform())
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# ---------------------------------------------------------------------------
# CIFAR_RESNET18 — the pretrained-backbone recipe (roadmap item 2).
#
# Exists because the frozen arm was measured to be useless without one: FROZEN_HEAD on a randomly
# initialised backbone sat at chance (10.0%) in the 2026-08-13 live run, since a linear probe on
# random features has almost nothing to separate. ImageNet features are the thing the research
# campaign's frozen arm actually froze.
# ---------------------------------------------------------------------------
RESNET18_IMG_SIZE = 112       # see the note in _resnet18_transform
RESNET18_HEAD_SEED = 0        # the fresh head is seeded so a run is reproducible


def _resnet18_transform():
    """CIFAR-10 -> ImageNet-normalised tensors at RESNET18_IMG_SIZE.

    Two deliberate choices:

    * **Resize.** 32x32 through an ImageNet backbone produces poor features — the network's stride
      schedule collapses such an input almost immediately. The resolution the weights were trained
      near is part of what makes them worth freezing. 112 rather than 224 is a measured compromise:
      224 costs 64.6 s per client-epoch of forward pass on this CPU versus 22.8 s at 112.
    * **ImageNet statistics**, not CIFAR's [-1,1]. The frozen backbone expects the normalisation it
      was trained under; feeding it a different one silently degrades every feature it produces.
    """
    import torchvision.transforms as T
    return T.Compose([
        T.Resize((RESNET18_IMG_SIZE, RESNET18_IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def build_cifar_resnet18(device="cpu", **kw):
    """ImageNet-pretrained ResNet-18 with a fresh 10-class head.

    The pretrained weights are REQUIRED, not best-effort. Falling back to a random init would
    reproduce the exact 10%-accuracy failure this recipe exists to fix, while still reporting
    itself as pretrained — so a missing checkpoint raises with the download command instead.
    """
    import torch
    import torch.nn as nn
    import torchvision

    try:
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
        model = torchvision.models.resnet18(weights=weights)
    except Exception as exc:                     # offline with a cold cache, mainly
        raise RuntimeError(
            f"CIFAR_RESNET18 could not load its ImageNet weights ({exc}). This recipe is defined by "
            f"those weights: a random init would put the frozen arm back at chance while still "
            f"calling itself pretrained. Pre-fetch them with:\n"
            f"  python -c \"import torchvision; "
            f"torchvision.models.resnet18(weights='IMAGENET1K_V1')\"") from exc

    # The 1000-class ImageNet head is discarded; this is the part that trains under FROZEN_HEAD.
    # Seeded so two clients starting from the same .npz agree, and so a run is reproducible.
    with torch.random.fork_rng():
        torch.manual_seed(RESNET18_HEAD_SEED)
        model.fc = nn.Linear(model.fc.in_features, 10)
    return model.to(device)


def load_cifar_resnet18_client_data(partition_id, num_clients, batch_size=CNN_BATCH_SIZE, **kw):
    """One CIFAR-10 client shard, transformed for the ImageNet backbone.

    Reuses the CNN recipe's IID shard so the two recipes are comparable on data: the ONLY
    difference between a CNN run and a CIFAR_RESNET18 run is the model and its transform.
    """
    from torch.utils.data import DataLoader
    partition = _cnn_iid_shard(partition_id)
    parts = partition.train_test_split(test_size=0.2, seed=42)
    tf = _resnet18_transform()

    def _apply(batch):
        batch["img"] = [tf(img) for img in batch["img"]]
        return batch

    parts = parts.with_transform(_apply)
    return (
        DataLoader(parts["train"], batch_size=batch_size, shuffle=True, num_workers=0),
        DataLoader(parts["test"], batch_size=batch_size, num_workers=0),
    )


def load_cifar_resnet18_server_test_data(batch_size=CNN_SERVER_TEST_BATCH, **kw):
    """Server-side CIFAR-10 test set under the ImageNet transform."""
    from torchvision import datasets as tv_datasets
    from torch.utils.data import DataLoader
    ds = tv_datasets.CIFAR10(root="./data", train=False, download=True,
                             transform=_resnet18_transform())
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


# ---------------------------------------------------------------------------
# LLM_LORA — federated LoRA sequence classification (Qwen2.5-0.5B / TinyLlama).
# ---------------------------------------------------------------------------
LLM_LORA_BASE_MODELS = {
    "qwen2.5-0.5b": "Qwen/Qwen2.5-0.5B",
    "tinyllama-1.1b": "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
}
LLM_LORA_HEAD = "score"  # *ForSequenceClassification head attr


def _resolve_llm_base(model_name):
    """Token (e.g. 'qwen2.5-0.5b') -> HF id. FEDLEARN_LLM_LORA_BASE overrides (tests/offline)."""
    override = os.environ.get("FEDLEARN_LLM_LORA_BASE")
    if override:
        return override
    key = (model_name or "qwen2.5-0.5b").lower()
    if key not in LLM_LORA_BASE_MODELS:
        raise ValueError(f"Unknown LLM_LORA base '{model_name}'. Known: {sorted(LLM_LORA_BASE_MODELS)}")
    return LLM_LORA_BASE_MODELS[key]


def apply_lora(base_model, lora_cfg, aggregation, task_type="SEQ_CLASSIFICATION"):
    """Wrap a base model with LoRA; under FFA_LORA freeze every lora_A. task_type selects the
    peft task + whether a classification head (modules_to_save) is trained."""
    if aggregation not in ("FFA_LORA", "FEDIT"):
        raise ValueError(f"unknown aggregation {aggregation!r}; expected FFA_LORA or FEDIT")
    from peft import LoraConfig, get_peft_model
    is_causal = task_type == "CAUSAL_LM"
    cfg = LoraConfig(
        r=lora_cfg["r"], lora_alpha=lora_cfg["alpha"], lora_dropout=lora_cfg["dropout"],
        bias="none", task_type=("CAUSAL_LM" if is_causal else "SEQ_CLS"),
        target_modules=list(lora_cfg["target_modules"]),
        modules_to_save=(None if is_causal else [LLM_LORA_HEAD]),
    )
    model = get_peft_model(base_model, cfg)
    if aggregation == "FFA_LORA":
        for n, p in model.named_parameters():
            if "lora_A" in n:
                p.requires_grad = False
    return model


def _load_llm_tokenizer(model_name=None):
    """Load and configure the tokenizer for the given LLM_LORA base model."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(_resolve_llm_base(model_name))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def llm_lora_adapter_keys(model, aggregation):
    """Keys the CLIENT uploads: FFA -> B+head; FEDIT -> A+B+head. Substring-matched."""
    if aggregation not in ("FFA_LORA", "FEDIT"):
        raise ValueError(f"unknown aggregation {aggregation!r}; expected FFA_LORA or FEDIT")
    from peft import get_peft_model_state_dict
    keep = set()
    for k in get_peft_model_state_dict(model, save_embedding_layers=False):
        is_a = "lora_A" in k
        is_b = "lora_B" in k
        # head appears as '...score...' (compact) or '...modules_to_save...' across peft versions
        is_head = (LLM_LORA_HEAD in k) or ("modules_to_save" in k)
        if aggregation == "FFA_LORA":
            if is_b or is_head:
                keep.add(k)
        else:  # FEDIT
            if is_a or is_b or is_head:
                keep.add(k)
    return keep


def _sst2_tokenize(split, max_length=64, model_name=None):
    from datasets import load_dataset
    tok = _load_llm_tokenizer(model_name)
    ds = load_dataset("glue", "sst2", split=split)
    cap = os.environ.get("FEDLEARN_LLM_LORA_SUBSET")
    if cap is not None:
        ds = ds.select(range(min(int(cap), len(ds))))

    def _tok(ex):
        out = tok(ex["sentence"], padding="max_length", truncation=True, max_length=max_length)
        out["labels"] = ex["label"]
        return out

    return ds.map(_tok, batched=True, remove_columns=ds.column_names).with_format("torch")


def load_sst2_client_data(partition_id, num_clients, batch_size=8, seed=42, model_name=None, **kw):
    import numpy as np
    from torch.utils.data import DataLoader, Subset
    ds = _sst2_tokenize("train", model_name=model_name)
    n = len(ds)
    if not (0 <= partition_id < num_clients):
        raise ValueError(f"partition_id {partition_id} out of range for {num_clients} clients")
    perm = np.random.default_rng(seed).permutation(n)
    shard = perm[partition_id::num_clients]  # deterministic, disjoint round-robin shards
    if len(shard) == 0:
        raise ValueError(f"client {partition_id} got an empty SST-2 shard")
    train = DataLoader(Subset(ds, shard.tolist()), batch_size=batch_size, shuffle=True, num_workers=0)
    return train, None


def load_sst2_server_test_data(batch_size=16, model_name=None, **kw):
    from torch.utils.data import DataLoader
    ds = _sst2_tokenize("validation", model_name=model_name)  # test labels are -1 -> unusable
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)


def _dolly_tokenize(split, max_length=256, model_name=None):
    from datasets import load_dataset
    tok = _load_llm_tokenizer(model_name)
    full = load_dataset("databricks/databricks-dolly-15k", split="train")
    cap = os.environ.get("FEDLEARN_LLM_LORA_SUBSET")
    if cap is not None:
        full = full.select(range(min(int(cap), len(full))))
    parts = full.train_test_split(test_size=0.1, seed=42)
    ds = parts["train"] if split == "train" else parts["test"]

    def _render(ex):
        ctx = (ex.get("context") or "").strip()
        body = f"{ex['instruction']}\n{ctx}" if ctx else ex["instruction"]
        text = f"### Instruction:\n{body}\n### Response:\n{ex['response']}"
        out = tok(text, padding="max_length", truncation=True, max_length=max_length)
        out["labels"] = [tid if m == 1 else -100 for tid, m in zip(out["input_ids"], out["attention_mask"])]
        return out

    return ds.map(_render, remove_columns=ds.column_names).with_format("torch")


def load_dolly_client_data(partition_id, num_clients, batch_size=4, seed=42, model_name=None, **kw):
    import numpy as np
    from torch.utils.data import DataLoader, Subset
    ds = _dolly_tokenize("train", model_name=model_name)
    n = len(ds)
    if not (0 <= partition_id < num_clients):
        raise ValueError(f"partition_id {partition_id} out of range for {num_clients} clients")
    perm = np.random.default_rng(seed).permutation(n)
    shard = perm[partition_id::num_clients]
    if len(shard) == 0:
        raise ValueError(f"client {partition_id} got an empty dolly shard")
    train = DataLoader(Subset(ds, shard.tolist()), batch_size=batch_size, shuffle=True, num_workers=0)
    return train, None


def load_dolly_server_test_data(batch_size=8, model_name=None, **kw):
    from torch.utils.data import DataLoader
    ds = _dolly_tokenize("test", model_name=model_name)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)


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
        self.task_type = meta.get("task_type")
        self.lora = meta.get("lora")

    @property
    def is_functional(self):
        """Whether this recipe's model AND data both live in recipes.py (vs legacy scripts).
        CNN/MLP joined after the DA-14 Phase-1 collapse (model + client/server data now in the
        registry). TRANSFORMER is NOT here yet — its model + tokenizer are collapsed but its data
        loading still lives in client.py/data.py."""
        return self.key in ("PNEUMONIA_CNN", "LLM_LORA", "BLOOD_CNN", "TINYNET_GOLDEN", "CNN", "MLP",
                            "FROZEN_DEMO")

    def build_model(self, device="cpu", model_name=None, aggregation="FFA_LORA",
                    task_type="SEQ_CLASSIFICATION"):
        if self.key == "CIFAR_RESNET18":
            return build_cifar_resnet18(device)
        if self.key == "CNN":
            # DA-14 Phase 1: CNN construction moved off the init_model.py if/elif onto the registry,
            # using the canonical CnnNet (models.CnnNet — the one client.py trains); byte-identical
            # state-dict keys to the legacy inline CnnNet (pinned by the state-key golden test).
            from models import CnnNet
            return CnnNet().to(device)
        if self.key == "MLP":
            # ECG MLP (140-dim heartbeat -> num_classes). Dims are intrinsic to this recipe; a later
            # phase moves them into the derivation record.
            from models.ecg_mlp import ECGModel
            return ECGModel(input_dim=140, hidden_dim=64, num_classes=len(self.classes)).to(device)
        if self.key == "TRANSFORMER":
            # DA-14 Phase 1 / FR-29: opt-125m sequence-classification model. num_labels is sourced
            # from the recipe's class list (=3, CB) — the single authority that ends the three-way
            # split (init_model.py hardcoded 3; client.py used the dataset's num_classes, so an sst2
            # 2-class head could never strict-load the 3-class global model init_model produces).
            # Byte-identical to the legacy from_pretrained(num_labels=3) build; the score head is
            # randomly initialised (no seed) in every legacy path, so only key order + head shape are
            # load-bearing on the wire.
            # DA-14 Ph3.1: the opt-125m-only guard lives here (was in init_model) so every caller
            # enforces it uniformly. A None model_name (client/infer, which don't pass one) is allowed.
            if model_name is not None and str(model_name).lower() != "opt-125m":
                raise ValueError(f"Unsupported Transformer model: {model_name}")
            from transformers import AutoModelForSequenceClassification
            return AutoModelForSequenceClassification.from_pretrained(
                "facebook/opt-125m", num_labels=len(self.classes), use_safetensors=True
            ).to(device)
        if self.key == "FROZEN_DEMO":
            # DA-14 Ph3.0: frozen backbone + trainable head (head-only federation). No blob here —
            # the backbone is frozen in place; the round-trip/BASE_REF path passes backbone_bytes.
            return build_frozen_backbone_model(num_classes=len(self.classes), device=device)
        if self.key == "TINYNET_GOLDEN":
            return build_tinynet_golden().to(device)
        if self.key == "PNEUMONIA_CNN":
            return build_pneumonia_cnn().to(device)
        if self.key == "BLOOD_CNN":
            return build_blood_cnn().to(device)
        if self.key == "LLM_LORA":
            base_id = _resolve_llm_base(model_name)
            tok = _load_llm_tokenizer(model_name)
            if task_type == "CAUSAL_LM":
                from transformers import AutoModelForCausalLM
                base = AutoModelForCausalLM.from_pretrained(base_id)
            else:
                from transformers import AutoModelForSequenceClassification
                base = AutoModelForSequenceClassification.from_pretrained(
                    base_id, num_labels=len(self.classes))
            base.config.pad_token_id = tok.pad_token_id
            model = apply_lora(base, self.lora, aggregation, task_type)
            return model.to(device)
        raise NotImplementedError(f"build_model not implemented in recipes.py for {self.key}")

    def input_transform(self, model_name=None):
        if self.key == "PNEUMONIA_CNN":
            return pneumonia_transform()
        if self.key == "BLOOD_CNN":
            return blood_transform()
        if self.key == "TRANSFORMER":
            # opt-125m tokenizer with a guaranteed pad token (opt has none by default -> eos).
            # The single tokenizer authority so infer.py can delegate instead of rebuilding it.
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained("facebook/opt-125m")
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            return tok
        if self.key == "LLM_LORA":
            return _load_llm_tokenizer(model_name)
        raise NotImplementedError(f"input_transform not implemented in recipes.py for {self.key}")

    def load_client_data(self, partition_id, num_clients, task_type="SEQ_CLASSIFICATION", **kw):
        if self.key == "PNEUMONIA_CNN":
            return load_pneumonia_client_data(partition_id, num_clients, **kw)
        if self.key == "BLOOD_CNN":
            return load_blood_client_data(partition_id, num_clients, **kw)
        if self.key == "CNN":
            return load_cnn_client_data(partition_id, num_clients, **kw)
        if self.key == "CIFAR_RESNET18":
            return load_cifar_resnet18_client_data(partition_id, num_clients, **kw)
        if self.key == "MLP":
            return load_ecg_client_data(partition_id, num_clients, **kw)
        if self.key == "FROZEN_DEMO":
            return load_frozen_demo_client_data(partition_id, num_clients, **kw)
        if self.key == "LLM_LORA":
            if task_type == "CAUSAL_LM":
                return load_dolly_client_data(partition_id, num_clients, **kw)
            return load_sst2_client_data(partition_id, num_clients, **kw)
        raise NotImplementedError(f"load_client_data not implemented in recipes.py for {self.key}")

    def load_server_test_data(self, task_type="SEQ_CLASSIFICATION", **kw):
        if self.key == "PNEUMONIA_CNN":
            return load_pneumonia_server_test_data(**kw)
        if self.key == "BLOOD_CNN":
            return load_blood_server_test_data(**kw)
        if self.key == "CNN":
            return load_cnn_server_test_data(**kw)
        if self.key == "CIFAR_RESNET18":
            return load_cifar_resnet18_server_test_data(**kw)
        if self.key == "MLP":
            return load_ecg_server_test_data(**kw)
        if self.key == "FROZEN_DEMO":
            return load_frozen_demo_server_test_data(**kw)
        if self.key == "LLM_LORA":
            if task_type == "CAUSAL_LM":
                return load_dolly_server_test_data(**kw)
            return load_sst2_server_test_data(**kw)
        raise NotImplementedError(f"load_server_test_data not implemented in recipes.py for {self.key}")

    def adapter_keys(self, model, aggregation):
        return llm_lora_adapter_keys(model, aggregation)

    def build_for_inference(self, model_name=None, task_type="SEQ_CLASSIFICATION"):
        """(net, classes, input_kind, transform) for the inference server (infer.py), so infer
        stays a one-line registry delegate (DA-14 Ph3.1). Data-driven default: build_model +
        (input_transform or None) + input_kind. Two per-recipe tweaks live here, not in infer:
        TRANSFORMER wires the model to its tokenizer's pad id; LLM_LORA reports 'generation' kind
        for the causal task. A recipe with no input_transform (CNN/MLP) yields transform=None and,
        because input_transform raises before importing anything, never drags transformers in."""
        net = self.build_model("cpu", model_name=model_name, task_type=task_type)
        try:
            transform = self.input_transform(model_name)
        except NotImplementedError:
            transform = None
        kind = self.input_kind
        if self.key == "TRANSFORMER" and transform is not None:
            net.config.pad_token_id = transform.pad_token_id
        if self.key == "LLM_LORA" and str(task_type).upper() == "CAUSAL_LM":
            kind = "generation"
        return net, self.classes, kind, transform


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
