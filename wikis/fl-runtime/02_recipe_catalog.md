# 02 — The Model-Recipe Catalog

> **Part of:** [FedLearn Platform Docs](../README.md) → [FL Runtime Wiki](./README.md)

## Table of Contents
- [What a Recipe Is](#what-a-recipe-is)
- [`RECIPE_METADATA` — the seven catalog keys](#recipe_metadata--the-seven-catalog-keys)
- [`--describe` Is Deliberately Torch-Free](#--describe-is-deliberately-torch-free)
- [From `recipes.py` to the Project-Creation Picker](#from-recipespy-to-the-project-creation-picker)
- [The Two Out-of-Catalog Recipes](#the-two-out-of-catalog-recipes)
- [The `Recipe` Dispatch Object](#the-recipe-dispatch-object)
- [How Far Registry Dispatch Actually Got](#how-far-registry-dispatch-actually-got)
- [Adding a Model Type: What You Actually Have to Wire](#adding-a-model-type-what-you-actually-have-to-wire)
- [The CIFAR-10 Shard, After `flwr`](#the-cifar-10-shard-after-flwr)
- [Dataset Resolution per Recipe](#dataset-resolution-per-recipe)

---

## What a Recipe Is

`fl-runtime/recipes.py` is the single source of truth for the trainable model types the platform offers. A **recipe** bundles, under one stable `key`:

- an **architecture** (`build_model`)
- a **dataset loader**, client-side and server-side (`load_client_data`, `load_server_test_data`)
- an **input transform** (`input_transform` — an image transform, or a tokenizer for text recipes)
- **class labels** (`classes`)
- an **input kind** (`image` / `vector` / `text`, plus `generation` for the causal-LM task)
- **UI metadata** — display name, selectable base models, offered optimizers, hardware requirements
- the **training arms** it supports, and what freezes under each (see [03](03_training_arms.md))

Keys are resolved case-insensitively: `get_recipe(key)` upper-cases before lookup (`recipes.py:1463`), and `is_recipe(key)` is the non-raising membership test.

---

## `RECIPE_METADATA` — the seven catalog keys

`RECIPE_METADATA` (`recipes.py:241`) is a list of plain dicts. **It currently holds seven entries** — any doc saying "six" predates `CIFAR_RESNET18`.

| Key | Display name | Input | Classes | `supported_arms` |
|---|---|---|---|---|
| `PNEUMONIA_CNN` | Pneumonia Chest X-ray | image | 2 (NORMAL / PNEUMONIA) | `FULL`, `FROZEN_HEAD` |
| `CNN` | Image classifier (CIFAR-10) | image | 10 | `FULL`, `FROZEN_HEAD` |
| `CIFAR_RESNET18` | Image classifier (CIFAR-10, pretrained ResNet-18) | image | 10 | `FULL`, `FROZEN_HEAD`, `OVA_LP` |
| `MLP` | ECG heartbeat (Normal/Abnormal) | vector | 2 | `FULL` |
| `TRANSFORMER` | Text classifier (OPT-125M) | text | 3 | `FULL` |
| `LLM_LORA` | Text LLM (LoRA fine-tune) | text | 2 | `FULL` |
| `TINYNET_GOLDEN` | On-device DeComFL demo (TinyNet) | vector | 3 | `FULL` |

The order of that list is the order of the catalog, and `catalog_keys()` (`recipes.py:418`) returns exactly those keys — which is also the `choices=` for `--model-type` on both `fl_server.py` and `client.py`. That is deliberate: a new catalog recipe automatically becomes an accepted model type with no argparse edit.

Per-recipe details worth knowing:

- **`CIFAR_RESNET18` is the pretrained-backbone recipe.** It declares its provenance explicitly under a `pretrained` block (`source: torchvision`, `weights: ResNet18_Weights.IMAGENET1K_V1`), with the ImageNet 1000-class head discarded and replaced by a fresh 10-class CIFAR head. Declaring it rather than leaving it implicit is what lets a result answer *"which backbone produced this?"* — the first question anyone asks of a frozen-arm number. It is also the only recipe carrying `arm_notes`, which state plainly that the paper's two-stage schedule is **not** implemented.
- **`CNN`'s `FROZEN_HEAD` prefix is `fc3.`, not `classifier.`.** The entry originally declared `classifier.`, which `CnnNet` (`conv1`/`conv2`/`fc1`/`fc2`/`fc3`) has no module for — so `FROZEN_HEAD` raised at model build. `fc3` is the real head.
- **`TINYNET_GOLDEN` freezes `fc2` by construction**, not by arm. It is `Linear(4,5) → ReLU → Linear(5,3)` with 25 trainable parameters (`fc1`), initialised deterministically at seed 0 so it byte-matches the golden `.pte` the mobile ExecuTorch client encodes. It is `FULL`-only precisely because an arm switch would change the trainable layout the on-device golden depends on.
- **`LLM_LORA` carries a `lora` block** (`r: 8, alpha: 16, dropout: 0.05, target_modules: ["q_proj","v_proj"]`) and `aggregation: FFA_LORA`, and offers two base models (`qwen2.5-0.5b`, `tinyllama-1.1b`).
- **`requirements` is a real gate, not decoration.** Every entry declares `min_ram_gb`, `min_storage_gb`, `mobile_safe`, `max_trainable_params`, and mobile recipes add `min_os_android` / `min_os_ios`.

---

## `--describe` Is Deliberately Torch-Free

`recipes.py`'s module-level imports are `argparse`, `json`, `os`, `sys` — **nothing else**. Every torch, torchvision, transformers, peft and datasets import lives *inside* a function body.

That is not tidiness; it is what makes serving the catalog cheap. `ModelRecipeService` spawns `run_recipes.sh --describe` in a fresh interpreter with a **30-second timeout**. If importing the module pulled in torch, that spawn would cost seconds of import time and megabytes of RSS for what is a metadata dump. `arm_tradeoff()` reads a JSON file for the same reason.

`describe()` (`recipes.py:400`) returns `RECIPE_METADATA` with one enrichment: recipes offering a **choice** of arms get their measured `arm_tradeoff` attached. Single-arm recipes deliberately do not — there is nothing to trade off, and attaching a figure would imply an un-offered arm had been evaluated for that recipe.

---

## From `recipes.py` to the Project-Creation Picker

```
recipes.py  RECIPE_METADATA
     │  describe()  →  JSON on stdout
     ▼
run_recipes.sh --describe          (bash on POSIX; the raw path on Windows)
     ▼
ModelRecipeService.runDescribe()   30s timeout, stderr DISCARDed so torch noise
     │                             cannot corrupt the JSON, result cached for the
     │                             JVM's lifetime (recipes are static config)
     ▼
GET /api/model-recipes  →  ModelRecipeDto[]
     ▼
Frontend CreateProjectModal  →  recipe picker (+ the arm selector when
                                 supportedArms.length > 1)
```

Two design decisions on the Java side are load-bearing:

1. **No hardcoded Java fallback (DA-10).** A duplicate catalog once existed in Java and had already drifted — it was missing `BLOOD_CNN` and `LLM_LORA`. Since the app spawns Python for all training and inference anyway, a broken `recipes.py` should surface loudly rather than be masked by stale data. A load failure throws `IllegalStateException` and is **not** cached, so a transient problem recovers on the next request.
2. **`requireModelTypeInCatalog` gates every spawn (SE-10).** `FlServerManager` refuses to start a gradient FL server whose `modelType` is not an **exact-case** catalog key. The exact-case requirement is a canonical-key consistency policy rather than crash prevention — `fl_server.py`'s `--model-type` uses `type=str.upper`, so a lowercase variant would in fact normalise correctly. Requiring the canonical key keeps the persisted `modelType` identical to the catalog and to the artifact's `recipeKey`. FoT runs carry no model type on their argv and are exempt.

`fl-runtime/tests/test_recipe_catalog_matches_runnable_choices.py` pins that the catalog and the two scripts' `--model-type` choices are the *same set* — the guarantee the SE-10 gate rests on.

---

## The Two Out-of-Catalog Recipes

`_NONCATALOG_METADATA` (`recipes.py:814`) registers two further recipes into `_METADATA_BY_KEY` **after** `describe()`'s source list. The consequence is precise: `get_recipe` and `is_recipe` resolve them, `--describe` and the project picker never see them, and `catalog_keys()` excludes them so they are not accepted `--model-type` values either. Dispatchable but not selectable.

### `BLOOD_CNN` — excluded because its dependency ships nowhere

BloodMNIST 8-class peripheral-blood-cell microscopy. The recipe is **fully functional** — `build_blood_cnn()` builds and `load_blood_*_data()` pulls a real batch — and it was verified to work.

It stays out of the catalog because `medmnist`, and its transitive `scikit-image` / `fire`, **are not declared in any requirements file in this repository** (`framework/requirements.txt` drives the actual `fl_server.py` spawn environment; `backend/fl-platform-api/requirements.txt`, `client-docker/requirements.txt` and `client-docker/packaging/requirements-client.txt` cover the rest). Advertising the key would let the SE-10 catalog gate *pass* and then crash the spawn on `ModuleNotFoundError` the moment `load_blood_server_test_data()` ran — the exact failure class SE-10 exists to prevent, just one import deeper.

Registering it as non-catalog also fixed a latent crash: `init_model.py` dispatched `BLOOD_CNN` through `recipes.get_recipe('BLOOD_CNN')` while the registry had no such key, so the call raised `ValueError`.

To promote it: add `medmnist` (+ `scikit-image`, `fire`) everywhere `fl_server.py`/`client.py` run — **verifying aarch64/Jetson wheel availability for `scikit-image` first** — then move the entry into `RECIPE_METADATA`.

### `FROZEN_DEMO` — excluded because it is a demo superseded by the real thing

A frozen `Linear(256, 128)` backbone plus a trainable `Linear(128, 3)` head, over a self-contained synthetic vector dataset (no external data, deterministic per partition). It exists to exercise the DA-11 partial-load path — `build_frozen_backbone_model` can reconstruct a content-addressed frozen backbone via `reconstruct_frozen_backbone` and re-freeze it — with only the head federated.

Its backbone is built under `torch.manual_seed(0)` inside a `fork_rng`, because **every federation peer must materialise an identical frozen backbone**: the head is aggregated over the wire and the backbone is not, so a per-peer random backbone would make the aggregated head meaningless. (`build_tinynet_golden` seeds itself for exactly the same reason.)

It is registered non-catalog as a demo superseded by the real derivation-record recipe. One trace of its history survives in `client.py`: `main()` still contains `if requested is None and mt == "FROZEN_DEMO": requested = "FROZEN_HEAD"` to preserve its pre-arm default — a branch that the `choices=recipes.catalog_keys()` on `--model-type` makes unreachable from the CLI today.

---

## The `Recipe` Dispatch Object

`get_recipe(key)` returns a `Recipe` (`recipes.py:1309`) — a thin object over the metadata dict with six methods:

| Method | Returns |
|---|---|
| `build_model(device, model_name, aggregation, task_type)` | the `nn.Module` |
| `input_transform(model_name)` | image transform, or tokenizer for text recipes; raises `NotImplementedError` when a recipe has none |
| `load_client_data(partition_id, num_clients, task_type, **kw)` | `(train_loader, val_loader)` |
| `load_server_test_data(task_type, **kw)` | the server's eval `DataLoader` |
| `adapter_keys(model, aggregation)` | LoRA adapter key set (LLM_LORA) |
| `build_for_inference(model_name, task_type)` | `(net, classes, input_kind, transform)` — so `infer.py` stays a one-line delegate |

`build_for_inference` is data-driven with exactly two per-recipe tweaks that live here rather than in `infer.py`: `TRANSFORMER` wires the model to its tokenizer's pad id, and `LLM_LORA` reports the `generation` input kind for the causal task. Because `input_transform` raises *before* importing anything, a recipe with no transform (`CNN`, `MLP`) never drags `transformers` into an inference process.

> **`Recipe.is_functional` is stale and unused outside tests.** It reports whether a recipe's model *and* data both live in `recipes.py`, and its hardcoded tuple omits `CIFAR_RESNET18` — even though `build_cifar_resnet18`, `load_cifar_resnet18_client_data` and `load_cifar_resnet18_server_test_data` are all in this file. Nothing in production reads the property (only `tests/test_recipes_registry.py`, `test_frozen_backbone_recipe.py` and `test_llm_lora_recipe.py` do), so the omission is a documentation-level inaccuracy rather than a behavioural bug — but do not trust it as a capability signal.

---

## How Far Registry Dispatch Actually Got

The registry is the catalog's source of truth. It is **not yet a full dispatch registry**, but the boundary has moved and older descriptions of it are wrong in a specific way. Believe this table, which reflects the code as of the `DA-14 Ph3.1` work:

| Consumer | Model construction | Data loading |
|---|---|---|
| `init_model.py` | **Fully registry-dispatched.** `get_model` (`init_model.py:38-54`) is a single `recipes.get_recipe(model_type).build_model(...)` call — no per-type `if`/`elif` at all | n/a |
| `infer.py` | **Fully registry-dispatched.** `build_model` (`infer.py:63-74`) delegates to `build_for_inference` | n/a |
| `data.py` | n/a | CNN delegates to the registry; the LLM branch still tokenises inline from `config.py` |
| `client.py` | **Still an `if`/`elif` chain** (`client.py:648-689`), but every branch now ends in `recipes.get_recipe(…).build_model(DEVICE)` | **Still an `if`/`elif` chain** (`load_data`, `client.py:197-338`); most branches delegate to the registry, but the `USE_LLM` branch tokenises and Dirichlet-splits inline |
| `fl_server.py` | Delegates to `init_model.get_model` (hence registry) — except `LLM_LORA`, which rebuilds per eval round | **Still an `if`/`elif` chain** (`fl_server.py:728-757`) selecting the server test loader |

So the "adding a recipe entry alone yields a broken model type" hazard is **real but no longer located in `init_model.py`** — there is no `raise ValueError(f"Unsupported model architecture: …")` anywhere in this directory any more. The gap moved *inside* `recipes.py`: `Recipe.build_model`, `input_transform`, `load_client_data` and `load_server_test_data` are themselves per-key `if`/`elif` chains that end in

```python
raise NotImplementedError(f"build_model not implemented in recipes.py for {self.key}")
```

Two chains outside `recipes.py` also still need a branch for a genuinely new model type: `client.py`'s `load_data` (unless the new recipe's data can ride an existing branch) and `fl_server.py`'s server-test-loader selection.

`tests/test_recipes_registry.py` pins the Ph3.1 property directly — it monkeypatches a fake `FAKE_NET` recipe into `recipes.get_recipe` and asserts `init_model.get_model` routes it through without any `init_model` edit. `tests/test_infer_registry_dispatch.py` does the same for `infer.py` with a `FAKE_IMG` recipe.

---

## Adding a Model Type: What You Actually Have to Wire

1. **Add the `RECIPE_METADATA` entry** — `key`, `display_name`, `input_kind`, `classes`, `base_models`, `optimizers`, `requirements`, `supported_arms`, `trainable_spec`. That alone gets the key into `--describe`, into the frontend picker, and into both scripts' `--model-type` choices.
2. **Add the `Recipe` branches it needs**, inside `recipes.py`: `build_model` always; `input_transform` if it has one; `load_client_data` / `load_server_test_data` if it is meant to be trainable rather than inference-only. Without `build_model`, the key is advertised and then raises `NotImplementedError` at the first spawn.
3. **Add a server-test-loader branch in `fl_server.py`** if the recipe's eval data does not fall through to `data.load_server_test_data`.
4. **Add a `load_data` branch in `client.py`** if it needs a loader no existing branch provides.
5. **Declare its arms honestly.** A declared arm whose payload cannot cross the wire is caught by `tests/test_declared_arms_are_wire_compatible.py`; a declared prefix that matches no parameter is caught by the same file. See [03](03_training_arms.md) and [04](04_the_federated_set.md).

**No Java or TypeScript edits are needed** for any of this — the catalog is data-driven end to end.

---

## The CIFAR-10 Shard, After `flwr`

`flwr` and `flwr-datasets` are **gone from this repository** (`65048b6`). They were never used for FL semantics; the only thing `flwr_datasets.FederatedDataset` did here was cut one CIFAR-10 IID shard. That single dependency capped `cryptography<45.0.0` (the SE-22 residual, below the framework's own `>=46.0.6` floor) and `protobuf<5.0.0` (which made the FoT path uninstallable, since `fot_pb2.py` is 5.29.0 gencode). Dropping it cleared both: `backend/fl-platform-api/requirements.txt` now pins `cryptography==46.0.7` and `protobuf==5.29.5`.

The shard is now implemented natively in this file:

```python
CNN_NUM_PARTITIONS = 10       # == client.py NUM_PARTITIONS; fixed, NOT num_clients
CNN_BATCH_SIZE     = 32       # == client.py BATCH_SIZE
CNN_SERVER_TEST_BATCH = 128   # == data.py CNN server test batch
CNN_SHUFFLE_SEED   = 42       # == the flwr FederatedDataset default it replaced
```

Three determinism traps are pinned by `tests/test_recipe_cnn_data_parity.py`, and every one of them is a way a "harmless" edit silently reshuffles every partition:

1. **The client partitioner is a fixed 10 shards, not `num_clients`.** `num_clients` is accepted and explicitly *ignored* — the shard count inside `load_cnn_client_data` stays `CNN_NUM_PARTITIONS`.
2. **The pipeline order is load-bearing**: `shuffle(seed=42) → shard(num_shards=10, index=partition_id, contiguous=True) → train_test_split(test_size=0.2, seed=42)`. That sequence is precisely what `FederatedDataset` (shuffle-before-partition, `seed=42` default) plus `IidPartitioner.load_partition(i)` did; both facts were read off the installed `flwr-datasets` 0.5.0 source and then verified empirically per partition.
3. **Source asymmetry, deliberately preserved**: the *client* shard comes from the HuggingFace `cifar10` **train** split, while the *server* test loader uses **torchvision** `CIFAR10(train=False)` at batch 128, unshuffled. Two different dataset libraries, on purpose — that is what the legacy `client.py` / `data.py` pair did, and unifying them would change the numbers.

The replacement was verified byte-identical per partition against the `flwr` implementation before the swap, so CIFAR-10 results recorded before and after remain comparable. `tests/test_requirements_security_floors.py` keeps `flwr` out of the lockfiles as a regression guard.

> Residual `flwr` *mentions* survive as comments (`client-docker/requirements.txt:23`) and as a `flwr_datasets` hidden-import in `client-docker/packaging/fedlearn-client.spec:37`. Those are stale strings, not dependencies — but see [05](05_testing.md#known-staleness-around-this-layer), because the spec one is not inert.

---

## Dataset Resolution per Recipe

| Recipe | Where the data comes from |
|---|---|
| `CNN` | HuggingFace `cifar10` **train** split for the client shard; torchvision `CIFAR10(train=False)` for the server's eval loader, cached under `fl-runtime/data/` — the source asymmetry described above |
| `CIFAR_RESNET18` | The **same** CIFAR-10 shard as `CNN` — only the transform differs: ImageNet normalisation at `RESNET18_IMG_SIZE = 112` px, a resolution the pretrained backbone can actually use. Keeping the shard identical is what makes a `CNN` run and a `CIFAR_RESNET18` run comparable on data. Evaluating pretrained features through CIFAR's `[-1,1]` normalisation would silently degrade every feature and understate the arm, which is why the server has its own branch for this recipe's test loader |
| `PNEUMONIA_CNN` | First match wins: (1) `FEDLEARN_PNEUMONIA_DIR` — a local ImageFolder layout `<dir>/train/{NORMAL,PNEUMONIA}/` + `<dir>/test/…`, the zero-network path; (2) HuggingFace `datasets`, repo from `FEDLEARN_PNEUMONIA_DATASET` (default `keremberke/chest-xray-classification`), config from `FEDLEARN_PNEUMONIA_CONFIG` (default `full`). `FEDLEARN_PNEUMONIA_SUBSET=<N>` caps samples per split for fast demo rounds; `FEDLEARN_PNEUMONIA_ALPHA` (default 0.5) sets the Dirichlet concentration |
| `MLP` | `fl-runtime/ecg_data/ecg.csv`, partitioned by `data_loaders/ecg_loader.py` with every hyperparameter sourced from `config.get_dataset_config("ecg")` |
| `TRANSFORMER` | HuggingFace `super_glue/cb` or `glue/sst2`, tokenised with the opt-125m tokenizer |
| `LLM_LORA` | SST-2 for `SEQ_CLASSIFICATION`, Dolly for `CAUSAL_LM` |
| `TINYNET_GOLDEN` | A committed golden fixture; the server skips eval entirely for this recipe (there is no golden eval dataset, and the default loader's image batches do not fit a 4-dim input) |
| `FROZEN_DEMO` | Self-contained synthetic vectors, deterministic per partition; no external dataset needed |
| `BLOOD_CNN` | MedMNIST auto-downloads BloodMNIST (~30 MB `.npz`) to `~/.medmnist` on first use |

> **SE-19 — remote code execution is off by default.** The `PNEUMONIA_CNN` HuggingFace fallback used to call `load_dataset(..., trust_remote_code=True)`, which downloads and *runs* the dataset repo's loader script on the host. The repo is unpinned, so a supply-chain compromise of it would be arbitrary code execution on the backend the moment a run started. `_hf_load_kwargs` now makes remote-code execution opt-in via an explicit operator flag (ideally alongside a pinned `revision`), and `tests/test_pneumonia_hf_load_kwargs.py` pins that.
