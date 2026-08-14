# FL Runtime — the executable layer

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-required-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](../LICENSE)

The scripts the Spring Boot backend actually runs. When you press **Start** on a project in the
dashboard, the backend does not import the FL framework — it shells out to a wrapper in *this*
directory, which launches `fl_server.py`; clients then run `client.py`.

> **`framework/` is the library, `fl-runtime/` is the executable layer.**
> `framework/` is what you `pip install -e .` to get `import fedlearn`. `fl-runtime/` is a set of
> entry-point scripts that *consume* it — `client.py:19` is literally `import fedlearn as fl`.
> The backend never runs `framework/` directly. This split is the thing people miss most often
> about the repo, so it is worth repeating: **library vs. entry points.**

For depth — per-entry-point reference, the recipe registry internals, the arm mechanics — see
**[`../wikis/fl-runtime/`](../wikis/fl-runtime/)**. This page is the entry point, not the manual.

## Layout

```
fl-runtime/
├── client.py           # THE canonical FL client (DA-5) — desktop, Docker and local runs all use it
├── fl_server.py        # The gradient FL server: strategy selection, aggregation, eval, DP
├── fl_fot_server.py    # The Federation-over-Text server (separate, additive, torch-free path)
├── init_model.py       # Builds the initial global model, writes it as .npz
├── infer.py            # Single-input inference on a trained model; result goes to --out, not stdout
├── recipes.py          # The recipe catalog — single source of truth for model types + arms
├── benchmarks.py       # Metric core shared by per-round eval and the offline scorer
├── data.py             # Server-side test-data loading
├── config.py           # Dataset + DeComFL hyperparameter configs (cb / sst2 / ecg)
├── device.py           # --device resolution (auto → cuda > mps > cpu), unit-testable in isolation
├── models.py           # CnnNet, kept for direct importers
├── arm_tradeoff.json   # Per-recipe MEASURED arm trade-off, rendered by the project-creation picker
├── run_*.sh / run_*.bat  # Cross-platform wrappers the backend invokes
└── tests/              # This unit's own pytest suite (gated in CI)
```

## How the backend finds these scripts

The paths live in `backend/fl-platform-api/src/main/resources/application.properties:149-154`:

```properties
python.executable.path=${PYTHON_EXECUTABLE_PATH:../../fl-runtime/run_init_model.sh}
python.flbat.path=${PYTHON_FLBAT_PATH:../../fl-runtime/run_fl_server.sh}
python.script.init-model.path=${PYTHON_SCRIPT_INIT_MODEL_PATH:../../fl-runtime/init_model.py}
python.script.fl-server.path=${PYTHON_SCRIPT_FL_SERVER_PATH:../../fl-runtime/run_fl_server.sh}
python.script.infer.path=${PYTHON_SCRIPT_INFER_PATH:../../fl-runtime/run_infer.sh}
python.script.recipes.path=${PYTHON_SCRIPT_RECIPES_PATH:../../fl-runtime/run_recipes.sh}
```

Two consequences worth internalising:

- **Those defaults are relative, and they resolve from the backend's working directory** — which is
  why the backend is normally launched from `backend/fl-platform-api/` (`../../fl-runtime/…` lands
  here). Launch it from somewhere else without overriding the `PYTHON_*` env vars and every spawn
  fails to find its script. Deployments override them via the `PYTHON_*` env vars.
- **The entry points go through `.sh` / `.bat` wrappers so the backend keeps one code path.** The
  Java side builds the same argv regardless of host OS; only the launcher differs
  (`FlServerManager` prepends `bash` on POSIX and invokes the script directly on Windows). The
  wrappers also `cd` into this directory first, so the scripts' sibling imports (`import recipes`,
  `import config`) resolve no matter where the caller was.

The FoT server has a path too, but only as an `@Value` default in
`FlServerManager.java:59` (`python.script.fot-server.path`, default `../../fl-runtime/run_fot_server.sh`) —
it is not listed in `application.properties`.

Every `.sh` wrapper the backend invokes honours **`FEDLEARN_PYTHON`** (default `python3`), which is
how you point a run at a specific interpreter or virtualenv without editing anything.
(`run_clients.sh`, the local multi-client helper the backend never spawns, is the exception — it
calls `python` directly.)

> **Honest caveat on the Windows wrappers.** Only `run_fl_server.bat` and `run_init_model.bat`
> exist (there is no `.bat` for recipes, infer or FoT), and both are stale: they activate a
> hard-coded Anaconda root and still invoke the pre-DA-6 path
> `src\main\resources\scripts\fl_server.py`, which no longer exists. Treat Windows spawning as
> unverified and fix the `.bat` files before relying on it.

## Entry points

| Script | Wrapper | Required flags | Notes |
|---|---|---|---|
| `client.py` | *(none — run directly, or via the Docker/desktop bundle)* | `--project-id`, `--server-address`, `--partition-id` | Also `--model-type`, `--model-name`, `--strategy`, `--training-arm`, `--dataset`, `--aggregation`, `--task-type`, `--device`, `--use-llm` (deprecated). **There is no `--client-id`.** |
| `fl_server.py` | `run_fl_server.sh` / `.bat` | `--model-path`, `--project-id`, `--model-type`, `--model-name` | Plus `--init-model-path`, `--num-rounds`, `--min-clients`, `--port`, `--strategy`, `--training-arm`, `--seed`, `--aggregation`, `--task-type`, `--dataset` and the `--dp-*` family. |
| `fl_fot_server.py` | `run_fot_server.sh` | `--port` | `--project-id`, `--num-rounds`, `--round-seconds`, `--quorum`, `--backend`. |
| `init_model.py` | `run_init_model.sh` / `.bat` | `--model-type`, `--model-name`, `--optimizer`, `--out` | Plus `--pretrain-epochs`, `--aggregation`, `--task-type`. Writes the initial global weights as `.npz` (keys escape `.` as `__DOT__`). |
| `infer.py` | `run_infer.sh` | `--model-path`, `--model-type`, `--model-name`, `--in`, `--out` | Plus `--task-type`, `--max-new-tokens`, `--temperature`. |
| `recipes.py` | `run_recipes.sh` | *(none)* | `--describe` prints the catalog as JSON. |
| `benchmarks.py` | `run_benchmark.sh` | `--predictions` | `--out` writes the metric report as JSON (else stdout). |

`--partition-id` is bounded by `NUM_PARTITIONS = 10` (`client.py:58`), so `argparse` rejects
anything outside `0..9` before the client starts.

**`infer.py` writes its result to the `--out` file, never to stdout** — deliberately, because
torch/CUDA banners and wrapper log lines pollute stdout. The Java side reads the out-file and
treats stdout as diagnostics only. For `CAUSAL_LM` generation, stdout additionally carries
`{"token": "…"}` lines that the backend rebroadcasts as a live stream.

Supporting modules (`config.py`, `device.py`, `models.py`, `data.py`, `data_loaders/`, `models/`)
are imported by the entry points and have no CLI of their own.

### One canonical client (DA-5)

`fl-runtime/client.py` is the **only** Python FL client in the repo — `mobile_client/` is a
separate on-device client with its own native C++ stack, not a fork of this one. The Docker image
(`client-docker/Dockerfile` `COPY`s `fl-runtime/` and sets `WORKDIR /app/fl-runtime`), the desktop
PyInstaller bundle (`client-docker/packaging/fedlearn-client.spec` points `CLIENT_ENTRY` here) and
a plain local run all execute this same file. There is no forked client under `client-docker/`.
Don't reintroduce one.

The container is configured by **env vars, not CLI flags** — `entrypoint.sh` requires
`PROJECT_ID` / `SERVER_ADDRESS` / `PARTITION_ID` and builds the flags itself. See
[`../client-docker/README.md`](../client-docker/README.md).

## The recipe catalog

A **recipe** bundles `{architecture + dataset loader + input transform + class labels + input
kind + UI metadata}` under a stable `key`. `recipes.py` is the single source of truth, and
`RECIPE_METADATA` currently advertises **seven** keys:

| Key | Display name | Input | `supported_arms` |
|---|---|---|---|
| `PNEUMONIA_CNN` | Pneumonia Chest X-ray | image | `FULL`, `FROZEN_HEAD` |
| `CNN` | Image classifier (CIFAR-10) | image | `FULL`, `FROZEN_HEAD` |
| `CIFAR_RESNET18` | Image classifier (CIFAR-10, pretrained ResNet-18) | image | `FULL`, `FROZEN_HEAD`, `OVA_LP` |
| `MLP` | ECG heartbeat (Normal/Abnormal) | vector | `FULL` |
| `TRANSFORMER` | Text classifier (OPT-125M) | text | `FULL` |
| `LLM_LORA` | Text LLM (LoRA fine-tune) | text | `FULL` |
| `TINYNET_GOLDEN` | On-device DeComFL demo (TinyNet) | vector | `FULL` |

```bash
cd fl-runtime
bash run_recipes.sh --describe      # the exact JSON the backend serves at GET /api/model-recipes
```

**`--describe` is deliberately torch-free** so that endpoint stays cheap: all catalog metadata
lives in plain dicts and every heavy import inside `recipes.py` is lazy. Measured on this
checkout, the subprocess completes in ~0.03 s and never imports torch. A test
(`test_describe_subprocess_is_torch_free_and_stable`) pins both the behaviour and the key order.

### Two recipes exist but are not in the catalog

`get_recipe()` / `is_recipe()` resolve them; `describe()` serves only `RECIPE_METADATA`, so they
stay dispatchable-but-not-selectable and never reach the project-creation picker.

- **`BLOOD_CNN`** — its `medmnist` dependency is in no requirements file. Advertising it would let
  a project pass the backend's catalog gate and then crash the spawn on `ModuleNotFoundError`
  (SE-10). It is registered rather than absent because `init_model.py` dispatches it, and a
  missing registry entry was a latent `ValueError`. Re-promote once `medmnist` (plus
  scikit-image/fire, with aarch64 wheels verified) ships everywhere the runtime runs.
- **`FROZEN_DEMO`** — a frozen-backbone/trainable-head demo, superseded by the real
  derivation-record recipe. Kept for the tests and the demo path, not offered as a product choice.

### Registry dispatch — where it stands

`catalog_keys()` feeds the `--model-type` `choices` on **both** `fl_server.py` and `client.py`, so
a new catalog entry becomes an accepted model type with no argparse edit, and the catalog can never
advertise a key the scripts reject. Two tests
(`test_recipe_catalog_matches_runnable_choices.py`) pin that equality in both directions.

Model construction is registry-driven at the two entry points that build from scratch:
`init_model.get_model()` and `infer.build_model()` are each a single
`recipes.get_recipe(key).build_*(…)` call with no per-type `if`/`elif` chain (DA-14 Ph3.1), and an
unknown key fails loudly through `get_recipe`'s `Unknown recipe key` error. A test substitutes a
fake recipe object and asserts it builds through `init_model.get_model` with no branch added.

**Still unfinished:** `client.py` has not been collapsed. It sets module-level flags from
`--model-type` (`USE_MLP`, `USE_LLM`, `USE_PNEUMONIA`, `USE_LLM_LORA`, `USE_DERIVED`) and keeps
`if`/`elif` chains for model construction (`client.py:648-689`) and data loading
(`load_data`, `client.py:197-338`) — every branch delegates to `recipes.get_recipe(...)`, but the
branch selection itself is still hand-written, and the `else` fallback only picks the recipe by
key when `recipes.is_recipe(MODEL_TYPE)` holds. So adding a recipe that needs its own loader or a
non-default build still means touching `client.py`.

And the dispatch that left `init_model.py` / `infer.py` moved *inside* `recipes.py` rather than
disappearing: `Recipe.build_model`, `input_transform`, `load_client_data` and
`load_server_test_data` are themselves per-key `if` chains ending in `raise NotImplementedError`.
There is no default build path, so a bare `RECIPE_METADATA` entry is advertised-but-not-buildable —
a new recipe needs its matching branches in `recipes.py` as well. **No Java or TypeScript edits are
needed either way** — the picker is generated from `--describe`.

## Training arms

An **arm** is a declared property of a recipe, not something inferred from its key:

```python
TRAINING_ARMS  = ("FULL", "FROZEN_HEAD", "OVA_LP")
ARM_OBJECTIVES = {"FULL": "cross_entropy", "FROZEN_HEAD": "cross_entropy", "OVA_LP": "one_vs_all"}
DEFAULT_ARM    = "FULL"        # an omitted arm means FULL, so existing projects are unchanged
```

Each recipe declares `supported_arms` plus, per arm, `trainable_spec[arm]` — the parameter-name
prefixes that stay trainable (`None` = train everything). `validate_arm()` rejects an arm the
recipe does not declare at process startup (`fl_server.py:667`, `client.py:1084`) — before round 1
rather than mid-run, and it is the only check that consults `supported_arms` at all: the Java side
validates the `TrainingArm` enum, never the recipe; `apply_arm()` writes *every* parameter's
`requires_grad` flag so applying an arm is idempotent, and raises if the prefixes match nothing
(a typo would otherwise freeze the whole model and look like a bad-but-converged run).

An arm carries an **objective**, not just a parameter subset. `OVA_LP` (one-vs-all linear probing
on a frozen encoder, [arXiv:2511.05028](https://arxiv.org/abs/2511.05028)) trains exactly what
`FROZEN_HEAD` trains but under C independent binary classifiers instead of one softmax — the first
arm that differs from another in objective rather than subset. Note the recipe's own `arm_notes`
disclaimer: the paper's two-stage schedule is **not** implemented, so results should be read as
OvA heads on a frozen encoder, not as a reproduction of the paper.

`--training-arm` exists on **both** `fl_server.py` and `client.py`, and the two must agree — the
server federates exactly the subset the clients send. The end-to-end flow is: picker → Java
`TrainingArm` enum → `projects.training_arm` → `--training-arm` on both processes.

`arm_tradeoff.json` holds the **measured** per-recipe trade-off the picker displays (currently
`PNEUMONIA_CNN`, `CNN`, `CIFAR_RESNET18`). It is generated by `scripts/build_arm_tradeoff.py` from
recorded results and carries `source`, `source_sha256` and explicit `caveats` — the picker renders
the record rather than a hand-written claim, and `describe()` attaches it only to recipes that
actually offer a *choice* of arms.

> Note: the `--training-arm` help text on both scripts still reads "FULL (default) or FROZEN_HEAD"
> and predates `OVA_LP`. The validation is data-driven from `supported_arms`, so the help string is
> the only thing out of date.

## What crosses the wire

The FL wire is the framework's deterministic **safetensors** codec — float32 only, so the
libtorch-free mobile C++ client can decode it. Two consequences show up here:

- **Non-float32 tensors are withheld from the federated set, not rejected.** BatchNorm's int64
  `num_batches_tracked` is a batch *counter*; averaging it is meaningless, and including it failed
  the run outright. `client.py:714-736` and `fl_server.py:653-673` apply the **same**
  `federable_state` / `non_federable_names` helper — one filter, both sides, agreeing by
  construction. `running_mean` / `running_var` are float32 and *are* still federated, so this is a
  wire fix, not FedBN. The server keeps the withheld tensors and merges them back at save time, so
  the saved model stays complete. This is what unblocked BatchNorm models on the `FULL` arm.
- **Under a subset arm only the trainable subset rides the wire.** The client uploads
  `trainable_state(net)`; the server applies the arm's prefix filter to its own initial parameters
  so `d_server == d_client` (a mismatch would silently misalign DeComFL's shared-seed perturbation).
  The frozen backbone stays local and is rebuilt deterministically from the recipe on every peer.

`fl_server.py --strategy` dispatches to the framework's strategies: `decomfl`, `fedlora`,
`fedprox`, `fedopt`, `robust`, plus the explicit `fedavg`. An **unrecognised value does not fall
back to FedAvg — it raises** (FR-28, `fl_server.py:449-457`), so a typo, or a factory-style name
like `fed_lora`, fails the spawn instead of silently training a different algorithm with every
strategy-specific flag ignored. (`select_strategy`'s own docstring still describes the old
FedAvg-fallback behaviour; the code is the truth.) `FedProx` / `FedOpt` / `RobustAggregator` are
constructed with sensible defaults for their extra hyperparameters; plumbing those through the run
config is a known follow-up.

## Testing

This unit has its **own** suite, separate from `framework/`:

```bash
cd fl-runtime
python -m pytest -q
```

`pytest.ini` sets `addopts = -m "not slow"`, so tests needing network access or cached model
artifacts are deselected by default. On this checkout the suite reports **259 passed, 28
deselected** in ~12 s.

CI runs it with **`FEDLEARN_FAIL_ON_UNEXPECTED_SKIP=1`**, which is stricter than it looks: a
**skipped** test fails the job (TE-10 — a silently skipped test is a false green, and this suite
allowlists no skip reasons at all). Deselection via `-m "not slow"` is **not** a skip, so the
slow-marker workflow is untouched. `tests/conftest.py` also walks up to put `framework/src` on
`sys.path` (and on `PYTHONPATH`, for tests that spawn subprocesses), so `import fedlearn` resolves
against the checkout without a separate editable install.

What the suite pins, in broad strokes:

- **The arm is real, not a label.** It is applied whichever branch built the model; it does not
  choose the dataset; client and server agree on which parameters are federated; a frozen backbone
  stays frozen (including its normalisation statistics) and survives the save; FedProx regularises
  the federated subset only; every declared arm is wire-compatible; the arm and its objective ride
  on the eval card and cannot collide with another arm's result.
- **The catalog is honest.** It equals the model types the scripts can actually spawn, advertises
  only base models `init_model` can build, keeps `--describe` torch-free and byte-stable, and
  dispatch stays registry-driven.
- **Entry-point behaviour.** Inference (text, chat, generation, registry dispatch), server
  strategy selection, perplexity, eval-card/DP tracing, server-side non-strict loading of a
  subset-federated model, LLM-LoRA and TINYNET_GOLDEN/DeComFL client paths, device resolution,
  benchmark metrics, the CIFAR-10 and ECG shard parity checks, and the security floors in the
  backend requirements file.

## Adjacent docs

- **[`../wikis/fl-runtime/`](../wikis/fl-runtime/)** — the deep reference for this unit
- **[`../framework/README.md`](../framework/README.md)** — the library these scripts import
- **[`../client-docker/README.md`](../client-docker/README.md)** — how `client.py` ships as a container
- **[`../backend/fl-platform-api/`](../backend/fl-platform-api/)** — the control plane that spawns these scripts
