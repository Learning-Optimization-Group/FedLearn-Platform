# 01 — Entry Points & the Backend Contract

> **Part of:** [FedLearn Platform Docs](../README.md) → [FL Runtime Wiki](./README.md)

## Table of Contents
- [How the Backend Resolves This Directory](#how-the-backend-resolves-this-directory)
- [Why a Shell Wrapper and Not a Python Path](#why-a-shell-wrapper-and-not-a-python-path)
- [`fl_server.py` — the FL aggregation server](#fl_serverpy--the-fl-aggregation-server)
- [`client.py` — the one canonical FL client](#clientpy--the-one-canonical-fl-client)
- [`init_model.py` — initial weights](#init_modelpy--initial-weights)
- [`infer.py` — single-input inference](#inferpy--single-input-inference)
- [`recipes.py` — the catalog CLI](#recipespy--the-catalog-cli)
- [`fl_fot_server.py` — Federation over Text](#fl_fot_serverpy--federation-over-text)
- [`benchmarks.py` — the metric core](#benchmarkspy--the-metric-core)
- [Support modules](#support-modules)
- [The wrappers, one by one](#the-wrappers-one-by-one)

---

## How the Backend Resolves This Directory

Six properties in `backend/fl-platform-api/src/main/resources/application.properties` point the Java side at this directory. The block sits at lines **149–154**:

```properties
python.executable.path=${PYTHON_EXECUTABLE_PATH:../../fl-runtime/run_init_model.sh}
python.flbat.path=${PYTHON_FLBAT_PATH:../../fl-runtime/run_fl_server.sh}
python.script.init-model.path=${PYTHON_SCRIPT_INIT_MODEL_PATH:../../fl-runtime/init_model.py}
python.script.fl-server.path=${PYTHON_SCRIPT_FL_SERVER_PATH:../../fl-runtime/run_fl_server.sh}
python.script.infer.path=${PYTHON_SCRIPT_INFER_PATH:../../fl-runtime/run_infer.sh}
python.script.recipes.path=${PYTHON_SCRIPT_RECIPES_PATH:../../fl-runtime/run_recipes.sh}
```

A seventh, `python.script.fot-server.path`, has no entry in the properties file at all — `FlServerManager` carries its default inline (`../../fl-runtime/run_fot_server.sh`, `FlServerManager.java:59`).

Which Java class reads which property:

| Property | Consumer | What it launches |
|---|---|---|
| `python.script.fl-server.path` | `FlServerManager.java:56` | `run_fl_server.sh` → `fl_server.py` |
| `python.script.fot-server.path` | `FlServerManager.java:59` | `run_fot_server.sh` → `fl_fot_server.py` |
| `python.executable.path` | `ModelInitializer.java:22` | `run_init_model.sh` → `init_model.py` |
| `python.script.infer.path` | `InferenceService.java:71` | `run_infer.sh` → `infer.py` |
| `python.script.recipes.path` | `ModelRecipeService.java:46` | `run_recipes.sh --describe` → `recipes.py` |

Two of these names are misleading and worth calling out, because they invite the wrong guess:

- **`python.executable.path` is not a Python interpreter.** Despite the name, `ModelInitializer` treats it as *the init-model wrapper script path* (the field is even called `initModelWrapperPath`). Nothing in the codebase reads it as an interpreter.
- **`python.script.init-model.path` is not read by anything.** It points at the raw `.py`, but `ModelInitializer` launches the wrapper from `python.executable.path` instead. The same is true of `python.flbat.path`. Both are vestigial; changing them changes nothing.

> **The relative defaults resolve from the backend's working directory.** `../../fl-runtime/…` is relative to wherever the JVM was started, so the backend is normally launched from `backend/fl-platform-api/` (`cd backend/fl-platform-api && ./gradlew bootRun`). Start it from the repo root and every one of these paths misses. Deployments override them with the `PYTHON_*` environment variables rather than relying on the working directory.

### The interpreter

Every `run_*.sh` wrapper that runs an entry point picks its interpreter the same way (the exception is the dev-only `run_clients.sh`, which sources a venv and calls a bare `python`):

```bash
PYTHON="${FEDLEARN_PYTHON:-python3}"
```

So `FEDLEARN_PYTHON=/opt/venv/bin/python` is the supported way to point the runtime at a virtualenv without editing scripts. The commented-out `source /home/ec2-user/app/venv/bin/activate` lines in the wrappers are disabled leftovers from an earlier EC2 layout — they do nothing today.

---

## Why a Shell Wrapper and Not a Python Path

The backend builds its command like this (`FlServerManager.buildServerCommand`, and identically in `ModelInitializer.buildInitCommand` / `ModelRecipeService.runDescribe`):

```java
List<String> command = new ArrayList<>();
if (!isWindows) {
    command.add("bash");
}
command.add(absoluteScriptPath);
// … then the flags
```

On macOS and Linux it prepends `bash` and runs the `.sh`. On Windows it prepends nothing and runs whatever the property points at — which is expected to be the `.bat` companion. That single indirection is what lets one Java code path work on all three platforms: the *wrapper* owns the platform-specific interpreter activation, and the Java side only ever knows about flags.

**Only two `.bat` companions exist** — `run_fl_server.bat` and `run_init_model.bat` — and both are stale: they hardcode a personal Anaconda root (`C:\Users\CHINMAY\anaconda3`), activate a `llm-gpu` conda env, and invoke `python src\main\resources\scripts\fl_server.py`, a path that no longer exists anywhere in the repository (the scripts moved to `fl-runtime/`). There are no `.bat` companions for `run_recipes.sh`, `run_infer.sh`, `run_fot_server.sh` or `run_benchmark.sh` at all. **Windows is therefore not a working target for this layer today**, despite the wrapper indirection being designed for it.

---

## `fl_server.py` — the FL aggregation server

The FL server the backend spawns for every gradient run. It loads initial weights from an `.npz`, constructs a framework `Strategy`, runs `fl.server.start_server(...)`, then saves, registers and reports the result.

**CLI surface** (`build_arg_parser`, `fl_server.py:473` — deliberately extracted from `main()` so the flag contract is unit-testable without booting a server):

| Flag | Required | Default | Notes |
|---|---|---|---|
| `--model-path` | yes | — | The run's `.npz`. **Write target** for the aggregated result, and the init source unless `--init-model-path` is given |
| `--init-model-path` | no | `None` | BA-11: read *initial* weights from the registry head instead. `--model-path` stays the write target, so the immutable content-addressed blob is never overwritten |
| `--project-id` | yes | — | |
| `--num-rounds` | no | `5` | |
| `--min-clients` | no | `1` | `min_fit_clients` for the strategy |
| `--model-type` | yes | — | `type=str.upper`; `choices=recipes.catalog_keys()` — data-driven from the catalog |
| `--model-name` | yes | — | |
| `--port` | no | `50051` | The backend passes a port reserved from `50000-50010` |
| `--strategy` | no | `FedAvg` | `fedavg` / `decomfl` / `fedlora` / `fedprox` / `fedopt` / `robust` |
| `--training-arm` | no | `None` → `FULL` | Validated against the recipe's `supported_arms`. See [03](03_training_arms.md) |
| `--seed` | no | `None` | Omitted ⇒ a fresh seed is drawn from the OS CSPRNG and recorded on the eval card |
| `--aggregation` | no | `FFA_LORA` | `FFA_LORA` / `FEDIT`; LLM_LORA only |
| `--task-type` | no | `SEQ_CLASSIFICATION` | `SEQ_CLASSIFICATION` / `CAUSAL_LM`; LLM_LORA only |
| `--dataset` | no | `cb` | `cb` / `sst2` / `ecg` |
| `--dp-enabled` | no | off | FedLoRA-only central DP (FR-13 / SE-11) |
| `--dp-clip-norm` | no | `None` | L2 clip bound S per client adapter delta |
| `--dp-noise-multiplier` | no | `None` | Raw z; mutually exclusive with `--dp-target-epsilon` |
| `--dp-target-epsilon` | no | `None` | ε budget, solved to z by the RDP accountant inside `FedLoRA` |
| `--dp-delta` | no | `None` | Required with `--dp-target-epsilon` |
| `--dp-num-clients` | no | `None` | Enrolled population N for q = cohort/N; omitted ⇒ q = 1 |
| `--dp-rounds` | no | `None` | Round count T the budget is accounted over |
| `--dp-seed` | no | `None` | Seeds only the DP noise RNG, deliberately independent of `--seed` |

**What the backend actually emits.** `FlServerManager.buildServerCommand` builds `--project-id`, `--model-path`, optionally `--init-model-path`, `--port`, `--strategy`, `--num-rounds`, `--model-type`, `--model-name`, `--min-clients`, then:

- `--training-arm` **only when the project's arm is not `FULL`**, so every pre-existing spawn's argv is byte-identical to before the arm feature landed. Both `fl_server.py` and `client.py` resolve an omitted arm to `FULL`.
- `--aggregation FFA_LORA` and `--task-type` when `modelType == LLM_LORA`.
- The `--dp-*` block when the project has DP enabled, re-validated at the seam so a null knob can never reach the argv as the string `"null"`.

The backend never passes `--seed` or `--dataset`; both fall to their defaults.

**Run shape** (`main`, `fl_server.py:540`):

1. `resolve_run_seed(args)` (`fl_server.py:515`) — seeds `random`, `numpy`, `torch` (and CUDA), and writes the resolved integer back onto `args` so `build_eval_card` commits the seed that actually seeded the RNGs.
2. Build the architecture through the recipe registry (`init_model.get_model`), except for `LLM_LORA`, which rebuilds its peft model per eval round instead.
3. Load the `.npz`, restore `__DOT__`-escaped keys, apply the wire filter and then the arm filter — see [04](04_the_federated_set.md).
4. Pick the server-side test loader, then define `server_side_evaluate`, which additionally emits a rich per-round benchmark record via `benchmarks.build_round_record`.
5. `select_strategy(args, initial_parameters, evaluate_fn)` (`fl_server.py:289`) — pure dispatch, extracted so it is unit-testable without a gRPC server.
6. `fl.server.start_server(...)` — the framework takes over.
7. Save (`merge_non_federated` first), register the artifact + eval card, POST per-round results and benchmark records to `/api/internal/**`, then mark the project finished.

**Strategy dispatch.** `select_strategy` recognises `decomfl`, `fedlora`, `fedprox`, `fedopt`, `robust`, and `fedavg`. **An unrecognised name raises** (FR-28) rather than silently falling back to FedAvg while every strategy-specific flag is ignored — the function's own docstring still describes the old fallback behaviour and is stale on that point; the code is the truth. `FedProx`/`FedOpt`/`RobustAggregator` are constructed with hardcoded sensible defaults (μ = 0.1; FedAdam with server_lr 1.0, β₁ 0.9, β₂ 0.99, τ 1e-3; coordinate-wise median with trim_ratio 0.1 and no clipping) — plumbing those through from the project config is unfinished work, not a completed feature.

**Backend callbacks.** The server talks back over `/api/internal/**` using two headers built by `_internal_headers()`: `X-Internal-Key` (`FEDLEARN_INTERNAL_API_KEY`, the shared secret) and `X-Internal-Run-Token` (`FEDLEARN_INTERNAL_RUN_TOKEN`, the SE-7 per-run token scoped to this project). Missing the shared key raises rather than silently no-op'ing. `FEDLEARN_BACKEND_URL` selects the callback host; it falls back to `http://<SERVER_HOST|AWS_HOST|localhost>:8081`.

---

## `client.py` — the one canonical FL client

**DA-5: there is exactly one FL client source in this repository, and this is it.** The Docker image (`client-docker/Dockerfile` → `COPY fl-runtime/ /app/fl-runtime/`, `WORKDIR /app/fl-runtime`, `entrypoint.sh` → `exec python3 -u client.py …`), the desktop PyInstaller bundle (`client-docker/packaging/fedlearn-client.spec` → `CLIENT_ENTRY = fl-runtime/client.py`) and the desktop dev-mode fallback (`docker.service.ts:271` → `path.join(repoRoot, 'fl-runtime', 'client.py')`) all consume this file. There is no `client-docker/scripts/` client fork; do not reintroduce one.

**CLI surface** (`build_arg_parser`, `client.py:1015` — extracted from `main()` for the same testability reason as the server's):

| Flag | Required | Default | Notes |
|---|---|---|---|
| `--project-id` | yes | — | |
| `--server-address` | yes | — | `host:port` |
| `--partition-id` | yes | — | `choices=range(0, 10)` — `NUM_PARTITIONS` is a fixed 10 |
| `--model-type` | no | `None` | `choices=recipes.catalog_keys()` |
| `--model-name` | no | `None` | |
| `--aggregation` | no | `FFA_LORA` | `FFA_LORA` / `FEDIT` |
| `--task-type` | no | `SEQ_CLASSIFICATION` | `SEQ_CLASSIFICATION` / `CAUSAL_LM` |
| `--dataset` | no | `cb` | `cb` / `sst2` / `ecg` |
| `--strategy` | no | `FedAvg` | `FedAvg` or `DeComFL` |
| `--training-arm` | no | `None` → `FULL` | Validated against the recipe's `supported_arms` at startup |
| `--use-llm` | no | off | Deprecated; use `--model-type TRANSFORMER` |
| `--device` | no | `$FEDLEARN_DEVICE` or `auto` | `auto` / `cpu` / `cuda` / `mps`; resolved by `device.resolve_device` |

**There is no `--client-id`.** The client id is derived: `f"project_{args.project_id}_client_{args.partition_id}"`.

**Mode flags.** `main()` translates `--model-type` into module-global booleans — `USE_LLM`, `USE_MLP`, `USE_PNEUMONIA`, `USE_LLM_LORA` — plus `MODEL_TYPE`, `TRAINING_ARM` and `USE_DERIVED`. `USE_DERIVED` means *"this arm federates a trainable subset"*, derived from `recipes.trainable_prefixes(mt, arm) is not None` rather than from an arm-name comparison. That derivation matters: an arm-name test silently excluded `OVA_LP`, which is also a subset arm.

**Two client classes, one file.** `--strategy DeComFL` constructs the framework's `DeComFLClient` (with per-model-type model + loader selection) and calls `fl.client.start_decomfl_client(...)`, whose return value is a terminal outcome string (`"completed"` / `"disconnected"` / `"error"`). Anything else constructs the local `ZOSLClient(fl.Client)` (`client.py:642`) and calls `fl.client.start_client(...)`. `LLM_LORA` + `DeComFL` is refused with `sys.exit(1)` on both the client and the server — the zeroth-order path needs a flat float-vector parameter space that adapter-only sync cannot provide.

**Hardcoded overrides.** Selecting `MLP` forces `--dataset ecg`, `--strategy DeComFL`, `num_clients = 5` (`ECG_NUM_CLIENTS`, `client.py:921` — note `fl_server.py`'s same-named constant is `3`) and the `ecg_data/ecg.csv` path, overriding whatever was passed. `PNEUMONIA_CNN` reads `FEDLEARN_NUM_CLIENTS` (default 2) and `FEDLEARN_PNEUMONIA_BATCH` (default 16). Everything else defaults to `num_clients = 2`.

**Container/desktop plumbing.** `client-docker/entrypoint.sh` hard-fails if `PROJECT_ID`, `SERVER_ADDRESS` or `PARTITION_ID` is unset, then builds those three flags itself and appends `--model-type`, `--strategy` and `--training-arm` from the `MODEL_TYPE` / `STRATEGY` / `TRAINING_ARM` environment variables before forwarding `"$@"` for extras. The desktop sets those env vars for the Docker path (`docker.service.ts:72`) and passes the flags directly for the native path (`docker.service.ts:336`).

---

## `init_model.py` — initial weights

Builds a model and writes its parameters to an `.npz` that `fl_server.py` later reads as round-0 global weights. Invoked by `ModelInitializer` at project creation, on a 300-second timeout (`python.script.init-model.timeout-seconds`, BA-1) so a hung init cannot pin a request thread holding a DB connection.

| Flag | Required | Default |
|---|---|---|
| `--model-type` | yes | — |
| `--model-name` | yes | — |
| `--optimizer` | yes | — |
| `--out` | yes | — (the `.npz` path) |
| `--pretrain-epochs` | no | `0` |
| `--aggregation` | no | `FFA_LORA` (`FFA_LORA` / `FEDIT`) |
| `--task-type` | no | `SEQ_CLASSIFICATION` (`SEQ_CLASSIFICATION` / `CAUSAL_LM`) |

Note there is **no `choices=` on `--model-type` here** — unlike `fl_server.py` and `client.py`, this script accepts any string and lets `recipes.get_recipe` raise on an unknown key.

Three things it does that are easy to miss:

- **`get_model` is a one-line registry delegate** (`init_model.py:38-54`): `recipes.get_recipe(model_type).build_model(device, model_name=…, aggregation=…, task_type=…)`. There is no per-type `if`/`elif` model construction here any more. See [02](02_recipe_catalog.md#how-far-registry-dispatch-actually-got).
- **The saved state differs by model type** (`init_model.py:158-171`). `LLM_LORA` saves `get_peft_model_state_dict(net, save_embedding_layers=False)` — the adapter, not a full checkpoint. `TINYNET_GOLDEN` saves `trainable_state(net)` (25 params = `fc1`) because DeComFL federates trainable parameters only, and a full `state_dict()` would give the server `model_dim=43` against 25-dim clients and reject every one. Everything else saves the full `state_dict()`.
- **Keys are escaped for `.npz`.** `.` is not legal in an `npz` member name, so every key is written as `key.replace('.', '__DOT__')` and un-escaped on load by `fl_server.py` and `infer.py`.

`--pretrain-epochs > 0` only does anything for `CNN` (1000 CIFAR-10 samples, batch 32). `TRANSFORMER`, `MLP` and `LLM_LORA` print an explicit warning and ignore it; `PNEUMONIA_CNN`, `CIFAR_RESNET18` and `TINYNET_GOLDEN` have no branch at all, so it is ignored silently.

> `init_model.py:18-36` still defines a private `CnnNet` class. Nothing references it — `get_model` routes `CNN` to `models.CnnNet` through the registry. It is dead code.

---

## `infer.py` — single-input inference

Rebuilds the training architecture, loads the aggregated `.npz`, runs one forward pass. Invoked by `InferenceService` via `run_infer.sh`.

| Flag | Required | Default |
|---|---|---|
| `--model-path` | yes | — |
| `--model-type` | yes | — |
| `--model-name` | yes | — |
| `--in` | yes | — (input payload JSON path) |
| `--out` | yes | — (result JSON path) |
| `--task-type` | no | `SEQ_CLASSIFICATION` |
| `--max-new-tokens` | no | `256` |
| `--temperature` | no | `0.7` |

**The result goes to the `--out` file, never to stdout.** That is a deliberate contract: torch/CUDA banners and wrapper log lines pollute stdout, so the Java side reads the out-file and treats stdout as diagnostics only.

Input payloads are one of `{"kind":"image","imagePath":…}`, `{"kind":"vector","values":[…]}`, `{"kind":"text","text":…}` or `{"kind":"generation","prompt":…,"history":[…]}`. The payload kind is checked against the recipe's declared `input_kind` and mismatches raise `InputError`.

Failures are always written as a structured result, tagged so the backend can map them: `errorKind: "input"` (an `InputError` — bad image, wrong vector length, empty text → HTTP 400) versus `errorKind: "internal"` (model load, arch import, torch → 502). Decoded-image size is capped at `MAX_IMAGE_PIXELS = 50_000_000` before any pixel is touched, so a decompression bomb is caller error rather than an OOM.

`build_model` (`infer.py:63-74`) is a one-line delegate to `recipes.get_recipe(model_type).build_for_inference(...)`, which returns `(net, classes, input_kind, transform)`.

---

## `recipes.py` — the catalog CLI

Importable as a module by every other script, and runnable as a CLI with exactly one flag:

```bash
bash run_recipes.sh --describe     # prints the catalog as JSON to stdout, exits 0
bash run_recipes.sh                # prints argparse help, exits 0
```

`ModelRecipeService` spawns it with a 30-second timeout, **discards stderr** so wrapper/torch noise cannot corrupt the JSON on stdout, and caches the parsed result for the JVM's lifetime. There is deliberately **no hardcoded Java fallback** (DA-10) — a previous duplicate had drifted out of sync with the catalog, so a load failure now throws `IllegalStateException` and is not cached, letting a transient problem recover on retry. Full detail in [02](02_recipe_catalog.md).

---

## `fl_fot_server.py` — Federation over Text

The spawn entry point for the FoT path, launched by `FlServerManager` exactly like `fl_server.py` (via `run_fot_server.sh`) so its stdout JSON events stream to the dashboard over STOMP.

| Flag | Required | Default |
|---|---|---|
| `--project-id` | no | `""` |
| `--port` | yes | — |
| `--num-rounds` | no | `5` |
| `--round-seconds` | no | `5.0` |
| `--quorum` | no | `2` |
| `--backend` | no | `stub` |

The backend emits only `--project-id`, `--port` and `--num-rounds` on the FoT branch — no model-type, no model-path, no DP flags (SE-11 refuses to spawn FoT for a DP-enabled project outright).

The file is a thin shim: it walks up to eight directory levels looking for `framework/src/fedlearn`, puts it on `sys.path` if `import fedlearn` fails, then calls `fedlearn.fot.fot_server.start_fot_server`. FoT is **additive and orthogonal to the gradient path** and torch-free by design.

> **Do not present FoT as a validated capability.** `--backend` defaults to `stub`, and `fedlearn.fot.backend.get_backend` wires only `DeterministicStubBackend` — `local-http`, `vllm` and `ollama` all raise `BackendError`. No LLM has ever run through this path, and there is no FoT result in `research/results/`.

---

## `benchmarks.py` — the metric core

The single source of truth for *how every benchmark metric is computed*, so the online (per-round, inside `fl_server.py`) and offline (deliberate, via the CLI) numbers are directly comparable.

```bash
bash run_benchmark.sh --predictions preds.json [--out report.json]
```

Its module docstring refers to a `run_benchmark.py` CLI; **no such file exists** — the CLI is `benchmarks.py`'s own `_cli`, reached through `run_benchmark.sh`.

Two properties are load-bearing:

- **ARM64 / Jetson safe.** Only numpy + scikit-learn, and every sklearn use is guarded so a missing install degrades to a pure-numpy fallback rather than crashing a live FL run. `fl_server.py` imports `benchmarks` (and therefore sklearn) **before torch**, mirroring `client.py`'s `import sklearn` on line 4 — the ARM64 static-TLS/libgomp allocation workaround.
- **camelCase output keys**, so the payload maps 1:1 onto the Java `BenchmarkRoundDto` with no renaming.

It computes classification macro metrics (accuracy, balanced accuracy, precision/recall/F1 in macro/micro/weighted, MCC, Cohen's κ, log-loss, ROC-AUC, expected calibration error, Brier score), per-class micro metrics and a confusion matrix, generative metrics (eval loss, perplexity = exp(loss), token accuracy), and a label-distribution + normalised Shannon entropy non-IID proxy.

---

## Support modules

| Module | Role |
|---|---|
| `config.py` | `DatasetConfig` / `ECGDatasetConfig` / `DeComFLConfig` dataclasses plus `DATASET_CONFIGS` (`cb`, `sst2`, `ecg`) and `DECOMFL_CONFIGS` (`default`, `ecg`). Accessed via `get_dataset_config()` / `get_decomfl_config()`, both of which raise on an unknown name. |
| `data.py` | `load_server_test_data(is_llm, dataset_name)`. The LLM branch tokenises `super_glue/cb` or `glue/sst2` with the opt-125m tokenizer straight from `config.py`; the CNN branch is a one-line delegate to `recipes.get_recipe("CNN").load_server_test_data(batch_size=128)`. |
| `device.py` | `resolve_device(choice)` — `auto` picks cuda > mps > cpu; an explicitly requested but unavailable accelerator warns and falls back to cpu rather than crashing. Kept out of `client.py` so it is unit-testable without the heavy client import. |
| `models.py` | `CnnNet` — the canonical CIFAR-10 CNN (`conv1`/`conv2`/`fc1`/`fc2`/`fc3`). The registry builds `CNN` from here. |
| `models/ecg_mlp.py` | `ECGModel(input_dim, hidden_dim, num_classes)` — the `MLP` recipe's architecture. |
| `data_loaders/ecg_loader.py` | ECG CSV loading and partitioning, now reached through `recipes.get_recipe("MLP")`. |
| `architecture/cnn/` | A legacy CNN package superseded by `models.CnnNet`. Still bundled by the PyInstaller spec. |

---

## The wrappers, one by one

| Wrapper | Runs | Behaviour worth knowing |
|---|---|---|
| `run_fl_server.sh` | `fl_server.py` | Truncates and `tee`s to `fl_server_deep_debug.log` in the script directory on every run, and preserves Python's exit code via `${PIPESTATUS[0]}` rather than `tee`'s |
| `run_fl_server.bat` | — | **Stale.** Hardcoded conda root; points at a `src\main\resources\scripts\` path that no longer exists |
| `run_init_model.sh` | `init_model.py` | Plain passthrough, `cd` to the script dir first |
| `run_init_model.bat` | — | **Stale**, same two problems as above |
| `run_recipes.sh` | `recipes.py` | Plain passthrough; stdout is the JSON catalog |
| `run_infer.sh` | `infer.py` | Plain passthrough; the wrapper's stdout is diagnostic only, the result is in `--out` |
| `run_fot_server.sh` | `fl_fot_server.py` | `tee`s to `fot_server_debug.log`, same `PIPESTATUS` handling as the FL server wrapper |
| `run_benchmark.sh` | `benchmarks.py` | Plain passthrough; its header documents the `preds.json` shape |
| `run_clients.sh` | `client.py` × N | Developer convenience only — spawns N clients in new terminals (iTerm / Terminal.app / gnome-terminal / xfce4-terminal). Carries a **hardcoded `PROJECT_ID`** and a `PROJECT_ROOT` computed by walking six levels up from a directory layout that no longer exists; edit before use |

All the `.sh` wrappers `cd` into `SCRIPT_DIR` before running Python. That is what makes the scripts' sibling imports (`import recipes`, `from config import …`, `from data import …`) resolve regardless of the caller's working directory.
