# 05 — Testing & CI

> **Part of:** [FedLearn Platform Docs](../README.md) → [FL Runtime Wiki](./README.md)

## Table of Contents
- [Running the Suite](#running-the-suite)
- [`pytest.ini` and the `slow` Marker](#pytestini-and-the-slow-marker)
- [`conftest.py` — making `import fedlearn` work](#conftestpy--making-import-fedlearn-work)
- [TE-10: a Skipped Test Fails the Job](#te-10-a-skipped-test-fails-the-job)
- [The CI Job](#the-ci-job)
- [What the Suite Pins](#what-the-suite-pins)
- [Known Staleness Around This Layer](#known-staleness-around-this-layer)

---

## Running the Suite

`fl-runtime/` has its **own** pytest suite, separate from `framework/tests/` and gated by its own CI job.

```bash
cd fl-runtime && python -m pytest -q
```

That is verbatim how CI runs it. As of this writing the suite collects **287 tests across 40 files**; `pytest.ini` deselects 28 as `slow`, leaving **259 selected**.

```bash
# a single file
cd fl-runtime && python -m pytest tests/test_training_arm.py -q

# include the slow tests (needs network and/or cached model artifacts)
cd fl-runtime && python -m pytest -q -m ""

# reproduce CI's skip-integrity gate locally
cd fl-runtime && FEDLEARN_FAIL_ON_UNEXPECTED_SKIP=1 python -m pytest -q
```

---

## `pytest.ini` and the `slow` Marker

The whole file is three directives:

```ini
[pytest]
addopts = -m "not slow"
markers =
    slow: marks tests that require network access or cached model artifacts (deselect with -m "not slow")
```

Two things follow, and the distinction between them matters:

- **`-m "not slow"` is a *deselection*, not a skip.** Deselected tests emit no skip report at all, so they never trip the TE-10 guard below. This is the sanctioned way to keep a network-dependent test out of the default run.
- **There is no coverage enforcement here.** Unlike `framework/pytest.ini`, which adds `--cov=fedlearn` and can fail its job on a coverage drop, this suite's job is a bare pytest with the skip guard.

Twelve files carry at least one `@pytest.mark.slow`: `test_arm_applies_to_every_recipe_branch`, `test_client_llm_lora`, `test_declared_arms_are_wire_compatible`, `test_frozen_means_frozen`, `test_infer_generate`, `test_infer_registry_dispatch`, `test_infer_text`, `test_init_model_llm_lora`, `test_llm_lora_recipe`, `test_pretrained_backbone`, `test_recipe_cnn_data_parity`, `test_recipes_registry`. In every case the reason is the same: a real dataset download (CIFAR-10, SST-2) or a real HuggingFace checkpoint pull.

Two files — `test_client_llm_lora.py` and `test_init_model_llm_lora.py` — contribute **zero** selected tests, because every test in them is slow-marked.

---

## `conftest.py` — making `import fedlearn` work

`tests/conftest.py` does two things before any test runs.

**1. It puts the in-repo framework on `sys.path`.** The trainer scripts do `import fedlearn`, which in a real deployment is pip-installed alongside them. A bare checkout and the CI job do not install it, so every test that imports one of those scripts would die at *collection* with `ModuleNotFoundError`. The conftest walks up to twelve directory levels looking for `framework/src`, prepends it to `sys.path`, **and also exports it on `PYTHONPATH`** — because several tests spawn `python init_model.py` (etc.) in a fresh interpreter that re-imports `fedlearn` but does not inherit the parent's `sys.path`. Only `PYTHONPATH` crosses a process boundary.

**2. It installs the TE-10 skip-integrity guard**, described next.

---

## TE-10: a Skipped Test Fails the Job

> A silently-skipped test is a **false green**: the suite reports success while the behaviour under test never ran.

When the guard is active, any test that ends up SKIPPED for a reason not explicitly allowlisted fails the run with a summary of the offenders.

```python
_ALLOWED_SKIP_REASONS = frozenset()   # this suite has no legitimate skips today
```

**Activation:** on when `$CI` is truthy (GitHub Actions always sets `CI=true`); force it with `FEDLEARN_FAIL_ON_UNEXPECTED_SKIP=1`, suppress it with `=0`.

**What does *not* trip it:**

- `-m "not slow"` deselection — deselected tests emit no skip report at all, so the slow-marker workflow is untouched.
- `xfail` / `xpass` — an expected failure is not a silent skip.
- Reasons listed in `_ALLOWED_SKIP_REASONS`. That set is currently empty, and each future entry needs a written justification.

Implementation-wise the guard hooks three points, which is what makes it hard to sneak past: `pytest_runtest_logreport` (marker skips surface in setup, runtime `pytest.skip`/`importorskip` in call), `pytest_collectreport` (module-level skips and module `importorskip`), and `pytest_sessionfinish` (flips a green `exitstatus` to 1). The terminal summary spells out the remedy:

> "A skipped test is a false green: make it run, or allowlist its exact reason in `tests/conftest.py` with a written justification."

The same guard exists in `framework/tests/conftest.py`; it is duplicated rather than shared so this directory stays self-contained in deployments.

---

## The CI Job

`.github/workflows/ci.yml`, job **`backend-scripts`**. CI is path-filtered — this job runs only when the filter matches:

```yaml
backend_scripts:
  - 'fl-runtime/**'
  - 'backend/fl-platform-api/requirements.txt'
```

The second path is there because `test_requirements_security_floors.py` reads that lockfile.

```yaml
- name: pytest (FL trainer scripts; pytest.ini deselects -m slow)
  working-directory: fl-runtime
  env:
    FEDLEARN_FAIL_ON_UNEXPECTED_SKIP: '1'
  run: python -m pytest -q
```

Environment notes worth knowing before you debug a red job:

- **Python 3.12.** Matches the repo pin (`.tool-versions` → 3.12.9); the framework declares a 3.10+ floor but 3.12 is what is actually tested.
- **CPU torch, pinned to `2.5.1` / torchvision `0.20.1` / torchaudio `2.5.1`** from the PyTorch CPU index, so pip treats them as satisfied and does not re-resolve torch off the default index. **This differs from the `framework` job, which pins `torch==2.12.0`** to match the DeComFL golden fixture — the two jobs deliberately run different torch versions.
- **`peft`, `scikit-learn` and `PyJWT` are installed explicitly**, on top of `backend/fl-platform-api/requirements.txt`. They are not in that lockfile (it predates the LoRA work), but the non-slow suite needs all three: `client.py` has a module-level `import sklearn` (line 4), so *any* test importing it fails at collection without it; the LoRA / eval-card / perplexity tests import `peft` at runtime; and the FL-server import chain reaches `security/token_verify.py` → `import jwt`. Without them the suite errors out with collection errors — a false red the moment the scripts change.
- The job installs the **backend** lockfile and reaches the framework via `sys.path`, which is why those three have to be named here rather than inherited.

---

## What the Suite Pins

Most of these files are named after a *property*, not a function, because most of them were written after a live federation surfaced a defect that unit tests had missed. Grouped by concern:

### Training arms

| File | Selected | Pins |
|---|---|---|
| `test_training_arm.py` | 21 | The arm as a first-class declared concept: `apply_arm` idempotence, unknown-arm rejection, `validate_arm` resolution, the federated payload following the arm, and the provenance stamp differing across arms of the same recipe |
| `test_arm_client_server_agreement.py` | 18 | The contract the whole feature rests on — client and server must agree on **which parameters are federated**. Nothing asserted this until P1-5, and the gap was real: the client kept its default `FULL` and uploaded a full state dict to a server holding only the head |
| `test_ova_lp_arm.py` | 17 | `OVA_LP` is declared, is offered only where `FROZEN_HEAD` is, trains the *same parameters* as `FROZEN_HEAD`, and carries the one-vs-all objective. Also documents what is deliberately **not** implemented (the two-stage schedule) |
| `test_arm_tradeoff.py` | 17 | The picker's numbers are a rendering of the record, not a hand-written claim — regenerates and compares where the record is present, and rejects a number shown without its caveat |
| `test_arm_eval_card.py` | 5 | The arm rides on every result and cannot silently collide with another arm's result. Written before the implementation |
| `test_arm_does_not_choose_the_dataset.py` | 4 | The arm must not decide which **dataset** the client loads. From a live crash: `Expected 3D or 4D input to conv2d, but got input of size: [32, 256]` |
| `test_fl_server_arm_scope.py` | 3 | The server's arm filter must be *reachable at runtime* — a function-local `import recipes` further down `main()` made the earlier use an unbound local |
| `test_arm_applies_to_every_recipe_branch.py` | 2 (3 slow) | The arm must be applied whichever branch built the model. From a live `FROZEN_HEAD` run whose backbone moved by 0.65 while reporting itself frozen |

### The frozen path and the wire

| File | Selected | Pins |
|---|---|---|
| `test_server_eval_strictness.py` | 8 | Server-side evaluation must load a subset-federated global model **non-strictly**, derived from the arm rather than a hardcoded recipe name |
| `test_frozen_save_preserves_backbone.py` | 7 | A frozen run must not destroy the backbone its own output depends on |
| `test_fedprox_respects_the_frozen_arm.py` | 7 | The proximal term regularises the federated subset and leaves the frozen backbone alone — `_apply_proximal_gradient` zips positionally, so a wrong `global_params` would pair backbone tensors with head values and stop |
| `test_frozen_means_frozen.py` | 6 (slow) | A frozen backbone must not adapt its BatchNorm statistics to local data |
| `test_frozen_backbone_recipe.py` | 6 | The `FROZEN_DEMO` recipe builder + the DA-11 partial-load path |
| `test_pretrained_backbone.py` | 5 (slow) | Pretrained-backbone recipes — and the honest negative that motivated them: `FROZEN_HEAD` on a randomly-initialised CIFAR-10 backbone sat at chance (10.05 / 10.02 / 10.00 %, loss 2.3042 against ln(10) = 2.3026) |
| `test_client_derived_subset_federation.py` | 2 | The production client federates a derived model **head-only** |
| `test_declared_arms_are_wire_compatible.py` | 1 (slow) | A recipe must not declare an arm whose payload cannot cross the wire — targets the defect *class*, having seen three instances |
| `test_frozen_backbone_federation_run.py` | 1 | Derivation actually runs: a real, converging, multi-round head-only FedAvg loop, seeded and self-contained |

### The recipe registry

| File | Selected | Pins |
|---|---|---|
| `test_recipes_registry.py` | 23 (some slow) | Every dispatched key resolves; the `--describe` catalog stays byte-stable; `BLOOD_CNN` is resolvable but un-advertised; and DA-14 Ph3.1's data-driven `init_model.get_model` routes an arbitrary fake recipe through with no `init_model` edit |
| `test_recipes_base_models_buildable.py` | 9 | Every advertised `(recipe, base_model)` pair instantiates without raising — an advertised base model is a user-facing promise. Pretrained downloads are stubbed with tiny `from_config` models, so the test proves the name-dispatch path reaches a real constructor without touching the network |
| `test_llm_lora_recipe.py` | 8 (some slow) | The LoRA recipe: adapter keys, tokenizer-per-base-model, `FFA_LORA` freezing |
| `test_recipe_cnn_data_parity.py` | 5 (slow) | The CIFAR-10 shard is byte-identical to the `flwr` implementation it replaced — the fixed-10-shard partitioner, the `shuffle→shard→split` order, and the train/test source asymmetry |
| `test_causal_lm_recipe.py` | 4 | The `CAUSAL_LM` task path |
| `test_recipe_mlp_data_parity.py` | 2 | The ECG shard reproduces the exact partition the hand-threaded `ecg_loader` call produced |
| `test_recipe_catalog_matches_runnable_choices.py` | 2 | The catalog equals the set of `--model-type` values the scripts accept — the guarantee SE-10's spawn gate rests on |

### Server, client, inference and infrastructure

| File | Selected | Pins |
|---|---|---|
| `test_fl_server_select_strategy.py` | 19 | Every `--strategy` branch constructs the right framework strategy with the right hyperparameters, and an unrecognised name fails loud (FR-28) |
| `test_fl_server_eval_card.py` | 12 | SE-11: a DP run's card carries an `accounted_epsilon` verbatim from the strategy; a non-DP run has **no** `dp` key at all. Plus DA-3's run-seed contract — the card's `seed` is always a concrete integer |
| `test_benchmarks.py` | 9 | The metric core, over pure `(y_true, y_pred[, y_score])` inputs with no torch and no GPU |
| `test_client_tinynet_golden_decomfl.py` | 7 | A desktop client can join a `TINYNET_GOLDEN` DeComFL federation with the phone as the other client — same recipe, byte-identical state-dict keys, deterministic frozen `fc2` |
| `test_client_fedprox_config.py` | 6 | FR-32: the production client **honors** FedProx (it used to refuse it), plus the string→int `local_epochs` coercion the `map<string,string>` gRPC config forces |
| `test_pneumonia_hf_load_kwargs.py` | 5 | SE-19: the HuggingFace fallback must not auto-execute remote code |
| `test_infer_registry_dispatch.py` | 5 (slow) | `infer.build_model` delegating to the registry is byte-identical: same state-dict keys, same labels, same input kind — and the returned module is now the registry's class |
| `test_device.py` | 4 | `resolve_device`: `auto` ordering and the fall-back-to-cpu-with-a-warning behaviour |
| `test_requirements_security_floors.py` | 3 | The backend lockfile must not pin below the security floors `framework/requirements.txt` documents — and `flwr` stays out of the lockfiles |
| `test_infer_chat.py` / `test_infer_text.py` / `test_infer_generate.py` | 2 / 1 / 1 | Chat-history rendering, text classification, and generative inference |
| `test_init_model_tinynet_golden.py` / `test_init_model_llm_lora.py` | 1 / 0 (slow) | `init_model.py`'s two non-default state-dict extraction branches |
| `test_fl_server_perplexity.py` | 1 | `perplexity_from_loss` is overflow-guarded — a diverged round reports `inf` rather than raising |

---

## Known Staleness Around This Layer

Recorded here because it is real, is not fixed by anything in this directory, and would otherwise be rediscovered the hard way. None of it is a wiki claim; all of it was read directly.

- **`run_fl_server.bat` and `run_init_model.bat` are broken.** Both hardcode a personal Anaconda root and invoke `python src\main\resources\scripts\<script>.py` — a path that does not exist anywhere in this repository. There are no `.bat` companions at all for `run_recipes.sh`, `run_infer.sh`, `run_fot_server.sh` or `run_benchmark.sh`, so the Windows branch of the backend's spawn logic has nothing valid to point at for those.
- **`client-docker/packaging/fedlearn-client.spec:37` still lists `flwr_datasets` in `FULL_COLLECT`, and the loop `raise`s on failure.** Since `flwr-datasets` is no longer installed anywhere, `collect_all('flwr_datasets')` will fail and take the PyInstaller build down with it. The `NOTE: matplotlib cannot be excluded — flwr_datasets/__init__.py …` comment at line 95 is stale for the same reason. `client-docker/requirements.txt:23`'s pyarrow note is stale but inert.
- **`application.properties` lines 145-147** describe the local-process Python paths as *"used by ModelInitializer for dev runs only — production uses the ECS Fargate FL-server path"*. There is no ECS path; OP-14 removed it and `FlOrchestrationModeValidator` now fails the boot if `ecs.cluster-name` is set. These paths are the only path, in every profile.
- **`python.script.init-model.path` and `python.flbat.path` are defined but read by no Java class.** `ModelInitializer` uses `python.executable.path` (which, despite the name, holds a *wrapper script* path, not an interpreter).
- **`select_strategy`'s docstring contradicts its code.** It says "FedAvg remains the fallback for both the explicit `fedavg` name and any unrecognized strategy"; FR-28 changed that to a hard `ValueError`. The code is right.
- **`benchmarks.py`'s module docstring refers to a `run_benchmark.py` CLI** that does not exist — the CLI is `benchmarks.py` itself, reached through `run_benchmark.sh`.
- **`init_model.py:18-36` defines a `CnnNet` that nothing references.** `get_model` builds `CNN` through the registry (`models.CnnNet`). Dead code.
- **`Recipe.is_functional`'s hardcoded tuple omits `CIFAR_RESNET18`**, whose model and data both live in `recipes.py`. Nothing in production reads the property, so this is an inaccurate signal rather than a behavioural bug.
- **`run_clients.sh` carries a hardcoded `PROJECT_ID`** and computes `PROJECT_ROOT` by walking six levels up from a directory layout that no longer exists. Edit before use.
- **`fl_server_deep_debug.log` (~93 KB) and `fl_server_deep_debug.log.FedProx.marker` sit in the working tree.** `run_fl_server.sh` truncates and rewrites the log on every run, so it is a run artifact rather than source.
- **The frontend's `Project.trainingArm` type is `'FULL' | 'FROZEN_HEAD'`** (`frontend/src/services/apiServices.ts:66`), which is missing `OVA_LP`. The picker itself renders `supportedArms` from the catalog, so the selector still offers the arm; it is the narrower response type that is out of date.
