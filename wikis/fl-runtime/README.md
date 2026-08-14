# FL Runtime Wiki

> **Part of:** [FedLearn Platform Docs](../README.md)

Welcome to the internal documentation for the **FL runtime** (`fl-runtime/`) — the **executable layer** of the platform, and the one the Spring Boot backend actually runs.

This is the unit most easily missed. It is not the FL library, it is not a helper directory, and it is not packaged inside the backend JAR. It is a first-class deployable unit with its own entry points, its own CLI contracts, its own pytest suite and its own CI job.

---

## The split that trips people up

| | `framework/` | `fl-runtime/` |
|---|---|---|
| What it is | An installable Python **library** | An **executable layer** of scripts |
| How you consume it | `pip install -e .` → `import fedlearn` | `python fl_server.py …`, `bash run_fl_server.sh …` |
| Entry points | none — it is imported | `client.py`, `fl_server.py`, `fl_fot_server.py`, `init_model.py`, `infer.py`, `recipes.py`, `benchmarks.py` |
| Depends on the other? | No | **Yes** — `fl-runtime/client.py:19` is literally `import fedlearn as fl` |
| Own test suite | `framework/tests/` (`PYTHONPATH=src pytest`) | `fl-runtime/tests/` (`cd fl-runtime && pytest`) |
| Own CI job | `framework` in `.github/workflows/ci.yml` | `backend-scripts` in the same file |

**The backend never runs `framework/` directly.** It shells out to `fl-runtime/`'s shell wrappers, which in turn run the Python entry points, which in turn `import fedlearn`. Every FL server the platform has ever started went through this layer.

```
Spring Boot (FlServerManager)
   └── bash fl-runtime/run_fl_server.sh --project-id … --port … --model-type …
          └── python3 fl_server.py "$@"
                 └── import fedlearn as fl   →   fl.server.start_server(...)
```

The same relationship holds on the client side: the desktop app and the Docker image both launch `fl-runtime/client.py`, which imports the framework. See [01 — Entry Points & the Backend Contract](01_entry_points.md).

---

## Documentation Index

| # | Document | Description |
|---|----------|-------------|
| 1 | [Entry Points & the Backend Contract](01_entry_points.md) | Every script and wrapper, its real CLI surface, and how `application.properties` resolves them |
| 2 | [The Model-Recipe Catalog](02_recipe_catalog.md) | What a recipe bundles, `RECIPE_METADATA`, torch-free `--describe`, the two out-of-catalog recipes, and how far registry dispatch actually got |
| 3 | [Training Arms](03_training_arms.md) | `FULL` / `FROZEN_HEAD` / `OVA_LP` end to end — objectives, `supported_arms`, `arm_tradeoff.json`, provenance |
| 4 | [The Federated Set](04_the_federated_set.md) | Which tensors cross the wire: the non-float32 exclusion, subset federation, save-time merge, and eval strictness |
| 5 | [Testing & CI](05_testing.md) | The suite, `pytest.ini`'s slow deselection, TE-10's skip-integrity guard, and what each test file pins |

---

## Quick Navigation

**"The backend says it can't find the script."** → [01 — Entry Points](01_entry_points.md#how-the-backend-resolves-this-directory).

**"I added a recipe and the model type is broken."** → [02 — Recipe Catalog](02_recipe_catalog.md#adding-a-model-type-what-you-actually-have-to-wire).

**"The frozen arm trained the whole model."** → [03 — Training Arms](03_training_arms.md#four-defects-only-a-live-federation-found).

**"A BatchNorm model dies on the first `GetGlobalModel`."** → [04 — The Federated Set](04_the_federated_set.md#the-non-float32-exclusion).

**"CI failed on a skipped test."** → [05 — Testing & CI](05_testing.md#te-10-a-skipped-test-fails-the-job).

---

## Directory At A Glance

```
fl-runtime/
├── client.py               ← THE canonical FL client (DA-5). 1328 lines.
├── fl_server.py            ← FL aggregation-server entry point. 1219 lines.
├── fl_fot_server.py        ← Federation-over-Text server entry point (separate path)
├── init_model.py           ← Builds + saves the run's initial .npz weights
├── infer.py                ← Single-input inference on a trained .npz
├── recipes.py              ← THE recipe catalog + training-arm authority. 1491 lines.
├── benchmarks.py           ← Metric-computation core (shared online + offline)
├── data.py                 ← Server-side test-data loading (LLM path + CNN delegate)
├── config.py               ← Dataset / DeComFL hyperparameter dataclasses
├── device.py               ← --device resolution (auto → cuda > mps > cpu)
├── models.py               ← CnnNet (the canonical CIFAR-10 CNN)
├── arm_tradeoff.json       ← Generated per-recipe measured arm trade-off (picker copy)
│
├── run_fl_server.sh/.bat   ← Cross-platform wrappers. The backend invokes THESE,
├── run_init_model.sh/.bat     not the .py files, so one Java code path works on
├── run_recipes.sh             macOS / Linux / Windows.
├── run_infer.sh
├── run_fot_server.sh
├── run_benchmark.sh
├── run_clients.sh          ← Developer convenience: spawn N clients in terminals
│
├── models/                 ← ecg_mlp.py (ECGModel)
├── data_loaders/           ← ecg_loader.py
├── architecture/           ← legacy cnn/ package (superseded by models.CnnNet)
├── ecg_data/               ← ecg.csv (the hardcoded MLP dataset path)
│
├── pytest.ini              ← addopts = -m "not slow"
└── tests/                  ← 40 test files, 287 tests (259 selected, 28 slow)
```

---

## Commands

```bash
# Run the suite exactly as CI does (pytest.ini deselects -m slow)
cd fl-runtime && python -m pytest -q

# Reproduce CI's skip-integrity gate locally (TE-10)
cd fl-runtime && FEDLEARN_FAIL_ON_UNEXPECTED_SKIP=1 python -m pytest -q

# Print the recipe catalog the backend serves at GET /api/model-recipes
bash fl-runtime/run_recipes.sh --describe | python -m json.tool

# Start an FL server by hand (what the backend does, minus the env plumbing)
bash fl-runtime/run_fl_server.sh \
  --project-id <uuid> --model-path /tmp/run.npz --port 50051 \
  --model-type CIFAR_RESNET18 --model-name resnet18 \
  --strategy FedAvg --num-rounds 3 --min-clients 2 --training-arm FROZEN_HEAD

# Join it as a client
python fl-runtime/client.py \
  --project-id <uuid> --server-address localhost:50051 --partition-id 0 \
  --model-type CIFAR_RESNET18 --training-arm FROZEN_HEAD
```

`FEDLEARN_PYTHON` overrides the interpreter every `run_*.sh` wrapper uses (default `python3`) — except the dev-only `run_clients.sh`, which calls a bare `python` from a sourced venv.

---

## Related Wikis

- [Framework Wiki](../framework/README.md) — the `fedlearn` library this layer imports; coordinator, strategies, the safetensors wire
- [Backend: Federated Orchestration](../backend/04_federated_orchestration.md) — the Java side of the spawn contract described in [01](01_entry_points.md)
- [Backend: Artifact Registry](../backend/07_artifact_registry.md) — where `fl_server.py`'s final model and eval card land
- [Client (Docker) Wiki](../client-docker/README.md) — packages this directory alongside `framework/`
- [Desktop Wiki](../desktop/README.md) — launches `client.py`, natively or in a container
- [`VERSIONS.md`](../VERSIONS.md) — per-unit release versions
