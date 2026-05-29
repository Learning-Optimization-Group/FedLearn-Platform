# C3 — ML Reproducibility & Experiment Lineage

**Date:** 2026-05-29 · **Branch:** `main-clean` (+ `origin/fed-mobile` for the C++ client) · **Scope:** Greenfield v2 design for reproducibility, deterministic-run guarantees, config/artifact capture, run lineage, and model registry — calibrated to a production-grade FL startup.

**Builds on:**
- `docs/audit/2026-05-27/03-framework.md` — cited as **[F]**. Specifically extends **[F]/M5** (the `np.random.seed` global-RNG mutation note), which I argue *understates* the problem.
- `docs/audit/2026-05-29/B3-observability.md` — cited as **[B3]**. B3 **selects MLflow as the experiment-tracking backend and explicitly defers reproducibility/lineage to this report** ([B3] §4 "Cross-ref C3 reproducibility", §6.1 channel 3, §10). I do **not** re-litigate MLflow-vs-W&B; I take MLflow self-hosted as decided and design *what gets logged into it* and *the determinism contract it records*.

> **Headline.** The single most important reproducibility finding is **C3-1: the DeComFL perturbation RNG is not device- or language-portable, and the platform actively mixes devices.** The server regenerates perturbation vector `z` on `cuda` whenever the host has a GPU (`decomfl_strategy.py:77,212`), while CPU clients and the mobile C++ core regenerate `z` on `cpu`. PyTorch **officially does not guarantee identical RNG output across CPU/GPU, releases, or platforms** ([PyTorch randomness notes](https://docs.pytorch.org/docs/stable/notes/randomness.html)). DeComFL's correctness depends on **every participant regenerating the *same* `z` from the same seed**, so this silently corrupts aggregation the moment a GPU server or a heterogeneous client fleet is involved. The mobile C++ code asserts the opposite ("bit-identical vectors") in a comment that is **unverified and contradicted by PyTorch's own documentation**. This is a *correctness* bug, not a polish item, and it is upstream of every reproducibility claim the product would make.

---

## 0. TL;DR for the synthesizer

1. **Determinism of the protocol is broken before we even get to "reproduce the paper."** The CPU/CUDA perturbation mismatch (C3-1) means a GPU-equipped server + CPU clients already disagree on `z`. The mobile cross-language parity claim (C3-2) is asserted, never tested. **Verdict on the seed-sync subsystem: rebuild** around an explicit, version-pinned, CPU-canonical, contract-tested RNG path.
2. **There is no "run" entity.** A run is an ephemeral `ProcessBuilder` child; the only persisted lineage is `Project` (`modelType`, `modelName`, `modelPath`, `optimizer`) and `RoundResult` (`serverRound`, `loss`, `accuracy`, `gpuUtilization`). **Seed, K, P, η, μ, dataset, framework/torch version, git commit, model hash — none are captured.** You cannot reproduce *your own* run, let alone a published one. **Verdict: rebuild (new `FlRun`/lineage layer).**
3. **Hyperparameters and the seed are hardcoded constants, not run inputs.** `seed=42`, `K=1`, `P=10`, `η=0.001`, `μ=0.001` live in `fl_server.py`'s sibling `config.py` (`config.py:121-133`); the backend never passes `--seed` and the user never sets it (`FlowerServerManager.java:167-181`). Every run is the same hidden config — which is *accidentally* reproducible but *deliberately* uncapturable and unchangeable.
4. **Can you reproduce a published DeComFL result today? No** — and not only because of seeds: model init isn't seeded on the canonical path, dataset partitioning seeds are uncaptured, no environment lockfile is recorded per run, and the GPU/CPU perturbation divergence breaks the algorithm itself. **Honest answer: not reproducible, and not *verifiably* deterministic across the heterogeneous fleet the product targets.**
5. **MLflow (decided in [B3]) is necessary but not sufficient.** MLflow gives you the params/metrics/artifact store and registry for $0. It does **not** give you the determinism *guarantee* — that requires fixing C3-1/C3-2 and a startup-time "determinism manifest" the framework computes and logs. This report specifies both.

---

## 1. The reproducibility question, decomposed

"Reproduce a published DeComFL result" decomposes into four independent requirements. The platform fails three of four today.

| Requirement | Needed for | Status | Evidence |
|---|---|---|---|
| **R1. Intra-federation RNG agreement** — server + every client regenerate the *identical* `z` from a shared seed | DeComFL *correctness* (not just reproducibility) | **BROKEN** | CPU/CUDA device split (§2.1); mobile C++ parity unverified (§2.2) |
| **R2. Deterministic single-run** — same inputs ⇒ same outputs on a fixed platform | Reproduce *your own* run | **PARTIAL / UNVERIFIED** | Seed plumbed to strategy (`fl_server.py:460`) but model init, dataloader, and CUDA nondeterminism unaddressed (§3) |
| **R3. Config & environment capture** — record everything needed to re-launch | Reproduce *later* / audit | **ABSENT** | No run entity; hyperparams hardcoded (§4) |
| **R4. Cross-environment portability** — same result on a different machine/version | Reproduce a *published* result elsewhere | **NOT GUARANTEED by PyTorch** | Official disclaimer (§2.3); must be *bounded*, not promised (§6) |

The product's research differentiator is DeComFL. R1 is therefore non-negotiable: it is the protocol, not a nice-to-have.

---

## 2. R1 — The cross-device / cross-language RNG mismatch (CRITICAL)

### 2.1 The live bug: server-on-GPU vs client-on-CPU regenerate different `z`

DeComFL never transmits the perturbation vector `z`. The client computes a scalar `g = (f(x+μz) − f(x))/μ` and sends only `g`; the server reconstructs `delta += g · z` by **regenerating the same `z` from the shared seed** (`decomfl_strategy.py:188,194`). If server and client produce different `z` for the same seed, every `g · z` term is multiplied against the wrong direction and the global update is garbage — silently, with no error.

The two sides generate `z` on **different devices**:

- **Server** (`decomfl_strategy.py:77`): `self.device = 'cuda' if torch.cuda.is_available() else 'cpu'`, then `torch.Generator(device=self.device)` + `torch.randn(..., device=self.device)` (`:212-218`). On any GPU host the server uses the **CUDA** Philox RNG.
- **Client estimator** (`zeroth_order.py:20,45-47`): `device='cpu'` default; `torch.Generator(device='cpu')` + `torch.randn(..., device='cpu')` → **CPU** Mersenne-Twister/Box-Muller path.
- **Docker client** (`client-docker/scripts/client.py:138-139,805`): `DEVICE = "cuda"` if a GPU is present, passed straight into `DeComFLClient(device=DEVICE)`. So a GPU docker client uses CUDA RNG, a CPU one uses CPU RNG — **two docker clients can disagree with each other.**

PyTorch is explicit that this does not work:

> *"Completely reproducible results are not guaranteed across PyTorch releases, individual commits, or different platforms. Furthermore, results may not be reproducible between CPU and GPU executions, even when using identical seeds."* — [PyTorch reproducibility notes](https://docs.pytorch.org/docs/stable/notes/randomness.html)

This is corroborated by [pytorch/pytorch#79496](https://github.com/pytorch/pytorch/issues/79496) (same seed, CPU vs CUDA → `tensor([-0.6014,-1.0122,-0.3023])` vs `tensor([-0.1029,1.6810,-0.2708])`) and [#158398](https://github.com/pytorch/pytorch/issues/158398) (CUDA `randn` differs *across machines*).

**Net:** the moment the platform runs the FL server on a GPU box (the obvious production choice) with any CPU client — or mixes GPU and CPU clients — DeComFL aggregation is mathematically wrong. It will *appear* to "train" (loss is computed locally and is real per-client) but the **global model the server builds is incoherent**, and results are non-reproducible by construction. This is strictly worse than **[F]/M5**, which only flagged two co-located servers clobbering the global numpy RNG; the device split breaks a *single* federation.

**Why it may not have surfaced yet:** the smoke tests (`run_local_test.py`, examples) run server+clients on the same CPU host, where both sides land on the CPU generator and accidentally agree. The bug is latent until the first GPU server or heterogeneous fleet — exactly the production topology.

### 2.2 The mobile C++ parity claim is asserted, not verified

`mobile_client/shared/src/ZerothOrderEstimator.cpp` (on `origin/fed-mobile`):

```cpp
// Matches Python: torch.Generator(device='cpu').manual_seed(seed) -> torch.randn(num_params, generator=gen)
// C++ torch::Generator uses the same Mersenne Twister as Python, producing identical outputs.
auto gen = torch::Generator();
gen.set_current_seed(seed);
return torch::randn({num_params}, gen);
```

and the header repeats it: *"produce bit-identical vectors to Python's torch.randn"* (`ZerothOrderEstimator.h`).

Problems with the claim:
1. **It is unverified.** There is no committed cross-language golden-vector test anywhere in `mobile_client/` or `framework/tests/`. The parity is a comment, not a guarantee.
2. **It is *only plausibly* true under tight conditions.** libtorch C++ and Python *do* share the same ATen RNG kernels **when they are the same PyTorch build/version on the same architecture**. The mobile path uses a **mobile libtorch build** (`scripts/build_libtorch_android.sh`, `build_libtorch_ios.sh`) cross-compiled for ARM, whose PyTorch version is pinned independently of the Python server's `torch`. PyTorch's own disclaimer covers "different platforms" and "individual commits" — an ARM mobile libtorch at version X is exactly the cross-platform/cross-version case where parity is **not guaranteed**. ([PyTorch notes](https://docs.pytorch.org/docs/stable/notes/randomness.html); the C++ forum thread on `torch::manual_seed` reaches no parity guarantee either — [discuss.pytorch.org](https://discuss.pytorch.org/t/unable-to-reproduce-the-same-result-with-torch-manual-seed/151348)).
3. **Seed-type asymmetry.** Python generates seeds in `[0, 2³¹−1)` via `np.random.randint(0, 2**31-1)` (`decomfl_strategy.py:107`), wire type is `repeated int32` (`fedlearn.proto:135`), C++ holds them as `int32_t` (`Utils.h: Seeds2D`) then widens to `int64_t` for `set_current_seed`/`generatePerturbation`. The *value* survives (positive, <2³¹), so this specific widening is safe — but it confirms nobody has reasoned about the seed domain as a contract; it works by luck of the chosen range.

**Verdict:** the mobile parity is *probably* achievable on CPU with a version-locked libtorch, but **it is currently a hope, not a tested invariant**, and the same-device caveat (§2.1) applies to mobile too — mobile is CPU, the server is likely GPU, so **mobile clients hit the C3-1 bug regardless of any C++/Python parity.**

### 2.3 What PyTorch actually guarantees (the bound we must design to)

From the official notes ([link](https://docs.pytorch.org/docs/stable/notes/randomness.html)): determinism is bounded to *"a specific platform, device, and PyTorch release."* It is **not** guaranteed across releases, commits, platforms, or CPU↔GPU. Therefore the *only* defensible design is:

- **Pin one canonical device for perturbation generation: CPU.** Generate `z` on CPU on **both** server and client, always, regardless of whether compute (the forward passes) runs on GPU. The RNG draw is cheap relative to the two forward passes; doing it on CPU costs almost nothing and removes the entire CUDA-divergence class.
- **Pin the PyTorch version across server + all client builds** (Python, docker, mobile libtorch) and record it per run.
- **Treat cross-version reproducibility as *bounded reproducibility*, not a promise.** Record the manifest (§5.2) so a mismatch is *detectable*, and refuse to mix versions in one federation (§6.3).

---

## 3. R2 — Deterministic single-run (partial / unverified)

Even setting aside R1, a single run is not demonstrably deterministic:

| Source of nondeterminism | Handled? | Evidence / fix |
|---|---|---|
| Strategy seed (perturbation schedule) | **Yes** | `seed=42` → `np.random.seed`/`torch.manual_seed` (`decomfl_strategy.py:82-83`); seeds generated via global numpy RNG (`:107`). |
| **Global RNG mutation** (two strategies/servers in one process clobber each other) | **No** | **[F]/M5** — use `np.random.Generator(np.random.PCG64(seed))`, not `np.random.seed`. Confirmed still present (`decomfl_strategy.py:82`). |
| **Initial model weights** | **No / unclear** | The canonical path loads weights from `--model-path` (`.npz`/exported `.pt`), but the *export* (`mobile_client/scripts/export_model.py`, framework examples) is **not seeded** — `SimpleCNN()` is constructed with default torch init under whatever ambient RNG state exists. Two exports ⇒ two different initial models ⇒ different runs. |
| **DataLoader shuffling / partitioning** | **No** | Dirichlet/data partition seeds are not captured or pinned on the platform path (cross-ref **[F]/H6** flwr_datasets, and C2 data-engineering). `data_iter = iter(self.train_loader)` (`decomfl_client.py:150`) order is whatever the loader's RNG yields. |
| **CUDA op nondeterminism** (atmospheric for the forward passes) | **No** | `torch.use_deterministic_algorithms(True)` is never called anywhere in `src/` (grep: 0 hits). For ZO the forward pass `f(x)` feeds `g`; nondeterministic CUDA kernels perturb `g`. |
| **`datetime.utcnow()` / wall-clock leakage into logs** | n/a to determinism | **[F]/M3** — cosmetic. |

**Conclusion:** R2 is *plausible on a single fixed CPU host with seed=42* (which is why local smoke tests pass) but is **not a guarantee** and **not verified by any test**. There is no `test_decomfl_determinism` that runs two identical configs and asserts identical loss curves.

---

## 4. R3 — Config, environment & artifact capture (ABSENT)

### 4.1 There is no run/experiment entity

A "run" is the `ProcessBuilder` child (`FlowerServerManager.java:160-185`); when it exits, everything in-process is gone ([B3] §1 confirms metrics are discarded at process end). Persisted lineage is only:

- **`Project`** (`Project.java`): `name`, `modelType`, `modelName`, `serverPort`, `modelPath`, `optimizer`, `orgId`, `status`, visibility/publish fields. **No seed, no hyperparameters, no dataset reference, no versions.**
- **`RoundResult`** (`RoundResult.java`): `serverRound`, `loss`, `accuracy`, `gpuUtilization`. **No per-round provenance** and (per [B3] §6.2) no communication-cost column either.

A Project is a *template*, not a *run*. Re-running a project overwrites nothing and records nothing distinguishing run #1 from run #2. **You cannot answer "what produced this loss curve?"**

### 4.2 Hyperparameters are hidden constants, not inputs

The backend launch command passes `--project-id`, `--model-path`, `--port`, `--strategy`, `--num-rounds`, `--model-type`, `--model-name`, optionally `--pretrain` (`FlowerServerManager.java:167-185`). It **never passes K, P, η, μ, or the seed.** Those come from a hardcoded `DeComFLConfig` keyed by dataset (`config.py:121-133`): `"default" = {K:1, P:10, η:0.001, μ:0.001, seed:42}`. Consequences:

- The user cannot change the seed or hyperparameters from the product at all.
- Two runs of the same project are byte-identical configs — *accidentally* reproducible, but the config is **invisible** to the lineage layer and **unchangeable** by the researcher whose product this is.
- The "reproducibility" that exists is the brittle kind: it survives only because nothing is configurable.

### 4.3 No environment / version capture

Nothing records `torch.__version__`, CUDA version, OS/arch, the framework git commit, the proto version (`fedlearn.v1`), the dataset hash, or the initial-model hash per run. `framework/pyproject.toml` has **no `[project]` dependency table** (**[F]/M1**), so there isn't even a lockfile to point at. The docker client logs `torch {__version__}` to stdout (`client.py:43`) — into the void, not into lineage.

---

## 5. v2 design

This is additive to [B3]'s observability stack (Prometheus/Loki/Tempo/MLflow). Reproducibility needs **three new substrates**: (a) a deterministic RNG contract, (b) a run-lineage data model, (c) MLflow logging + a model registry with lineage edges.

### 5.1 Deterministic RNG contract (fixes R1; the core of the rebuild)

**Design principles**
1. **CPU is the canonical RNG device, universally.** Generate every perturbation on CPU on server, docker client, and mobile — even when forward passes run on GPU. Move the generated `z` to the compute device only for the forward pass. This deletes the CPU/CUDA divergence class (§2.1) at ~zero cost (the RNG draw is negligible vs two forward passes; `z` is the model dimension, materialized either way).
2. **Pin PyTorch to one version across all participant builds** (Python server, docker client, mobile libtorch) and record it. A federation must refuse to mix versions (§6.3).
3. **Make the RNG path a tested invariant, not a comment.** Ship a committed **golden-vector test**: for fixed seeds `[..]` and dimensions `[..]`, assert `torch.randn(generator=cpu_gen)` produces a checked-in reference vector; run the *same* assertion in C++ (`gtest`) against the *same* reference (e.g., a committed `.npy`/JSON of the first N values + a hash). CI fails if Python or C++ drifts. This is the only thing that converts "bit-identical" from hope to guarantee.
4. **Counter-based RNG as the long-term hardening (flagged, not mandated).** The clean, provably portable solution is a counter-based generator (Philox/Threefry — [Random123](https://www.thesalmons.org/john/random123/), used by JAX) implemented identically in Python and C++, independent of any framework RNG. **Uncertainty:** this is a larger lift and changes the perturbation values vs the current torch path, so it must be coordinated with B1 (paper-alignment) before adoption. v2 floor = CPU-canonical torch RNG + golden test; counter-based = a later correctness-hardening epic.

**Concrete edits**
- `zeroth_order.py:45-47` and `decomfl_strategy.py:77,210-218`: force `torch.Generator(device='cpu')` and generate `z` on CPU; `.to(compute_device)` only where the forward pass needs it.
- Mobile `ZerothOrderEstimator.cpp`: keep CPU `torch::Generator`, **delete the unverified "bit-identical" comment**, and add it to the golden-vector test instead.
- Add `framework/tests/test_rng_parity.py` + `mobile_client/shared/tests/rng_parity_test.cpp` sharing one committed reference fixture.

### 5.2 Determinism manifest (computed once per run, logged to lineage + MLflow)

At server startup the framework computes and emits a manifest; the backend persists it on the run row and the framework logs it as MLflow tags/params:

```json
{
  "framework_git_sha": "…",
  "proto_version": "fedlearn.v1",
  "torch_version": "2.5.1",
  "torch_cuda_version": "12.4 | null",
  "rng_device": "cpu",
  "rng_engine": "torch.Generator(cpu)",
  "use_deterministic_algorithms": true,
  "seed": 42,
  "decomfl": {"K": 1, "P": 10, "eta": 0.001, "mu": 0.001},
  "initial_model_sha256": "…",
  "dataset_ref": {"name": "ecg", "partition_seed": 0, "dirichlet_alpha": 0.5, "split_sha256": "…"},
  "platform": {"os": "linux", "arch": "x86_64"}
}
```

`initial_model_sha256` and `dataset_ref.split_sha256` are the lineage anchors that let you assert "this run started from *that* model and *that* data split." Both are computable today (hash the `.npz`/`.pt` and the partition index arrays — note replacing the pickle cache per **[F]/C2** makes the split hashable cleanly).

### 5.3 Run-lineage data model (new — fixes R3)

Introduce an explicit `FlRun` aggregate (Flyway `V6` per the schema-ownership invariant; coordinate the migration index with [B3]'s proposed `V6` comm-cost migration — pick distinct version numbers):

```
fl_runs
  id UUID PK
  project_id UUID FK -> projects
  status, started_at, ended_at
  strategy                       -- 'DeComFL' | 'FedAvg'
  seed BIGINT
  k_local_steps INT, p_perturbations INT, eta DOUBLE, mu DOUBLE
  num_rounds INT, min_clients INT
  torch_version, framework_git_sha, proto_version  -- determinism manifest core
  initial_model_sha256, dataset_ref JSONB
  mlflow_run_id                  -- link-out to MLflow
  created_by_user_id
```

Then **re-point `RoundResult` at `fl_run_id`, not `project_id`** (or add `fl_run_id` and keep `project_id` for back-compat), so each loss curve belongs to a *run*, not a template. This is the difference between "compare run A vs B" being possible vs impossible. The user-facing hyperparameters (K/P/η/μ/seed) become **run inputs** flowing `frontend → POST /start → FlowerServerManager → --seed/--k/--p/… → fl_server.py` instead of hidden constants (deleting the `config.py` hardcode, §4.2).

### 5.4 MLflow logging + model registry with lineage (extends [B3] §6.1 channel 3)

[B3] decided MLflow self-hosted. C3 specifies the *lineage* it must carry:

- **At run start:** `mlflow.set_experiment(project_id)`, start a run, `mlflow.log_params(determinism_manifest)`, `mlflow.set_tags({framework_git_sha, torch_version, dataset_split_sha256, initial_model_sha256, mlflow_run_id ↔ fl_runs.mlflow_run_id})`.
- **Per round:** `mlflow.log_metrics({loss, accuracy, uplink_bytes, scalars_transmitted}, step=round)` — same numbers as the STOMP feed and the comm-cost KPI [B3] §6.2 (single source of truth).
- **At run end:** log the final aggregated model as an MLflow **artifact** and register it in the **Model Registry** as a new version, with the run's lineage tags attached. Registry version → `fl_runs.id` → determinism manifest is the full provenance chain: *"model v7 came from run R, seed 42, torch 2.5.1, git abc123, data split hash def…".* This is the [B3]-cited reproducibility substrate, made concrete.
- **Healthcare/data-residency note:** MLflow self-hosted keeps artifacts in-VPC (S3/MinIO), which matters for the pneumonia/clinical federations the product targets — consistent with [B3]'s rationale for MLflow over W&B-managed.

**Best-practice alignment** (industry consensus on what a run must log to be reproducible): code version (git), data version (hash/DVC), seed, environment, params, metrics, and the model artifact linked back to all of it — see [MLflow tracking docs](https://mlflow.org/docs/latest/ml/tracking/), [MLflow dataset tracking](https://mlflow.org/docs/latest/ml/dataset/), and the 2025 practice summaries ([ML Journey](https://mljourney.com/mlflow-experiment-tracking-best-practices/), [KDnuggets](https://www.kdnuggets.com/mlflow-mastery-a-complete-guide-to-experiment-tracking-and-model-management)). The §5.2 manifest is exactly this set, specialized for DeComFL.

### 5.5 Artifact storage with lineage

- **Initial models & datasets:** content-addressed (sha256) in object storage (MinIO/S3); `fl_runs` references hashes, not paths. Today `modelPath` is a bare filesystem string (`Project.java:34`) with no integrity or versioning.
- **Final models:** MLflow Model Registry (above) is the system of record; the registry version is the immutable artifact ID surfaced to users.
- **Per-run config snapshot:** persist the launch command + manifest JSON as an MLflow artifact, so "re-run exactly this" is a button, not archaeology.

---

## 6. Honest answer to "can you reproduce a published DeComFL result?"

**No, not today, and v2 must scope the promise carefully.**

- **6.1 Same protocol, same paper math?** The DeComFL/ZerothOrder implementation matches Algorithm 3/4 structurally ([F] confirmed; mem obs: "implement Algorithm 4 correctly; averaging confirmed in reference cezo_fl"). So the *algorithm* is right.
- **6.2 Same *numbers* as the paper?** Not achievable as a guarantee. PyTorch's own disclaimer means exact loss curves won't match the paper's environment ([PyTorch notes](https://docs.pytorch.org/docs/stable/notes/randomness.html)). The defensible product claim is **"reproducible within a pinned environment (our torch version, CPU-canonical RNG, recorded seed/data hash)"**, plus *statistical* reproduction of the paper's trends (accuracy within CI over seeds), **not** bit-exact replication of a third-party run.
- **6.3 Within our own pinned federation?** Achievable *after* C3-1/C3-2 are fixed: CPU-canonical RNG (§5.1), version-pinned + manifest-recorded (§5.2), seed/data/model hashed in `fl_runs` (§5.3). Add a **federation version-compatibility gate**: the server rejects a client whose `torch_version`/`proto_version`/`rng_engine` disagree with the run manifest (carry it in gRPC metadata — the same channel [B3] §5 uses for `traceparent`). Mixing versions silently is how reproducibility dies in production.

**Product framing recommendation:** sell "**deterministic, auditable runs within a pinned environment**" (true and valuable for regulated/healthcare FL), not "bit-exact reproduction of arbitrary published results" (false for any PyTorch system).

---

## 7. Decision table (verdicts)

| Module / subsystem | Verdict | One-line rationale |
|---|---|---|
| **Perturbation seed-sync / RNG path** (`zeroth_order.py`, `decomfl_strategy.py`, mobile `ZerothOrderEstimator.cpp`) | **rebuild** | CPU/CUDA divergence breaks DeComFL aggregation on any GPU server or mixed fleet; mobile "bit-identical" is unverified. Make CPU-canonical + golden-vector tested. |
| **Run/experiment lineage** (`Project`, `RoundResult` as the only persistence) | **rebuild** | No run entity, no seed/hyperparameter/version/hash capture; cannot reproduce your own run. New `FlRun` + manifest. |
| **Hyperparameter/seed config flow** (`config.py` hardcode + `FlowerServerManager` args) | **refactor** | Hyperparams/seed are hidden constants; lift them to run inputs persisted in `fl_runs` and passed via CLI. |
| **DeComFL/ZerothOrder algorithm core** | **salvage** | Algorithm 3/4 structurally correct ([F]); only the RNG *device* and the missing global-RNG generator ([F]/M5) need fixing, not the math. |
| **Initial-model export + dataset partition seeding** (`export_model.py`, data loaders) | **refactor** | Unseeded model init + uncaptured partition seeds; seed them and hash the artifacts into the manifest (cross-ref C2). |
| **Determinism verification (tests)** | **rebuild (build new)** | Zero determinism/parity tests exist; add `test_decomfl_determinism` + Python↔C++ golden-vector parity in CI. |
| **Experiment-tracking backend choice (MLflow)** | **salvage** | Decided in [B3]; correct. C3 only specifies the lineage payload + registry edges. |
| **Model registry / artifact lineage** | **rebuild (build new)** | None exists; MLflow Model Registry + content-addressed artifacts wired to `fl_runs`. |
| **Mobile cross-language RNG parity comment/claim** | **kill** | Delete the unverifiable "bit-identical" assertion; replace with the enforced golden-vector test. |

---

## 8. Prioritized recommendations (calibrated to startup runway)

**P0 — Correctness (the protocol is wrong without these; days).**
1. **CPU-canonical perturbation RNG** on server + all clients (§5.1 edits to `zeroth_order.py`, `decomfl_strategy.py`, mobile). This is the single highest-impact fix in this report — it makes DeComFL *correct* on GPU servers and heterogeneous fleets.
2. **Fix [F]/M5**: `np.random.Generator(np.random.PCG64(seed))` instead of global `np.random.seed` (`decomfl_strategy.py:82`).
3. **Golden-vector RNG parity test**, Python + C++, sharing one committed fixture; wire into CI. Delete the mobile "bit-identical" comment.
4. **Pin PyTorch version** across server/docker/mobile builds; add `framework/pyproject.toml [project]` deps + lockfile ([F]/M1).

**P1 — Lineage you can ship a product on (1–2 weeks).**
5. **`FlRun` entity + `V_n` Flyway migration**; re-point `RoundResult` at the run; persist the §5.2 determinism manifest. (Coordinate version number with [B3]'s `V6`.)
6. **Lift hyperparameters + seed to run inputs** (frontend → `/start` → CLI → `fl_server.py`), deleting the `config.py` hardcode.
7. **Compute + log the determinism manifest** (torch version, git sha, model/dataset hashes) at run start.

**P2 — MLflow lineage + registry (1 week; on top of [B3]'s MLflow stand-up).**
8. `log_params`(manifest) + `log_metrics`(per round) + final-model artifact + **Model Registry** version tagged with `fl_runs.id` and all lineage hashes.
9. Content-address initial models & dataset splits in object storage; reference hashes from `fl_runs`.
10. **Federation version-compatibility gate**: server rejects clients whose torch/proto/rng manifest disagrees (gRPC metadata).

**P3 — Determinism hardening (research-grade; later).**
11. `torch.use_deterministic_algorithms(True)` + seeded dataloaders on the canonical path; add `test_decomfl_determinism` asserting identical loss curves across two runs.
12. **Counter-based RNG (Philox/Threefry, Random123-style)** as the provably portable cross-language perturbation source — coordinate with B1 (changes perturbation values vs current torch path).

---

## 9. Risks & uncertainties (flagged explicitly)

| # | Risk / uncertainty | Note |
|---|---|---|
| 1 | **CPU-canonical RNG still isn't guaranteed bit-identical across PyTorch *versions*** | True even on CPU ([PyTorch notes](https://docs.pytorch.org/docs/stable/notes/randomness.html)). Mitigated by version-pinning + the compatibility gate (§6.3), not eliminated. The counter-based RNG (P3.12) is the only true cross-version guarantee. |
| 2 | **Mobile libtorch CPU RNG parity with Python is *probable* but untested** | I did not (cannot here) run the cross-build comparison. The golden-vector test (P0.3) is what turns probability into a CI-enforced fact. Do not ship the mobile claim until that test is green. |
| 3 | **Moving `z` generation to CPU on a GPU server adds a host↔device copy per perturbation** | The copy is O(model dim) and already paid (the forward pass needs params on device anyway). Negligible vs two forward passes per perturbation; verify on Jetson/M4 if K·P is large. |
| 4 | **Changing `RoundResult`'s FK from project to run is a schema migration with data-migration cost** | Existing rows have no run; backfill a synthetic run per project or add `fl_run_id` nullable and dual-write during transition. Flyway-owned per invariant. |
| 5 | **Counter-based RNG changes the actual perturbation values** | Not paper-aligned without B1 review; explicitly a later epic, not the v2 floor. |
| 6 | **Determinism vs throughput** | `use_deterministic_algorithms(True)` can disable fast kernels; offer it as a per-run "reproducible mode" flag rather than always-on, recorded in the manifest. |

---

## 10. Sources

- Prior audits: `docs/audit/2026-05-27/03-framework.md` (**[F]**), `docs/audit/2026-05-29/B3-observability.md` (**[B3]**).
- PyTorch reproducibility guarantee (CPU/GPU, version, platform): [PyTorch — Reproducibility notes](https://docs.pytorch.org/docs/stable/notes/randomness.html).
- Same-seed CPU vs CUDA divergence: [pytorch/pytorch#79496](https://github.com/pytorch/pytorch/issues/79496); cross-machine CUDA `randn`: [pytorch/pytorch#158398](https://github.com/pytorch/pytorch/issues/158398).
- libtorch C++ `manual_seed` (no parity guarantee reached): [discuss.pytorch.org thread](https://discuss.pytorch.org/t/unable-to-reproduce-the-same-result-with-torch-manual-seed/151348).
- Counter-based RNG (portable cross-language): [Random123 / Philox](https://www.thesalmons.org/john/random123/).
- MLflow tracking & dataset/lineage: [MLflow tracking](https://mlflow.org/docs/latest/ml/tracking/), [MLflow dataset tracking](https://mlflow.org/docs/latest/ml/dataset/).
- ML reproducibility best practices (log code/data/seed/env, registry↔commit lineage): [ML Journey](https://mljourney.com/mlflow-experiment-tracking-best-practices/), [KDnuggets](https://www.kdnuggets.com/mlflow-mastery-a-complete-guide-to-experiment-tracking-and-model-management).
- Codebase evidence cited inline by `file:line` against `main-clean` and `origin/fed-mobile`.
