# Projection mechanism implementation plan

**Goal:** Execute the approved generated-only projection audit with bounded qualification and a frozen 136-cell study.

**Architecture:** Reuse LocalTrainer, ComposedStrategy, GDP calibration and owned-data ascent. Reproduction-only modules own generation, explicit cross-width noise coupling, source-bound one-attempt execution and independent analysis.

**Tech stack:** Existing clean Python 3.12 CPU environment, PyTorch, NumPy, SciPy and pytest; existing separate Matplotlib environment for aggregate plotting.

**Spec:** `docs/superpowers/specs/2026-10-27-projection-mechanism-design.md` (approved 27 October 2026).

## Global constraints

- Generated data only; no medical bundles, data paths, network calls or hardware services.
- D=512, train=4096, evaluation=8192, dataset seed=20261027, teacher=e0, labels Bernoulli(sigmoid(2*x0)).
- Twenty clients, twenty rounds, one local epoch, batch128, SGD LR=.03, zero head, C=.05, no decay/momentum.
- Main training seeds24001–24004; random bases20261101–20261105; retention0/.25/.75/1; GDP epsilon4, delta1e-5; full participation.
- Exact136 main cells,2720 rounds,54400 client fits; at most12 qualification cells and30minutes; main12hours; no automatic retries or output overwrite.
- Existing source/studies remain unchanged. Canonical pipeline runs with noise disabled only inside the study adapter; all adapters supply the same justified isotropic law.
- Budget-specific attack checks; no novelty, external-validity, production-privacy or population-inference claims.

## Review focus

- Equal seeds must not masquerade as active-coordinate coupling across shapes; test both rows/biases and inactive covariance.
- Cached transformed data must never depend on training seed or expose evaluation examples to fitting/attacks; test split and seed isolation.
- Missing/duplicate clients and failed stages cannot publish success; require exact cohort admission and no head on failure.
- A changed source/manifest or reused output must be refused before fitting; tamper tests must cover both.
- Attack search sees only coalition inputs and previous releases; test budget counters and fixed-state prefixes, not trajectory monotonicity.

All test commands use `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=framework/src:docs/artifacts/capstone/repro .temp/capstone-clean-reproduction-20261026/mac-env/bin/python -m pytest -q -o addopts=''` followed by named files.

### Task 1: Resolved design and public generated workload

**Files:** Create `docs/artifacts/capstone/repro/projection_study_design.py`, `projection_study_workload.py`, `test_projection_study_workload.py`.

**Interfaces:** `Cell` frozen dataclass; `cells()->list[Cell]`; `qualification_cells()->list[Cell]`; `seed_for(seed:int,*parts:object)->int`; `dataset(seed=20261027)->dict[str,Tensor]`; `basis(cell:Cell)->Tensor`; `features(data,cell)->dict`; `partitions(seed)->list[Tensor]`; `coalition(seed)->tuple[int,...]`.

- [ ] Write tests for exact block counts/uniqueness, deterministic separate streams, orthonormal teacher retention, projection-independent dataset, zero padding, complete partition and global RNG preservation. Run before creating modules. Expected: missing-module failure, followed by assertions failing against empty stubs if needed.
- [ ] Implement typed cells and SHA256-derived stream seeds; independent named PCG64 feature/label splits; existing QR basis; explicit controlled transforms.
- [ ] Run workload tests and existing projection checks. Expected: all pass. Commit focused files.

### Task 2: Canonical training and coupled-noise strategy

**Files:** Create `projection_study_strategy.py`, `test_projection_study_strategy.py`.

**Interfaces:** `initial_parameters(width)->OrderedDict`; `noise_draw(seed,round,width,std)->OrderedDict`; `StudyStrategy(parameters,seed,private)`; `local_fit(parameters,x,y,seed,round,client)->(params,count)`; `score(parameters,x)->Tensor`.

- [ ] Write tests for canonical no-noise parity, exact active/bias noise mapping, finite analytical score variance and independent covariance, cohort refusal, no global RNG mutation, real LocalTrainer padding parity and native Gaussian marginals. Expected: fail before implementation.
- [ ] Implement subclass override calling parent pipeline with apply_noise=False; reject incomplete/duplicate cohort, add one noise draw with namespaced generators. Canonical aggregate_fit retains state handling. Add read-only diagnostics before return.
- [ ] Use LocalTrainer with an explicitly seeded DataLoader and zero-head architecture constructed inside fork_rng. No alternate optimizer implementation.
- [ ] Run both new test files plus existing owned-data attack tests. Expected: all pass. Commit focused files.

### Task 3: Frozen runner, independent metrics and qualification

**Files:** Create `projection_study_metrics.py`, `run_projection_study.py`, `test_run_projection_study.py`.

**Interfaces:** `metrics(y,score)->dict`; `run_cell(cell,data,out,deadline)->dict`; `preflight(output,stage)->dict`; `execute(plan_dir,output,checkpoint)->dict`; CLI `preflight --stage qualification|main --output PATH`, `execute --plan PATH --output PATH --checkpoint COMMIT`.

- [ ] Write tests for tied AUROC vs explicit pairwise oracle, invalid scores/labels, prior-output refusal, source/manifest tampering, incomplete-round failure without head, attack budget/prefix integrity, and generated miniature smoke execution. Expected: fail before implementation.
- [ ] Add atomic JSON, generated array/head digests, saved score/head output, per-round diagnostics and explicit failure journals. Source binding includes all study modules/tests, canonical dependencies and approved spec. Reject missing/dirty checkpoint bindings. Main preflight requires qualified receipt and projected runtime below12hours.
- [ ] Freeze qualification allocation: all seed25001 and public basis20261111 (separate from main). Cells1–2 full/no-noise exact replay;3 full/private;4–5 random/no-noise and matching padded;6–7 random/private and matching padded;8–10 full/no-noise attacks20/80/160;11 random/private attack160;12 controlled eta0/private. Each has20rounds and20clients. Same public generated dataset; separately labelled qualification outcomes cannot become main fits.
- [ ] Retain best surrogate candidates and verify counters using existing ascent_upload. For fixed-state prefix checks, run short and long search on the same generated state before fitting; trajectory results are not compared monotonically.
- [ ] Run the complete reproduction-directory test suite with clean environment; document unavailable optional dependencies separately and run the generated projection suite without importing them. Expected: all available profile tests pass; actual test totals recorded, no claims about unavailable dependencies.
- [ ] Commit source/tests. Freeze qualification manifest at that checkpoint, execute once with30minute ceiling, validate replay/padding/metrics and extrapolate using maximum observed cell time×136. Expected: all12 qualification cells complete with tolerances and projection below12hours; otherwise stop main fitting and retain failure.

### Task 4: Measurement, independent analysis and report

**Files:** Create `analyze_projection_study.py`, `test_analyze_projection_study.py`, `plot_projection_study.py`; public aggregate/verification/report files under `docs/artifacts/capstone/`.

**Interfaces:** `analyze(run_dir)->dict` reconciles exact manifest, raw journals, saved arrays, metrics and separate seed/basis summaries; plot CLI consumes only public aggregate JSON.

- [ ] Write tests for missing cells, altered scores/heads/journals, wrong padding pair, baseline reuse and per-budget damage qualification. Expected: fail before implementation.
- [ ] Implement independent rank-sum AUROC using scipy ranks vs runner's sorted ROC integration; independently rescore heads from generated inputs. Plot aggregate-only retention, basis and attack-budget figures with existing Matplotlib environment.
- [ ] Commit/freeze all measurement and analysis sources. Prepare main136-cell manifest only after qualified receipt. Execute once within12hours. Expected: complete136 cells or honest partial status; never retries.
- [ ] Run independent analysis and actual affected-profile/full framework checks as appropriate; request one fresh review of new scientific code and evidence. Resolve material findings with regression tests; source changes after fitting require explicit interpretation or amendment rather than silent rebinding.
- [ ] Write report distinguishing analytical identities, measured outcomes, failed damage gates, single-dataset scope and unestablished novelty. Commit reviewed source/evidence locally. No push, merge or next project implied.
