# Capstone review handoff

Written 23 October 2026; claim/status corrections added 24–26 October 2026. This handoff describes
the MSc capstone built on FedLearn-Platform. Read this file first, then follow section 5.
Default to read-only review unless implementation is explicitly authorized. Section 8 lists the rules.

The phased research remediation is complete: see the [phase plan](../../superpowers/plans/2026-10-24-capstone-evidence-remediation.md)
and [selection-only seed reanalysis](selection-seed-reanalysis-2026-10-24.md). Historical measurements
remain pinned; interval targets and conclusions are corrected retrospectively.

**25 October checkpoint:** All five accepted research-remediation phases are complete.
Qualification matched native/ByzFL/prior-reference heads exactly; all126 pilot and196 finalist
cells completed. Projection's private-clean gain was+0.0192 AUROC, seedt95[+0.0110,+0.0273],
less than half the pilot point; its clean no-noise cost was0.0149. All31 final contrasts,
150 source files and3,920 final rounds passed separate audit. No holdout or official test was reopened.

**26 October checkpoint:** The approved CPU extension completed all 52 nonprivate selection-only cells across Mac/G14/Orin: 320 rounds and 13,180 acknowledged client fits. Every placement pair and historical anchor passed the prospective 1e-5 parameter tolerance (observed maxima 2.98e-8 and 3.73e-9 respectively). The five-round three-host jobs were 1.87–2.93 times slower end to end; steady three-client rounds improved, while 24/80-client rounds worsened. See the [results and limits](multihost-results-2026-10-26.md), [verification](multihost-verification-2026-10-26.json), [prospective protocol](multihost-protocol-2026-10-26.md) and [reproduction instructions](multihost-reproduction.md). Source/input/runtime checkpoint `8ac59c4` preceded fitting. GPU, DP and institutional deployment remain unqualified.

Start with the [integrated working report](capstone-final-report-v5-working.md),
[PDF](capstone-final-report-v5-working.pdf), [HTML](capstone-final-report-v5-working.html),
[completion audit](remediation-completion-audit-2026-10-25.md), and
[reproduction guide](remediation-reproduction-guide-2026-10-25.md).
The [phase4](phase-4-verification-2026-10-25.json) and [phase5](phase-5-verification-2026-10-25.json)
receipts define verification scope. Earlier v2/v3/v4 sources and reading copies remain preserved as historical versions.
Advisor format/approval, presentation and actual submission remain separate.
The contribution-first narrative is integrated in v4/v5; v5 adds the verified CPU extension. Proposed defense explanations must still be checked against source and privacy mathematics before reuse.

## 1. Where everything is

| What | Where |
| --- | --- |
| Working tree (use this one) | The existing isolated `capstone-dp-trimmed-mean` worktree; resolve its root with `git rev-parse --show-toplevel`. |
| Branch | `capstone`. It is **local only**: there is no `origin/capstone`, and nothing from this phase is pushed or merged. |
| Capstone docs | `docs/artifacts/capstone/`. `docs/` is gitignored, but these files are force-added and committed on `capstone`. |
| Reproduction harnesses (each with a `test_*.py`) | `docs/artifacts/capstone/repro/` |
| Platform code touched | `framework/src/fedlearn/` (library and simulator), `fl-runtime/` (scripts the backend runs), `backend/fl-platform-api/` (Spring Boot) |
| Private evidence (raw run records, label plans, caches) | `.temp/capstone-effusion-2026-10-22/` and `.temp/capstone-nih-2026-10-19/` (gitignored) |
| Paper working area (partial mirror) | The main checkout's ignored `research/` directory (gitignored). It holds the 5–6 October results and A22/A23 only. |
| Python | Current CPU-extension checks use `.temp/capstone-multihost-env/bin/python` in this worktree. It overlays SciPy 1.17.1 without changing the shared environment; the shared environment's SciPy binary failed to import. Historical studies retain their manifest-pinned environments. |
| Extra deps for the repro harnesses | `PYTHONPATH=.temp/capstone-medical-backbone-20261021/deps` |
| Repo-wide conventions | The main checkout's ignored `CLAUDE.md` (gitignored, so it is not inside the worktree). It covers architecture, test commands and conventions. |

## 2. What the capstone is

**Question:** in federated learning, how do differential privacy, robust aggregation and
attacking clients interact? What should a platform like FedLearn choose? The test bed is chest
X-ray classification on NIH ChestX-ray14, using frozen image features and a trainable linear head.

**Deadline:** 5 December 2026 (report, presentation and rehearsal). The advisor has not yet
confirmed the title or the format; a draft message is in `advisor-title-confirmation-draft.md`
and has not been sent.

### Phases

1. **Milestone 2 (early October):**
   - 945 exploratory runs. See `capstone-milestone-2-report.md`.
   - The in-process simulator (`framework/src/fedlearn/simulation/`) was shown to reproduce the
     earlier harness bit for bit.
2. **Phase 2, the NIH "any finding" study (about 19–21 October):**
   - **Data audit:** patient-separated splits had exact-duplicate image leakage; 111 images were
     quarantined (`nih-data-audit-2026-10-19.md`).
   - **The study:** a frozen 360-cell design over mean vs trimmed mean, central DP and attacks,
     with 7,200 rounds and no failures.
   - **Primary interaction:** +0.0023 AUROC [+0.0007, +0.0038]. Small, and conditional on the
     fixed cohorts (`nih-phase2-study-v1-report.md`).
   - **Working final report:** `capstone-final-report-v2-working.md` / `.pdf` describes **this
     study only**. The later Effusion work is not in it yet.
3. **Effusion phase (22–23 October):** the bulk of the recent work.
   - One condition (Effusion) with a **locked internal holdout**.
   - The pre-registered protocol is `effusion-baseline-protocol-v1.md`.
   - Every experiment is an amendment (A1–A23), committed before its run.
   - R0 is the primary representation: 512-wide frozen ResNet-18 ImageNet features, a linear
     head, and 80 whole-patient IID simulated clients.
   - Results are on the "selection" half of development data. **The holdout was opened exactly
     once (A10) and must never be reopened.**

## 3. Effusion results in one table (selection half unless marked)

| Amendment | Question | Headline | Note |
| --- | --- | --- | --- |
| Gate, A1–A3 | Does a medical encoder (R1, CheXpert DenseNet) beat R0? | 0.829 vs 0.822, interval includes 0, so **the gate failed**. Federated ≈ central (0.797 vs 0.800 on R1). | `effusion-gate-v1-result…`, `effusion-federated-v1-result…` |
| A4, A5 | Central DP cost; secure aggregation | ε 8/4/1 costs 0.035/0.088/0.232 on R1; a 2× wider head costs more. LightSecAgg is utility-equivalent but uploads 158× more. | `effusion-privacy-secure-v1-result…` |
| A6 | Representation decision | R0 becomes primary (user decision after the gate failed) | protocol |
| A7, A8 | Recipes | Clean 0.819 (≈ centralised). DP recipe (LR 0.03, clip 0.05): 0.743 at ε 8, 0.695 at ε 4. | `effusion-recipe…`, `effusion-dp-recipe…` |
| A9 | Attacks | Without DP, trimmed mean helps. Under DP, trimming is worse than the mean. | `effusion-attacks-r0-v1-result…`, heatmaps in `figures/` |
| A10 | **Holdout, opened once** | FedAvg 0.836 vs central 0.835. DP keeps sensitivity at about 0.91 but specificity falls from 0.55 to 0.39/0.30/0.14. | `effusion-holdout-v1-result…` |
| A11, A12 | Local DP | Near chance at ε ≤ 8 (needs about ε 128). Robust rules do not help, and sign flip goes below chance. | `effusion-local-dp…`, `effusion-local-robust…` |
| Literature check | Did we test the right things? | A ranked 10-item shortlist, only partially covered; several methods remain untested on Effusion | `literature-review-robust-private-fl-2026-10-22.md` |
| #1 adjacency | Is production DP's ε what it claims? | Correct for add/remove with a fixed cohort; about 2× weaker if read as replace-one. Now selectable. | `production-dp-adjacency-analysis-2026-10-22.md` |
| A13 | Exact Gaussian-DP accounting (q = 1) | Same ε with 13–24% less noise: +0.017 at ε 4 | `effusion-gdp-calibration…` |
| A14 | Attacks built for robust rules (Fang, min-max, IPM) | Without DP, Fang at 20% costs the mean 0.155 and the median 0.063. With nominal aggregate-input noise calibration the mean beats the tested trimming variants; oracle attacked cells lack the complete client-data sensitivity guarantee. | `effusion-strong-attacks…` |
| A15 | Fewer DP rounds | Lose with a random-init head | `effusion-dp-rounds…` |
| A16 | Canary privacy audit | **Underpowered** (the clean control failed). A18 was withdrawn before running (the estimator telescopes). | `effusion-canary-audit…` |
| A17 | Zero-initialised head | +0.019 to +0.034 at T 20; +0.15 to +0.20 at T 5. ε 4 goes from 0.695 to 0.738 cumulatively (exploratory). | `effusion-zero-head…` |
| A19 | Non-IID (Dirichlet) | At alpha 1 clean results are close to IID; Fang/mean declines more. At alpha 0.1, private AUROC drops0.038; additional privacy cost is about 0.03253; under Fang the mean falls to chance and the median keeps +0.129. | `effusion-noniid…` |
| A20, A21 | Filtering local-DP uploads | The norm filter catches noise-skipping attackers. Norm-matching attackers evade it (up to 0.54). The KS test is only partial. | `effusion-local-filter…`, `effusion-local-ks…` |
| A22 | Stronger encoders (DINOv2, MIMIC DenseNet) | Only DINOv2 passes, and barely (+0.013, lower bound +0.0007). **DP-cost difference unresolved**; width effects are not causally isolated. | `effusion-encoder-gate-v1-result-2026-10-23.md` |
| A23 | Distributed DP (Skellam over coded LightSecAgg) | At16 bits calibrated noise is close to central RDP; utility equivalence is unestablished and collusion is unqualified. **4 of 80 clip-skipping clients drive AUROC to 0.25.** Coded shares cost 4.4× plain, against 81× uncoded. | `effusion-distributed-dp-r0-v1-result-2026-10-23.md` |

The one-page synthesis is `fedlearn-practical-guidance.md`: each row is a decision, its evidence,
the platform action and its boundary.

## 4. Platform changes made (local commits on `capstone`, not pushed)

**DP correctness and defaults:**
- `5fad502`: strategies refuse privacy flags they cannot honour. Before this, a DP-enabled
  project on several strategies trained without DP.
- `969755a`: `DPFedAvg` and a DP budget planner (`fedlearn.privacy.planner`).
- `edad4aa` / `2d529be`: selectable DP adjacency (`--dp-adjacency`) and an exact GDP accountant
  (`fedlearn/privacy/gdp_accountant.py`, `--dp-accountant gdp`).
- `573fb2f` / `7bb28af`: `--head-init zero` for frozen-head arms. The backend
  (`FlServerManager.java`) now passes `--dp-accountant gdp` for every DP run, and
  `--head-init zero` for DP runs on FROZEN_HEAD/OVA_LP that are not continued runs.

**Simulator additions** (`framework/src/fedlearn/simulation/`):
- attacks: min-max, Fang trimmed-mean, canary clients;
- partitioning: patient-whole Dirichlet;
- local-DP filtering: norm filter and KS test, plus attackers that add the protocol's noise or
  match the honest norm;
- `privacy.mode: distributed`.

**New modules:**
- `fedlearn/privacy/distributed_skellam.py` and `fedlearn/security/lightsecagg_coded.py` (`4d4ff3d`);
- `fedlearn/privacy/audit.py` (canary estimator).

**Simulation Lab:** the Effusion workload is served by the Lab API/UI. A Lab run reproduces A7
exactly, and a real-browser pass is recorded. The full integrated simulator is **not** complete
(see the checklist).

**Last measured gates:**
- framework: 2,348 passed, 81.21% coverage (23 October);
- fl-runtime: 698 passed;
- backend: 1,218 tests, 0 failures, 84.17% line coverage.

## 5. Reading order

1. `docs/artifacts/capstone/fedlearn-practical-guidance.md` (15 min). Every conclusion with its
   evidence link and boundary. If you read one thing, read this.
2. `docs/artifacts/capstone/capstone-todo.md` (10 min). What is done and open, with links.
3. `docs/artifacts/capstone/effusion-baseline-protocol-v1.md`:
   - §1–9 (the frozen design);
   - then skim the amendments, each written before its run;
   - check that every result note matches its pre-registration.
4. `docs/artifacts/capstone/literature-review-robust-private-fl-2026-10-22.md`. Why the later
   experiments were chosen, and two self-corrections.
5. **The result notes**, in order of importance:
   - `effusion-holdout-v1-result-2026-10-22.md`
   - `effusion-distributed-dp-r0-v1-result-2026-10-23.md`
   - `effusion-zero-head-r0-v1-result-2026-10-22.md`
   - `effusion-gdp-calibration-r0-v1-result-2026-10-22.md`
   - `effusion-strong-attacks-r0-v1-result-2026-10-22.md`
   - `effusion-noniid-r0-v1-result-2026-10-22.md`
   - `effusion-local-ks-r0-v1-result-2026-10-22.md`
   - `effusion-encoder-gate-v1-result-2026-10-23.md`
   - `production-dp-adjacency-analysis-2026-10-22.md`
6. **For the older study and the report draft:**
   - `nih-data-audit-2026-10-19.md`
   - `nih-phase2-study-v1-report.md`
   - `capstone-final-report-v2-working.md`
   - `final-report-evidence-map.md`
7. **Code under the newest claims:**
   - `framework/src/fedlearn/privacy/gdp_accountant.py`
   - `framework/src/fedlearn/privacy/distributed_skellam.py`
   - `framework/src/fedlearn/security/lightsecagg_coded.py`
   - `framework/src/fedlearn/simulation/engine.py` (`run`, `round_hook`, `distributed_sum`)
   - `framework/src/fedlearn/server/composed_strategy.py`

## 6. Where a reviewer should push hardest

**Scope:**
- One workload (Effusion), mostly IID, five seeds, frozen features with a linear head.
- Results are on the selection half. Only A10 touched the holdout.

**Accounting:**
- The DP unit is the simulated client, under replace-one adjacency.
- Production uses add/remove adjacency with a fixed public cohort.
- Is the GDP formula μ = √T/z (q = 1) applied correctly?
- Is Skellam's Corollary 3.6 transcribed correctly? A test compares it with the exact Rényi
  divergence: `framework/tests/test_distributed_skellam.py`.

**Statistics:**
- Intervals are paired patient-cluster bootstraps. They do **not** resample DP noise.
- A23 found this matters when comparing independently noised arms, so seed-level t intervals
  were added after the run and labelled as such.
- Are any earlier cross-mechanism contrasts similarly overconfident? Candidates are A11/A12
  (local vs central) and A19.

**Exploratory, not pre-registered:**
- the A17 T5-vs-T20 comparisons;
- the A23 seed intervals;
- the zero-head default that went into the platform from an exploratory comparison. The zero
  head itself was pre-registered; T 5 was not.

**Known weak or negative results:**
- A16 audit underpowered; A18 withdrawn.
- One A21 cell (ε 8 label flip) got worse with KS (−0.056), unexplained.
- A22 passed by 0.0007.
- The A23 prediction "b 20 within ±0.005" was not met on bootstrap intervals.

**Threat model:**
- Attacks are non-adaptive to the clip and the filters.
- A23 secure aggregation is in-process, with analytic bytes, no dropouts, and key agreement assumed.

**Delivery gap:** the working final report does not yet include any Effusion result. Deciding
the report's narrative is open.

## 7. How to verify things yourself

Run from the worktree root.

```bash
ROOT="$(git rev-parse --show-toplevel)"
PY="$ROOT/.temp/capstone-multihost-env/bin/python"

# Framework tests (about 10 minutes for all; coverage is enforced)
(cd $ROOT/framework && PYTHONPATH=src $PY -m pytest -q -p no:cacheprovider)
(cd $ROOT/framework && PYTHONPATH=src $PY -m pytest -q --no-cov tests/test_distributed_skellam.py tests/test_lightsecagg_coded.py tests/test_gdp_accountant.py)

# fl-runtime tests
(cd $ROOT/fl-runtime && $PY -m pytest -q)

# Harness tests (synthetic data, no private inputs)
(cd $ROOT/docs/artifacts/capstone/repro && PYTHONPATH=$ROOT/.temp/capstone-medical-backbone-20261021/deps $PY -m pytest -q -p no:cacheprovider test_run_effusion_distributed_dp.py)

# Backend (needs Docker for Testcontainers)
(cd $ROOT/backend/fl-platform-api && SPRING_PROFILES_ACTIVE=test ./gradlew test jacocoTestCoverageVerification)
```

**Recomputing a reported number:**
- Each result note ends with the SHA-256 of its pinned `summary.json` under
  `.temp/capstone-effusion-2026-10-22/<study>/`.
- Hash the file, then read `contrasts` / `runs` from it. For example:
  `shasum -a 256 .temp/capstone-effusion-2026-10-22/distributed-dp-r0-v1/summary.json`
- Per-run JSON records sit beside it.
- **Do not re-run the full harnesses.** They write new output directories and take 5–90 minutes
  each. If one must be re-run, write to a new output path; the harnesses refuse an existing one.

## 8. Rules for work in this checkout

- **Read-only by default.** Do not commit, push, merge, rebase, stash or delete anything without
  Anurag's explicit say-so. Never use bare `git stash`; other worktrees share the stash.
- **Never read or evaluate the holdout** (`.temp/capstone-effusion-2026-10-22/holdout-lock-v1`
  manifests beyond what the harnesses load, or anything labelled test/holdout). It was opened
  once by design.
- **Keep private evidence within the authorized study boundary.** The CPU extension explicitly authorized assigned training arrays and logical-client metadata on G14/Orin. Selection and patient mappings remain on the Mac; no holdout/test payload or patient-ID CSV is transferred. Task-owned remote study directories are restricted to the training user. Do not paste patient IDs, credentials or raw private contents into external services. New transfers or studies need their own scope.
- **Write review findings to a new file,** for example `docs/artifacts/capstone/review-notes-<date>.md`.
  Cite `file:line` or the result note and its contrast key for each finding, and say whether you
  verified it by running something or only by reading.
- **Repo conventions apply:**
  - no authorship footers, signatures or co-author trailers unless explicitly requested;
  - do not mention the author's personal infrastructure in tracked files;
  - local HTML deliverables go in `docs/artifacts/`.
- **Do not trust older status lines over newer ones.** `capstone-todo.md` and this file are the
  current state. Older reports, including the working final report, are historical where they
  disagree.
