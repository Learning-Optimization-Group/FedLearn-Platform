# Capstone plan — current as of 19 October 2026

This is the current schedule and completion checklist. Earlier milestone decks,
reports and design notes remain historical evidence. Confirmatory phase 2 has not
been pre-registered or run. The NIH validation pilot and Milestone 2 grids are
exploratory; pilot readiness issues below must be resolved before test evaluation.

The [final-report evidence map](../../docs/artifacts/capstone/final-report-evidence-map.md)
tracks which claims are currently supported, the remaining study/report/demo
gates and an eight-page working outline. It is not a finished final report.

## Objective and boundaries

Study how client-level differential privacy, robust aggregation, adversarial
clients and trainable head size interact in a reproducible federated simulation.
The current dataset is NIH ChestX-ray14 with **0 = No Finding** and **1 = any of the
listed findings**. These are report-derived labels, not verified clinical diagnoses.
Pneumonia-only results from the mobile project answer a different task.

The simulation engine was implemented at `8a9b89d`. Its 1,000-client execution check
does not establish performance or scientific results on NIH. Mobile/XGBoost PR #14
is merged into `main` at `3012d1e`, with post-merge CI green. Release/TLS work stays
parked for the capstone. The dedicated `capstone` branch preserves the original
research commits and integrates this verified main baseline at `660576a`.

## Verified starting point

- The presented Milestone 2 HTML deck was last edited on 7 October. Its older
  PowerPoint and the first generated HTML deck are not the current presentation.
- The three retained grids contain 420 + 480 + 45 = 945 run JSON files, excluding
  their index files. The research note explicitly calls them exploratory.
- The original 18 capstone commits, including the consolidated plan/design, remain
  preserved on `worktree-capstone-dp-trimmed-mean` as well as the new `capstone`
  branch. Capstone branch publication is still outstanding. Its code, ignored
  results/notes, presentation assets, experiment configurations
  and original Kermany feature cache are included in a private safety backup.
  Git publication alone does not include ignored evidence.
- NIH metadata has 112,120 unique image IDs: 60,361 No Finding and 51,759 with
  findings. Official train/validation has 86,524 images from 28,008 patients;
  official test has 25,596 images from 2,797 patients. The lists cover metadata,
  with zero image-ID or patient overlap. All images decode, but the later full-content audit found exact RGB overlap
  across splits; the original cohort fails content qualification. Metadata checks
  alone do not establish content integrity.
- The legacy engine workload partitions image rows. A separate NIH workload now
  loads a pinned verified cache, requires explicit evaluation selection and
  allocates whole patients. NIH configurations must explicitly preserve prevalence;
  the old Kermany defaults, config hashes and reproduction anchors remain unchanged.

## Schedule and completion gates

| When | Priority | Required output |
| --- | --- | --- |
| 18–19 October | NIH data and pilot | Explicit labels, validated patient-disjoint manifests, frozen ResNet-18 feature cache with provenance, patient-preserving simulated clients, clean validation pilot |
| Before confirmatory runs | Phase-2 protocol | Versioned hypotheses, primary endpoint, exact configurations/seeds, tuning/test separation, run budget, inference and failure-handling rules; approved and frozen before examining confirmatory results |
| 20 October–2 November | Experiments | Complete the core matched NIH study first, then separately budgeted scale/head-size studies; verified records, failures and analysis |
| 3–16 November | Simulation Lab | Design review first; minimum GUI loads a resolved config, starts/stops a run, streams results and exports the record |
| 17–30 November | Final deliverables | Report, current presentation and reproducible demonstration; keep writing alongside experiments |
| 1–5 December | Buffer | Reproduction checks, corrections and submission; 5 December is the stated deadline |

October 20 is a target start, not permission to call a retrospective analysis
pre-registered. If the data/pilot/protocol gates are incomplete, move confirmatory
runs rather than mislabeling their status. Advisor title approval remains open;
the [title/scope message](../../docs/artifacts/capstone/advisor-title-confirmation-draft.md)
is drafted for review and has not been sent. The working title is not approved.

## NIH adapter and historical integration checks

The [NIH workload design](../../docs/superpowers/specs/2026-10-18-capstone-nih-workload-design.md)
defines the proposed implementation boundary and acceptance checks. Preserve the
old Kermany behavior and exact reproduction anchors. Do not alter the mobile
pneumonia recipe, reuse its class mapping, overwrite an old feature cache or inspect
official-test performance to select settings.

The [first implementation plan](../../docs/superpowers/plans/2026-10-19-capstone-nih-manifests.md)
is implemented and verified. It covers source validation, patient-separated manifests,
whole-patient client allocation and independent verification. Feature extraction
and engine integration are implemented and qualified in later slices.
The original failed content audit remains preserved; only the qualified curated
cohort enters the new cache and engine.

After integrating main, 60 targeted workload/config/engine tests pass, with the
synthetic scale test deselected; another 40 reproduction/strategy/behavior/privacy
tests pass. A fresh bounded comparison reproduces all eight stored cells from
repetition 0 of `main_n80_eps1` exactly under torch 2.12.0. The original 160-cell
torch 2.9.1 verification remains historical evidence; this new eight-cell check
does not replace or expand its scope. That integrated suite passed 1,787 tests,
14 deselected, 81.21% coverage. The latest participation-accounting suite passes
1,837 tests, 14 deselected, 81.31% coverage, with unhandled thread exceptions
treated as errors. Tests alone do not establish NIH learning or content integrity.

The no-DP/no-attack NIH validation pilot and the separate 1,000-client/two-round
execution check are complete. V1 found useful ranking but poor calibration and
probability-saturation AUROC error. Scoring is now versioned and corrected;
the separately frozen v3 schedule probe establishes a stable eighty-client
baseline across three seeds. The attack-knowledge/accounting audit, larger-head
cost and prospective confirmatory protocol still precede the privacy/attack
matrix; successful baseline execution alone does not complete the study.

## NIH manifest qualification — 19 October

The source/manifest foundation is implemented. Preparation reads the original
dataset without moving images and assigns the No Finding/any-finding label
explicitly. A separate verification invocation reconstructs every declared split
and client assignment; a separate CSV/metadata audit checks labels, complete row
coverage, official-test membership and patient ownership against original inputs.

| Split | Images | Patients | Finding images | Finding prevalence |
| --- | ---: | ---: | ---: | ---: |
| Training | 68,758 | 22,407 | 28,554 | 41.53% |
| Validation | 17,766 | 5,601 | 7,470 | 42.05% |
| Official test | 25,596 | 2,797 | 15,735 | 61.47% |

Seed 42, 20% patient validation within patient-level any-finding strata, and ten
whole-patient IID clients. Independent splits have zero patient/image overlap;
clients are pairwise disjoint and their union is exactly training. Clients have
2,240–2,241 patients but 6,492–7,244 images. Equal patient counts do not mean equal
sample counts. The preparation records actual counts and source/manifest/assignment
hashes; the initial assignment SHA-256 is
`1f2979922e4ae5af7130c007603d9546d040015c8a516bde0b3b75b728bb8593`.

**Practical consequence:** the majority class reverses between training/validation
and official test. Do not silently balance evaluation or choose a new threshold on
official test. Preserve prevalence, select thresholds on validation, and report
discrimination, balanced accuracy, confusion counts and calibration. These are
metadata findings, not evidence of classifier performance or a complete explanation
of distribution shift. Test prevalence was inspected during this data audit,
before the phase-two protocol; disclose that fact in the protocol.

**Content audit result:** all 112,120 images decode, but three exact RGB-content
groups cross independent splits. The original preparation report correctly remains
`image_content_verified: false`; the separate completed audit is `qualified: false`.
Across the complete cohort there are 25 RGB duplicate groups, eight crossing
declared patient IDs and six with contradictory labels (categories overlap).
A separate direct pixel/metadata check confirmed every reported group. See the
[NIH audit](../../docs/artifacts/capstone/nih-data-audit-2026-10-19.md) for evidence
and the [quarantine design](../../docs/superpowers/specs/2026-10-19-capstone-nih-cohort-curation-design.md)
for the next gate. The separate curated development cohort is now prepared and independently
reverified: it excludes 111 images from nine non-test patients and retains the
complete official test set.
No source files or original manifests have been changed. Do not extract features
from the failed original cohort or label it content-separated.

From the capstone worktree, set `NIH_DATA_ROOT` to the existing dataset directory and
`NIH_PREPARATION` to a fresh directory outside that dataset, then run:

```bash
export CAPSTONE_PYTHON="$(git rev-parse --git-common-dir)/../.venv/bin/python"
export PYTHONPATH=framework/src:framework
"$CAPSTONE_PYTHON" -m benchmarks.nih_data --data-root "$NIH_DATA_ROOT" \
  --output "$NIH_PREPARATION" --clients 10 --validation-fraction 0.2 --seed 42
"$CAPSTONE_PYTHON" -m benchmarks.nih_data --data-root "$NIH_DATA_ROOT" \
  --output "$NIH_PREPARATION" --verify-only
```

Existing outputs are never overwritten. An interrupted output without a completed
report refuses verification; choose a fresh output path. Keep real-data manifests
and local qualification reports outside git.

The retrospective [Milestone 2 report](../../docs/artifacts/capstone/capstone-milestone-2-report.md) now records the exploratory methods, source-linked tables, anomalies, reproduction scope and practical phase-2 questions. It does not reclassify historical experiments as confirmatory.

## Curated NIH development cohort — 19 October

The explicit `nih-development-quarantine-v1` policy is implemented in
`framework/benchmarks/nih_cohort.py`. It quarantines whole non-test patients
implicated in cross-patient or contradictory-label exact RGB groups. It does not
redraw validation, relabel images or modify official test. The failed original
audit remains preserved; this is a new cohort with a narrower development-content
gate, not a replacement claim that all original labels are clean.

| Split | Selected images | Patients | Finding images | Excluded images |
| --- | ---: | ---: | ---: | ---: |
| Training | 68,650 | 22,399 | 28,515 | 108 |
| Validation | 17,763 | 5,600 | 7,470 | 3 |
| Official test | 25,596 | 2,797 | 15,735 | 0 |

All 111 excluded images (39 findings, 72 No Finding) and their patient/group reasons
are retained privately. Separate full reconstruction re-read all source images and
completed in 252.69 seconds with exit zero. An independent CSV/set audit checked
exact retained rows, every exclusion/reason, client union/whole-patient ownership,
zero independent-split RGB overlap and unchanged official-test bytes. A separate
original-metadata check covered all 112,120 selected/quarantined image labels and
patient IDs. Three ambiguous exact RGB groups remain inside official test; they
are disclosed, not removed or corrected.

Cohort report SHA-256:
`287dca8359300f08cea79098d367793a73e43205bcfe3213e9d22115e3bb7267`.
Training client assignment SHA-256:
`be4cd7d1352726f325a04590c4b2f4440a5c3f5408cddaaf20160b9900ab787d`.

Set `NIH_CURATED_COHORT` to a fresh output directory and `NIH_IMAGE_AUDIT` to the
completed original audit. Keep `NIH_PREPARATION` pointing to the original manifests:

```bash
"$CAPSTONE_PYTHON" -m benchmarks.nih_cohort --data-root "$NIH_DATA_ROOT" \
  --original-preparation "$NIH_PREPARATION" --image-audit "$NIH_IMAGE_AUDIT" \
  --output "$NIH_CURATED_COHORT" --audit-workers 4
"$CAPSTONE_PYTHON" -m benchmarks.nih_cohort --data-root "$NIH_DATA_ROOT" \
  --original-preparation "$NIH_PREPARATION" --image-audit "$NIH_IMAGE_AUDIT" \
  --output "$NIH_CURATED_COHORT" --audit-workers 4 --verify-only
```

Both commands require the full original source tree; verification rechecks the
original audit and reconstructs the policy. There is no ignore/force flag.
Frozen feature extraction now consumes this explicit cohort and keeps quarantined
development rows out. Explicit engine split selection is now implemented and tested;
ranking information is now demonstrated, while stable optimization remains open.
See the [feature plan](../../docs/superpowers/plans/2026-10-19-capstone-nih-frozen-features.md)
and the qualification below.

A private second-device checkpoint through `bf1c9cc`, including the curated
cohort implementation, 98 NIH evidence files/source-metadata files and the retained
1,021-file historical capstone archive, has been hash-verified after transfer.
Raw NIH image files are not included. This is separate from branch publication.

## Qualified NIH frozen features — 19 October

The source-bound pretrained ResNet-18 cache is complete: **68,650 training, 17,763
validation and 25,596 official-test vectors**, each 512 float32 values. Preparation
freshly reverified the full audit/cohort and exited zero after 641.24 seconds on
MPS (batch 64, four image/audit workers, four CPU threads). The cache is about
228 MiB; the OS child maximum RSS excludes GPU allocation and whole-tree peak memory.

Independent reload checked every retained label/identity against original metadata,
unchanged official-test membership, all quarantined patients absent and disjoint
patient ownership. A separately constructed model and direct MPS reference matched
all 192 sampled feature vectors exactly within `rtol=1e-5, atol=1e-6`; no cross-device
equality is claimed. **No official-test classifier performance has been inspected.**

Feature report SHA-256:
`6c9e6969c6fd762156ef2246323fbdbc5e9cb272c9910a8b1c9b691ab33783fd`.
The [qualification note](../../docs/artifacts/capstone/nih-frozen-features-2026-10-19.md)
records the actual model state, versions, integrity checks, measured cost and limits.

For a fresh cache directory outside all source/evidence inputs:

```bash
"$CAPSTONE_PYTHON" -m benchmarks.prepare_nih_features --data-root "$NIH_DATA_ROOT" \
  --original-preparation "$NIH_PREPARATION" --image-audit "$NIH_IMAGE_AUDIT" \
  --curated-cohort "$NIH_CURATED_COHORT" --output "$NIH_FEATURE_CACHE" \
  --device mps --batch-size 64 --image-workers 4 --audit-workers 4
```

Choose an explicitly available device; no silent fallback. Real caches/manifests
stay private and ignored. The [next implementation plan](../../docs/superpowers/plans/2026-10-19-capstone-nih-engine-integration.md)
requires an explicit evaluation split and pinned cache digest, whole-patient
clients, unchanged legacy reproduction, and a bounded validation-only learning pilot.

The private checkpoint through `98ba88a` now includes the full qualified cache and
95 hash-verified evidence/source-metadata files, plus a verified complete Git bundle.
The earlier 1,021-file historical evidence archive remains separately preserved.
Raw NIH image files are not included. Later engine integration is newer than that
checkpoint and will be included in the next one.

## NIH engine adapter — 19 October

The engine now selects `NihFeatureHeadWorkload` for `dataset: "nih"`. It requires
`evaluation_split: "validation"` or `"test"`, the trusted `features_report_sha256`,
`balance: false`, and `partition.type: "patient_iid"`. Random-weight controls require
explicit `allow_random_features: true`; the real pilot keeps this false. Backdoor
configurations refuse because aligned triggered NIH features do not exist.

The workload copies verified read-only arrays into owned CPU tensors and records
actual evaluation/cache/source/model/manifest identities, client counts and a
versioned assignment digest. Patient/image lists stay out of the run record.
Synthetic tests exercise both explicit evaluation choices; no real test performance
has been inspected. Another 92 targeted engine/config/reproduction tests pass.

A separate constructor-only real-cache audit checked complete row coverage,
disjoint whole-patient clients and exact counts at 1, 10 and 1,000 clients. The
ten-client allocation matches the curated client manifests at seed 42. At 1,000
clients, each has 22–23 patients but 27–198 images. This is an allocation check,
not itself a training/cost result. The separately completed pilot and execution
check below supply that later evidence. Whole-slice review follows these results.

## Validation pilot and current blocker — 19 October

The [prospectively fixed exploratory pilot](../../docs/artifacts/capstone/nih-validation-pilot-protocol.md)
completed all eleven cells and 102 rounds in 68.63 seconds, with no failure,
omission or timeout. Independent reconciliation checked every config/hash/round,
rebuilt patient assignments from CSV and confirmed the fixed selection rule.
The [full report and curves](../../docs/artifacts/capstone/nih-validation-pilot-v1-report.md)
retain all cases, including poor results.

The selected ten-client LR 0.1 gave probability-score AUROC 0.718259, loss 6.127444
and ECE 0.349491. Three-seed AUROC mean/range was 0.718472 / 0.718259–0.718697,
but loss/calibration remained poor. Pooled LR 0.001 gave AUROC 0.719872, loss
0.612794 and ECE 0.025980, showing useful representation signal. Do not treat the
runner's narrow AUROC readiness flag as a stable model or a deployment result.

The separately frozen [three-replay diagnostic](../../docs/artifacts/capstone/nih-score-diagnostic-protocol.md)
matched original final model digests and every ten-round utility metric. In pooled
LR 0.1, 15,729 of 17,763 probabilities saturated at exactly 1: probability AUROC
0.564571 versus decision-margin AUROC 0.718015 on identical logits. The selected
federated case's scoring difference was negligible; its poor calibration remains
real. NIH now defaults to versioned `margin_v2` scoring and retains
`auc_probability` as a diagnostic. Explicit `probability_v1` preserves v1's metric
semantics; original records and source-locked configs remain unchanged. All three
saved real-logit comparisons exactly match the independent margin calculations,
with unchanged loss, predictions and calibration. Kermany defaults/hashes and
numerical reproduction checks are unchanged. Historical impact outside these
replays is unknown. A bounded optimization diagnostic remains higher priority
than the privacy/attack matrix. Before confirmatory execution, harden source
identity beyond the exploratory runner's HEAD-only check.

The 1,000-client/two-round execution check completed in 7.063 seconds end-to-end
(3.560 seconds inside simulation), peak process RSS 735,051,776 bytes. This is a
sequential CPU tiny-head simulator over cached features, not 1,000 physical devices,
CNN training, convergence or deployment latency. Large-head costs remain unmeasured.

## Optimization pilot v2 — 19 October

The [separately frozen v2 budget](../../docs/artifacts/capstone/nih-optimization-pilot-v2-protocol.md)
ran six initial cells and 120 rounds in 50.496 seconds. All execution/source/cache
checks qualified, but no eighty-client case passed the joint ranking, balanced
accuracy, loss and ECE gates. The four conditional repeat/clipping cases remained
explicitly unrun; no LR was selected. Preserve this planned negative result.

The [report and all six curves](../../docs/artifacts/capstone/nih-optimization-pilot-v2-report.md)
show a useful ten-client LR 0.003 candidate: AUROC 0.705962, loss 0.618022,
balanced accuracy 0.628419 and ECE 0.007593, in one seed only. Its sensitivity
0.425301 is not a clinical-quality result. Ranking alone would again prefer
overconfident LR 0.01. The eighty-client LR 0.003 is less overconfident but weak
(AUROC 0.640542); it does not justify starting the privacy/attack matrix yet.

The smaller shards give 3–5 local SGD steps per client/round at eighty clients,
versus 26–30 at ten, despite the same nominal image passes. Total minibatch calls
increase, so the confound is the local trajectory/averaging schedule, not simply
less total computation. This led to a separate bounded LR 0.003 schedule probe:
eighty clients at seven local epochs/twenty rounds versus one epoch/140 rounds. The
[v3 protocol](../../docs/artifacts/capstone/nih-schedule-pilot-v3-protocol.md) is now
completed with its budget frozen before training; all v2 results remain unchanged.

The reviewed runner enforces the fixed cache pin and isolated child source
origins/byte fingerprints; a separate checker independently verified all records,
140 source/protocol fingerprints, assignments, counts and the failed selection
gate. Full framework:1,817passed,14deselected,81.26%coverage. Deferred minor:
source-check/control overhead can exceed a strict global wall deadline; actual
v2 execution is far below its 1,800-second cap. Official-test performance is unseen.

## Stable eighty-client schedule — 19 October

The [v3 report and exposure-aligned curves](../../docs/artifacts/capstone/nih-schedule-pilot-v3-report.md)
reconcile all six cells and 240 rounds. Seven local epochs/twenty rounds at
LR 0.003 pass the joint baseline gate across all three declared seeds:
AUROC mean 0.703844 (0.699003–0.706996), loss 0.616709–0.621308 and ECE
0.008377–0.019428. Clean clipped mean and trimmed mean also pass in the base seed.
These are exploratory research results, not clinical acceptance or a confidence
interval. Finding sensitivity is only about 0.47–0.50 across the core seeds.

At the same 140 nominal image passes, one epoch/140 rounds has similar validation
utility but seven times the modeled payload: 47.6 MB each direction versus
6.8 MB for twenty rounds. These are application-payload estimates, not measured
network traffic or battery cost. Recorded training time is nearly equal; the
longer schedule spends more time evaluating and orchestrating. Child execution
is 14.024 versus 30.379 seconds. All children together take 98.405573 seconds,
plus separately measured parent overhead 3.594475 seconds.

At C 0.4 no update was clipped. Clipped mean matches the unclipped model digest
and all round utility metrics; clean trimmed-mean AUROC is 0.705190 versus
0.705532. This supports matched clean controls, not an optimal clipping setting
or attack resistance. Independent checks bind 142 source/protocol fingerprints,
both per-child proof phases, all assignments/counts, selection and timing totals.
The interrupted-child proof and timing-accounting review findings are fixed.

The [attack-knowledge/privacy boundary audit](../../docs/artifacts/capstone/attack-knowledge-privacy-boundary-audit.md)
is complete. The subsequent [participation contract](../../docs/artifacts/capstone/participation-accounting-contract.md)
fixes rejected admissions that previously caused repeated unresolved rounds and
separates submitted, admitted, validated and successfully aggregated input counts.
These failures were reproduced in synthetic regressions; no corresponding NIH
baseline failure was observed. New records identify participation schema version 2;
historical records and their numerical results remain unchanged.

Next enforce permitted-information attacks and the fixed accepted cohort in the
prospective runner, bound larger-head costs, then freeze the confirmatory protocol. IPM/ALIE
read raw honest updates, so their oracle stress results cannot silently inherit
a one-final-upload sensitivity claim at the raw-client data boundary. V3 has no
attacks or noise and is unaffected. Official-test performance remains unseen.

## Practical research outputs

The capstone should produce evidence for project decisions, alongside the report:

- Privacy settings: show the measured privacy/utility frontier and declared
  client-level accounting assumptions; avoid presenting one epsilon as universally safe.
- Robust aggregation: quantify attack resistance and clean-data cost under matched
  clipping, cohort and privacy settings.
- Head size: compare trainable dimensions on the same frozen representation so
  backbone quality does not explain a claimed dimension effect.
- Cohort size: distinguish reduced aggregate noise from changed per-client data,
  communication cost and execution time. Simulated clients are not verified institutions.
- Evaluation under shift: retain official prevalence and examine the transfer of a
  validation-selected threshold; include patient-aware sensitivity checks because
  repeated images per patient are uneven across splits.

Expand experiments when a pilot or uncertainty requires them. Keep each resulting
recommendation linked to configurations, uncertainty, measured costs and limits;
avoid adding an unrestricted Cartesian experiment matrix.

The [methods source register](../../docs/artifacts/capstone/phase-2-methods-sources.md) distinguishes checked primary references from claims still needing full theoretical review.

## Protocol decisions to freeze

- Primary cohort, rounds, local optimizer/steps, learning rate, clipping norm,
  attacker knowledge/fraction/strength, aggregation rules and DP settings.
- One primary endpoint and contrast, paired seeds, interval method and treatment
  of multiple secondary comparisons. Seed intervals describe simulation variation,
  not uncertainty across clinical populations.
- Training/validation/test identities, patient-to-client allocation and whether
  balancing is applied. Keep primary evaluation at its declared prevalence;
  report majority baseline, balanced accuracy, AUROC and confusion counts.
- Pilot/tuning budget restricted to training/validation. Freeze settings before
  confirmatory test evaluation; never choose a favorable intermediate round.
- Exact cell list and repetitions, resource/time cap, stop/refusal criteria and
  reporting of incomplete/failed cells. Avoid an unrestricted Cartesian product.
- Release scope must distinguish a protected model release from research diagnostics and published randomness. The engine records the noise seed (also derived from public base seeds), pre-noise clipping/shrink statistics, raw-update summaries, training-based fairness and an initial upload digest. A [code-backed output audit](../../docs/artifacts/capstone/privacy-output-boundary-audit.md) inventories the paths and demonstrates seeded-noise reconstruction in one synthetic round. The current JSON has no raw model parameters; do not imply that seed disclosure alone exposes an aggregate from that JSON. The model accountant does not establish privacy for the entire diagnostic record or a future model released with reconstructible noise. Label research outputs accordingly, and design independently secret production randomness and an export allowlist before a deployment guarantee. Retain original public-dataset logs unchanged.
- DP unit is a simulated client under replace-one adjacency. Patient-disjoint
  allocation does not by itself turn that into a patient-level DP guarantee.
  Keep the existing conservative no-subsampling-credit accounting explicit.

The recommended head-size comparison holds frozen ResNet-18 features fixed and
uses a linear head plus MLP hidden widths 8, 128 and 512: 1,026, 4,122, 65,922 and
263,682 trainable parameters. Head capacity and optimization still change; compare each head against its matched no-DP baseline rather than calling every difference a pure dimension effect. This is the working comparison under the continuous-work mandate; freeze it in the prospective protocol after the pilot. A ResNet-50
linear head has 4,098 parameters but also changes feature dimension/backbone;
retain it only as a separately named representation comparison.

## Scope control

LoRA/full image-network training is already outside this capstone phase. Do not
reintroduce it. Keep a matched core comparison of mean/trimmed mean, central
DP/no DP and clean/declared attack conditions. Treat scale, head size and the
broader bad-client suite as separately bounded studies. If the schedule slips,
cut the local-DP extension first, then GUI extras. Preserve the core study,
honest reporting, evidence backup and final reproducibility.

A useful ImageNet-feature baseline is an empirical question. Do not assume that
a domain-trained backbone would necessarily improve it, or transfer the mobile
pneumonia model's AUROC to this different label/task.
