# Capstone plan — current as of 19 October 2026

This is the current schedule and completion checklist. Earlier milestone decks,
reports and design notes remain historical evidence. Phase 2 has not been
pre-registered or run; the Milestone 2 grids are exploratory.

## Objective and boundaries

Study how client-level differential privacy, robust aggregation, adversarial
clients and trainable head size interact in a reproducible federated simulation.
The next dataset is NIH ChestX-ray14 with **0 = No Finding** and **1 = any of the
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
- The existing workload repartitions feature rows and lacks patient ownership.
  The current feature-preparation CLI expects ImageFolder data. Setting its
  dataset label to `nih` does not implement an NIH loader.

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
draft the message for review, without treating a working title as approved.

## Next work: NIH adapter

The [NIH workload design](../../docs/superpowers/specs/2026-10-18-capstone-nih-workload-design.md)
defines the proposed implementation boundary and acceptance checks. Preserve the
old Kermany behavior and exact reproduction anchors. Do not alter the mobile
pneumonia recipe, reuse its class mapping, overwrite an old feature cache or inspect
official-test performance to select settings.

The [first implementation plan](../../docs/superpowers/plans/2026-10-19-capstone-nih-manifests.md)
is implemented and verified. It covers source validation, patient-separated manifests,
whole-patient client allocation and independent verification. Feature extraction,
engine evaluation-split integration and features remain later slices. The complete content audit is implemented; its failed cohort qualification requires explicit quarantine before extraction.

After integrating main, 60 targeted workload/config/engine tests pass, with the
synthetic scale test deselected; another 40 reproduction/strategy/behavior/privacy
tests pass. A fresh bounded comparison reproduces all eight stored cells from
repetition 0 of `main_n80_eps1` exactly under torch 2.12.0. The original 160-cell
torch 2.9.1 verification remains historical evidence; this new eight-cell check
does not replace or expand its scope. The full integrated framework suite passes:
1,695 tests, 14 deselected, 81.04% coverage after the NIH curated-cohort slice, with unhandled thread exceptions
treated as errors. These checks do not establish NIH learning or content integrity.

The first useful run is a small, no-DP/no-attack validation pilot with real NIH
features. It must establish that labels, learning, patient ownership and evaluation
work before scaling to 1,000 clients or adding privacy/attacks. The next gate is
a 1,000-client, two-round execution/cost check, explicitly distinct from a utility
claim. Measure elapsed time and memory before expanding the matrix.

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
Frozen feature extraction, explicit engine split selection and useful learning
remain unverified. The [feature plan](../../docs/superpowers/plans/2026-10-19-capstone-nih-frozen-features.md)
now consumes this explicit cohort and keeps quarantined development rows out.

A private second-device checkpoint through `88ce88b`, including the completed
content audit and 55 archived evidence files, has been hash-verified. Newer cohort
work still requires its next checkpoint; no capstone branch publication is implied.

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
