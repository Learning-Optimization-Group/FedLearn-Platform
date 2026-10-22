# Effusion follow-up: baseline and evaluation protocol v1

22 October 2026. **Status: FROZEN, 22 October 2026** — approved by the user as
proposed, including the 0.80 AUROC floor, the sensitivity-0.90 operating point and the
50/50 validation carve. Written after the condition decision and holdout lock, before
any Effusion label was used to fit, select or score a model. Later changes must be
dated amendments, and any analysis outside this plan is labelled post-hoc. The completed any-finding study,
its cache and its results stay unchanged as the historical reference.

## 1. Target and intended use

- **Decision (user, 22 October):** detect the NIH report-derived `Effusion` label.
- **Intended use, research only:** an image-level triage aid that flags frontal
  chest X-rays for prioritised human review of possible pleural effusion. This is not
  a diagnosis, and the study offers no clinical qualification.
- **Label:** positive iff the exact token `Effusion` appears in the pipe-split
  `findings`. Everything else is negative, including `No Finding` and other findings.
  Labels are mined from reports, so negatives are weak and unverified.

## 2. Data partitions and the locked holdout

| Partition | Source | Patients / images | Use |
| --- | --- | ---: | --- |
| Train | Existing curated train split | 22,399 / 68,650 | Fitting, standardisation, federated clients |
| Selection | Validation half | 2,800 / 9,075 | Model/hyperparameter selection, operating point, recalibration |
| **Locked holdout** | Other validation half | 2,800 / 8,688 | One final evaluation of the frozen set (§8) |
| Official NIH test | — | — | **Excluded entirely**; already inspected, stays historical |

The holdout was carved with `repro/lock_effusion_holdout.py` (5 tests, written
failing first). Validation patients were ordered by SHA256(`capstone-effusion-holdout-v1`
NUL patient_id) and the first half went to the holdout. Patients stay whole, label and
findings columns were dropped on read, and assignment is proven label-invariant by
test. No cross-patient duplicate-content group touches validation. Pins:
lock receipt `083aa7ce95e40af7bdc00d27d396b0ac5954e7f6ebd444200ce97651baccc794`;
holdout manifest `e55f703f6038426659261866b2dc067e98cbc0b673a235dde6639f1ff77e6647`;
selection manifest `af8c98bb994fc6c7400496bd0d6e66a96d2ba42b8b6e9728b6e75462bab3770e`.
The private manifests are write-protected; they stay outside the repository.

**Holdout rules.** Holdout labels, features and scores are not read until §8. All
fitting/selection code must refuse any image index in the holdout manifest. The
holdout's Effusion prevalence is deliberately not yet known. By expectation it holds
about half of the validation split's 709 positive patients.

**Honest scope of "independent".** This is an internal, patient-disjoint and
Effusion-untouched holdout, not an external cohort. The validation split as a whole
was used for selection in the historical any-finding study. The feasibility audit
also counted its Effusion positives in aggregate. No model has been fitted, selected
or scored on Effusion. The CheXpert backbone is documented upstream as non-NIH, but
its training membership was not independently reconstructed. Site, scanner and
population generalisation therefore remain untested.

## 3. Representations and controls

| ID | Representation | Status |
| --- | --- | --- |
| R0 | Historical ImageNet ResNet-18, 512-wide (existing cache; Effusion labels derived from its manifests, arrays untouched) | Older-baseline comparator |
| R1 | TorchXRayVision `densenet121-res224-chex` `features2()`, 1,024-wide | Candidate. Needs a new versioned cache built under the [probe contract](medical-input-probe-2026-10-22.md); train and validation only |
| X | The same checkpoint's own published `Effusion` output, untrained on NIH | Reference control only; untuned, with its threshold and calibration taken from the selection half like every other model |
| C0 | Constant score / always-negative | Shows imbalance; zero sensitivity |

## 4. Centralised candidate budget (fixed)

Per representation: L2 logistic regression on features standardised with
**train-only** mean/std; C ∈ {0.001, 0.01, 0.1, 1, 10}; class weight ∈ {none, balanced};
lbfgs, max_iter 2,000, fixed seed. That is 10 fits per representation, 20 in total. Choose
by selection-half image AUROC; ties go to the smaller C. Every fit, including any
non-convergence, is retained, and a failed fit is not rerun under new settings.
No MLP, fine-tuning or extra backbone in v1; those need a dated amendment after a
failed gate. Fitting also includes a matched central-SGD control: the same linear
head and optimiser budget as the federated recipe (§6), to separate optimisation
loss from federation loss.

## 5. Operating point, calibration and metrics

- **Recalibration:** Platt scaling fitted on the selection half, applied unchanged later.
- **Operating point (triage):** the selection-half threshold giving sensitivity ≥ 0.90,
  chosen as the highest such threshold. The Youden threshold is reported as secondary.
- **Metrics:** AUROC; average precision alongside prevalence; sensitivity,
  specificity, PPV and NPV at the frozen threshold; Brier score; calibration
  intercept and slope; ECE (10 equal-mass bins); reliability diagram.
- **Units and uncertainty:** image-level is primary; patient-level (maximum score
  per patient) is secondary. 95% intervals come from a 2,000-replicate
  patient-cluster percentile bootstrap. Model differences use the paired bootstrap
  on the same resamples.

## 6. Matched clean federated baseline

The federated recipe mirrors the historical primary one: the gate-selected representation,
the same frozen standardiser, a linear two-output head (2,050 parameters for R1),
80 clients with whole-patient IID allocation, all clients in each of 20 rounds,
7 local epochs, SGD LR 0.003, batch 256, update clipping C 0.05, no DP and no
attacks. Five declared seeds. Reported as federated minus central-SGD minus convex
optimum, with paired intervals. The shared standardiser assumes public/shared
normalisation statistics; this is disclosed, not hidden. At 80 clients, about 280
patients each, no client is forced to lack a positive. The bound only bites at the
1,000-client scale, which is a simulation check rather than physical devices.

## 7. Learning gate (pre-registered, selection half only)

R1 qualifies as the stronger centralised baseline only if both hold:
(a) the paired ΔAUROC (R1 − R0) 95% interval lies entirely above 0, and
(b) R1's AUROC is ≥ **0.80**. That floor was approved with the protocol; it is a
research threshold, not derived from a clinical requirement.
Specificity at sensitivity 0.90 and calibration are always reported but not gated:
calibration is only non-circular on the holdout. A failed gate is recorded as the
result. Proceeding to privacy/attack studies on a failed baseline requires an explicit
decision, and any added candidate requires an amendment.

## 8. Final evaluation (once)

Before the holdout is opened, record the exact frozen set: C0, X, the selected R0
and R1 heads, the clean federated head per seed, and every privacy/attack
configuration later frozen for the heatmaps. Each is then scored once on the
holdout with §5 metrics. Outputs are raw per-image scores (private) plus aggregate
tables, generated by a committed script. Any rerun or added model after opening is
labelled post-hoc. If the holdout itself proves unusable, report that.

## 9. Order and time budget (deadline 5 December)

1. Sign-off: given 22 October.
2. Build the versioned R1 cache for train + selection (77,725 images, deterministic
   CPU) with its own report; holdout features are extracted only at §8, then a target-specific duplicate/contradictory
   label audit on train and selection only.
3. §4 fits, §7 gate, then §6 federation. Privacy (central DP, local DP, secure
   aggregation where supported) and bounded attacks follow on the frozen recipe.
4. §8 once, then heatmaps, report, presentation and simulator integration.

## Amendment A1 — 22 October 2026, before any federated run or gate result

Fixes details §6 left open; no result of any kind had been observed when this was written.

- **Seeds:** 20261101–20261105 (the first five of the historical study's declared
  list), with the historical scheme: partition = init = client = s, attack = s + 1000,
  noise = s + 2000.
- **Engine:** the production simulation engine (`ComposedStrategy` + `FLCoordinator`
  via `SimulatedFederation`) through a new optional `workload=` argument, so FedAvg is
  not re-implemented. Uniform mean rule, no server optimiser, AUROC on float64
  logit margins (`margin_v2`), evaluated on the selection half after each round.
- **Central-SGD control:** the same engine with one client holding all of train,
  so local SGD, seven epochs, batch 256, LR 0.003 and the 0.05 update clip per round
  are identical. Only data distribution and averaging differ. The clip bounds the
  per-round movement of both arms equally.
- **Convex optimum:** the gate-selected R1 logistic head.
- **Reported:** per-seed AUROC and AP; the seed mean and range; and paired
  patient-bootstrap intervals (2,000 resamples, seed 20261023) for federated −
  central-SGD (same seed), federated − convex and central-SGD − convex. There is
  no pass/fail threshold; the gap is the result.
- Runs only if the §7 gate passed.

## Amendment A2 — 22 October 2026, after the gate result, before any federated run

The §7 gate failed criterion (a): ΔAUROC for R1 − R0 was +0.007, 95% CI
[−0.004, +0.018]. Criterion (b) passed with R1 AUROC 0.829 ≥ 0.80. See the
[gate result](effusion-gate-v1-result-2026-10-22.md).

By explicit user decision, §6 onward proceeds on **R1**, the gate-selected CheXpert
probe, with its frozen standardiser. The decision is recorded, not inferred.

- R1 is **not** claimed to be a stronger baseline than R0. Every later report states
  that the representations were statistically tied on selection.
- R1's costs are disclosed wherever relevant: a 2,050-parameter head (twice R0's
  payload) and the unverified CheXpert training-membership caveat.
- R0's selected head stays available as a reference at the §8 holdout evaluation.
  Nothing else in §4–§8 or A1 changes.
- The federated harness refuses a failed gate unless it is given `--amendment A2`,
  and it records that authority in its output.

## Amendment A3 — 22 October 2026, before any federated result was opened

This is a reading rule for §6 only. It changes no procedure, and no run setting
was changed. The §6 recipe was fixed for raw 512-wide any-finding features. Here it
runs on standardised 1,024-wide Effusion features, and the 0.05 update clip bounds
per-round movement in both arms.

- **Primary contrast:** federated − central-SGD on the same seed. Both arms share the
  recipe and the clip, so this gap is attributed to federation (partitioning and
  averaging).
- **Federated − convex and central-SGD − convex** are reported as federation cost
  only if the recipe is not limiting. The recipe counts as limiting if either of these
  holds:
  - the mean last-round clip rate is ≥ 0.95 in either arm;
  - the central-SGD seed mean is more than 0.02 AUROC below the convex R1 head (0.829).

  If it is limiting, those gaps are labelled **recipe-limited** and are not called
  the cost of federation.
- **No retuning after seeing results.** LR, clip, epochs and rounds are not tuned
  after these results. Any optimiser check needs its own declared amendment, with the
  outcome reported as exploratory.

## Amendment A4 — 22 October 2026, central-DP sweep with a declared R0 arm, before any DP run

**Question.** R0 and R1 tied on clean selection AUROC, and R1 is twice as wide (2,050
vs 1,026 head parameters). Does width change the utility cost of central DP at equal
ε? This is the declared secondary contrast. R1 stays primary (A2).

- **Arms:**
  - R1 uses the A2 setup.
  - R0 uses the historical 512-wide arrays, standardised with the gate's R0 train
    mean/std, and a 1,026-parameter head.
  - Everything else is the §6/A1 recipe: 80 whole-patient IID clients, 20 rounds,
    7 epochs, LR 0.003, batch 256, clip 0.05, mean rule, seeds 20261101–20261105.
- **Cells:** clean, plus central DP at ε ∈ {1, 4, 8} with δ = 10⁻⁵. That is
  2 arms × 4 conditions × 5 seeds = 40 runs.
- **DP mechanism:** the engine's existing mechanism.
  - Gaussian noise is added to the clipped mean, using the replace-one sensitivity.
  - The multiplier is calibrated for all 20 rounds at q = 1, with no amplification credit.
  - **The privacy unit is one simulated client (about 280 patients), not a patient.**
    ε here is not patient-level DP.
- **Reported:**
  - Per-cell AUROC and AP.
  - DP cost: clean − DP, same arm and seed, with a paired patient bootstrap
    (2,000 resamples, seed 20261024).
  - **Width contrast:** cost(R1) − cost(R0), seed-averaged inside each bootstrap
    replicate.
  - Recorded noise SD and noise-to-signal ratio.
  - There is no pass/fail threshold.
- **Replay check:** the R1 clean cells must reproduce the §6 federated margins exactly.
- **Excluded:** local DP, until a separate qualification amendment. Secure aggregation
  is covered by A5. There is no retuning.

## Amendment A5 — 22 October 2026, secure-aggregation scope, before any secure run

**Decision (user, 22 October):** do not build a new secure path for the head. Instead:
show equivalence, measure overhead, and mark incompatible combinations.

- **Mechanism under test:** the existing LightSecAgg implementation, run in-process:
  - The real `SecureAggregationClient`: X25519 key agreement, Shamir shares of a
    fresh `os.urandom` mask, sealed-share relay through the real servicer handlers,
    and `quantize(x) + z` masking at scale 10⁶.
  - The real `SecureAggregationSession`: masked submissions and a one-shot recovery.
  - Threshold t = ⌊n/2⌋ + 1 = 41 of 80; no dropouts.
  - Two harness adaptations:
    - Summed shares are filed through the same holder-index check as the servicer,
      but skip its DeComFL-specific completion hook (which applies a zeroth-order
      update).
    - No network: channel latency, retries and interceptors are not measured.
- **Placement:** inside the §6 engine run (R1, seeds 20261101–20261105). Only the
  `mean` reduction is swapped, within a scoped context, for the secure sum divided by n.
  Everything else in training is identical.
  - Each round runs the protocol once. The engine's noise-free previews of the same
    inputs reuse that result.
- **Trust change, disclosed:** clipping is applied per client before masking. Numerically
  this matches server-side clipping, but under secure aggregation the server cannot
  verify it; clients are trusted to self-clip.
- **Reported:**
  - Per-round recovered-vs-float-mean error against the fixed-point bound.
  - Final margin difference and AUROC difference from the §6 plain runs, same seed
    (paired bootstrap, 2,000 resamples, seed 20261025).
  - Per-round protocol wall time by phase.
  - Bytes per client per round, from serialised request and response protobufs,
    compared with the plain 8,348-byte upload.
- **Unavailable, by construction:** secure aggregation combined with trimmed mean,
  median or Krum. The server learns only the sum, and those rules need individual
  coordinates. The harness refuses them rather than silently falling back to a plain
  path. Central DP on the recovered sum would be mechanically possible but is not run
  here.

## Amendment A6 — 22 October 2026: R0 becomes primary for all remaining work

User decision. From here on, **R0 (historical ImageNet ResNet-18 features) is the primary
representation** for recipe work, attacks, local DP, the frozen §8 list and the
platform-facing outputs.

Reasons:
- R0 tied with R1 at the gate.
- A4 measured a significantly smaller DP cost for the narrower head at every ε.
- Its payload is half the size.
- It carries no CheXpert training-membership caveat.
- It matches the simulator's qualified 512-wide workload.

Completed R1 results (§6 baseline, A4 arm, A5) are kept unchanged as secondary
evidence. Nothing already run is re-labelled.

## Amendment A7 — 22 October 2026: bounded federated recipe study on R0, before any A7 run

**Why:** under the fixed §6 recipe, R0 reaches 0.755 federated against 0.822 for its
convex head. The §6 results were recipe-limited (A3). A default that is limited this
way is not useful to the platform. Per A3, this is a declared optimiser check: it is
exploratory, run on the selection half, and leaves the holdout untouched.

- **Grid:** 12 configurations, R0, no DP, all retained.
  - Client LR ∈ {0.003, 0.01, 0.03, 0.1} × update clip ∈ {0.05, 0.25, 1.0}.
  - Fixed: 80 whole-patient IID clients, 20 rounds, 7 local epochs, batch 256, mean rule.
- **Selection:** the highest mean selection AUROC of the federated arm over seeds 20261101
  and 20261102. Ties go to the smaller clip, then the smaller LR. Non-finite or refused
  runs are ineligible.
- **Confirmation:** the chosen recipe runs on all five seeds 20261101–20261105:
  - the federated arm and the central-SGD control;
  - central DP at ε 8/4/1 with δ 10⁻⁵ (client-level, q = 1, as in A4).
- **Reported:**
  - the gain of the chosen recipe over the §6 recipe (same seeds, paired);
  - federated − central-SGD;
  - the gap to the convex R0 head (0.822);
  - the DP cost at the chosen clip compared with the A4 R0 cost at clip 0.05. Noise
    scales with the clip, so this is a declared clip–DP interaction, not a fresh
    privacy result.
- **Honest limits:** seeds 1–2 choose the recipe, so the five-seed means are optimistic.
  Seeds 3–5 are separate training seeds on the same selection data, not new data. The
  unbiased number comes only from the §8 holdout.

## Amendment A8 — 22 October 2026: DP-specific recipe study on R0, before any A8 run

**Why:** A7 showed that the clean-optimal clip (1.0) makes central DP about 0.05 AUROC
worse than clip 0.05 at ε 8 and ε 4. The platform needs evidence-backed DP defaults.

- **Grid:** 12 configurations, all retained. Client LR ∈ {0.03, 0.1, 0.3} × clip ∈
  {0.01, 0.025, 0.05, 0.1}. Central DP at ε = 4, δ = 10⁻⁵, client-level, q = 1.
  Otherwise the §6 recipe: 80 whole-patient IID clients, 20 rounds, 7 epochs, batch 256,
  mean rule.
- **Selection:** the highest mean selection AUROC over seeds 20261101 and 20261102 at
  ε = 4. Ties go to the smaller clip, then the smaller LR.
- **Confirmation:** the chosen recipe on all five seeds at ε 8, 4 and 1, plus its clean
  (no-DP) arm. Reported as paired contrasts against the A4 R0 cells (§6 recipe) and the
  A7 DP cells (clean-tuned recipe).
- **Honest limits:**
  - Choosing hyperparameters by looking at DP runs on the same data consumes privacy
    that the stated ε does not account for. The selection half is development data, not
    the training data, but this is still disclosed. Production guidance must fix
    defaults in advance or account for tuning.
  - The ε = 4 choice is applied to ε 8 and ε 1 without retuning.
  - Seeds 1–2 choose, so the means are optimistic; only the holdout is unbiased.

## Amendment A9 — 22 October 2026: bounded attack study on R0, before any A9 run

**Question:** which aggregation rule should the platform recommend when some clients may
be malicious, with and without DP, given each rule's cost when nobody attacks?

- **Recipes** (A7/A8 defaults): no-DP cells use LR 0.1 with clip 1.0. DP cells use
  LR 0.03 with clip 0.05 and central DP at ε 4 (δ 10⁻⁵, client-level, q = 1).
  Otherwise the §6 setup: R0, 80 whole-patient IID clients, 20 rounds, 7 epochs, batch 256.
- **Rules:**
  - No DP: `mean`, `trimmed_mean` (trim 0.2 per side), `median`.
  - DP: `mean` and `trimmed_mean`. These are the rules with a replace-one sensitivity
    bound in the engine; median under DP is unavailable, not failed.
- **Conditions:** 16 of 80 attacker slots (fraction 0.2), fixed per seed by the attack
  seed stream and identical across rules and conditions.
  1. **clean** — no attackers.
  2. **sign flip** — each attacker multiplies its own update by −4 before clipping, as in
     the historical study.
  3. **label flip** — attackers train on reversed labels.
  4. **ALIE** — `alie`, z = 0.4888, the published bound Φ⁻¹((n − s)/n) with
     s = ⌊n/2⌋ + 1 − m = 25 for n = 80, m = 16.

  No attacker reads another client's raw update except as the engine's ALIE
  implementation defines it.
- **Size:** seeds 20261101–20261105, all cells retained. No DP: 3 rules × 4 conditions ×
  5 seeds = 60 runs. DP: 2 × 4 × 5 = 40 runs. Total 100.
- **Reported** (selection half, paired patient bootstrap, 2,000 resamples, seed 20261028):
  - absolute AUROC per cell (heatmap 1);
  - attack-induced drop, clean minus attacked under the same rule and seed (heatmap 2);
  - each robust rule's clean cost relative to `mean`;
  - refusals and non-finite runs, reported as such, never as zero.
  - There is no pass/fail threshold, and no winner is declared from a single seed.
- **Limits:** client-level, simulated, IID; one attacker fraction; the attacks are not
  adaptive to the defence beyond ALIE's construction.
