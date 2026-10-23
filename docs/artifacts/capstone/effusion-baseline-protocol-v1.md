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

## Amendment A10 — 22 October 2026: frozen §8 holdout list, before any holdout feature or label is read

The holdout is opened **once**, by `repro/evaluate_effusion_holdout.py`, for exactly these
22 models. All are R0 (A6) with the gate's train-only standardiser.

1. **C0:** constant score.
2. **Convex R0 logistic head** (the gate-selected fit).
3. **R0 FedAvg, no DP**, A7 recipe (LR 0.1, clip 1.0): five seeds, 20261101–20261105.
4. **R0 central DP**, A8 recipe (LR 0.03, clip 0.05): ε 8, 4 and 1, five seeds each
   (15 models).

**Excluded, declared now:** R1 heads and the X control. R1 is secondary since A6, and both
would need holdout R1 feature extraction. Also excluded: attack cells (reported on
selection only), central-SGD controls, the A8 clean arm, and the §6-recipe runs.

**Frozen before the holdout is read:**
- Each model's selection-half margins are recomputed from its saved final head and must
  equal the stored margins exactly.
- On those margins: the Platt recalibration and the operating point (the highest
  Platt-probability threshold with selection sensitivity ≥ 0.90).
- These go into a hashed calibration file, written before any holdout label is loaded.

**Holdout metrics** (§5):
- AUROC; AP with prevalence;
- sensitivity, specificity, PPV and NPV at the frozen threshold;
- Brier score; calibration intercept and slope; ECE over 10 equal-mass bins;
- patient-level AUROC (maximum score per patient).
- Uncertainty: 2,000 patient-cluster bootstrap resamples, seed 20261029.
- Seed groups: per seed, plus the seed mean with its interval taken inside each replicate.
- Paired contrasts: federated − convex, and DP − no-DP federated at each ε.

**Rules:** the harness refuses to write over an existing evaluation. Any later scoring of
the holdout, for any model, is labelled post-hoc.

## Amendment A11 — 22 October 2026: local DP on R0, before any local-DP run

**Question:** what does removing the trusted server (local DP) cost, compared with central
DP at the same ε?

- **Mechanism:** the engine's existing local DP.
  - Each honest client clips its own update to C and adds N(0, (2zC)²) per coordinate;
    the noise multiplier z is calibrated for 20 rounds at q = 1, δ = 10⁻⁵.
  - The server applies only the non-shrinking post-processing clip C + 2zC(√d + 4).
  - The privacy unit is one simulated client, with replace-one adjacency, as for central DP.
  - The trust model differs: no trusted aggregator. Equal ε is therefore **not** an
    equivalent claim across the two modes.
- **Reporting fix first:** round records now carry explicit `client_noise_std` and
  `server_noise_std`. The historical `noise_std` keeps its server-only meaning and old
  records are not reinterpreted.
- **Cells:** 25 runs, all retained.
  - R0, A8 DP recipe (LR 0.03, clip 0.05), 80 whole-patient IID clients, 20 rounds,
    mean rule, seeds 20261101–20261105.
  - ε ∈ {1, 4, 8, 32, 128}. ε 32 and 128 are weak-privacy values, included only to find
    where local DP regains utility; they are not recommended settings.
- **Reported** (selection half; the holdout was opened once in A10 and is not reused):
  - AUROC per cell;
  - local − central at ε 1/4/8 against the A8 confirmation runs;
  - local cost against the A8 clean arm;
  - client noise SD.
  - Paired patient bootstrap, 2,000 resamples, seed 20261030.

## Amendment A12 — 22 October 2026: robust aggregation under local DP, before any A12 run

**Question:** does robust aggregation help or hurt when clients add local-DP noise, with
and without attackers? A9 covered robust rules without DP and under central DP; A11 covered
local DP with the plain mean only.

- **Cells:** 135 runs, all retained.
  - R0, A8 DP recipe (LR 0.03, clip 0.05), 80 whole-patient IID clients, 20 rounds,
    seeds 20261101–20261105.
  - Local DP ε ∈ {8, 32, 128} with δ 10⁻⁵, as in A11. Honest clients clip and add
    N(0, (2zC)²); the server applies only the non-shrinking post-processing clip.
  - Rules: mean, trimmed mean (0.2 per side) and median. Robust rules are server-side
    post-processing of already-privatised updates, so they cost no extra privacy.
  - Conditions: clean, sign flip ×−4, and label flip at 16 of 80 attacker slots,
    as in A9.
  - Attackers do not add local noise, since the engine privatises honest clients only.
    This is the realistic threat: a malicious client need not follow the protocol.
  - ALIE is excluded because it had no measurable effect in A9.
- **Reported** (selection half; paired patient bootstrap, 2,000 resamples, seed 20261031):
  - absolute AUROC per cell;
  - attack drop: clean − attacked, same rule, ε and seed;
  - rule − mean, same condition, ε and seed.
  - Refusals and non-finite runs are reported as such. The ε 8, mean, clean cells repeat
    A11's configuration and must reproduce its margins exactly (replay check).

## Amendment A13 — 22 October 2026: exact Gaussian-DP calibration on R0, before any A13 run

**Question:** how much selection AUROC does exact noise calibration recover at the same claimed
budget? All central-DP runs so far calibrated z with the RDP accountant. At q = 1 the T releases
compose exactly to μ-GDP (μ = √T/z). The RDP z therefore over-delivers: at T = 20 and δ 10⁻⁵, the
claimed ε 8/4/1 are really 6.76/3.25/0.74.

- **Cells:** 15 runs, all retained.
  - R0, A8 DP recipe (LR 0.03, clip 0.05), 80 whole-patient IID clients, 20 rounds,
    seeds 20261101–20261105.
  - Central DP, replace-one, at claimed ε ∈ {8, 4, 1} with δ 10⁻⁵, as in A8.
  - The only change from A8: z is the smallest multiplier whose **exact** GDP ε meets the claim
    (`fedlearn.privacy.gdp_accountant.gdp_required_noise_multiplier`). The engine is unchanged. The
    harness passes the RDP-equivalent budget whose RDP-calibrated z equals that GDP z. Each run must
    then record a z within 10⁻⁶ (relative) of the GDP z, with an exact ε at most the claim. Any run
    that fails this is refused.
  - Same seeds as A8, so noise draws are common random numbers up to the scale change.
- **Reported** (selection half; paired patient bootstrap, 2,000 resamples, seed 20261032):
  - absolute AUROC per cell;
  - gain over A8 at the same claimed ε: GDP-calibrated − RDP-calibrated, same seed;
  - DP cost against the A8 clean cells.
  - Refusals and non-finite runs are reported as such. The A8 margins are read from the pinned
    A8 output (`summary.json` `bd9868ae…`).

## Amendment A14 — 22 October 2026: attacks built for robust rules, attacker-fraction sweep and small-k trimming on R0, before any A14 run

**Questions:**
1. Do A9's robustness conclusions survive attacks designed against the trimmed mean and median,
   at production-realistic attacker fractions?
2. Under central DP, does trimming only a few values per side keep most of the mean's clean
   utility while still bounding attacks? A9 found trimming 0.2 per side costs utility under DP.

- **Recipes and setup:** as A9.
  - No-DP cells: LR 0.1, clip 1.0.
  - DP cells: LR 0.03, clip 0.05, central DP ε 4 (δ 10⁻⁵, replace-one, RDP-calibrated as in A9).
  - R0, 80 whole-patient IID clients, 20 rounds, seeds 20261101–20261105.
- **Rules:**
  - No DP: `mean`, `trimmed_mean` (k = 16 per side, ratio 0.2), `median`.
  - DP: `mean`, and `trimmed_mean` with k ∈ {1, 4, 16} per side (ratios 0.0125, 0.05, 0.2).
    Sensitivity is 2C/(n − 2k).
- **Attacks** (full knowledge of the honest updates, as for IPM/ALIE in the engine):
  - `min_max`: Shejwalkar and Houmansadr, NDSS'21, with std perturbation.
  - `fang_trim`: Fang et al., USENIX Security'20, trimmed-mean attack, b = 2.
  - `ipm` with scale 0.1: Xie et al., the small-ε variant aimed at median-type rules.

  Uploads are clipped by the server like any other update. The attacks do not adapt to the
  clip, which is a stated limit.
- **Attacker fractions:** 4, 8 and 16 of 80 slots (5%, 10%, 20%), drawn by the engine's
  attack-seed stream.
- **Size:**
  - No DP: 3 rules × (1 clean + 3 attacks × 3 fractions) × 5 seeds = 150 runs.
  - DP: 4 rules × 10 conditions × 5 seeds = 200 runs.
  - Total 350, all retained.
  - The no-DP clean cells and the DP mean and trimmed-0.2 clean cells repeat A9's configuration
    and must reproduce A9's margins exactly (replay check).
- **Reported** (selection half; paired patient bootstrap, 2,000 resamples, seed 20261033):
  - absolute AUROC per cell;
  - attack drop: clean − attacked, same rule, recipe and seed;
  - each rule's clean cost relative to `mean`;
  - drop as a function of attacker fraction.
  - Refusals, non-finite runs and the engine's attack notes are reported as such. No pass/fail
    threshold.
- **Limits:** IID, simulated, client-level, non-adaptive to clipping, one ε under DP.

## Amendment A15 — 22 October 2026: number of DP releases on R0, before any A15 run

**Question:** under central DP at a fixed claimed budget, does training for fewer rounds beat
20 rounds? With exact GDP accounting at q = 1, z scales as √T. At claimed ε 4, z falls from
4.84 (T = 20) to 1.08 (T = 1). Fewer releases therefore each carry much less noise, at the
cost of fewer optimisation steps.

- **Cells:** 75 runs, all retained.
  - R0, A8 recipe (LR 0.03, clip 0.05, 7 local epochs), 80 whole-patient IID clients,
    seeds 20261101–20261105.
  - Rounds T ∈ {1, 2, 5, 10, 20}.
  - Privacy ∈ {clean, claimed ε 8, claimed ε 4}, δ 10⁻⁵, replace-one central DP.
  - z is calibrated exactly with GDP for each T, as in A13 (the same RDP-equivalent budget
    mechanism and the same z check and refusal).
  - The recipe is held fixed across T. It was tuned at T = 20, which is a stated limit.
- **Reported** (selection half; paired patient bootstrap, 2,000 resamples, seed 20261034):
  - absolute AUROC per cell;
  - each T against T = 20 at the same privacy level and seed;
  - DP cost against the clean cell with the same T.
  - Refusals and non-finite runs are reported as such. The T = 20 ε 4 and ε 8 cells repeat
    A13's configuration and must reproduce its margins exactly. The T = 20 clean cells must
    likewise reproduce A8's clean cells.

## Amendment A16 — 22 October 2026: one-run canary audit of central DP on R0, before any A16 run

**Question:** does the simulator's central-DP mechanism leak no more than its accountant
claims, against an adversary who knows the canary directions and sees the final model?
Method: Andrew et al., ICLR'24 (arXiv:2302.03098).

- **Canaries:**
  - 8 of the 80 client slots are canary clients, drawn by the engine's attack-seed stream.
  - Each canary uploads global + C·u, every round, where u is its own fixed random unit
    direction over the trainable head and C = 0.05 is the clip, so the clip leaves it unchanged.
  - Each final model is also scored against 1,000 random unit directions that were never inserted.
- **Cells:** 40 runs, all retained.
  - R0, A8 recipe, T = 20.
  - Privacy ∈ {clean (positive control), claimed ε 8, 4, 1}, with z calibrated exactly with GDP as in A13.
  - Seeds 20261101–20261110.
- **Score:** ⟨final head parameters, u⟩ for inserted canaries (80 per privacy level) and unseen
  directions (10,000 per level).
  - μ̂ = (mean inserted − mean unseen) / sd unseen, with a bootstrap 95% interval
    (`fedlearn.privacy.audit.canary_audit`, 2,000 resamples, seed 20261035).
  - ε̂ is the GDP ε of μ̂ at δ 10⁻⁵.
- **Reference:**
  - The noise is calibrated for replace-one sensitivity 2C/n. "Canary inserted vs not inserted"
    is a shift of C/n per round, so its theoretical μ is √T/(2z).
  - μ̂ is expected at or below that, because the learning signal adds spread. A lower interval
    bound above √T/(2z) would mean the mechanism leaks more than claimed, and is reported as a
    finding.
- **Also reported:**
  - AUROC with canaries present;
  - the clean control's μ̂, which must be large, or the audit has no power;
  - refusals and non-finite runs.

## Amendment A17 — 22 October 2026: zero-initialised head under central DP on R0, before any A17 run

**Why:** A15 found that with the A8 recipe the head barely leaves its random initialisation:
- at T = 1 the weight norm is 0.81, the random-init scale;
- from T = 1 to T = 20 the weights move only 0.24 while the bias moves 0.65.

AUROC depends only on the weight direction, so the random init acts as a fixed perturbation
that a binding clip cannot overcome. Zeroing the initial head is data-independent, so it costs
no privacy.

**Question:** does a zero-initialised head raise AUROC under central DP, and does it make fewer
rounds viable?

- **Cells:** 40 runs, all retained.
  - R0, A8 recipe (LR 0.03, clip 0.05, 7 local epochs), 80 whole-patient IID clients,
    seeds 20261101–20261105.
  - Rounds T ∈ {5, 20}.
  - Privacy ∈ {clean, claimed ε 8, 4, 1}, z calibrated exactly with GDP as in A13.
  - The only change from A13/A15 is the initial global head: weights and bias are zero.
    Everything else, including every seed, is identical.
- **Reported** (selection half; paired patient bootstrap, 2,000 resamples, seed 20261036):
  - absolute AUROC per cell;
  - zero-init minus random-init, same T, privacy and seed. The random-init counterparts are
    A13 (T = 20, DP), A8 (T = 20, clean) and A15 (T = 5, clean/ε 8/ε 4). T = 5 at ε 1 has no
    counterpart and is reported as absolute only.
  - DP cost against the zero-init clean cell with the same T.
  - Refusals and non-finite runs are reported as such.

## Amendment A18 — 22 October 2026: all-iterates canary audit of central DP on R0, before any A18 run

**Why:** A16's final-model audit failed its positive control (clean μ̂ 0.15 [−0.07, 0.36]).
The parameter vector's own projection (sd 0.030) swamped a canary shift that honest training had
already partly erased (0.005 of an expected 0.0125). See the
[A16 result](effusion-canary-audit-r0-v1-result-2026-10-22.md).

**Question:** against an adversary who sees every released global model, does the simulator's
central-DP mechanism leak no more than claimed?

- **Cells, canaries and calibration:** as A16. That means 8 canaries of norm C = 0.05 per run,
  clean and claimed ε 8/4/1 with exact-GDP z, T = 20, and seeds 20261101–20261110: 40 runs.
- **Observation:** the global head at the start of each round and the final head, captured
  through a new engine hook. It is read-only and adds no config key, so every earlier config hash
  is unchanged. The released updates are Δ_t = θ_t − θ_{t−1}, t = 1…20.
- **Score:** s(u) = Σ_t ⟨Δ_t, u⟩ / T, the per-round mean of the released updates projected onto
  a direction.
  - A canary's own contribution is exactly C/n per round.
  - Inserted canaries give 80 scores per level; 1,000 unseen directions per run give 10,000 per
    level, as in A16.
- **Estimate:** `canary_audit` applied directly to the round-averaged scores s(u): μ̂ =
  (mean inserted − mean unseen) / sd of the unseen averaged scores.
  - Because the spread is that of the averaged score itself, μ̂ already describes all T releases
    together. No √T scaling is applied, and no independence across rounds is assumed; correlated
    honest projections simply widen the unseen spread.
  - The interval uses 2,000 bootstrap resamples, seed 20261038. ε̂ is the GDP ε of μ̂ at δ 10⁻⁵.
  - Diagnostic only: the ratio of the averaged score's sd to (per-round sd / √T), where 1 means
    independent rounds.
  - *Correction before any run, 22 October:* an earlier draft of this bullet said to scale by √T,
    which would have double-counted the composition.
- **Reference:**
  - Theory for inserted vs not is √T/(2z).
  - The clean control must have a lower interval bound above 1, or the audit is again
    underpowered and is reported as such.
  - A lower bound above √T/(2z) under DP is a leak finding.
  - μ̂ below theory is expected only to the extent that honest updates add spread.

**A18 withdrawn before any run (22 October).**
- The round-averaged all-iterates score telescopes: Σ_t ⟨θ_t − θ_{t−1}, u⟩/T = ⟨θ_T − θ_0, u⟩/T.
  So A18 is A16 with the public initial model subtracted.
- That removes the init spread but not the erasure by honest training.
- Projected from A16's measurements:
  - The bias still moves about 0.65, which leaves an unseen spread of about 0.02 against a canary
    shift of about 0.005, so clean μ ≈ 0.2. The positive control would fail again.
  - Even weights-only canaries give clean μ ≈ 0.6, below the ε 8 theory of 0.83, so no leak
    could be detected there.
- No A18 run was made, and the engine hook written for it was removed uncommitted.
- In its place, a deterministic mechanism check over existing records
  (`repro/check_effusion_dp_noise_records.py`) is recorded in the
  [A16 result](effusion-canary-audit-r0-v1-result-2026-10-22.md).

## Amendment A19 — 22 October 2026: non-IID clients on R0, before any A19 run

**Question:** every earlier federated result used IID whole-patient clients. Under label skew:
- does FedAvg still match its IID result?
- do robust rules cost more without attackers (as reported for the median under heterogeneity)?
- does skew change the cost of central DP and the damage of Fang's attack?

- **Partition** (`fedlearn.simulation.partition.patient_dirichlet_partition`, partition seed as
  in A9):
  - Patients are kept whole. A patient is positive if any of their training images is Effusion-positive.
  - Shuffled patients are dealt one per client first, so every client has at least one patient.
  - Each class's remaining patients are then split across the 80 clients in Dirichlet(α)
    proportions, drawn independently per class.
  - α ∈ {1.0 (moderate skew), 0.1 (strong skew)}.
- **Cells:** 60 runs, all retained. Seeds 20261101–20261105, 20 rounds, R0, random-init head.
  - No DP (LR 0.1, clip 1.0): rules mean, trimmed (k = 16), median × α × clean (30 runs);
    plus mean and median under Fang's attack at 16 of 80 attackers, b = 2 (20 runs).
  - Central DP ε 4 (LR 0.03, clip 0.05, RDP z, replace-one): mean, clean × α (10 runs).
  - Attacker slots and every other seed are exactly as in A14, so each cell differs from its A14
    IID counterpart only in the partition.
- **Reported** (selection half; paired patient bootstrap, 2,000 resamples, seed 20261039):
  - absolute AUROC per cell;
  - non-IID − IID (same rule, condition and seed, against A14);
  - rule − mean under each α;
  - attack drop under each α;
  - heterogeneity per seed: clients without a positive patient, image-count coefficient of
    variation, and min/median/max positive share.
  - Refusals, non-finite runs and attack notes are reported as such.

## Amendment A20 — 22 October 2026: noise-aware norm filtering under local DP on R0, before any A20 run

**Question:** A12 found that under local DP, 20% non-compliant attackers break every aggregation
rule. Does the noise-aware filter of Xiang et al. (SIGMOD'23) restore robustness? What remains
against attackers that add the protocol's noise themselves?

- **Filter** (`server.norm_filter = {"k": 4}`):
  - The server drops any upload whose delta norm is outside the band an honest privatised update
    clip(Δ, C) + N(0, s²I_d) occupies, with s = 2zC. The band is
    `engine.local_dp_norm_band(C, s, d, k)`: s²·χ²_d ± k standard deviations plus the clip and
    cross terms.
  - It reads only privatised uploads, so it is post-processing and costs no privacy.
  - Expected honest false-drop rate is of order 10⁻⁴ per upload. It is measured and reported.
- **Attackers** (16 of 80, as A12):
  - **non-compliant**: no local noise, exactly A12's sign flip ×−4 and label flip.
  - **noise-matching**: the same attacks, but each attacker clips its own malicious delta to C
    and adds N(0, s²I) (`privatize: true`), which places it inside the band by construction.
- **Cells:** 105 runs, all retained. R0, A8 recipe (LR 0.03, clip 0.05), 20 rounds, local DP
  ε ∈ {8, 32, 128}, δ 10⁻⁵, rule `mean`, seeds 20261101–20261105.
  - Filter on: clean, non-compliant sign flip, non-compliant label flip, noise-matching sign
    flip, noise-matching label flip (75 runs).
  - Filter off: noise-matching sign flip and label flip (30 runs).
  - Filter-off clean and non-compliant cells are A12's, compared seed by seed from the pinned A12
    output (`summary.json` `5de3c331…`).
- **Reported** (selection half; paired patient bootstrap, 2,000 resamples, seed 20261040):
  - absolute AUROC per cell;
  - filter on − off (same ε, condition and seed);
  - attack drop with the filter on;
  - honest and attacker drop counts per round;
  - refusals and non-finite runs.
- **Limit stated in advance:** a noise-matching attacker is the strongest attacker this filter can
  face cheaply. An attacker that spends the band's slack on a larger malicious component is not run.

## Amendment A21 — 22 October 2026: norm-matching attackers and a coordinate-distribution test under local DP, before any A21 run

**Why:** A20 showed a norm-only filter passes an un-noised update whose norm happens to fall in
the band: 82 of 1,600 at ε 128, costing 0.073. A deliberate attacker can do this every round.

- **Attacker** (`match_norm: true`, 16 of 80):
  - The A9 sign flip ×−4 and label-flip deltas are rescaled, without noise, to the expected
    honest privatised norm √(C² + s²d), with s = 2zC.
  - This places them at the centre of the norm band.
- **Filter variants:**
  - **norm only** (A20, k = 4);
  - **norm + KS**: an upload in the band is also dropped if the Kolmogorov–Smirnov distance between
    its delta coordinates and N(0, s²) exceeds the asymptotic critical value at α = 10⁻⁴.
  - Both read only privatised uploads, so both are post-processing.
- **Cells:** 75 runs, all retained. R0, A8 recipe, local DP ε ∈ {8, 32, 128}, rule `mean`,
  seeds 20261101–20261105.
  - Norm-matching sign flip and label flip under each filter variant (60 runs).
  - Clean under norm + KS (15 runs), to measure the KS test's honest false drops and utility cost
    against A20's norm-only clean cells.
- **Reported** (selection half; paired patient bootstrap, 2,000 resamples, seed 20261041):
  - absolute AUROC;
  - norm + KS − norm only, same cell and seed;
  - attack drop against the matching clean cell;
  - honest and attacker drop counts;
  - refusals and non-finite runs.

## Amendment A22 — 22 October 2026: stronger frozen encoders on R0's protocol, before any A22 cache is built

**Question:** does a stronger frozen encoder raise centralised Effusion AUROC over R0? If it does,
does it shrink the central-DP cost? Tramèr and Boneh (arXiv:2011.11660) report that better
features make DP cheaper.

**Leakage rule:** an encoder is admissible only if its documented training data excludes NIH
ChestX-ray14 and datasets derived from it (for example RSNA Pneumonia). Checked before any build:
- **RAD-DINO: excluded.** Its released checkpoint was trained on all 112,120 NIH-CXR images,
  holdout included (model card).
- **XRV `all`, `nih`, `rsna` and `resnet50-res512-all`: excluded.** Their training sets include
  NIH or RSNA.
- **R2 = DINOv2 ViT-B/14 (`facebook/dinov2-base`): admitted, with a caveat.**
  - Self-supervised on LVD-142M: web images retrieved by similarity to ImageNet-22k/1k, Google
    Landmarks and fine-grained natural-image datasets; no medical source datasets.
  - Web chest X-rays cannot be ruled out entirely, but NIH ChestX-ray14 was not a training dataset.
  - Feature: the final-layer CLS token, 768-wide.
- **R3 = TorchXRayVision `densenet121-res224-mimic_ch`: admitted.**
  - Trained on MIMIC-CXR only; the library's weights naming lists only `mimic_ch`.
  - Feature: `features2()`, 1,024-wide, the same interface and preprocessing as R1.

**Caches:**
- **Splits:** built by `build_xrv_cache.build_cache` over train and the selection half only, with
  the same pinned image, decode and partition contract as R1. Holdout and test images are never
  read.
- **R2 settings:**
  - device: MPS, float32;
  - preprocessing: decoded grayscale, divide by 255, bilinear antialiased resize to 224 × 224,
    replicated to three channels, ImageNet mean/std.
  - MPS was checked to give bit-identical features across repeats and batch sizes before the build.
- **R3 settings:** CPU, deterministic, with the R1 contract.
- **Weights:** pinned by SHA-256 at build.
- **Verification:** each cache is verified by exact recomputation of a hash-chosen sample, as for R1.

**Gate** (centralised, selection half, as the original gate):
- Same L2 logistic grid, selection rule, standardisation and patient-cluster bootstrap
  (2,000 resamples, seed 20261042), for R0, R1, R2 and R3.
- A representation passes if its AUROC − R0 interval is above 0 and its AUROC is at least 0.80.
- All results are reported, whether or not they pass.

**Conditional DP follow-up (pre-registered now):** for each passing representation:
- clean and central DP at claimed ε 8/4/1 with the best R0 configuration (A8 recipe LR 0.03,
  clip 0.05, zero-initialised head, exact-GDP z, T = 20), five seeds;
- each compared seed by seed with A17's R0 `zero_T20` cells (seed 20261043).
- **Limits:** the recipe was tuned on R0. Wider heads (768/1,024 vs 512) add noise norm √d, as
  in A4, and that cost is part of what is measured.

## Amendment A23 — 23 October 2026: distributed DP (Skellam over coded LightSecAgg) on R0, before any A23 run

**Question:** can the trusted server be removed while keeping central-DP utility? If so, what does
it cost in accuracy, bits and bytes, and what can a client that skips the clip do?

**Mechanism:**
- **Encoding:** distributed Skellam (Agarwal, Kairouz and Liu, NeurIPS'21, arXiv:2110.04995, Algorithm 1),
  in `fedlearn.privacy.distributed_skellam`. Each client:
  - clips its update to C;
  - scales it by s and applies a shared randomized Hadamard rotation (d 1,026 padded to 2,048);
  - rounds it stochastically, retrying until the Proposition 4.2 bound holds (β = e^(−1/2));
  - adds Skellam(μ/n) noise and reduces modulo the largest prime below 2^b.
- **Scale:** s is the largest scale with 2kσ̃ inside the field, where k = 4 and σ̃ follows Section 5's heuristic.
- **Secure sum:** coded LightSecAgg (So et al., arXiv:2109.14236), in `fedlearn.security.lightsecagg_coded`.
  - Privacy threshold T = 40 colluding clients and recovery target U = 64 (up to 16 dropouts tolerated).
  - Each mask is cut into 24 pieces, so every coded share is 86 field elements instead of 2,048.
  - Run in-process every round. Each round's decoded sum is checked against the direct modular sum.
- **Accounting:**
  - Corollary 3.6 RDP at integer orders, composed over T = 20 full-participation rounds.
  - Converted to (ε, δ) with the same Mironov bound as the RDP accountant, at δ = 10⁻⁵.
  - Replace-one adjacency: Δ₂ is twice the rounded-norm bound and Δ₁ = min(√d·Δ₂, Δ₂²).
  - μ is the smallest value meeting the claimed ε.
  - The modulo is post-processing.
- **Checked before any run:**
  - the bound dominates the exact numerically computed Rényi divergence of 1-D Skellam;
  - at 31 bits the effective noise multiplier is within 1% of Gaussian RDP;
  - the T-subset invertibility property of the coded shares holds.

**Configuration:** A17's T = 20 zero-head configuration, with only the privacy mechanism changed.
- R0, A8 recipe (LR 0.03, clip 0.05), 80 clients, mean rule, five seeds, selection half.
- The server never sees an individual update, so it cannot clip.

**Cells:**
1. **Distributed DP:** b ∈ {12, 14, 16, 20} × claimed ε ∈ {8, 4, 1} × 5 seeds.
   - b = 12 at ε 1 cannot fit the field at any scale. It is recorded as "not run: infeasible", not as a zero.
2. **Central Gaussian, RDP-calibrated z, zero head:** ε ∈ {8, 4, 1} × 5 seeds. This is the
   matched-accountant control.
3. **Clip-skipping attackers:** sign flip ×10 (`sign_flip_scale`, scale −10) at 4 and 16 of 80
   clients, ε 4, 5 seeds.
   - Distributed DP at b = 16: the attackers skip the clip. Nothing on the wire reveals it.
   - Central DP with RDP z: the server clips every upload.

**Pre-registered contrasts** (paired patient bootstrap, 2,000 resamples, seed 20261044):
- distributed(b, ε) − central RDP(ε): the price of removing the trusted server, at the same accountant;
- distributed(b, ε) − A17 exact-GDP(ε): the total price against the best central configuration;
- central RDP(ε) − A17 exact-GDP(ε): the accounting gap with a zero head;
- attack damage (attacked − unattacked) per mode at ε 4, and distributed damage − central damage.

**Also reported:**
- analytic bytes per client per round for coded shares, full-length Shamir shares (the
  `lightsecagg.py` scheme behind A5's 158×) and a plain float32 upload, all on the same
  bit-packed basis;
- secure-sum exactness every round;
- wrapped-coordinate counts.

**Predictions** (stated before running):
- At b = 20, distributed DP is within ±0.005 of central RDP at every ε.
- Fewer bits cost utility first at ε 1.
- Against A17, distributed DP loses about A13's accounting gain (around 0.017 at ε 4).
- Clip-skipping attackers do far more damage under distributed DP than under central DP.
- Any secure-sum mismatch is an integrity failure.

**Limits:**
- Honest-but-curious server. Key agreement and authenticated channels are assumed, not modelled.
- The secure sum runs in-process: bytes are analytic, not measured on a network.
- No dropouts or stragglers.
- The Hadamard padding doubles the masked upload; a non-power-of-two rotation would avoid it.
- IID, one workload. The recipe was tuned for central DP.
