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
