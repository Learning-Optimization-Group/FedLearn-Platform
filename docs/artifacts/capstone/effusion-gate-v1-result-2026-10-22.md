# Effusion learning gate v1: result

22 October 2026. Executes [protocol v1](effusion-baseline-protocol-v1.md) §2–§5 and
§7 on the selection half. **The pre-registered gate failed on criterion (a).** That
is recorded as the result. The matched federated baseline (§6) was not started,
because it was conditional on a pass. The locked holdout and official NIH test were
not touched.

## Inputs that ran

- **R1 cache** `capstone-nih-xrv-chex-features-v1`. It holds 68,650 train and 9,075
  selection images (77,725 total) and was built in 66 minutes on CPU with no image
  refused. The 411 RGBA images (359 train, 52 selection) all had identical colour
  channels and full opacity, so they passed the decode contract. Report SHA256
  `6cf097b14866a31c1be8f654e5d1b16d1345822ae88e301d6090a130ada4b3d7`. The source hashes
  recorded by the build equal the committed files, and the model-state and XRV module
  hashes equal the [input probe](medical-input-probe-2026-10-22.md).
  - **Fresh-process verify:** all payload hashes and pins check out. 32 train and 32
    selection images chosen by hash re-extracted to exactly the stored rows.
  - **Disclosure:** the builder file was edited and then restored while the build was
    running. The recorded hash equals the committed file, and the exact replay above is
    the functional check.
- **R0:** the historical ImageNet ResNet-18 arrays, unchanged. Only train and
  selection rows were indexed; the full files were hashed for integrity.
- **Labels:** [`effusion_labels.py`](repro/effusion_labels.py) with exact-token,
  closed-vocabulary labels.

  | | Images | Positive images | Patients | Positive patients |
  | --- | ---: | ---: | ---: | ---: |
  | Train | 68,650 | 6,779 | 22,399 | 2,790 |
  | Selection | 9,075 | 992 | 2,800 | 364 |

  The train counts equal the feasibility audit. No duplicate-content group (7 in
  train, 1 in selection) has contradictory labels.

## All 20 fits

All 20 fits converged and produced finite outputs. The selected margins match a
recomputation without BLAS to 8e-15. That check was added because numpy 2.1.2 with
Apple Accelerate raises spurious matmul warnings.

| C | Weight | R0 train / selection AUROC | R1 train / selection AUROC |
| ---: | --- | --- | --- |
| 0.001 | none | 0.8310 / 0.8196 | 0.8448 / 0.8288 |
| 0.001 | balanced | 0.8351 / 0.8215 | 0.8504 / 0.8280 |
| 0.01 | none | 0.8361 / 0.8216 | 0.8534 / **0.8289** ✓ |
| 0.01 | balanced | 0.8381 / **0.8220** ✓ | 0.8577 / 0.8267 |
| 0.1 | none | 0.8367 / 0.8216 | 0.8571 / 0.8273 |
| 0.1 | balanced | 0.8383 / 0.8216 | 0.8606 / 0.8253 |
| 1 | none | 0.8367 / 0.8216 | 0.8580 / 0.8260 |
| 1 | balanced | 0.8383 / 0.8216 | 0.8612 / 0.8240 |
| 10 | none | 0.8367 / 0.8216 | 0.8581 / 0.8255 |
| 10 | balanced | 0.8383 / 0.8216 | 0.8613 / 0.8234 |

✓ = selected by the frozen rule. The grid barely matters for R0. For R1, weaker
regularisation raises train AUROC while selection AUROC falls slightly.

## Selected models on the selection half

The bootstrap used 2,000 patient-cluster resamples with seed 20261022; none were
skipped. The operating point is the highest Platt-probability threshold that gives
sensitivity ≥ 0.90.

| Model | AUROC [95% CI] | AP (prevalence 0.109) | Spec / PPV / NPV at sens 0.90 | Patient-level AUROC |
| --- | --- | ---: | --- | ---: |
| R0 ImageNet ResNet-18 | 0.822 [0.804, 0.839] | 0.374 | 0.544 / 0.195 / 0.978 | 0.904 |
| R1 CheXpert DenseNet | 0.829 [0.811, 0.847] | 0.397 | 0.565 / 0.203 / 0.979 | 0.906 |
| X CheXpert Effusion head, untrained on NIH | 0.787 [0.767, 0.808] | 0.268 | 0.503 / 0.182 / 0.976 | 0.886 |
| C0 constant | 0.500 | — | 0.000 / 0.109 / — | — |

**Gate:**

- (a) The paired ΔAUROC for R1 − R0 is **+0.007, 95% CI [−0.004, +0.018]**. The
  interval includes zero, so **(a) fails.**
- (b) R1's AUROC of 0.829 is at least 0.80, so (b) passes.
- R1 − X is +0.042 [+0.029, +0.054]: fitting a probe on NIH beats CheXpert's own head.

## What this does and does not show

- **No evidence that the medical backbone is better.** Under this frozen-probe
  protocol, CheXpert features are not distinguishably better than ImageNet
  ResNet-18 features. The R1 point estimate is higher, but within resampling
  uncertainty.
- **The Effusion target is learnable from the existing frozen features.** The same
  R0 arrays the historical any-finding study used reach 0.82 AUROC with a centralised
  probe. That study scored a different target on a different split with a federated
  head, so this is not a like-for-like comparison. It does suggest that target
  definition, not only representation, limited the earlier result.
- **Triage usefulness is modest.**
  - At 90% sensitivity, roughly half of negative images are also flagged, and about one
    flagged image in five is positive.
  - These numbers are in-sample for the threshold and the Platt fit, so calibration
    is not assessed here. Only the holdout can assess it (§8).
  - Labels are report-derived and weak.
  - None of this is clinical qualification.
- **Patient-level AUROC is secondary.** It takes the highest score per patient, so it
  favours patients with many images.

## Decision required (protocol §7)

The protocol allows privacy and attack work on a failed gate only by explicit
decision, and any new candidate needs a dated amendment. The two representations are
statistically tied and both clear the 0.80 floor. The realistic choices:

1. **Amend to proceed on R0** (recommended for feasibility):
   - Half the head size: 1,026 versus 2,050 parameters, so half the payload.
   - It already matches the simulator's qualified 512-wide workload.
   - Its pretraining is ImageNet only, so it carries no CheXpert-membership caveat.
2. **Amend to proceed on R1:** higher point estimate and domain pretraining, at
   twice the communication.
3. **Add a bounded fine-tuning candidate**, as §4 allows after a failed gate. This
   needs GPU time, risks the 5 December deadline, and needs a new amendment and budget.

The R0 and R1 heads, their selection scores and the Platt and threshold parameters are
retained privately for the one-time holdout evaluation (§8).

Private evidence: `gate.json` `aac1714a8f17eae05a639017776b82dbc128559031bebc2adc5df7954782921c`;
selected heads `000af03c…a18c`; selection scores `55a6b823…c112`; label audit `621072d9…6de4`;
cache verification receipt in the same private directory.
