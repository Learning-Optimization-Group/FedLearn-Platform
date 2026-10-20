# Condition feasibility audit — development labels only

20 October 2026. The requested follow-up is one specific condition; its identity
is still pending. This audit informs that choice without fitting classifiers,
changing the historical labels or scoring the official test. It is not a frozen
experiment protocol or evidence that any candidate model performs well.

## What the available development cohort supports

The curated training and validation CSV hashes match the existing cohort report.
Patient and image membership are disjoint. Independent Python-set and SQL
counts agree for every listed condition in both splits. Counts below describe
report-derived labels, not clinically verified disease status. A patient can
contribute several images and several conditions.

| Condition | Train positive images / patients | Validation positive images / patients | Always-negative training accuracy |
| --- | ---: | ---: | ---: |
| Atelectasis | 6,619 / 3,335 | 1,654 / 844 | 90.36% |
| Cardiomegaly | 1,438 / 1,019 | 268 / 208 | 97.91% |
| Consolidation | 2,211 / 1,262 | 637 / 353 | 96.78% |
| Edema | 1,067 / 587 | 311 / 160 | 98.45% |
| Effusion | 6,779 / 2,790 | 1,876 / 709 | 90.13% |
| Emphysema | 1,114 / 605 | 309 / 157 | 98.38% |
| Fibrosis | 987 / 787 | 264 / 216 | 98.56% |
| Hernia | 114 / 81 | 27 / 21 | 99.83% |
| Infiltration | 11,001 / 5,705 | 2,761 / 1,402 | 83.98% |
| Mass | 3,146 / 1,683 | 884 / 431 | 95.42% |
| Nodule | 3,674 / 2,274 | 1,027 / 579 | 94.65% |
| Pleural Thickening | 1,753 / 1,236 | 489 / 323 | 97.45% |
| Pneumonia | 688 / 551 | 188 / 146 | 99.00% |
| Pneumothorax | 2,053 / 854 | 582 / 225 | 97.01% |

Training contains 68,650 images from 22,399 patients; validation contains 17,763
images from 5,600 patients. The always-negative rule has zero sensitivity for every
condition here. Its high accuracy demonstrates class imbalance, not useful detection.

## Candidate tradeoffs, before a condition is selected

- **Pneumonia:** only 551 positive-label training patients. If the cohort is
  partitioned into 1,000 nonempty, patient-disjoint clients, at least 449 clients
  must have no positive-labelled training image. This is a counting lower bound,
  not a measured federation result. The earlier mobile pneumonia scores do not
  qualify a new capstone baseline.
- **Pneumothorax:** 854 positive-label training patients and 225 in validation.
  At 1,000 patient-disjoint clients, at least 146 must have no positive label.
  Expert-reviewed labels exist for a subset of NIH, but their IDs/patients must be
  mapped against prior training, selection and pretraining before use. Reannotation
  does not make previously used images an untouched final evaluation.
- **Pleural effusion:** 2,790 positive-label training patients and 709 in validation
  provide a larger positive cohort than those two alternatives. This favors
  feasibility of a bounded federated study, but does not establish cleaner labels,
  easier generalization or clinical utility. Its NIH label is named `Effusion`.

Condition selection should reflect the intended use and evaluation evidence, not
which label produces the largest accuracy number. Do not select the best condition
by fitting and scoring all fourteen against the same validation set.

## Reference standards and transfer-learning constraints

Google documents 4,374 NIH images with adjudicated labels for four findings,
including pneumothorax; obtaining the labels requires its access form. These are
additional annotations of NIH images, not an independent source population.
The fourteen-finding expert set is a smaller PA-view subset of the same test images.
No access form was submitted or labels downloaded by this audit.
[NIH additional-label documentation](https://docs.cloud.google.com/healthcare-api/docs/resources/public-datasets/nih-chest).

VinDr-CXR supplies radiologist annotations and a separate Vietnamese source cohort.
Its files require a credentialed PhysioNet account, training and a signed data-use
agreement. Access is not established, and it is only a potential external cohort;
patient/content independence, condition definitions, sample size and eligibility
would still need qualification before a final evaluation.
[VinDr-CXR release and access requirements](https://physionet.org/content/vindr-cxr/1.0.0/).

TorchXRayVision distinguishes NIH-, CheXpert- and mixed-data pretrained models.
A mixed or NIH checkpoint cannot be assumed independent of NIH evaluation images.
A CheXpert-only checkpoint is a candidate to audit, not yet an approved model;
its training-source overlap must also be excluded from any proposed final cohort.
[Model provenance documentation](https://mlmed.org/torchxrayvision/models.html).

## Required decisions before fitting

1. Select the condition and intended use. Specify whether absence of a label is
   treated as a weak negative, and retain other conditions in the negative class
   unless a different target population is explicitly justified. Comparing only
   positive cases with No Finding would define a narrower, potentially easier task.
2. Audit target-specific duplicate/contradictory-label groups. The old broad-label
   content audit does not automatically establish consistency of every new label.
3. Establish the independent final-evaluation route, overlap exclusions and access.
   The already inspected official NIH test stays historical. If an untouched
   evaluation cannot be obtained, report the follow-up as exploratory.
4. Freeze a small development-only model/optimization budget, class-imbalance
   handling, training-only preprocessing and validation-based operating point.
   Evaluate AUROC, average precision with prevalence, sensitivity/specificity,
   predictive value and calibration, with patient-level uncertainty. Set the
   learning gate prospectively; no single arbitrary accuracy target proves utility.
5. Run the centralized baseline and then a matched federated comparison. Only
   after a viable baseline should privacy/security and attack studies expand.

## Evidence identity

Private aggregate inventory SHA256:
`51a80fbba99346bb87d0b7cc86aac2299a77faff2911fae8bfb2c03dac7537aa`.
The inventory records both input manifest hashes, aggregate label/co-occurrence
counts and scope. No patient identifiers, images or prediction arrays appear in
this public-facing note. The raw study, broad-label cache and original protocol
were not modified.
