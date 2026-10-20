# Learning baseline audit and immediate decision

20 October 2026. Read-only diagnosis after the approved baseline-first priority
change. No new classifier was fitted and no additional official-test scoring was
performed for this audit. The original 360-cell study remains unchanged.

**Subsequent target decision:** the follow-up should detect one specific condition.
Which condition remains open. The any-finding diagnostics below describe the
existing workload; they do not qualify labels or predictive performance for a new
condition. Select and audit that target before executing the proposed fitting
screen. Preserve the earlier study and cache identities.

## Verdict

The present evidence does not identify the source of the modest prediction
quality. The NIH experiments train classifier heads over frozen ImageNet ResNet-18
features. They do not establish a converged centralized NIH baseline, a task-tuned
image representation, or a clinical decision threshold. Expanding the privacy or
attack grid now would repeat that unresolved learning limitation.

The next experiment should therefore diagnose centralized learning on the existing
development features before investing in another large study or changing datasets.
This is an optimization/representation diagnostic, not yet the matched causal
comparison between centralized and federated training.

## What the inspected implementation actually does

The [feature preparation](nih-frozen-features-2026-10-19.md) uses ImageNet-pretrained
ResNet-18 with the image classifier removed, producing 512 values per image.
Images are resized to 224×224 with ImageNet RGB normalization. The backbone stays
frozen; only a linear or small multilayer classifier is trained.

`framework/src/fedlearn/simulation/workloads.py` loads those features unchanged
into `NihFeatureHeadWorkload`, preserves natural prevalence, and partitions whole
patients. `engine.py` trains local heads using plain SGD. Image normalization is
present; training-feature centering/scaling is not. That is a possible optimization
factor, not evidence that scaling will improve clinical performance.

The earlier [optimization pilot](nih-optimization-pilot-v2-report.md) found a real
cohort/schedule confound: changing from ten to eighty clients changed each client's
minibatch trajectory. The [schedule pilot](nih-schedule-pilot-v3-report.md) then
established a stable research baseline around validation AUROC0.704, using seven
local epochs and twenty rounds. It did not establish convergence or clinical utility.

The completed study's clean eighty-client linear mean has test accuracy0.6420 and
AUROC0.6622 without DP, versus majority accuracy about0.6147. Those are historical
reference results. The larger-head comparison did not improve the observed private
ranking in its tested schedule; it does not establish that larger or fine-tuned
models cannot perform better under an independently selected training recipe.

Searching the NIH benchmark modules found no dedicated centralized NIH fitting
comparison. The older `frozen_vs_finetune_xray.py` harness uses a pneumonia-image
entry point and optional other workloads; its high reported AUROCs must not be
presented as NIH No Finding/any-finding results.

## Fresh development-data checks

The feature report still matches its trusted SHA256:
`6c9e6969c6fd762156ef2246323fbdbc5e9cb272c9910a8b1c9b691ab33783fd`.
Training and validation NPY/CSV payload hashes were rechecked against that report.
Only these development arrays were inspected by the new diagnostic.

| Check | Training | Validation |
| --- | ---: | ---: |
| Images | 68,650 | 17,763 |
| Patients | 22,399 | 5,600 |
| Finding-label images | 28,515 | 7,470 |
| Finding prevalence | 41.54% | 42.05% |
| Always predict No Finding: accuracy | 58.46% | 57.95% |
| Feature width | 512 | 512 |
| Zero-variance features | 0 | 0 |
| Feature SD: minimum / median / maximum | 0.05893 / 0.45600 / 1.41454 | 0.05904 / 0.45439 / 1.41539 |
| Raw image paths checked / present | 3 / 3 | 3 / 3 |

Every inspected feature value was finite. Feature means and marginal standard
deviations are broadly similar between these development splits; this is not a
distribution-equivalence test. Feature SDs span about24× within training, so a
train-fitted scaling control is justified. These observations do not establish a
cause of poor utility, exclude correlated/irrelevant features, revalidate every
raw image, or certify report-derived labels as clinical truth.

The private diagnostic retains exact payload hashes and all aggregate statistics;
its SHA256 is `61d3319202ae2b8064035b1e60efa8c98b843c6eaa4f923cf8c51eb398abb8ef`.
No patient identities or prediction arrays are included here.

## Bounded next experiment

Write/freeze a development-only diagnostic protocol before fitting. Start with
training-prior/majority controls and a small predeclared regularized logistic
regression screen on raw versus train-standardized features. Record solver
termination, training and validation performance, calibration, operating-point
behavior, resource use and failures. Fit all preprocessing to training data only.
Retain every candidate rather than report only the best result. Scikit-learn's
[logistic regression reference](https://scikit-learn.org/1.7/modules/generated/sklearn.linear_model.LogisticRegression.html)
documents the solver/regularization controls; installed version is1.7.1.

This tests whether a stronger optimized linear probe extracts substantially more
signal from the existing representation. It is not a mathematical AUROC ceiling,
an exact equivalent of the existing two-logit SGD trajectory, or an isolated causal
estimate of federation's effect. A matched centralized/federated training comparison
must hold architecture, initialization, data, preprocessing and relevant exposure
controls explicit in its own subsequent design.

If the optimized probe remains weak, investigate a medically pretrained image
representation or bounded fine-tuning next. Do not add DP/attacks to a baseline
that still fails the declared learning gate. Define that gate prospectively as a
research decision, without presenting an arbitrary score as clinical acceptance.

## Pretraining and final-evaluation constraints

TorchXRayVision provides separately identified
[NIH-, mixed-dataset- and CheXpert-trained models](https://mlmed.org/torchxrayvision/models.html).
A NIH or mixed-data checkpoint could have seen images/labels in the evaluation
cohort. It is not an eligible clean transfer baseline without an overlap audit.
An explicitly CheXpert-trained candidate is worth investigating, with pinned
weights, preprocessing and data provenance; no performance gain has been measured
here, and those weights have not been installed or downloaded by this audit.

The same overlap check must cover any proposed external final evaluation: a dataset
used to pretrain a candidate is not automatically an unseen test. The current NIH
official-test outcomes have already been inspected, so they cannot serve as a new
untouched selection/evaluation split. Availability and permissions for genuinely
independent final data remain unresolved. Do not silently obtain new credentials,
accept data agreements, or relabel an already examined subset as untouched.

The proposed purpose is flagging finding-labelled X-rays for human review, pending
the user's clarification. It is a research task using weak report labels. A higher
accuracy or AUROC alone would not prove clinical usefulness; intended use, missed
findings, false alarms, calibration and independent validation still matter.

## Simulator handoff

The simulator remains required at the end. Its worker is committed and verified;
the persistent API has a32-case focused pass after the reproduced review fixes,
and the full backend gate is running. New UI expansion should wait behind the
learning-baseline work. This audit neither completes that API gate nor changes
the original study, its numerical code, models, scores or conclusions.
