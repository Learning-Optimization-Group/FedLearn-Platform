# Medical feature candidate: source and synthetic execution checks

21 October 2026. This prepares one domain-pretrained candidate for the approved
learning-baseline investigation. It selects no disease, fits no classifier, opens
no NIH images and reports no new accuracy/AUROC. The completed historical study,
its cache and the working report retain their scientific scope.

## Verdict and observed execution

TorchXRayVision's explicit `densenet121-res224-chex` checkpoint loads successfully
with the tested Python/Torch environment and extracts finite pooled features.
It is a candidate for a later bounded development comparison, not a selected
winning representation or a clinically qualified model. The
[receipt](medical-backbone-candidate-2026-10-21.json) binds checkpoint/package/source
identities, the synthetic control and its fresh-process replay.

| Actual control | Result |
| --- | --- |
| Input | Two generated grayscale ramps; no patient data |
| Device | This Mac's CPU, four threads |
| Parameters | 6,966,034, frozen during the check |
| Intermediate feature map | 2 × 1,024 × 7 × 7 |
| Pooled features | 2 × 1,024, float32, all finite |
| Same-process repeat | Exactly equal |
| Fresh-process repeat | Feature/state/preprocessing hashes equal |
| Independent ReLU/average-pool projection | Exactly equals `features2()` |
| Parameters and buffers after inference | Unchanged |
| New learning metrics | None |

The model is identified upstream as trained on Stanford CheXpert. NIH and mixed-data
alternatives are separate choices; the mixed model includes NIH. A source label is
not an independent reconstruction of the checkpoint's original patient membership
or selection history. Modern inspected training scripts construct patient-group
splits and save the best validation checkpoint, but we have not established that
these exact script revisions produced the old release artifact.
[Model documentation](https://mlmed.org/torchxrayvision/models.html),
[pinned model source](https://github.com/mlmed/torchxrayvision/blob/17726dd8f5e756eec3e5ae958281a289b05b1dae/torchxrayvision/models.py),
[training entry point](https://github.com/mlmed/torchxrayvision/blob/17726dd8f5e756eec3e5ae958281a289b05b1dae/scripts/train_model.py),
[validation/selection code](https://github.com/mlmed/torchxrayvision/blob/17726dd8f5e756eec3e5ae958281a289b05b1dae/scripts/train_utils.py).

## Inputs that must change from the current ImageNet feature path

The checked XRV path takes one grayscale channel, maps eight-bit endpoints to
-1024 and +1024, center-crops the long dimension, then resizes to 224 × 224 using
the declared skimage path. The synthetic test exercised those transformations.
Real-image color conversion, bit-depth handling and content/row binding have not
been qualified by generated grayscale ramps. The existing RGB/ImageNet
normalization must not be silently reused.
[Normalization source](https://github.com/mlmed/torchxrayvision/blob/17726dd8f5e756eec3e5ae958281a289b05b1dae/torchxrayvision/utils.py),
[crop/resize source](https://github.com/mlmed/torchxrayvision/blob/17726dd8f5e756eec3e5ae958281a289b05b1dae/torchxrayvision/datasets.py).

Use the pooled feature interface. `features()` returns a spatial feature map;
`features2()` applies ReLU and average pooling. The pretrained classification
interface has eighteen slots, seven without a trained CheXpert target, and applies
published operating-point rescaling when thresholds are present. This check uses
pooled features and does not reuse its diagnostic scores or thresholds as NIH
predictions or operating points.
[Pinned feature/classifier implementation](https://github.com/mlmed/torchxrayvision/blob/17726dd8f5e756eec3e5ae958281a289b05b1dae/torchxrayvision/models.py).

The existing `simulation/nih_cache.py` validates a 512-column cache, and
`core_qualification.py` binds that workload identity. The dashboard's head table
also assumes that width. The candidate produces 1,024 values, so any later selected
workload needs a new versioned cache/report and dimension-aware qualification and
parameter accounting. A two-output linear head would contain 2,050 parameters,
compared with the current 1,026. The historical cache is not overwritten or relabelled.

## Reproduce the technical check

The exact library is TorchXRayVision 1.5.7. Its installed models, datasets and utils
files match the inspected upstream commit byte-for-byte. PyPI wheel bytes matched
their published digests. The checkpoint download came from the model source's
official release URL and matched the release asset size. The asset had no publisher
checksum; the observed SHA256 is a reproducibility pin, not an independent signature.
[PyPI release](https://pypi.org/project/torchxrayvision/1.5.7/),
[checkpoint release](https://github.com/mlmed/torchxrayvision/releases/tag/v1).

Checkpoint pin:
`1a914ae802170187a4852e594708206c12468eeeefd66aa4e8e26141649d61a4`.
The exact URL, filename, package artifacts, hashes and tested runtime versions are
in the receipt. Downloads and optional dependencies were kept in a separate
project-local directory; the existing framework environment was unchanged.

Install the optional packages in `repro/requirements-xray-candidate.txt` into an
isolated environment or target directory, alongside the recorded Torch/torchvision
and other base dependencies. Place the checkpoint in a chosen cache directory,
then run from the repository root using fresh receipt paths:

```sh
python docs/artifacts/capstone/repro/qualify_xray_backbone.py \
  --cache-dir "$CAPSTONE_XRAY_WEIGHT_DIR" \
  --output .temp/xray-candidate-check/receipt.json
```

The command checks the exact checkpoint pin before importing the optional model
runtime or deserializing the checkpoint. It refuses to replace existing evidence.
Those two refusal regressions were watched failing before implementation and then
passed; all nine standalone reproduction tests also passed. The actual positive
check and a fresh-process replay passed with the recorded dependencies. This is
same-environment synthetic execution evidence, not a hardware-general result.

## Next learning gate

Choose the condition and intended use, audit its labels/content groups, and establish
an independent final-evaluation route. CheXpert is a pretraining source for this
candidate and cannot automatically serve as its untouched final cohort. The inspected
NIH official test remains historical. Patient/content and pretraining independence
must cover the proposed final cohort as well as the development split.

Then freeze the bounded development protocol already called for in the
[baseline audit](learning-baseline-audit-2026-10-20.md): first diagnose raw versus
train-standardized optimized linear probes, then compare a justified medical
representation or bounded fine-tuning if required. Preserve all candidates/failures
and fit preprocessing on training data only. Qualify the clean matched federated
baseline before spending the study budget on privacy and attack utility sweeps.
