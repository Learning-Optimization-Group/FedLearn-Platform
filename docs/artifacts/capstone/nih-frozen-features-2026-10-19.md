# NIH frozen feature preparation — 19 October 2026

Status: implemented, full real-data preparation completed and independently
qualified. No classifier has been evaluated on official test by this preparation.

## What the cache represents

Every selected image in the separately curated NIH cohort receives one ordered
512-dimensional float32 vector from a frozen ImageNet-pretrained ResNet-18, with
the classifier replaced by an identity layer. This is a fixed representation for
later trainable-head comparisons, not a trained NIH classifier. Training,
validation and official test retain their original curated order, image identities,
patient identities and report-derived binary labels.

The original failed content audit and original manifests remain unchanged. The
selected cohort excludes 111 images from nine implicated development patients and
retains every official-test image. Three ambiguous exact-RGB groups remain within
official test and are disclosed in the [data audit](nih-data-audit-2026-10-19.md).
The cache does not resolve clinical label errors, near-duplicates or true patient
identity uncertainty.

## Extraction and identity

Preparation fully reconstructs the original image audit and curated policy once,
then checks each selected image's raw bytes, dimensions and decoded RGB identity
again at extraction. Model construction uses a scoped seed and restores CPU,
available CUDA and available MPS RNG state. All parameters are frozen, every module
is in evaluation mode and forwards use inference mode. Ordered preprocessing
workers receive at most one forward batch at a time; the model stays in the parent
thread. There is no full-resolution image collection in memory.

Preprocessing is RGB, bilinear antialiased 224×224 resize, float32 conversion from
uint8 divided by 255, then ImageNet mean `[0.485, 0.456, 0.406]` and standard deviation
`[0.229, 0.224, 0.225]`. Provenance records versions, actual device, thread/worker
counts, batch size, resolved weight enum and actual model state digest. Random
weights are explicitly labelled as non-pretrained test controls.

The model-state digest starts with `fedlearn-frozen-state-v1\0`. For each sorted
state-dictionary name, canonical JSON describes name, torch dtype and shape; its
eight-byte unsigned big-endian length precedes those bytes. The eight-byte raw
tensor byte count precedes contiguous flattened state bytes. Parameters and buffers
both contribute. The digest is checked again after extraction.

## Reload contract and limits

The directory contains exactly nine split payloads (one CSV, feature NPY and label
NPY per split), the exact curated cohort report and the final feature report.
The final report is published after all payloads are flushed, using atomic rename;
an interrupted directory is retained and refuses reload. Existing outputs are
never overwritten.

Reload streams payload hashes, validates fixed filenames and canonical reports,
binds split CSV digests to the copied curated report, and checks nonempty disjoint
patient/image ownership, labels, dimensions, float32/int64 dtypes and finiteness.
Arrays are read-only memory maps and object-pickle loading is disabled. Future
engine configurations must pin the trusted feature report SHA-256.

These are integrity/provenance checks, not digital signatures. Replacing both
features and an unpinned report with mutually consistent hashes cannot be detected
without the trusted digest or feature recomputation. The standalone loader does
not re-read original images or certify the feature values computationally; the
separate extraction and direct-reference checks supply that evidence.

## Verification so far

The final full framework run passed 1,750 tests, with 14 slow tests deselected
and 81.09% coverage; unhandled thread exceptions were errors. Another 209 focused
NIH, workload, reproduction and RNG tests passed. Focused checks cover
all 24 synthetic NIH images against a separately built random ResNet-18 reference,
source mutation after verification, whole-patient quarantine exclusion, bounded
frozen forwards, RNG restoration, malformed/tampered arrays, interruption, pinned
digest mismatches and fresh-process CLI behavior. An additional regression first
reproduced, then fixed rejection of an NPZ archive disguised as a NPY payload.

The first fresh-process backbone test also reproduced an existing circular import:
privacy helper → server package → strategy factory → composed privacy pipeline.
Factory exports now load on demand; fresh privacy-first, backbone-first and
factory-first import orders all pass, including public named/star imports.

## Scientific boundary

No learning quality, attack resistance, privacy/utility frontier or head-size
effect follows from a completed feature cache. Next come explicit engine split
selection, patient-preserving clients and a bounded training/validation-only pilot.
Only then can a prospective, resource-bounded confirmatory protocol be frozen.

## Real cache qualification

Preparation exited zero in **641.24 seconds**, including fresh full-source audit
and curation reconstruction. It used MPS, batches of 64, four preprocessing workers,
four audit workers and four torch CPU threads. The cache occupies approximately
228 MiB. OS-reported child-process maximum RSS was 1,034,797,056 bytes; this excludes
GPU allocation and is not a whole-process-tree peak-memory measurement.

| Split | Feature rows | Patients | Feature width |
| --- | ---: | ---: | ---: |
| Training | 68,650 | 22,399 | 512 |
| Validation | 17,763 | 5,600 | 512 |
| Official test | 25,596 | 2,797 | 512 |

An independent invocation reloaded the cache with the report identity emitted by
the separate preparation process. It checked all **112,009** retained image/patient
identities and labels against original metadata, exact official-test membership,
absence of all nine quarantined patients, finite array values and disjoint patient
ownership. A separately constructed pretrained ResNet-18 matched the actual state
digest. Direct same-device forwards on the first 64 images of each split (192
vectors) had maximum absolute error **0.0**, within declared `rtol=1e-5, atol=1e-6`.
This sample equality is not a claim of cross-device or cross-version equality.

- Feature report SHA-256: `6c9e6969c6fd762156ef2246323fbdbc5e9cb272c9910a8b1c9b691ab33783fd`.
- Actual model state SHA-256: `5def31088ffcf5a8629286d3a79223fcd2f56edbd281524d99a2c1f047513c68`.
- Weights: `ResNet18_Weights.IMAGENET1K_V1`.
- Versions: torch 2.12.0, torchvision 0.27.0, NumPy 2.1.2, Pillow 12.3.0.

The private cache, exact manifests, execution log, independent audit and source
metadata remain outside git. The checked-in code, aggregate findings and report
identity define the reproducible preparation contract.
