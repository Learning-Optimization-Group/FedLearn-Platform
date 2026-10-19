# NIH workload for capstone phase 2

Date: 18 October 2026. Status: proposed for review before implementation.
Parent plan: [current capstone plan](../../../wikis/capstone/README.md).

## Purpose

Add a reproducible NIH No Finding/any-finding feature workload to the existing
simulation engine, preserving patient identities through splits and client
allocation. The output is usable simulation data and a validation pilot, not a
clinical model or completed phase-2 result.

## Data contract

Read the original metadata, image tree and official train/validation/test lists
without moving or modifying images. Reuse the validated source/split checks from
the published mobile NIH preparation as a reference, with an explicit separate
label policy: exactly `No Finding` maps to 0; one or more recognized finding labels
maps to 1. Reject unknown, duplicate, empty or contradictory labels, duplicate image
IDs, inconsistent patient IDs, missing images, and official lists that overlap or
fail to cover metadata.

Keep the official test image/patient membership unchanged. Derive validation only
from official train/validation, assigning a patient wholly to training or validation.
The proposed fixed preparation uses seed `42` and a 20% patient validation fraction,
with patients stratified by whether any of their images has a finding and ordered
deterministically within each stratum. A patient with both image labels is still
one group. Record realized image/patient/class counts. Hash metadata and list files;
retain ordered image IDs, patient IDs, split identities and the preparation version.
Check corruption and cross-split content duplication while reading images; refuse
unresolved collisions rather than silently changing the cohort.

## Frozen feature preparation

Use the existing frozen-backbone construction and ImageNet preprocessing conventions:
ResNet-18 pretrained weight identity/digest, 224×224 RGB input, declared resize,
normalization, eval mode and no gradients. Record precise transforms, dtype,
library versions and execution device. Forward-pass batches are bounded; do not
load every full-resolution image into memory or construct an ImageFolder copy.

Write aligned float32 feature/label tensors and image/patient IDs for training,
validation and official test. Use a fresh output path and atomic completion;
interrupted preparation must not leave a cache that looks complete. Bind the cache
to source/split/weight/transform hashes. Refuse mismatches and overwrites.

Validation pilots evaluate validation only. Confirmatory configurations identify
their evaluation split explicitly and are frozen before official-test performance
is examined. Do not let the legacy `test_x` field silently stand for an unspecified
split. Triggered/backdoor feature preparation is a separate extension; refuse such
a configuration if aligned triggered features do not exist.

## Engine integration

Add a patient-grouped allocation path for NIH. IID allocation distributes whole
patient groups reproducibly; every selected training row belongs to exactly one
client, no patient crosses clients, and clients are nonempty or initialization
refuses clearly. Store assignment hashes and actual client class/sample counts.
The original image-row IID/Dirichlet algorithms and Kermany defaults stay unchanged.
Patient-grouped non-IID allocation must be named/documented separately from the
existing image-level Dirichlet law before it enters the protocol.

Preserve all metadata alignment if balancing/subsetting occurs. The first NIH
baseline keeps the observed class prevalence rather than invoking the existing
global balancing of both training and evaluation. Add the feature/source/split
and assignment identities to the run's workload provenance; do not claim patient
separation merely because the source manifest once had it.

## Intended code boundaries

- `framework/benchmarks/nih_data.py`: metadata validation, explicit labels and
  deterministic patient train/validation manifests. No training or downloading.
- `framework/benchmarks/prepare_nih_features.py`: bounded extraction using existing
  frozen backbone utilities; cache creation, integrity and alignment metadata.
- `framework/src/fedlearn/simulation/partition.py`: a separate whole-patient IID
  partition function with no change to existing row partition behavior.
- `framework/src/fedlearn/simulation/workloads.py`, `config.py`, `engine.py`:
  explicit NIH/evaluation/grouping support and bound provenance.
- Corresponding framework tests cover every contract below; existing Kermany
  anchor/reproduction tests remain required.

Do not merge the whole mobile branch or duplicate the canonical FL client. Keep
data preparation independent of aggregation, DP and attack implementations.

## Acceptance checks

1. A synthetic NIH-shaped image/metadata tree verifies each label, invalid-label
   refusal, complete list coverage, duplicate detection and patient consistency.
2. Seeded train/validation and client allocation cover the intended rows without
   patient/image overlap; repeated seeds reproduce assignments. Too many clients
   or missing grouping metadata refuse before training.
3. Features match a direct frozen-backbone reference on the same images/transforms;
   clean, validation, test and any triggered rows stay aligned. Cache identity or
   payload tampering, mismatched source/weights and existing outputs are refused.
4. Engine configurations select validation/test explicitly. A pilot cannot silently
   evaluate official test; run metadata binds the actual cache and assignment.
5. Existing Kermany workload behavior and capstone reproduction anchors remain
   unchanged. The full framework checks and coverage gate pass.
6. A small real NIH validation pilot completes with a no-DP/no-attack baseline and
   independently reloadable outputs. Report weak learning if observed.
7. A separate 1,000-client, two-round case measures execution, counts, time and
   memory. It adds no population-scale or useful-learning claim by itself.

## Explicit exclusions

No confirmatory matrix before a frozen protocol; no LoRA, full backbone training,
GUI implementation, clinical efficacy claim, patient-level DP claim, secure
aggregation, new mobile support or changed historical evidence. External advisor
messages, branch publication and final scientific choices remain separate actions.
