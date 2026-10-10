# NIH preparation and private image clients

## NIH ChestX-ray14

```sh
python scripts/prepare_nih.py --source data/NIH_Chest_Xray \
  --out output/nih-pneumonia --clients 2 --validation-fraction 0.2 --seed 42
```

This standard-library command writes train, validation, test and client CSV
manifests, plus `report.json`. Image paths are relative to `--source`; images
are neither copied nor moved. Existing output and inconsistent metadata/splits
are refused.

Labels mean `PNEUMONIA_ABSENT` or `PNEUMONIA_PRESENT`: whether `Pneumonia`
appears in the report-derived finding list. Other findings remain in the
manifest. `No Finding` is not relabeled clinically normal. This is a binary
research task, not the full 14-label NIH task.

The official test list stays unchanged. Official train/validation patients are
stratified by whether any image has the pneumonia finding, ordered by seeded
SHA-256, then divided into validation and training. Training patients are
allocated round-robin within each stratum to clients. All images of a patient
stay together. Patient strata are balanced; image counts need not be.
No image records are undersampled or duplicated.

The report records input and manifest SHA-256 values, algorithm, seed,
image/patient/positive counts and overlap checks. Filename/metadata checks do
not establish that all image contents are uncorrupted.

Use the separate `NIH_PNEUMONIA_CNN` recipe, base model
`nih_pneumonia_cnn`, Adam, FULL and zero pretraining. Its supported training
methods are FedAvg and Robust, with `LOCAL_SNAPSHOT` data. The existing
`PNEUMONIA_CNN` retains `NORMAL/PNEUMONIA` semantics.

The Python client accepts its own `client-N.csv` via `--dataset-path`, with
`--nih-source` pointing to the original image root and `--partition-id N`.
It verifies metadata, manifest hashes, labels and patient ownership, then loads
images lazily without materializing the full prepared partition in memory.
Image contents are hashed at binding and checked again when read.

Configure server evaluation with `FEDLEARN_NIH_SOURCE` and
`FEDLEARN_NIH_PREPARED`. It evaluates `validation.csv`; the official test
manifest is never used for round monitoring. By default all validation images
are evaluated. An explicit `FEDLEARN_NIH_EVAL_MAX_IMAGES` limit and
`FEDLEARN_NIH_EVAL_SEED` (default 42) produce a recorded proportional sample.
The registry card includes actual counts, source/manifest/sample/image hashes,
accuracy, balanced accuracy, AUROC, average precision, confusion counts and
majority baseline. Ranking metrics are null when the sample has one class.
NIH has no generic accuracy target in the benchmark dashboard.

A phone or a Python client can instead import a bounded image package:

```sh
python scripts/make_nih_image_package.py --source data/NIH_Chest_Xray \
  --prepared output/nih-pneumonia --manifest client-0.csv \
  --out output/nih-phone.zip --count 32 --seed 42 --balanced --patient-unique
```

Balancing and patient-unique selection are explicit sample choices, intended
here for execution qualification. Omitting them preserves proportional image
sampling. The sidecar records selected records, counts and hashes. Held-out
packages require `--purpose eval`, and must not be used as training data.
Outputs are never overwritten. Pixels use the recipe's PIL grayscale/resize;
contracted float32 conversion and normalization reproduce Python tensor bits.
XGBoost additionally needs a fixed numeric feature extractor; raw NIH images
do not supply its tabular schema.

## Python image clients

The canonical client accepts a raw image ZIP or an Android snapshot directory
under a published `LOCAL_SNAPSHOT` execution contract:

```sh
python fl-runtime/client.py --model-type CNN --strategy FedAvg \
  --project-id "$PROJECT_ID" --run-id "$RUN_ID" \
  --partition-id 0 --server-address "$GRPC_ENDPOINT" \
  --execution-contract contract.json --dataset-path private-images.zip --device cpu
```

Supply the enrollment connection token through the existing client auth
configuration, never shared scripts. A ZIP contains exactly `dataset.json` and
`records.jsonl` in the format written by `scripts/make_image_package.py`.
Pixels are already resized 8-bit HWC values. The client applies only the
contract's image-to-tensor and normalization steps, in float32, once.

An Android snapshot directory contains `snapshot.json`, `inputs.f32` and
`targets.i64`. Prepared CHW float32 tensors are loaded without another transform.
Schema, preparation fingerprint, hashes and tensor lengths must agree.
Shape and labels alone do not identify preprocessing.

All local records are trained. Recipe downloads, partitioning and implicit
validation splits are bypassed. Batches use the authoritative server round and
contracted seeded epoch order, retaining partial batches as declared.
Compressed sources, decompressed records and prepared tensors are each bounded
at 256 MiB; large datasets need smaller local packages. Malformed packages,
different classes/transforms, non-finite tensors and unsupported contracts are
refused before training.

This source-client support does not qualify the packaged desktop executable or
every recipe/strategy combination. Preserve official held-out images when
preparing client packages.

## Qualification — 9 October 2026

CNN/FedAvg completed three rounds on an Android phone and a Linux CPU Python
client, each using a separate 320-image CIFAR training partition. Neither client
had image-content overlap with the official 10,000-image test split. Each of the
six accepted contributions replayed exactly on its original runtime, and their
aggregates reconstructed the registered final checkpoint exactly. Changed-label
controls produced different updates.

This run continued from a preceding three-round checkpoint. Independent registry
reload gave 25.83% test accuracy, matching server evaluation. This establishes
the mixed-device training path; it does not establish exact numerical agreement
between Android ExecuTorch and PyTorch. Phone uploads were 248,805 bytes per
round; measured training took about 3.2 seconds per round.

NIH preparation with two clients, seed 42 and a 20% validation fraction produced
68,959 training images, 17,565 validation images and all 25,596 official test
images. Client partitions contain 34,328/34,631 images and 346/338 pneumonia
positives. All 112,120 source images are accounted for, with no patient or image
overlap among clients, validation and test. The later NIH qualification below verifies its separate label contract and
evaluation loader.

Validation: 533 runtime tests passed (29 slow tests deselected); 63 focused
data/client tests also passed on Linux. The source loader is unit-tested for
grayscale pneumonia and resized RGB ResNet preparation. Later qualifications below and in the
[mobile guide](MOBILE_QUALIFICATION.md) supersede the earlier live-run gaps.

## NIH qualification — 9 October 2026

The new NIH recipe completed two FedAvg rounds on vivo Android and g14 CPU,
with separate patient-unique 32-image samples (16 positives/16 negatives each).
All four accepted updates replay bitwise on their original runtimes, and the
aggregate matches the registered checkpoint bitwise. Existing phone datasets
remain unchanged. Probe 137.2 seconds; training about 4.6 minutes/round; updates
about 103 MB each way. The Android catalog is cleared, with each device still
required to qualify its staged programs before training.

The final model predicts pneumonia-present for every image in the recorded
512-image validation sample: accuracy 1.17%, balanced accuracy 50%, AUROC 0.4384.
Model quality is poor. The official test set remains unused. See the
[qualification note](../../research/notes/on-device/2026-10-09-nih-pneumonia-mixed.md)
and [machine-readable result](../../research/results/on-device/nih-pneumonia-mixed-2026-10-09.json)
for hashes, counts, provenance and limits.


## Larger NIH learning controls — 9 October 2026

A second experiment used 128 balanced, patient-unique training images per
client (256 pooled), keeping the same proportional 512-image validation
sample and all official test patients held out. All six CPU contributions
replayed bitwise on their original Mac/g14 runtimes. The aggregate and
independently reloaded validation metrics match the registered checkpoint.

The three-round FedAvg model reached 91.80% accuracy, below the 98.83%
majority-class baseline. Balanced accuracy was 46.44%, AUROC 0.4417 and
average precision 0.01369. It missed all six pneumonia-positive cases and
produced 36 false positives. Accuracy alone would conceal this failure.

Three centralized epochs with persistent Adam at 0.001 collapsed to predicting
positive for every image. Lowering the rate to 0.0001 avoided that collapse,
but AUROC was still only 0.4292. Resetting Adam each epoch at 0.001 produced
AUROC 0.5277 and almost all-positive predictions. Neither control supplies a
useful classifier. Centralized and federated batch orders differ; these are
diagnostic controls, not a causal comparison of their algorithms.

The [retained learning report](../../research/results/nih/learning-controls-2026-10-09.json)
records each epoch, probabilities, metrics and checkpoint hashes. The
[control driver](../../research/benchmarks/nih_learning.py) verifies package
receipts and disjoint patients/images before training. Raw inventory and
ownership were checked on the source host at packaging; g14 verified the
bounded packages. Only six validation positives make estimates uncertain.
Larger samples and an appropriate training setup remain necessary.


## Bounded learning diagnostics

The research control driver keeps its original three-epoch comparisons by
default. For a predefined longer comparison, supply explicit rates and a fixed
CPU thread count:

```sh
python research/benchmarks/nih_learning.py --packages output/nih-diagnostics \
  --initial initial.npz --out output/nih-controls --seed 42 --epochs 10 \
  --persistent-rates 0.0001 0.00001 --reset-rates --threads 4
```

An empty `--reset-rates` disables reset-per-epoch controls. Budgets are bounded
to 20 epochs, four distinct rate/lifetime combinations and eight CPU threads.
The result records the actual PyTorch thread count; exact numerical replay
requires the same thread configuration. Existing outputs are refused.

Packages must bind to the verified source preparation and have disjoint
training/validation patients, raw image hashes and prepared pixel contents.
Balanced held-out packages are diagnostic case samples. Their accuracy and
precision do not estimate performance at NIH's disease prevalence. The
proportional reference sample and official test set must remain separate from
training or parameter selection. These research flags do not change the
platform's published NIH execution contract.


## Extended diagnostic results — 9 October 2026

The continuation doubled pooled training to 512 balanced, unique-patient
images and used a separate balanced validation sample with 128 positive and
128 negative patients. Both predefined persistent-Adam configurations completed
ten epochs from the same retained initial state and seed. Four CPU threads
were recorded after a two-sample synthetic batch timing probe; the incomplete
one-thread pilot was retained
and excluded from completed-control claims.

| Fixed rate | Training AUROC | Balanced validation AUROC | Balanced validation accuracy | Proportional reference AUROC |
| --- | --- | --- | --- | --- |
| 0.0001 | 0.8313 | 0.6410 | 61.72% | 0.4107 |
| 0.00001 | 0.7109 | 0.6286 | 59.38% | 0.3883 |

The untrained initial model scored AUROC 0.4606 on balanced validation. The
new controls avoid all-positive collapse and show modest discrimination in
that diagnostic sample. They still do not establish useful NIH performance.
The original proportional reference has only six positive cases: the first
control found three but produced 264 false positives; the second found one and
produced 188 false positives. Neither is a useful classifier at that threshold.

The balanced and proportional held-out samples share 169 patients and 134
source images; they are not independent replications. Neither overlaps training.
Both final checkpoint reloads reproduce their balanced validation metrics
exactly with the recorded thread count. The official test set was unused.

The [extended report](../../research/results/nih/extended-diagnostics-2026-10-09.json)
retains all twenty epochs, initial-model controls, package/checkpoint hashes,
thread timing and reference metrics. Research rates do not change the current
platform NIH contract. A suitable pretrained image baseline and stronger
validation remain the next learning investigation before platform integration.
