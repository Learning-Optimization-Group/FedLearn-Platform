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

NIH manifests are **not** a direct input to the existing `PNEUMONIA_CNN`
loader: its contract states `NORMAL/PNEUMONIA`. NIH needs an explicit
dataset-specific label contract and manifest evaluation loader before platform
training. Do not create `NORMAL` folders from NIH negatives to bypass that
distinction. XGBoost additionally needs a fixed numeric feature extractor;
raw image manifests do not supply its tabular schema.

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
overlap among clients, validation and test. NIH platform training remains
pending the dataset-specific label contract and evaluation loader.

Validation: 533 runtime tests passed (29 slow tests deselected); 63 focused
data/client tests also passed on Linux. The source loader is unit-tested for
grayscale pneumonia and resized RGB ResNet preparation. Those recipes have not
yet been qualified in mixed-device live runs.
