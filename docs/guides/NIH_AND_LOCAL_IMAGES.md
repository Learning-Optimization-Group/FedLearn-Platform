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
