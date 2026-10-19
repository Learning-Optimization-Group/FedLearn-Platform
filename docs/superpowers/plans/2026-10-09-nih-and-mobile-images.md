# NIH preparation and mixed mobile image qualification

Goal: prepare the downloaded NIH data for reproducible binary pneumonia experiments and close the Python image-snapshot gap before a mixed phone/g14 image run.

Authorized direction: prepare NIH and continue mobile image support. Continue inline in the existing isolated checkout; preserve the root checkout and phone datasets.

## Design

NIH preparation writes small CSV manifests referencing the original images, never copies or renames the dataset. Labels mean PNEUMONIA_ABSENT and PNEUMONIA_PRESENT, based on presence in the report-derived finding list. Keep the official test split unchanged. Divide official train/validation patients into validation and training, stratified by whether a patient has any pneumonia image; divide only training patients among clients. Save input hashes, split/label counts, manifest hashes and zero-overlap checks. Refuse malformed metadata, missing/duplicate images and overlapping/incomplete official splits before creating output.

The canonical Python client accepts an image package or an Android image snapshot under an exact LOCAL_SNAPSHOT contract. Packages use the contract's float32 image transforms; snapshots must carry the exact transform identifier, classes, shape and content hashes. All supplied records are trained with the existing seeded round/epoch order. Local data takes precedence over every recipe loader. No protocol, model class, optimizer or pretrained-weight changes.

NIH's new label names deliberately differ from NORMAL/PNEUMONIA. This slice does not relabel negatives as normal or bypass the current recipe's label contract. A future NIH training integration needs an explicit dataset-specific label contract. Qualify the existing CNN recipe using disjoint CIFAR training images, leaving its official test set for evaluation.

## Tasks

- [x] Add test-first NIH preparation and CLI. Test repeated patient images, multi-label positives, deterministic results, exhaustive patient separation, malformed splits/data and refusal to overwrite. Run on the full downloaded inventory and independently check output.
- [x] Add test-first image package/snapshot loading, size limits, exact transform/schema/hash binding and first-order CLI dispatch. Reproduce and fix pneumonia local-loader ordering. Test original image bits, changed preparation, corrupted metadata/tensors, zip traversal/duplicate entries/count mismatches and unsupported inputs. Run the complete runtime suite.
- [x] Run real CNN/FedAvg on vivo plus g14 with separate local data; preserve snapshots and capture accepted updates. Check original-runtime contributions and final aggregation by replay. Record measured payloads, held-out results and limits. Update status and guide.
- [x] Review final source changes and run required checks; commit locally without merge/push.

## Review focus

- A patient's negative and positive images stay in the same split/client.
- Validation/test images never enter training manifests or balanced qualification samples.
- NIH No Finding is not represented as a clinically normal label.
- A supplied image snapshot cannot fall back to a download/partitioner or different preprocessing.
- Large/malformed packages are rejected before tensor allocation, and cross-runtime replay claims match measured evidence.

## Work log

Ruling: qualify CNN first because it exercises the same image snapshot, seeded batch and update path in seconds; pneumonia requires much longer rounds and its NIH label contract is not yet representable. Cost if wrong: pneumonia-specific training remains unqualified; report it explicitly.


Completed: NIH full manifests independently account for 112120 images and preserve all 25596 official test images. Train/validation/client patient and image overlap is zero. Final files are delivered under the original checkout's ignored output/nih-pneumonia-seed42 directory; input images are untouched.

Completed: canonical image ZIP and Android snapshot clients pass 533 runtime tests (29 slow deselected), and 63 focused tests on g14. Python-produced tensors and snapshot IDs agree with Android's exact preparation format. NIH non-file images, oversized ZIP metadata and conflicting ZIP64 footers are refused. All protocol mirrors match.

Completed: vivo plus g14 CNN/FedAvg run ec6e3fd7-00b8-4bd9-a4c9-709265cd0a94, three rounds, two separate 320-image partitions, six accepted updates. The retained registry parent matches the published initial-state hash. Replays on both original runtimes and aggregate reconstruction are exact; changed-label controls differ. Registered checkpoint a0263f43fbb68417c482a21b3c096982711fc3f0508f3d25b4948969c423f00f gives 25.83% on the official 10000-image test set; independent reload agrees. Phone uplinks are 248805 bytes; three rounds take about 3.2 seconds each. Qualification receipts remain private/local.

Ruling: stop a cache-miss attempt before clients train, verify the original torchvision cache by its built-in integrity checks and copy it into the isolated runtime cache. Cost if wrong: held-out provenance; disjoint client image hashes and independent official-test evaluation verify it.
Ruling: repeat the first completed CNN run from its registered checkpoint because its starting state was not retained. The repeated run preserves the starting checkpoint before training and verifies its contract hash. Cost if wrong: extra training rounds; report this as a warm-started qualification, not a fresh-initialization accuracy comparison.
Review: independent review found two malformed-input issues; both reproduced before fixes. Follow-up exposed a ZIP64 override and it also received a failing regression then a fix. Final focused review found no remaining concrete defect. Packaged desktop, NIH label-contract integration, mixed ResNet/pneumonia and cross-runtime numerical equality remain outside this qualification.
