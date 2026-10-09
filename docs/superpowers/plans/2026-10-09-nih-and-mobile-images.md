# NIH preparation and mixed mobile image qualification

Goal: prepare the downloaded NIH data for reproducible binary pneumonia experiments and close the Python image-snapshot gap before a mixed phone/g14 image run.

Authorized direction: prepare NIH and continue mobile image support. Continue inline in the existing isolated checkout; preserve the root checkout and phone datasets.

## Design

NIH preparation writes small CSV manifests referencing the original images, never copies or renames the dataset. Labels mean PNEUMONIA_ABSENT and PNEUMONIA_PRESENT, based on presence in the report-derived finding list. Keep the official test split unchanged. Divide official train/validation patients into validation and training, stratified by whether a patient has any pneumonia image; divide only training patients among clients. Save input hashes, split/label counts, manifest hashes and zero-overlap checks. Refuse malformed metadata, missing/duplicate images and overlapping/incomplete official splits before creating output.

The canonical Python client accepts an image package or an Android image snapshot under an exact LOCAL_SNAPSHOT contract. Packages use the contract's float32 image transforms; snapshots must carry the exact transform identifier, classes, shape and content hashes. All supplied records are trained with the existing seeded round/epoch order. Local data takes precedence over every recipe loader. No protocol, model class, optimizer or pretrained-weight changes.

NIH's new label names deliberately differ from NORMAL/PNEUMONIA. This slice does not relabel negatives as normal or bypass the current recipe's label contract. A future NIH training integration needs an explicit dataset-specific label contract. Qualify the existing CNN recipe using disjoint CIFAR training images, leaving its official test set for evaluation.

## Tasks

- [ ] Add test-first NIH preparation and CLI. Test repeated patient images, multi-label positives, deterministic results, exhaustive patient separation, malformed splits/data and refusal to overwrite. Run on the full downloaded inventory and independently check output.
- [ ] Add test-first image package/snapshot loading, size limits, exact transform/schema/hash binding and first-order CLI dispatch. Reproduce and fix pneumonia local-loader ordering. Test original image bits, changed preparation, corrupted metadata/tensors, zip traversal/duplicate entries/count mismatches and unsupported inputs. Run the complete runtime suite.
- [ ] Run real CNN/FedAvg on vivo plus g14 with separate local data; preserve snapshots and capture accepted updates. Check original-runtime contributions and final aggregation by replay. Record measured payloads, held-out results and limits. Update status and guide.
- [ ] Review final source changes and run required checks; commit locally without merge/push.

## Review focus

- A patient's negative and positive images stay in the same split/client.
- Validation/test images never enter training manifests or balanced qualification samples.
- NIH No Finding is not represented as a clinically normal label.
- A supplied image snapshot cannot fall back to a download/partitioner or different preprocessing.
- Large/malformed packages are rejected before tensor allocation, and cross-runtime replay claims match measured evidence.

## Work log

Ruling: qualify CNN first because it exercises the same image snapshot, seeded batch and update path in seconds; pneumonia requires much longer rounds and its NIH label contract is not yet representable. Cost if wrong: pneumonia-specific training remains unqualified; report it explicitly.
