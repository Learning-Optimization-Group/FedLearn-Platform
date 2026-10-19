# NIH binary pneumonia training integration

## Intent and scope

Continue the authorized NIH/mobile roadmap: make prepared patient-separated
NIH manifests runnable through the platform with their actual label semantics,
and qualify the existing pneumonia training architecture on phone plus CPU.
Keep the isolated branch, source images, existing projects and phone snapshots.

## Choice

Add a distinct `NIH_PNEUMONIA_CNN` recipe with classes
`PNEUMONIA_ABSENT/PNEUMONIA_PRESENT`. Reuse the existing 224-pixel grayscale
PneumoniaCNN architecture, dropout, float32 image preparation and Adam/FULL
contract. This makes dataset meaning explicit in project, enrollment, model
registry and inference. A generic label-override API would introduce semantic
negotiation without a need for it; changing the existing recipe would invalidate
its NORMAL/PNEUMONIA data and checkpoints.

Add recipe enum value 9 without renumbering existing values. Neural contract
version remains 1 and minimum round protocol 2: computation and wire updates
are unchanged, and older readers reject the unfamiliar recipe. Approve only
FedAvg and Robust on FULL; require LOCAL_SNAPSHOT for NIH participants. No
fixture data, FedProx/FedOpt/DeComFL, frozen-arm, DP or XGBoost-image expansion.

## Data

Validate prepared report/input/manifest hashes, all relative paths, CSV schema,
labels against source finding metadata, official test membership and patient
separation before using NIH manifests. The loader requires explicit source and
prepared-manifest roots, with no download or unrelated-data fallback.

Python clients may supply their own image package/snapshot as already supported.
For full local manifests, support lazy image loading through the seeded batch
loader so client partitions do not require multi-gigabyte eager tensors. A local
NIH manifest must be a client manifest from the verified preparation; test or
validation manifests must never be accepted as client training data.

Image decoding converts raw NIH PNGs with the existing grayscale/resize recipe
before applying the published float32 transforms. The package tool records
source image IDs, patients, manifest/source hashes, selected rows, preprocessing,
class counts and package hash. Count/selection/balancing are explicit options,
never hidden training-loader resampling. Training packages may only use client
manifests; evaluation packages may only use validation/test manifests.

## Evaluation

Round evaluation uses validation patients; the official test split is never used
by the training server. Default validation uses all its rows. An explicit fixed
size limit may select a reproducible proportional sample, recording actual
counts and manifest/sample hashes. Compute accuracy, balanced accuracy, AUROC,
average precision, confusion counts and majority baseline; return undefined
ranking metrics explicitly when only one label is present. Registry evaluation
cards identify NIH semantics, validation split, sample and metrics.

Do not imply useful classification from high accuracy on imbalanced data. Full
official-test evaluation belongs to a later fixed experimental protocol, not
adaptive round monitoring during this engineering qualification.

## Platform and mobile

Catalog initialization, model export/staging, contract assembly/validation,
canonical Python client/server, registry and inference use the new recipe.
Python, Java and TypeScript approve its exact first-order image matrix. Java
stubs regenerate from canonical proto; Python and TS committed stubs use pinned
generators. Proto mirrors remain identical. Existing pneumonia semantics and
XGBoost version-2 refusal remain intact.

Mobile uses the run's published class names when importing a new NIH image
package. No native kernels or model layout change; debug JS may update while
the installed native library remains the qualified build. Preserve existing
datasets and record the JS/source/native versions used. Do not advertise mobile
safety without actual program/device qualification.

## Evidence

Test-first coverage: altered manifests/labels/paths/hashes, leakage, source
fallback, seeded lazy-image order and partial batches, exact labels/transforms,
new/old contract acceptance, validation-only evaluation and metric provenance.
Run applicable full suites and a read-only independent review.

Live: two authenticated clients, two rounds, separate small explicit training
samples from distinct patient client partitions, passing phone model probe,
accepted updates, retained parents, registry artifact and original-runtime plus
aggregate replay. Use explicit small validation evaluation for qualification.
Record limits rather than claiming full NIH training or clinical quality.
