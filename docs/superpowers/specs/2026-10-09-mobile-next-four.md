# Mobile qualification and real-data learning

Complete the four requested deliverables inline on the isolated feature branch,
starting from cc9a6c9. Preserve original checkout edits and imported phone data.
Keep all g14 files inside codebase/FedLearn_Platform. No push or merge.

## 1. ResNet mixed qualification

Use CIFAR_RESNET18/FedAvg/FULL and separate32-image CIFAR training samples on
vivo and g14. Retain the initial full checkpoint, contract, dataset identities,
accepted updates, registry artifact and phone telemetry. Two rounds suffice for
execution qualification. Replay each contribution on its original runtime,
including BatchNorm running statistics, and reconstruct the registered model.
Compare float32 bits and preserve withheld integer buffers separately. Changed
labels must alter a contribution. Clear mobile support only after actual probe,
training and exact original-phone replay. Do not claim model quality from this
small experiment or exact numerical agreement between different runtimes.

## 2. Larger NIH baseline

Use128 balanced patient-unique training images from each existing client split
(256 pooled), with the same512-image proportional validation sample used in the
NIH qualification. Verify rows against raw metadata/manifest ownership on the
source host and bind prepared tensors to provenance. Official test patients stay
unused. Run three epochs on g14, comparing centralized Adam at1e-3 and1e-4;
record optimizer persistence/reset, fixed seeds, train/validation loss, binary
metrics, confusion counts, probabilities and checkpoint hashes every epoch.
Compare the default-rate result with three real platform FedAvg rounds over the
same two client samples. Report collapse, weak ranking and sampling uncertainty
honestly. No automatic hyperparameter tuning against the official test set.

## 3. FedOpt/FedProx real-model support

Share server settings with the execution plans and make every supported Python
neural client honor the contracted server learning rate. Implement the fixed
global-anchor proximal gradient before Adam moments on device; mu0 keeps the
existing optimizer unchanged. Extend MLP/CNN/ResNet/pneumonia first-order plans
and Python/Java/TypeScript matrices, corpus and client checks. Include NIH only
when its closed backend/UI settings are updated consistently. Old native
libraries must refuse Adam+proximal training before provisioning via an explicit
runtime capability. Rebuild/install the same-signed APK in place, verifying all
dataset hashes. Qualify multi-batch ECG/FedOpt and ECG/FedProx on vivo+g14 with
original-runtime replay and wrong-rate/zero-mu controls. Leave unsupported arms,
DeComFL image behavior, DP and secure aggregation outside this change.

## 4. Real-data XGBoost

Default to binary Iris versicolor/virginica if no further dataset preference is
provided. Its four raw measurements fit the current closed CPU contract; map
f0..f3 to the named centimeter measurements and labels0/1 to species explicitly
in provenance. Do not silently expand the platform schema or imply multiclass
support. Group duplicate feature rows before deterministic train/validation/test
splits; preserve sample identities and hashes, keep heldout data external to the
server. Two CPU clients, four rounds/eight accepted trees, exact local and
aggregate replay, registered inference and a centralized same-tree-count baseline.
Record the local source/version/hash and small-sample limits. A clinical task or
broader feature schema is separate unless the user steers the dataset choice.
