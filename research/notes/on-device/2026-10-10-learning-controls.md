# Bounded image-learning controls — 10 October 2026

The longer controls do not establish useful image classifiers. They separate
successful execution from weak learning, retain every round or epoch, and report
the final fixed-budget checkpoint rather than selecting a favorable intermediate
result.

## CIFAR ResNet18

The existing two 64-image training partitions were replayed on Linux CPU for six
rounds each under FedAvg, FedProx and FedOpt. All profiles share the original
initial state, seed and client Adam rate 0.001. FedProx uses mu 0.1; FedOpt uses
server rate 0.01. These are offline Python counterfactuals, not new phone runs.

A fixed 320-image balanced validation package comes from the CIFAR training
split. Prepared image contents do not overlap either training partition. The
official test split was not read. Final validation accuracy is 97/320 for
FedAvg, 108/320 for FedProx and 33/320 for FedOpt. Some classes still have zero
recall. Longer training did not produce reliable convergence, and the near
chance FedOpt result requires further experimental work before a quality claim.

The [complete receipt](../../results/on-device/resnet-longer-controls-2026-10-10.json)
binds the initial state, packages, driver, all trajectories and class-wise
metrics. Fresh participants reload the final checkpoints and reproduce the
reported predictions. The driver is
[resnet_client_rates.py](../../benchmarks/resnet_client_rates.py); `--longer`
requires an explicit validation package and checks prepared-image separation.

## NIH pneumonia finding

The follow-up pools the two existing 256-image training packages and fine-tunes
ImageNet V1 ResNet18's last block and binary head for a fixed ten epochs. Adam
persists between batches and epochs; layer4 rate is 0.00001 and head rate is
0.001, batch16, seed42. Earlier layers and every BatchNorm statistic remain
frozen, with their state checked after each epoch.

Training patients and image contents are disjoint from the 256-image balanced
validation package and the 512-image proportional reference package. The two
heldout packages themselves share 169 patients, so they are not independent
replications. The proportional package has only six positive images. Neither
package trains the model, and official test images remain unused.

Final balanced-validation AUROC is 0.631958 and accuracy is 59.375%. On the
proportional reference, AUROC is 0.554018; 284/506 negative images are false
positives and one of six positives is missed. Accuracy is 44.34%, far below the
98.83% majority baseline. The balanced-validation result does not establish a
useful classifier or clinical efficacy, and it does not justify selecting a
threshold using the reference set.

The [complete receipt](../../results/on-device/nih-last-block-controls-2026-10-10.json)
includes all ten epochs, preprocessing, patient/image separation, official
weight digest, package hashes, frozen-state checks and exact final-checkpoint
reload. The bounded driver is [nih_finetune.py](../../benchmarks/nih_finetune.py).
Further quality work needs a prospective protocol with adequate positive
validation samples and a reserved patient-disjoint test evaluation.
