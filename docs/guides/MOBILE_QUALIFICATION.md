# Mobile qualification — 9 October 2026

Mixed-device training is now verified for ECG, CIFAR CNN, ResNet-18 and the
separate NIH pneumonia recipe. A successful run proves that clients contributed
reproducible updates and the server saved the correct aggregate. It does not
establish useful predictive performance.

| Recipe | Mixed phone/Python qualification | Remaining evidence |
| --- | --- | --- |
| MLP / ECG | FedAvg; multi-batch FedOpt and FedProx; two-round Robust | FedOpt server settings need tuning; broader datasets |
| CNN / CIFAR-10 | FedAvg; two-round FedOpt/FedProx/Robust, 320 images/client | Useful adaptive learning and subset arms |
| CIFAR_RESNET18 | FedAvg; two-round FedOpt/FedProx, 64 images/client | Useful adaptive learning, subset arms, larger runs |
| NIH_PNEUMONIA_CNN | FedAvg, two rounds, 32 images/client | Useful learning, live Robust; remains FedAvg/Robust only |
| PNEUMONIA_CNN | Earlier phone training; source/contract support | A mixed run with the original NORMAL/PNEUMONIA dataset |
| XGBOOST | Two Python CPU clients with real Iris rows | Android trees, broader schemas and clinical features |

MLP, CNN, ResNet and the original pneumonia recipe have tested FULL-arm
FedAvg/Robust/FedOpt/FedProx contracts. The new live adaptive-strategy evidence
covers ECG, CNN and ResNet. Other combinations must not be described as live-qualified solely
because their execution plans pass validation.

## ResNet-18

The first mixed run used separate CIFAR training samples and the official
10,000-image test set for evaluation. All four contributions replayed bitwise
on their original runtimes, including floating-point BatchNorm running
statistics. Changed labels produced a different update. Independent aggregation
matched the registered checkpoint and preserved integer BatchNorm counters.
The phone probe took 102.6 seconds; each upload was about 44.8 MB.

Slow synchronous server evaluation initially delayed upload acknowledgments and
heartbeats, causing an upload deadline and reconnection. The server now
acknowledges an accepted cohort before evaluating it in a worker, and keeps the
current global round until evaluation finishes. Worker failures and stops do
not register a successful final artifact. Regression tests cover heartbeat
responsiveness, early next-round updates, completion events and per-round
metrics.

A repeat completed after the upload fix. Its four updates reconstruct the
registered checkpoint exactly; independent registry reload gives 34.85% test
accuracy. Individual replay evidence is from the first run, and the repeat's
full phone telemetry was not retained. Neither run establishes convergence or
bitwise equivalence between PyTorch and ExecuTorch.

ResNet/FedOpt keeps BatchNorm running mean and variance out of server optimizer
moments and instead averages them with client example weights. This prevents
valid variances from becoming negative under an adaptive step. Two-round
regressions verify unequal client weights and finite BatchNorm evaluation;
the mixed-device adaptive continuation below now verifies this behavior live.

The catalog now allows ResNet on Android. Each device must still pass its own
program self-test and resource checks. This decision does not qualify iOS.
See the [ResNet receipt](../../research/results/on-device/resnet-mixed-2026-10-09.json).

## Adaptive strategies

Both two-round ECG runs used 300 local rows per client, batch size 128 and fresh
Adam state each round. Three local batches make the FedProx term measurable:
its fixed round-start anchor contributes `mu * (weight - anchor)` to gradients
before Adam moments. Changing the learning rate, removing the proximal term or
changing a label produces a different phone update. Both clients' original
runtime replays and server aggregation match their registered artifacts bitwise.

The Python client now honors and checks the selected optimizer's contracted
learning rate for every first-order recipe. FedOpt and FedProx require both
server rate and epoch settings; incomplete pairs are refused on every contracted
first-order run, matching native behavior. Contracted finite losses are trained
even above 100; non-finite losses refuse an update instead of silently skipping
batches. The latter mismatch was found during the first FedOpt run and verified
in a new run after its fix.

The same-signed Android update advertises `supportsAdamProximal`. Older native
cores refuse positive Adam proximal training before model provisioning. The
in-place installation preserved all 27 dataset files by SHA-256.

FedProx's recorded server ECG accuracy was 98.40%. FedOpt's default server rate
of 1.0 produced severe loss growth and 58.40% accuracy. These small execution
runs do not compare algorithm quality: they have different initial weights and
seeds. Server settings are now configurable as described below; appropriate rates still
need qualification for each workload.
See the [strategy receipts](../../research/results/on-device/ecg-adaptive-strategies-2026-10-09.json).

## Adaptive CNN continuation

CNN/FedProx and CNN/FedOpt each completed two mixed vivo/g14 rounds with
320 existing private CIFAR training images per client and ten local batches.
All eight contributions replay bitwise on their original runtimes, and both
aggregates reproduce their registered checkpoints bitwise. Changed labels and
learning rates alter phone updates; removing the proximal coefficient also
changes the FedProx update. All 27 phone dataset hashes remain unchanged.

Independent registry reload gives 15.09% official test accuracy for FedProx
and 10% for default FedOpt. The latter's phone loss rose from 2.30 to 283.27;
server rate 1.0 remains unsuitable here. Fresh projects use different seeds and
weights, so this is execution qualification rather than an algorithm comparison.
The original g14 clients pin OpenMP/MKL to one CPU thread; exact desktop replay
requires that setting. A default-thread comparison differed by 1.32e-6 and was
retained as a numerical reproducibility check.

See the [adaptive CNN receipts](../../research/results/on-device/cnn-adaptive-2026-10-09.json).

## Adaptive ResNet continuation

FedOpt and FedProx each completed two mixed vivo/g14 rounds with 64 private
CIFAR training images per client and two local batches of 32. The native phone
importer and Python preparation agree on the snapshot hash. All eight accepted
contributions replay bitwise on their original runtimes, including floating-point
BatchNorm statistics. Both registered aggregates are exact and preserve integer
counters. Native changed-label/rate controls differ; removing the proximal term
also changes the FedProx update, proving it affects training after the first batch.

FedOpt used explicitly recorded server Adam rate 0.01, betas 0.9/0.99 and tau
0.001. Those runs used client Adam rate 0.01. This did not
produce useful learning: independent official test reload scores 1000/10000
(10%) for each. Phone losses rose from 32108 to 48104681 for FedOpt and from
9388 to 30350732 for FedProx. The initial model states match, but round seeds
and aggregation differ; these runs do not establish algorithm superiority.
The lower client-rate controls below follow up this failure; these historical runs retain their original contracts.

Phone training compute was approximately 389/385 seconds for FedOpt and
402/485 seconds for FedProx, with about 44.8 MB uploaded per round. The app
reported a cached qualification probe; no fresh probe timing is claimed.
Offline native comparisons use the installed core in isolated processes and
compare parameter/buffer arrays, not full wire-blob metadata or cross-runtime
numerical equality. All 27 existing dataset files retain their hashes; only
three files were added for the 64-image snapshot. An unused prematurely staged
FedProx run was stopped and excluded from completed-run evidence.

See the [adaptive ResNet receipt](../../research/results/on-device/resnet-adaptive-2026-10-09.json).

## ResNet client-rate controls

New CIFAR_RESNET18 FedOpt and FedProx runs now use client Adam at 0.001,
the base recipe rate. The shared resolver supplies both the execution plan and
server round settings. Other recipes retain their existing rates; FedOpt's server
rate remains independently configured. Historical contracts keep their recorded
values, and a different server round rate is rejected before training.

A bounded CPU diagnostic compared 0.01, 0.001 and 0.0001 with identical initial
weights, seeds, two 64-example partitions and two rounds. Both partitions ran
on Python CPU. With 0.01, final aggregate loss on the 128 training examples was
about 8.25 million for FedProx and 286.49 for FedOpt. At 0.001 it was 0.332 and
0.648 respectively. The smaller 0.0001 control also remained bounded. FedOpt's
server rate was explicitly 0.01 and FedProx mu was 0.1 in all matching controls.

All six controls reproduced numerically. This diagnoses the rate's effect on
these CPU runs; training-set diagnostics do not establish generalization or
native-runtime stability. Official test images were unused for rate selection.
The retained driver checks the contracted initial state and refuses a different
recipe, round/batch budget or snapshot size.

See the [matched client-rate controls](../../research/results/on-device/resnet-client-rate-controls-2026-10-09.json)
and [benchmark driver](../../research/benchmarks/resnet_client_rates.py).

## ResNet FedProx at the corrected client rate

A fresh mixed vivo/g14 run completed two rounds at client Adam 0.001 and mu
0.1, with the same prepared 64 images per client and two batches of 32. The
phone reported losses 0.9980 and 0.7992, with approximately 400 and 408 seconds
of training compute. The independently reloaded registered checkpoint scores
4333/10000 (43.33%) on the untouched official CIFAR test set; mean cross-entropy
is 1.77117. Official test data was unused in selecting the rate.

All four accepted contributions reconstruct the registered checkpoint exactly,
and both original g14 updates replay bitwise with OMP/MKL pinned to one thread.
Both original phone updates also replay bitwise, including BatchNorm statistics.
The native replay exits successfully; these comparisons cover model arrays,
not full wire metadata or cross-runtime numerical equality.
All 30 pre-run phone dataset files retain their hashes. The installed APK hash
matches the earlier qualified runtime, and the app reused its cached probe.

This short run does not establish convergence or FedProx superiority. Its initial
state matches the earlier FedProx run, but the round seed differs; the earlier
10% result is historical context, not a matched mixed-runtime rate comparison.

See the [corrected-rate FedProx receipt](../../research/results/on-device/resnet-fedprox-client-rate-2026-10-10.json).

## ResNet FedOpt at the corrected client rate

A fresh mixed vivo/g14 run completed two rounds with client Adam 0.001 and
explicitly saved server Adam rate 0.01, betas 0.9/0.99 and tau 0.001. Each client
used its preserved 64 CIFAR training images, with two batches of 32 per round.
The phone reported losses 0.7840 and 0.8353, with approximately 392 and 389
seconds of training compute and 44.8 MB uploaded per round. Independent reload
of the registered checkpoint scores 3383/10000 (33.83%) on the official test set,
with mean cross-entropy 2.26524. Official test images were unused for rate selection.

Both original phone contributions and both original Python CPU contributions
replay bitwise on their respective runtimes. The accepted updates reconstruct
both aggregate parents and the registered checkpoint exactly. Floating BatchNorm
variances remain finite and nonnegative, and integer counters remain unchanged.
The installed APK and replay binary hashes match; all 30 pre-run phone dataset
files retain their hashes. Isolated native replay processes exited successfully
and their temporary files were removed. Replay equality covers model arrays,
not full wire metadata or numerical identity between the phone and Python.

The initial state matches both the earlier FedOpt run and corrected-rate FedProx,
but round seeds differ. These scores do not establish algorithm superiority or a
matched mixed-runtime rate comparison. Two rounds establish bounded execution,
not convergence or good classification across all classes: the final model has
zero recall for two classes. The default FedOpt server rate 1.0 remains unqualified.
The app reused a cached qualification probe, and no new native counterfactual
controls were run; earlier controls retain their original provenance.

An earlier attempt stopped after host sleep interrupted forwarding following
round 1. Its accepted updates and original desktop replay are retained separately;
it has no final checkpoint and is excluded from completed-run evidence. Keeping
the host awake enabled the fresh run; mobile suspension and interruption recovery
remain unqualified.

See the [corrected-rate FedOpt receipt](../../research/results/on-device/resnet-fedopt-client-rate-2026-10-10.json)
and [interrupted-attempt receipt](../../research/results/on-device/resnet-fedopt-host-sleep-interruption-2026-10-10.json).

## Remaining work

- Improve NIH generalization: extended lower-rate controls use 512 training
  images and 128 positive validation patients, but proportional reference
  performance remains poor. A frozen pretrained baseline also remains weak.
  The earlier federated model missed every positive.
- Qualify newly enabled adaptive strategies on image recipes and the remaining
  real-data/strategy combinations. Real-model subset arms and DeComFL image
  training remain outside the verified matrix.
- Qualify packaged desktop clients, iOS, and mobile lifecycle behavior such as
  background suspension and interruption recovery across supported models.
- Extend XGBoost only with an explicit shared feature schema and dataset choice.
  The current real-data test is binary Iris on Python CPU clients.

[NIH and private image guide](NIH_AND_LOCAL_IMAGES.md) records sampling and
learning limits. [XGBoost guide](FEDERATED_XGBOOST.md) records the tree workload.

## Recorded FedOpt server settings

Starts can now specify `fedOpt` with `variant` (`adam` or `yogi`),
`serverLearningRate`, `beta1`, `beta2` and `tau`. The resolved values are persisted
with the run and returned in its manifest, and the server receives those stored
values. The dashboard exposes variant and server rate. Omitted values preserve
Adam, rate 1.0, betas 0.9/0.99 and tau 0.001; omission is not a stability guarantee.
Settings on other strategies and non-finite or out-of-range values are refused.
The server rate is separate from the device learning rate published in the
execution contract. No protobuf change is needed for these server-only settings.

## Android interruption checks — 10 October

An actual screen-off run exposed paused React Native timers despite an active
foreground service. Android training waits now use a bounded native scheduler;
Stop cancels pending waits and older APKs without that scheduler are refused.
The installed APK retains the same numerical core and all 30 pre-existing
phone dataset files have identical hashes.

Fresh three-round TinyNet runs passed brief background/screen-off training,
an actual six-second socket outage with rejected reconnects, and Stop/rejoin.
Stop took 69 ms; checkpoint 1 survived, rejoin skipped the already accepted
round, and the resumed run reached checkpoint 3. Each successful run accepted
six unique contributions. Removing an ADB reverse mapping did not break an
existing socket and is retained as an ineffective outage probe.

[Lifecycle receipt](../../research/results/on-device/android-lifecycle-2026-10-10.json)
retains the failed attempt and successful cases. These USB-powered fixture
checks do not qualify forced Doze, long suspension, unplugged battery operation,
mid-upload acknowledgement loss, or image-duration interruptions.

The actual mounted provider also completed ECG/CNN Robust runs and cleared
obsolete server telemetry. Both original devices replayed their contributions
bitwise, changed labels/rates failed the controls, and the aggregate matched
the registered model. [Robust receipt](../../research/results/on-device/mobile-robust-real-models-2026-10-10.json).
Two clients using median aggregation establish execution support only.
