# Mobile qualification — 9 October 2026

Mixed-device training is now verified for ECG, CIFAR CNN, ResNet-18 and the
separate NIH pneumonia recipe. A successful run proves that clients contributed
reproducible updates and the server saved the correct aggregate. It does not
establish useful predictive performance.

| Recipe | Mixed phone/Python qualification | Remaining evidence |
| --- | --- | --- |
| MLP / ECG | FedAvg; multi-batch FedOpt and FedProx | FedOpt server settings need tuning; broader datasets |
| CNN / CIFAR-10 | FedAvg; two-round FedOpt/FedProx, 320 images/client | Useful adaptive learning and subset arms |
| CIFAR_RESNET18 | FedAvg, two rounds, 32 images/client | Live adaptive strategies, subset arms, larger training runs |
| NIH_PNEUMONIA_CNN | FedAvg, two rounds, 32 images/client | Useful learning, live Robust; remains FedAvg/Robust only |
| PNEUMONIA_CNN | Earlier phone training; source/contract support | A mixed run with the original NORMAL/PNEUMONIA dataset |
| XGBOOST | Two Python CPU clients with real Iris rows | Android trees, broader schemas and clinical features |

MLP, CNN, ResNet and the original pneumonia recipe have tested FULL-arm
FedAvg/Robust/FedOpt/FedProx contracts. The new live adaptive-strategy evidence
covers ECG and CNN. Other combinations must not be described as live-qualified solely
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
this image/adaptive combination still awaits a mixed-device live test.

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
seeds. Exposing and qualifying appropriate FedOpt server settings remains work.
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

## Remaining work

- Improve NIH learning with a larger, adequately positive held-out sample and a
  suitable training configuration; current results miss every positive case.
- Qualify newly enabled adaptive strategies on image recipes and the remaining
  real-data/strategy combinations. Real-model subset arms and DeComFL image
  training remain outside the verified matrix.
- Qualify packaged desktop clients, iOS, and mobile lifecycle behavior such as
  background suspension and interruption recovery across supported models.
- Extend XGBoost only with an explicit shared feature schema and dataset choice.
  The current real-data test is binary Iris on Python CPU clients.

[NIH and private image guide](NIH_AND_LOCAL_IMAGES.md) records sampling and
learning limits. [XGBoost guide](FEDERATED_XGBOOST.md) records the tree workload.
