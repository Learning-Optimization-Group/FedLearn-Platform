# Android qualification matrix — 10 October 2026

Android portable CPU training is implemented for the 23 FULL-arm plans below.
Declared plans and live qualification are separate facts. The
[machine-readable matrix](../../research/results/on-device/mobile-qualification-matrix-2026-10-10.json)
enumerates all 90 combinations of six recipes, five strategies and three arms.
Every unavailable plan is checked against the actual resolver and retains its
refusal and required work. Nineteen plans have retained live receipts; four TinyNet plans retain September
live reports without a fresh rerun. The other 67 combinations are refused.

| Recipe | Declared FULL-arm strategies | Evidence scope |
| --- | --- | --- |
| TinyNet | FedAvg, FedOpt, Robust, FedProx, DeComFL | September live reports for all five; fresh FedAvg interruption checks. One-step FedProx does not discriminate its proximal term. |
| ECG MLP | FedAvg, FedOpt, Robust, FedProx | Own-data phone/Linux runs and original-runtime replays; 300 examples per client. |
| CIFAR CNN | FedAvg, FedOpt, Robust, FedProx | Own-data phone/Linux runs and original-runtime replays; 320 images per client. |
| CIFAR ResNet18 | FedAvg, FedOpt, Robust, FedProx | Own-data phone/Linux runs and original-runtime replays; corrected adaptive client rate 0.001, 64 images per client. |
| Original pneumonia CNN | FedAvg, FedOpt, Robust, FedProx | Current mixed-device cases use 16 balanced images per client; all listed strategies have original phone/Linux replay and aggregate proofs. |
| NIH pneumonia finding CNN | FedAvg, Robust | Separate NIH class schema and patient/content-disjoint client packages, 32 images per client; all listed strategies have original phone/Linux replay and aggregate proofs. |

The image/vector clients use their recipe optimizer. FedOpt adapts the aggregate
on the server; its client training is still ordinary local training. The current
adaptive client rate is 0.01 for ECG, CIFAR CNN and original pneumonia; ResNet
uses its corrected rate 0.001. Each receipt binds the actual contract and source
settings. Older ResNet runs at0.01 are retained as unstable experiments and do
not qualify the corrected setting.

Two clients running median Robust aggregation prove its execution path. They
do not establish tolerance of Byzantine participants. A one-batch pneumonia
FedProx check proves its declared contract path; the proximal term is zero on
that first local step, so that case does not measure a proximal effect.

## Reliability and packaged clients

The [interruption receipt](../../research/results/on-device/android-lifecycle-2026-10-10.json)
retains the original background failure and successful short screen-off,
actual socket-outage recovery and Stop/rejoin checks. The implementation uses
native scheduling because Android pauses JavaScript timers in the background.
Terminal status retires polling, and submitted-round state prevents duplicate
contributions after rejoin. The
[picker recreation receipt](../../research/results/on-device/android-picker-recreation-2026-10-10.json)
binds the native activity fix and preserved datasets. Automatic recovery of a
JavaScript import promise after process death is not claimed.

The [longer image interruption receipt](../../research/results/on-device/android-image-lifecycle-2026-10-10.json)
adds a three-round,16-image-per-client run: forced idle/screen off through the
first phone round, cooperative Stop/rejoin, and recovery from a20second active
socket outage while waiting for the peer. All six original contributions replay
exactly. Stop now keeps the stopping state until native work quiesces; failed
rejoins back off instead of exhausting recovery immediately. This forced-idle
case uses simulated unplug and USB routes; natural wireless Doze, hour-scale
suspension and sustained thermal behavior remain unqualified.

Final review found that this retained run's recovered status RPCs did not prove
renewed heartbeat after failed registration. The subsequent
[lifecycle corrections](../../research/results/on-device/android-lifecycle-review-fixes-2026-10-11.json)
require accepted re-registration before further rounds and serialize native
client replacement/heartbeat lifecycle against Stop. Regression tests and a
real native cancellation/re-registration fixture verify those corrections;
the [fresh final-source image run](../../research/results/on-device/android-image-reconnect-final-2026-10-11.json)
now separately observes accepted re-registration and an acknowledged upstream
heartbeat after a failed rejoin, before releasing the held peer. Another heartbeat
is acknowledged while resumed native work is pending. The new run also repeats
forced idle and Stop/rejoin, without relabeling the older receipt. Native heartbeat
round/step fields remain fixed at registration; this proves liveness rather than
accurate per-round telemetry. Pending native work does not distinguish graph
computation from a model RPC.

[Release readiness](14-release-readiness.md) records explicit signing gates,
local signed packaging, update-signature controls, preserved datasets and native
TLS positive/negative cases. Production signing and deployed HTTPS/gRPC checks
remain open.

The fresh Mac package qualified its normal login, native CSV file picker,
enrollment, selected model preset and Start flow in the
[breast-cancer XGBoost run](../../research/results/xgboost/breast-cancer-mixed-cpu-2026-10-10.json).
This exercises the embedded native client, not a Python-source substitute.
Directory and CSV/ZIP handoff, preset forwarding and Docker file mounts also
have regression coverage. Linux/Windows packaging and actual Docker training
are not qualified by this Mac run. The local Mac package is unsigned.

The [tree deadline check](../../research/results/xgboost/tree-deadline-cleanup-2026-10-10.json)
confirms failed-run reconciliation, server exit and absent final artifact. A
separate deterministic regression covers a client RPC detecting the deadline
before the server polling loop, which previously left a failed run waiting.

## What remains outside this qualification

- FROZEN_HEAD and OVA_LP need subset/frozen-state and objective bindings,
  exported programs, native projection and original-runtime checks.
- Real-model DeComFL needs recipe-specific zeroth-order settings, appropriate
  dropout/BatchNorm semantics and a practical compute budget. TinyNet support
  does not enable image DeComFL.
- Text and LoRA need on-device tokenization, sequence contracts, exported
  programs and memory-bounded device training.
- iOS needs a native training bridge and physical-device qualification.
- GPU/XNNPACK variants need probes, fallback/quarantine and replay evidence.
- Android secure aggregation is not implemented; the phone refuses secure runs.
- A persistent release identity, same-release-key update, deployed HTTPS/TLS
  training, natural wireless Doze, long suspension and sustained thermal behavior
  remain open. Local packaging/TLS and the bounded image interruption case above
  are now measured.
- Android tree training needs a tree runtime, transport and feature-schema data
  path. The working Mac/Linux XGBoost path does not enable it.

Debug APK execution over authenticated development plaintext is the measured
scope. Each device still has to pass the published program's probe. Model
quality is evaluated separately in the
[bounded learning controls](../../research/notes/on-device/2026-10-10-learning-controls.md),
which show that successful training does not yet yield useful NIH classification
or reliable longer ResNet convergence.
The [learning follow-up](../../research/notes/on-device/2026-10-11-learning-followup.md)
records an improved offline ResNet server-step candidate and a failed NIH
threshold adjustment, without changing platform defaults or claiming deployment.
