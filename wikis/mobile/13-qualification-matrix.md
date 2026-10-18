# Android qualification matrix — 15 October 2026

Android portable CPU training is implemented for the 24 FULL-arm plans below.
Declared plans and live qualification are separate facts. The
[machine-readable matrix](../../research/results/on-device/mobile-qualification-matrix-2026-10-15.json)
enumerates all 90 combinations of six recipes, five strategies and three arms.
Every unavailable plan is checked against the actual resolver and retains its
refusal and required work, with an explicit data source. Twenty-one plans have retained live receipts; three
TinyNet plans retain September live reports without a fresh rerun. The other 66 combinations are refused.

| Recipe | Declared FULL-arm strategies | Evidence scope |
| --- | --- | --- |
| TinyNet | FedAvg, FedOpt, Robust, FedProx, DeComFL | September live reports for all five; fresh FedAvg interruption checks. One-step FedProx does not discriminate its proximal term. |
| ECG MLP | FedAvg, FedOpt, Robust, FedProx | Own-data phone/Linux runs and original-runtime replays; 300 examples per client. |
| CIFAR CNN | FedAvg, FedOpt, Robust, FedProx, DeComFL | First-order phone/Linux runs and original-runtime replays use 320 images per client. DeComFL is separately qualified with 32 local images per client, one sequential batch and same-process recovery. |
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
forced idle and Stop/rejoin, without relabeling the older receipt. Those retained runs used native heartbeat
round/step fields fixed at registration, establishing liveness rather than
accurate per-round telemetry. Pending native work does not distinguish graph
computation from a model RPC.

A separate [partial-upload qualification](../../research/results/on-device/android-image-upload-interruption-2026-10-11.json)
cuts an original-pneumonia phone update after two complete chunks, before its
final chunk. No contribution or submitted checkpoint is accepted for that partial
stream. After an eight-second refusal window, the real provider re-registers and
retries; round one counts once. Three rounds finish with six unique updates, exact
original phone/Linux/aggregate replay and all 33 dataset files preserved. Matching
client-declared hashes do not prove the interrupted attempt's complete contents;
replay verifies the accepted retry tensors. This foreground USB case leaves
long suspension and production Release/TLS training unqualified; complete-upload
acknowledgement loss is tested separately below. The new observer tests terminal-response CONTINUATION;
the earlier hash-bound observer and receipt remain unchanged.

The [accepted-upload/lost-completion qualification](../../research/results/on-device/android-image-acknowledgement-loss-2026-10-11.json)
then exercises the other boundary: the server accepts the complete phone update,
but its successful response is withheld while the peer remains held. The phone
has no checkpoint, reconnects and submits again. Both full attempts complete at
the upstream server; the accepted recorder counts round one once, and the phone
saves its checkpoint before the peer is released. Three-round completion, exact
accepted phone/Linux/aggregate replay and all 33 preserved dataset files are
separately bound. Duplicate retry hashes are declarations; its tensor equivalence
is not independently captured. This covers one eight-second foreground USB
response outage, leaving server restart, client process death, prolonged partitions
and natural wireless background behavior unqualified.

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
- DeComFL beyond the small CIFAR CNN needs recipe-specific zeroth-order settings,
  appropriate dropout/BatchNorm semantics and a practical compute budget. CNN
  DeComFL currently accepts only 1–32 local images per client in one sequential batch.
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


The [native heartbeat progress qualification](../../research/results/on-device/android-heartbeat-progress-2026-10-11.json)
uses updated native source `c331e59`. Two real foreground USB vivo/Linux
pneumonia rounds acknowledge actual round advancement and `TRAINING 0/1` to
`WAITING 1/1` after upload completion, with fresh counts in round two. Four
accepted updates reconstruct the final aggregate exactly and all 33 datasets
remain unchanged. Seven host regressions cover step/phase/error semantics;
all 154 native, 681 mobile and 61 Android checks pass. This changes heartbeat
fields, without adding a progress UI or extending earlier physical receipts to
the new APK. Multiple steps/DeComFL are host-verified; fresh physical numerical
replay and Release/TLS qualification remain open. The separate background run
below qualifies a longer USB screen-off case.

The [ten-round screen-off qualification](../../research/results/on-device/android-screenoff-longrun-2026-10-11.json)
uses the unchanged progress APK and real vivo/Linux clients. The screen stays
asleep for 32.8 minutes, with no further debugger calls from the harness until
completion. Every round has acknowledged `TRAINING 0/1`, successful upload and
`WAITING 1/1` before the held Linux peer proceeds. Twenty unique accepted states
reconstruct the registered final model exactly; checkpoint ten and all 33 saved
data files are verified. The retained samples show app/service-record presence,
USB power and battery temperature of 25–26°C. They can miss brief wake events
and temperature peaks. This strengthens USB-powered Debug background evidence;
natural wireless Doze, long suspension, thermal endurance, physical Release
training and deployed TLS remain unqualified.

The [local-training Stop qualification](../../research/results/on-device/android-training-stop-2026-10-12.json)
requests Stop after two acknowledged `TRAINING round1 0/1` heartbeats. The app
retains its stopping flag and dataset pin and rejects another Start or Join.
Stop completes cooperatively after **106.993 seconds**; it is not immediate
during local computation. The cancelled interval has no accepted update or
checkpoint, and the observer sees no decoded upload chunks or submissions.
Same-run rejoin then completes two phone/Linux rounds with four unique updates,
an exact reconstructed aggregate and unchanged saved data. This verifies Stop
in the reported local-training phase, without identifying exact kernel entry,
proving mid-operator preemption or establishing a worst-case latency bound.


The [battery-powered Wi-Fi attempt](../../research/results/on-device/android-wireless-background-2026-10-13.json)
**failed before the first completed round**. Android records the active FedLearn
process being killed by `com.vivo.abe` at 20:05:43 UTC, before the harness stopped
the project. Across the 8.2-minute interval before failure detection, all retained samples
showed the display asleep and external power absent; light idle was sampled,
deep idle was not.
Acknowledged training heartbeats preceded the kill, but no accepted update or
durable checkpoint exists. The APK and all 33 saved files are unchanged.
Temporary Wi-Fi debugging and test helpers are removed; the original USB route
state is restored. This leaves unplugged background training unqualified under
the tested vivo policy. Inspect the app-specific background-power policy and
repeat the same contract with any changed setting explicitly recorded. A wake
lock or allowlist fix has not been validated; earlier successful USB receipts
retain their narrower scope.

The [charger-powered Wi-Fi attempts](../../research/notes/on-device/2026-10-14-wireless-powered-attempts.md)
did not complete the planned ten rounds after a FedLearn-only OEM background
opt-in. The first completed one round with two accepted updates and checkpoint
one before an ADB monitoring timeout; the fresh retry lost training heartbeat
acknowledgments during model download and accepted no update. The same app PID was observed
after both failures, with unchanged saved data. The current shared connection
also failed a bounded 10 MiB upload check; the original pneumonia model streams
about 98 MiB. This is separate transport-failure evidence, not a successful
background qualification or proof that the policy prevents OS kills. Charger
power, diagnostic Wi-Fi ADB and the opt-in are recorded conditions. Battery-only
survival and natural deep Doze remain open.

A fresh SSH comparison completed the same 10 MiB operation in 1.75 seconds.
The owned development tunnel was refreshed and API/Metro responses verified;
training over the refreshed connection still needs a separate qualification.

The [seven-round timeout follow-up](../../research/notes/on-device/2026-10-14-wireless-powered-refreshed-failure.md)
also failed the ten-round target after reopening the persistent connection.
Fourteen updates and checkpoint seven are retained. Round eight's upload was
partial: 62 chunks were observed, the server logged 61 of 99, and none was
accepted. The phone journal confirms a native upload deadline followed by a
clean-stop message; the app process survived. Code inspection finds heartbeat
failure and genuine Stop share the same abort flag, allowing retry to exit
without rejoining. This recovery defect needs a scoped fix and regression tests.
The internal flag reason was not directly sampled, and the underlying network
stall remains unexplained. No successful final aggregate, new app build or
ten-round qualification is claimed.

After stopping the owned test, temporary Wi-Fi ADB was disabled over its existing
connection. The original empty reverse/forward routes were verified before
shutdown; the USB-mode command was acknowledged and three connections to the
old port were refused. Saved files and APK matched before shutdown. Physical USB
identity, TCP-property and route postchecks remain pending. This cleanup does
not change the failed training verdict.

The [heartbeat recovery correction](../../research/notes/on-device/2026-10-14-heartbeat-recovery.md) separates transient heartbeat loss
from terminal Stop and routes it through bounded re-registration. Local mobile,
native transport/core and server-boundary checks plus the Android Debug build
pass. The candidate is not installed; powered physical requalification remains
pending. Review also found that the pre-existing DeComFL rejoin path discards its
trajectory state, so native continuation after round one remains OPEN. This does
not turn any failed physical qualification into a pass.

The [physical heartbeat recovery test](../../research/notes/on-device/2026-10-15-heartbeat-recovery-physical.md)
now qualifies the installed correction on vivo: three deliberately failed
heartbeat calls are followed by actual server re-registration and continuation
to round two, preserving checkpoint one without a duplicate contribution. Four
accepted states reconstruct the registered final model exactly. All 33 saved
files survive the app update and test. This is a controlled, USB-powered recovery
case; the internal native flag was not directly sampled.

The subsequent [powered Wi-Fi screen-off test](../../research/notes/on-device/2026-10-15-heartbeat-recovery-longrun.md)
completes ten rounds with twenty unique accepted updates, checkpoint ten and an
exact reconstructed final aggregate. Over 36.6 minutes, retained samples show
the same app process, asleep display, USB charging and battery temperature of
24–26°C. No debugger evaluation occurs during that window. Temporary Wi-Fi
debugging is disabled afterward and the original empty USB routes and all 33
saved files are verified. This qualifies externally powered Debug operation
over Wi-Fi with diagnostic ADB active; battery-only survival, natural deep Doze,
production Release/TLS and DeComFL native continuation after round one remain
open. Earlier failed cases retain their verdicts; this is not a controlled
causal comparison of the fix alone.

The [DeComFL reconnection correction and physical qualification](../../research/notes/on-device/2026-10-15-decomfl-recovery.md)
now closes the previously open continuation defect for an app process that stays
alive. The bridge retains the trajectory across network-client replacement, and
the loop applies each completed aggregate once, rejecting incomplete history
before committing it. Nine new native cases cover retention, repeated history,
contract isolation, malformed responses and frozen Python trajectory parity.
Three real vivo/Linux TinyNet rounds survive both a three-heartbeat failure and
four rejected round-two scalar uploads. Five identical round-two payloads,
six unique accepted contributions, checkpoint three and an exact reconstructed
final model are verified. Saved data and USB-only debugging are preserved.
This short powered, awake Debug fixture test does not qualify real-model DeComFL
recovery, app process restart, fresh late joining, response loss after server
acceptance, background endurance or production Release/deployed TLS. Previous
receipts retain their original APK bindings.

## CIFAR CNN DeComFL — 15 October

The [CNN DeComFL receipt](../../research/results/on-device/android-cnn-decomfl-2026-10-15.json)
and [qualification note](../../research/notes/on-device/2026-10-15-cnn-decomfl.md)
add one bounded image plan: FULL arm, LOCAL_SNAPSHOT, one sequential batch of
1–32 images per participant. Fixture runs and larger snapshots are refused.
Fresh functional exports correct the old loss-program bound of 15 despite a
nominal maxBatch of 32; older staged programs must be replaced before reuse.

A fresh three-round vivo/Linux run used 32 disjoint CIFAR training images per
client. Three failed heartbeat calls triggered accepted re-registration, then
four rejected round-two scalar uploads exhausted retries and triggered another
rejoin. All five round-two payloads were identical. Six accepted updates
reconstructed the registered final model exactly. Sampled screen-off recovery
retained one phone process, USB power and 26°C battery temperature. This was a
98.93-second Debug exercise, not battery-only endurance or natural deep Doze.
All 33 original dataset files remained unchanged; the new dataset added three
files. The original empty ADB routes and stopped helpers/guards were verified.

## Standalone internal Android package — 16 October

The [standalone qualification](../../research/notes/on-device/2026-10-16-standalone-debug.md)
adds packaging evidence for two existing cells, CNN/FULL/LOCAL_SNAPSHOT with
FedAvg and DeComFL. Both completed ten real vivo/Linux rounds with Metro stopped,
normal UI enrollment/import/Start, and exact final aggregate reconstruction.
The opt-in Debug APK embeds Hermes JavaScript; ordinary Debug retains Metro.
No support-matrix cells or production Release/TLS qualifications are added.

## Authentication lifecycle — 16 October

The [sign-out qualification](../../research/notes/on-device/2026-10-16-signout-lifecycle.md)
fixes local participation outliving its authenticated provider. Unmount now
cancels native work and the foreground service, with a shared guard preventing
new-provider overlap until old enrollment/training and final abort have drained.
Seven regression cases pass. A powered vivo/Linux CNN FedAvg run verifies no
phone round-two update while signed out, followed by explicit rejoin and exact
six-update final aggregate reconstruction. This is same-process cancellation and
manual rejoin, not active process-death recovery; no support cells are added.

## Manual recovery after process death — 16 October

The [process-death qualification](../../research/notes/on-device/2026-10-16-process-death.md)
covers a forced VM crash after an accepted round in CNN/FULL/LOCAL_SNAPSHOT/FedAvg.
The corrected standalone app leaves no orphan foreground service; explicit cold
launch, rejoin, dataset selection and Start complete the same three-round run.
All six accepted updates reconstruct the final model exactly. Seven Android service
regressions pass within the 68-test JVM suite. This adds no support cells, automatic
resume, low-memory/in-flight-compute qualification or DeComFL process recovery.

## Lost completion before process death — 16 October

The [lost-completion qualification](../../research/notes/on-device/2026-10-16-acknowledgement-crash.md)
holds the successful upload response after the server accepts the phone's complete
round-one update, then crashes the app before the native call returns. Explicit
cold launch and same-round rejoin produce an identical second upload; only the
first is counted. No third upload occurs during a 10.38-second observation. All
six unique updates reconstruct the completed three-round CNN FedAvg model exactly.
The installed standalone Debug APK and product source are unchanged. Checkpoint
absence follows source ordering; encrypted storage was not inspected and write
durability was not tested. No automatic resume, interrupted computation/partial
upload, DeComFL process recovery or new support cells are qualified.

## Incomplete transport upload before process death — 16 October

The [partial-upload qualification](../../research/notes/on-device/2026-10-16-partial-upload-crash.md)
forwards only the start of the first HTTP/2 upload frame, then crashes the phone
with zero server-counted updates. Manual cold launch, same-run rejoin and dataset
selection complete three CNN FedAvg rounds with six unique updates and an exact
reconstructed final model. The original and recomputed complete phone-side payloads
match, while the original upstream request remains incomplete and is discarded.
This tests transport before the first application chunk, not a partly populated
server upload buffer, interrupted computation or checkpoint-write durability.
Product source and APK are unchanged; no support cells or automatic recovery are added.

## Forced exit during native backward computation — 16 October

The [native-computation qualification](../../research/notes/on-device/2026-10-16-native-compute-exit.md)
verifies a round-one federated backward stack, then exits the unchanged app through
its mapped libc while the debugger holds it stopped. Android confirms actual
process disappearance and no orphan training service. Manual cold launch and rejoin
finish three CNN FedAvg rounds; all six accepted updates and actual desktop parents
reconstruct the final artifact bitwise. Each completed phone payload is bound to
its run, round, client and counted tensor state. This adds no support cells,
automatic resume, natural low-memory/SIGKILL qualification or DeComFL process recovery.

## DeComFL cold process restart — test 16 October, verified 18 October

The [DeComFL process-death test](../../research/notes/on-device/2026-10-18-decomfl-process-death.md)
confirms a recovery gap: after round one and actual process death, explicit cold
rejoin at round two refuses initialization. The mobile trajectory is held only in
memory and its initial bootstrap requires round one. No round-two scalar request
or counted update is observed in the bounded window; the isolated run is stopped.
Both accepted round-one grids reconstruct the captured server model bitwise, and
complete rebuild history is available. This adds no recovery support cell. Current
server-snapshot bootstrap and permanent-error handling need work before claiming
DeComFL cold recovery; checkpoint storage alone is not established as the solution.

## DeComFL current-snapshot cold recovery — 18 October

The [corrected cold-recovery qualification](../../research/notes/on-device/2026-10-18-decomfl-cold-restart-recovery.md)
closes the preceding first-round-only gap for CNN/FULL/LOCAL_SNAPSHOT. After a
powered-phone app crash and server advancement while the phone is absent, manual
launch/rejoin at round two downloads the current validated snapshot and completes
three rounds with six unique updates. The round-two phone snapshot matches the
independent server capture byte for byte; native replay exactly reproduces the ten
phone scalars. Incorrect initial-state and double-replay controls differ. Round three
needs no additional global download. The final model matches production reconstruction
and independent aggregation arithmetic bitwise. This adds bounded manual DeComFL
recovery evidence, not model support cells, automatic resume, natural-kill coverage,
checkpoint durability or production release qualification.
