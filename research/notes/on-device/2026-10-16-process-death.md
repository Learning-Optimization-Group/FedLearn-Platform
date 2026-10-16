# Manual recovery after Android process death — 16 October 2026

A controlled crash now qualifies manual rejoin for one first-order image case:
CNN/FULL/LOCAL_SNAPSHOT/FedAvg, two participants, three rounds and 32 disjoint real
CIFAR training images per participant. Both the baseline and corrected app completed
the same run after process death. This supports testing the existing global-model
reload path before adding local model checkpoints. It does not establish automatic
resume or recovery for every first-order recipe or interruption point.

The baseline also exposed an actual lifecycle defect. Android killed the phone's
original process, then restarted `FlForegroundService` in a new process and posted
its training notification. The app reopened as Idle, with no joined run or active
training loop. Its foreground service had returned `START_STICKY` and treated an
empty restart intent as an ordinary start.

Source `609d2cf` returns `START_NOT_STICKY`, stops empty restart requests and rejects
system retry/redelivery flags. Rejection and explicit Stop remove the notification
and pending device sampler. They pass the received start ID to `stopSelf(startId)`
so an older request cannot stop a newer queued explicit start. These choices follow
[Android's service delivery and stop semantics](https://developer.android.com/reference/android/app/Service).
The service remains a companion to the current process's training loop; it does not
attempt to reconstruct that loop or start training on its own.

Seven Robolectric regressions exercise the real service against Robolectric's notification
and handler shadows, an isolated plain Application and a paused SDK-29 looper.
Advancing the looper after stop checks that the device sampler was removed before
it could reach unavailable JNI. The original implementation failed three cases;
two further cases caught non-null retry/redelivery requests, and another caught an
unconditional stop that lost the start ID. The final Android suite freshly passed
68 JVM tests. The mobile JavaScript suite passed 699 tests across 45 suites, and
three existing native DeComFL guard/reconnect cases passed. The standalone Debug
assembly passed; its Hermes bundle and native archive are unchanged from the
previous installed package. Robolectric is a test-only dependency.

The [qualification receipt](../../results/on-device/android-process-death-2026-10-16.json)
binds source, the installed APKs, process/service observations, recorder windows,
actual desktop parents/configurations, UI actions and preservation evidence.

| Case | Phone process behavior after crash | Training result |
| --- | --- | --- |
| Baseline | PID 22159 died; Android restarted service in PID 28568 while Home showed Idle | Manual rejoin completed all three phone rounds |
| Corrected app | PID 18696 died; no phone process or foreground service in 19 samples over 15.90 seconds | New PID 21396 manually rejoined at round two and completed the run |

The active-run termination used Android's `am crash` command on the verified app
PID. Android's process-death events confirm termination. Direct shell SIGKILL was
denied by the phone and did not kill the app; no device security settings were
weakened. The result is a forced VM-crash qualification, not low-memory/SIGKILL,
reboot, force-stop or kernel interruption qualification. A final force-stop was
used only for an idle cold-launch check after both training runs had completed.

For each live case, the canonical desktop's first fit was held while the phone
submitted round one. The crash happened while waiting after that accepted upload,
not during native computation or an incomplete upload. Releasing the desktop
advanced the server to round two. Thirteen observations over another 10.8 seconds
showed no phone round-two submission before explicit restart. Cold launch restored
authentication and the contribution ledger, but the joined run and dataset choice
were lost. Normal Join → choose the existing dataset → Start completed rounds two
and three. The new session correctly reports two rounds; the persisted ledger
contains all three contributions.

Each completed run contains six unique accepted updates, three from each client,
with 32 examples per update. Reaggregation with the production FedAvg implementation
matches every float32 bit of the registered final model, including actual desktop
parent models at all three rounds. This proves the retained aggregate; it is not
independent numerical replay of the phone's training.

Test-driver limitations are retained explicitly. The baseline's first desktop hold
expired during setup, so that desktop was restarted with the same identity before
advancement; seamless desktop continuity is not claimed. Before the corrected case,
the previous SSH control connection had expired. The first desktop registration
failed and UI login could not reach the API. Those failures occurred before any
update in that case. A fresh loopback route was verified and the desktop restarted;
the successful fit then remained on the canonical training implementation. These
setup failures are retained and do not count as extra successful training runs.

Both tests used temporary USB reverse and SSH loopback routes. Vivo remained on USB
power for its weak battery, and Metro stayed stopped. No Wi-Fi, deployed TLS,
production signing, endurance, natural Doze or clinical-quality claim is added.
No support-matrix cells are added.

All 36 saved phone dataset files kept their hashes. The original saved account and
server setting were restored through the normal UI, and an idle Metro-free cold
launch succeeded. The test UI package, USB routes, SSH forwards, desktop clients
and local guards were removed or stopped. Vivo was left asleep on USB power. The
1,102 prior bindings were audited: existing evidence/helpers/artifacts remain
unchanged, while authorized source revisions are checked against their previous
and new commits instead of claiming those source files were unchanged. Historical
default build outputs and the previous installed APK remain preserved.

The next interruption tests should target first-order computation/upload boundaries,
including accepted updates whose client checkpoint is not durably saved. A local
submitted-round checkpoint is not an authoritative receipt that the server counted
an update. An explicit recovery offer could later retain run/contract/dataset and
account/server identity; the present result does not justify automatic training.
DeComFL remains separate: its reconstructed model and applied-history cursor must
survive together and reconcile with server history/pruning. Its late-start refusal
must remain until that design is implemented and qualified.
