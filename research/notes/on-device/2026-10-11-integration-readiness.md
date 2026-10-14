# Mobile continuation integration candidate

The feature branch is prepared locally for integration into `mobSupp`. The
checked base is`5f5ee82`; it is an ancestor of the feature branch. An actual
`git merge-tree --write-tree mobSupp HEAD` produced a clean tree. This is a
fast-forward-compatible candidate, not an executed merge or push.

The continuation begins at`8c91746`. Its implementation commits add cooperative
Stop completion, late-registration cleanup, failed-rejoin backoff, explicit
release signing, native release TLS enforcement, Android allocator-symbol
isolation and bounded learning controls. The root working copy's existing
one-line test edit and untracked output directory remain outside this branch's
changes. g14 retains the same project path and receives committed source only;
local datasets and qualification outputs are preserved.

Evidence is split by claim:

- [Image interruptions](../../results/on-device/android-image-lifecycle-2026-10-10.json):
  three rounds, six unique contributions, exact phone/Linux/aggregate replay,
  forced idle, Stop/rejoin and a20second outage while waiting for the peer.
- [Signing and native TLS](../../results/on-device/android-release-tls-2026-10-11.json):
  actual signing validation, local signed packaging, signature mismatch refusal,
  same-key Debug update, native trusted/refused TLS controls and preserved data.
- [Learning follow-up](2026-10-11-learning-followup.md): one improved offline
  ResNet candidate and a NIH threshold that worsens false positives.

The numerical core remains identical to that used by the original image replay.
Later transport/signing/lifecycle edits do not extend the old replay's scope to a
physical Release training run. No production key, HTTPS endpoint, release-to-release
update, natural wireless Doze, hour-scale suspension or sustained thermal run is
qualified. Android tree training, alternate training arms, real-model DeComFL,
secure aggregation, iOS and text/LoRA retain the limitations listed in the
[qualification matrix](../../../wikis/mobile/13-qualification-matrix.md).

Full continuation checks and the final review outcome are recorded below before
handoff. Publishing and merging require a separate decision.

Verification:679 mobile tests, lint and TypeScript checks passed;661 runtime
tests passed with30 slow tests deselected; the host native core CTest binary
passed;61 Android unit tests and Debug/Release builds passed. Eight real Gradle
signing cases and both APK variants' allocator ELF checks passed. The90-cell
resolver matrix still has19 retained live receipts,4 historical reports and67
refusals, and every referenced evidence hash matches. No backend, desktop or
protocol behavior changed in this continuation.

The final review identified two Important lifecycle gaps: native Stop could race
client replacement/heartbeat startup, and failed registration could resume rounds
without a renewed heartbeat. Both are fixed in`73ca4bf`. The native lifecycle lock
covers ownership and heartbeat changes independently of the round/RPC state lock.
Three native interleaving regressions fail with that protection removed; all four
lifecycle tests and the complete host native suite pass. Three reconnect regressions
failed before the explicit registration-required state; all681 mobile tests,
lint and TypeScript checks pass after the fix.

The retained image run predates this review correction. Its completed updates and
numerical replays remain valid, but its recovered status RPCs do not establish a
renewed native heartbeat. The final fix is verified separately; this continuation
does not relabel the old image run as a new physical image-outage qualification.

The original review retained these boundaries: production deployment/physical Release training still
need identity and endpoint verification; natural Doze/thermal/long suspension and
upload interruption need further device runs; generalized ResNet/clinical NIH
performance need independent evaluation; unchanged unsupported mobile features
remain unavailable; hash-bound private physical evidence was inspected without an
independent repetition of the raw fixture/datasets. No Minor findings were deferred.

The [review-fix receipt](../../results/on-device/android-lifecycle-review-fixes-2026-10-11.json)
records the regression failures and final real-device native cancellation/heartbeat
checks. The final Debug and locally signed Release APKs contain`73ca4bf`; the
signing/TLS receipt binds their hashes. Release app execution and deployed secure
training remain unperformed.

Final phone verification passed after restoring an expired development tunnel:
Stop cancelled the blocked registration in371ms, no heartbeat started for that
cancelled registration, and a fresh accepted registration produced a server-observed
heartbeat. Both native fixture processes exited successfully and removed their
owned TLS forwarding route. All33 dataset files remain hash-identical and the
training service is absent. Development API/Metro routes remain available.

Signing checks preserve the installed identity and app data; same-signer Debug
updates were allowed to qualify native fixes. This clarifies the plan's initial
APK-freeze wording. The cost of that choice is retaining earlier physical runs
against their original APK instead of extending them to the updated transport.

A subsequent [fresh final-source image qualification](2026-10-11-final-source-image-reconnect.md)
closes the explicitly retained image-outage gap. It uses native `73ca4bf` and
JavaScript/Linux `50a01ca`: after a failed rejoin, the real upstream server accepts
registration and acknowledges a heartbeat before the held peer is released.
Continued liveness during pending native work, six unique contributions, exact
original phone/Linux/aggregate replay and preserved datasets are separately bound
in the new receipt. This supplements the earlier source-bound run. That run used heartbeat
round/step telemetry fixed at registration; pending native work does not
identify graph computation. That run left natural wireless Doze, graph-boundary
Stop, upload interruption and production release/security qualification open.

The subsequent [partial-upload qualification](2026-10-11-image-upload-interruption.md)
closes the bounded incomplete-image-upload case: no partial contribution or
checkpoint, recovery on the same run, one accepted phone round-one update and
three-round completion. Original phone/Linux outputs and the registered aggregate
replay exactly, with preserved datasets and owned helpers removed. The new
observer fixes/tests terminal-response CONTINUATION without modifying earlier
hash-bound drivers. This adds no application behavior. That partial-stream test left complete-upload
acknowledgement loss, long suspension, natural wireless Doze, accurate progress
telemetry and production release/security qualification open.

A separate [accepted-upload/lost-completion run](2026-10-11-image-acknowledgement-loss.md)
now verifies full server acceptance with a withheld response and no phone checkpoint.
The real provider reconnects and resubmits, the coordinator counts its contribution
once, and the phone checkpoints before the held Linux peer proceeds. All six
accepted outputs and their aggregate replay exactly; datasets and helper cleanup
are verified. The scope is one eight-second foreground USB response outage,
with unchanged application source. That run left accurate heartbeat progress, natural wireless
Doze, long suspension, server restart, client process death, thermal endurance
and production release/security qualification open.


The [native heartbeat progress follow-up](2026-10-11-heartbeat-progress.md) now
fixes round/phase/step reporting in native source `c331e59`, using the existing
wire and heartbeat cadence. Seven regression tests and two real vivo/Linux
image rounds verify completed-step reporting, post-upload waiting and round-two
reset. Four accepted updates reconstruct the final aggregate exactly; the app's
33 dataset files are preserved. Code review corrected two phase/round defects
before qualification. No new progress UI or physical numerical replay was added.
Earlier physical receipts remain bound to their original runtime, which was
preserved before rebuilding. The separate run below strengthens background
evidence; production Release/TLS training remains unqualified.

The [ten-round screen-off follow-up](2026-10-11-screenoff-longrun.md) completes
32.8 minutes of USB-powered Debug background operation with the unchanged
progress APK. No further debugger calls are made by the harness during that
window. Acknowledged round/step progression and successful upload-before-waiting
are verified for each round before the held Linux peer proceeds. Twenty unique
contributions reconstruct the registered final model exactly; checkpoint ten,
all 33 dataset files and post-run source/historical-evidence preservation pass.
Independent samples show the display asleep, app/service records present and
battery temperature of 25–26°C. The run does not establish natural wireless
Doze, prolonged suspension, peak processor temperature, thermal endurance,
physical numerical client replay or production Release/TLS training.

The [local-training Stop follow-up](2026-10-12-training-stop.md) verifies the
actual provider's cancellation and same-run recovery after native `TRAINING`
is acknowledged. Stop takes 106.993 seconds in this case: the current local
computation settles before teardown, with the dataset pinned and another
Start/Join rejected during that wait. No cancelled contribution or checkpoint
is recorded; resumed training produces four unique updates and an exact final
aggregate. All saved data/runtime and earlier evidence are preserved. This
qualifies cooperative Stop in the local-training phase, not exact operator
entry, immediate preemption or a worst-case latency bound.


The [wireless battery follow-up](2026-10-13-wireless-background.md) failed during
the first local image update. Android's event log attributes the process kill
to `com.vivo.abe`, preceding harness cleanup. There are no accepted updates or
checkpoint. All 33 saved files, installed runtime and earlier evidence remain
unchanged, and USB-only debugging is restored. Mobile background support is
therefore incomplete under this vivo policy even though the earlier USB
screen-off case passed. Next is an explicit app-specific background-policy
check and controlled retest; a power-management setting or code fix is not yet
proven.

The [charger-powered Wi-Fi attempts](2026-10-14-wireless-powered-attempts.md)
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

The [seven-round timeout follow-up](2026-10-14-wireless-powered-refreshed-failure.md)
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

The [heartbeat recovery correction](2026-10-14-heartbeat-recovery.md) separates transient heartbeat loss
from terminal Stop and routes it through bounded re-registration. Local mobile,
native transport/core and server-boundary checks plus the Android Debug build
pass. The candidate is not installed; powered physical requalification remains
pending. Review also found that the pre-existing DeComFL rejoin path discards its
trajectory state, so native continuation after round one remains OPEN. This does
not turn any failed physical qualification into a pass.
