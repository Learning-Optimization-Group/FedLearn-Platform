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
in the new receipt. This supplements the earlier source-bound run. Heartbeat
round/step telemetry remains fixed at registration; pending native work does not
identify graph computation. That run left natural wireless Doze, graph-boundary
Stop, upload interruption and production release/security qualification open.

The subsequent [partial-upload qualification](2026-10-11-image-upload-interruption.md)
closes the bounded incomplete-image-upload case: no partial contribution or
checkpoint, recovery on the same run, one accepted phone round-one update and
three-round completion. Original phone/Linux outputs and the registered aggregate
replay exactly, with preserved datasets and owned helpers removed. The new
observer fixes/tests terminal-response CONTINUATION without modifying earlier
hash-bound drivers. This adds no application behavior. Complete-upload
acknowledgement loss, long suspension, natural wireless Doze, accurate progress
telemetry and production release/security qualification remain open.
