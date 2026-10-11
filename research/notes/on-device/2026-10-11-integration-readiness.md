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
