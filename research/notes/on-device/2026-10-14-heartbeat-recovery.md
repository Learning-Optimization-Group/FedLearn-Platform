# Heartbeat loss requires bounded re-registration

The native client now distinguishes a dead heartbeat from genuine user/server
Stop. Exhausting the heartbeat failure limit raises a recoverable registration
requirement at status and round checks. The JavaScript loop enters its existing
bounded re-enrollment/registration path before another round call. A later
heartbeat failure cannot overwrite explicit Stop, and restarting a stopped or
heartbeat-dead client does not reset its participation state. The production
60-second RPC deadline, same-run/contract validation, durable submitted-round
checkpoint and server deduplication rules remain unchanged.

Implementation: f24e2cd. This is a local regression/build result, not a completed
physical background qualification. The previous ten-round powered Wi-Fi attempt
remains failed after seven accepted federated rounds. Its original network stall
cause remains unknown. The [receipt](../../results/on-device/android-heartbeat-recovery-2026-10-14.json)
binds the seven changed source files, private verification logs/drivers and the
separately retained candidate APK/native archive.

Verification:

- All 44 mobile JavaScript suites / 687 tests pass, including heartbeat-loss routing,
  partial-upload checkpoint preservation, retry bounds, Stop and submitted-round safety.
- All 154 native core tests pass. A real Linux loopback gRPC fixture passes all
  12 transport/marshalling tests; the old transport fails four of the final tests.
  These exercise heartbeat exhaustion, a real partial-upload deadline, replacement
  registration/heartbeat acknowledgement, terminal server Stop and user Stop.
- Replacing the Stop-preserving atomic transition with an unconditional connection-loss
  store fails the concurrent Stop regression; the other eleven native tests pass.
- The existing coordinator/stream-upload/run-end boundary tests pass (37 tests).
  ESLint and TypeScript checks pass.
- Android Debug assembly succeeds. Its JVM test task is up to date; retained XML
  reports contain 61 passing tests across five suites. Gradle and upstream native
  dependency deprecation warnings remain. The loopback test's plaintext permission
  is isolated to its test executable; the production Release transport policy is unchanged.

The candidate APK was retained separately and has **not been installed on vivo**.
The historically bound local APK/core were restored after the build; all 516
previous evidence bindings and the root working-copy edit remain intact. g14
currently has no ADB device. Wi-Fi debugging was already disabled; physical USB
identity/property/route postchecks are pending. The former development SSH master
is no longer present, while API and Metro remain present; the bridge must be
re-established before a physical test. No new phone data or runtime observation
is claimed after the earlier debugging shutdown.

## Open DeComFL continuation limitation

Review found a pre-existing problem in the generic rejoin path: the bridge
recreates FederatedLoop on registration, discarding its owned DeComFL trajectory
state. A contracted DeComFL round beyond round one then refuses to start from an
empty trajectory. The new JavaScript DeComFL case tests error routing only; it
does not prove native DeComFL continuation. Recovery across all strategies remains
unqualified. Address this by preserving or restoring verified same-run trajectory
state and testing a real native continuation after an accepted DeComFL round,
including interrupted/repeated replay and changed-run/contract refusal. Do not
remove the initial-state contract check to bypass the refusal.

Next physical check is the first-order pneumonia case with the new native app,
charger power and a bounded interruption, followed by the ten-round screen-off
Wi-Fi qualification. Production signing/TLS, natural deep Doze, battery-only
survival and process-death recovery retain their previous open status.
