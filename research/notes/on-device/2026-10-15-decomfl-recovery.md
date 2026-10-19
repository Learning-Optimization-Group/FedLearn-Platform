# DeComFL continuation across native reconnection

DeComFL now retains its in-memory trajectory when the mobile bridge replaces a
network client. Source c3bdb29 binds that trajectory to the run, client and all
zeroth-order contract settings. Loading a manifest or model clears it. A fresh
participant still refuses to start after round one; this change does not persist
the trajectory across app process death.

The loop also tracks the last aggregate round it applied. The server advances
its client cursor only when a contribution is aggregated, so a failed submission
can make the next configuration repeat history already applied locally. The
mobile loop skips that prefix, requires a complete ordered suffix through the
current round's parent, and rejects malformed shapes or non-finite values.
Reconstruction occurs on a copy; the weights and cursor commit together only
after the complete reconstruction succeeds.

Three new regressions failed against the original code: a retry changed the model
and scalar payload, missing history was accepted, and a non-finite history entry
corrupted the model. The final suite passes 163 native tests, including nine new
cases and a frozen three-round Python trajectory check. Removing the replay skip
fails two selected regressions; discarding the external trajectory fails three.
After both mutations, the committed source was restored and all 163 tests passed.
All 687 mobile tests across 44 suites, lint and TypeScript checks pass. Android
Debug assembly passes. Its JVM task was up to date; five retained reports contain
61 passing tests. There is no protocol, server or JavaScript interface change.

The updated app completed a real three-round TinyNet DeComFL run with vivo and
the canonical Linux CPU participant. Each trained on the eight-example golden
fixture. After the phone checkpointed round one, the test held the Linux peer
and deliberately failed three heartbeat calls. The actual server accepted
re-registration and acknowledged a new heartbeat before the peer was released.
The phone continued into round two without downloading the initial model again.

Round two then deliberately failed four scalar submissions before forwarding
them to the server. The default three retries were exhausted and a second rejoin
completed. All five round-two request payloads, including the successful fifth
attempt, were identical. Each attempt received the same round-one rebuild
history. This directly exercises retention and repeated-history handling after
the trajectory has advanced, rather than merely demonstrating that a reconnect
message appears.

All three real upstream scalar submissions succeeded. The server recorded six
unique expected client/round contributions and the phone reached checkpoint
three. Those accepted scalars and actual server seeds reconstruct the registered
final model bit for bit; delivered history matches their accepted averages. The
project completed and the training service ended. The measured exercise lasted
76.15 seconds. Four active samples retained the same app process, service record,
USB charging and battery temperature of 27°C; this is a short awake Debug test.

All 33 saved dataset files matched before installation, after the same-package
update, after training and at final cleanup. The original empty USB reverse and
forward routes were restored, TCP ADB remained disabled, and all owned test
helpers and guards ended. The root workspace edit, earlier runtime artifacts and
679 prior evidence bindings were preserved. An initial postflight assertion
treated ADB's trailing blank line as a route; it was corrected before any route
was removed. The initial driver is retained. The first Gradle invocation needed
access to its existing cache outside the workspace; the permitted final build
succeeded.

The [receipt](../../results/on-device/android-decomfl-recovery-2026-10-15.json)
binds the committed source, APK/core, regression and mutation logs, exact raw
accepted-record window, protocol observations, registered artifact and cleanup.
The installed APK SHA256 begins cdab9ffb; its core archive begins 9948ab3f.
Earlier physical receipts remain bound to their earlier runtimes, including the
separate successful ten-round FedAvg screen-off case.

The controlled application-level gRPC proxy forwards successful protobuf payloads
and the connection token, drops other metadata/trailers and caps upstream
deadlines at 60 seconds. It does not simulate a submission accepted by the server
whose response is subsequently lost. Independent numerical replay of every phone
training step was not repeated. UI session totals omit failed-attempt compute
and traffic. Real-model DeComFL recovery, longer partitions, server restart,
process-death recovery, battery-only operation, natural deep Doze and production
Release/deployed TLS remain unqualified. Mobile XGBoost and secure aggregation
also retain their open status.
