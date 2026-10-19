# Physical heartbeat recovery after the native correction

The updated Debug app completed two original-pneumonia FedAvg FULL rounds with
vivo and the canonical Linux CPU peer. After phone round one was accepted and
its durable checkpoint was saved, the test deliberately failed three heartbeat
RPCs while keeping the real server available. The phone re-registered, received
an actual upstream heartbeat acknowledgement and continued to round two. Its
round-one checkpoint remained one and no duplicate contribution was accepted.

The three failures occurred at 01:31:34.922, 01:31:39.982 and 01:31:45.037 UTC
on 15 October. The server accepted re-registration at 01:31:54.393 and a new
heartbeat at 01:31:54.431. The journal records one reconnect attempt and both
completed rounds. It does not directly expose the internal native participation
state or capture the RECONNECT_REQUIRED rejection itself. The native source and
host regressions describe that mechanism; the phone evidence establishes the
actual failure/re-registration/continuation sequence.

All four unique 16-example accepted updates reconstruct the registered final
aggregate bit for bit. Both phone upload payload hashes match their accepted
states. The final checkpoint is two, the project completed, the training service
ended and all owned test helpers are absent. The same app PID20200 remained in
33 active samples, with USB power, battery100% and temperature22–24°C. All33 saved
files matched before and after. The same-signer update also preserved those files
and completed the earlier USB-debugging restoration postcheck.

The [receipt](../../results/on-device/android-heartbeat-recovery-physical-2026-10-15.json)
binds installation, protocol events, accepted states, journal, final artifact,
verification and exact drivers. Source is f497dd0, including native fix f24e2cd;
APK SHA256 begins e31d467c. The first attempted test remains recorded as an
aborted harness attempt: an immediate assertion consumed a pre-training monitor
sample. It accepted no update and never injected the heartbeat fault. The fresh
retry waited for a timestamped active training sample before making that check.
A second independent deadline guard handled potentially partial event-file reads;
both deadline guards ended after completion. No product change was made during
the physical test.

This qualification is deliberately narrow. A gRPC proxy returned UNAVAILABLE for
three heartbeat calls and forwarded successful protobuf payloads and connection
tokens to the real server. It does not preserve all metadata/trailers, and it
caps forwarded deadlines at60seconds. One heartbeat failed after final peer
release and terminal completion as the server exited; the recovery interval had
no other upstream RPC failure. The model qualification cache was reused; the
journal's old probe duration is not a new measurement.

This is USB-powered functional recovery, with debugger observations, rather than
a screen-off endurance result. Independent local numerical training replay was
not repeated. DeComFL continuation beyond round one, physical production
signing/deployed TLS, process-death recovery, battery-only survival and natural
deep Doze remain open. The separate ten-round powered Wi-Fi screen-off test uses
a byte-forwarding proxy and retains its own outcome.
