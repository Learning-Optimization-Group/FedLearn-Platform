# DeComFL recovery after a lost scalar acknowledgement — 18 October 2026

The unchanged Android app completes the tested DeComFL run after the server counts
its first scalar update, the successful response is withheld, and the app process
is crashed. Manual same-round rejoin sends the identical request again; the server
counts that contribution once. Three rounds finish with six unique updates and an
exactly reconstructed final model. The
[evidence receipt](../../results/on-device/android-decomfl-acknowledgement-crash-2026-10-18.json)
binds the actual boundary, captured protobufs, counted updates and cleanup.

This extends the [current-snapshot recovery qualification](2026-10-18-decomfl-cold-restart-recovery.md)
to a lost completion before native return. It uses the same standalone internal
Debug APK and implementation `748ba8a`, with qualification checkout `eec4c24`.
No product source, main APK or checkpoint-storage changes were needed. Metro stayed
off. The case is CNN/FULL/LOCAL_SNAPSHOT/DeComFL with two clients, each using 32
disjoint real CIFAR images. The canonical Linux client is held before each fit.

A temporary gRPC relay forwards the phone's complete first scalar request and
captures the real successful response. The unfiltered recorder and independent
server status both show one counted phone update at round one. The relay withholds
that response permanently. It preserves serialized protobuf request bytes, not
HTTP/2 packet framing.

Before crash dispatch, the held RPC is observed active on the same host monotonic
clock, with an 18 ms observation age and 57.77 seconds before its native deadline.
The crash command returns in 86 ms; cancellation is observed after dispatch with
57.54 seconds remaining. These checks exclude an earlier RPC timeout being mistaken
for a crash during the blocked call. Android `am crash` targets the verified app PID.
Nineteen samples over 15.8 seconds show no app process or foreground training service.
The old successful response is discarded, with no original-call success returned.

Manual launch creates a new PID and idle Home while the server remains at round
one with one received update. Explicit rejoin, dataset selection and Start produce
a second round-one request. The two captured request files are byte-for-byte
identical, including run, client, round, seeds, scalars and example count. The retry
receives success, while thirteen samples over 10.56 seconds retain exactly one
counted update and show no third request. The normal round-one completion log then
appears after the awaited submitted-round save returns.

The initial save cannot be reached through the blocked native call, based on source
ordering and the controlled RPC boundary. Encrypted storage was not inspected, and
checkpoint-write durability is not established. The result is observed manual
recovery and server deduplication for this case, not a general exactly-once transport
guarantee.

Releasing the desktop completes all three rounds. The phone's request rounds are
`[1, 1, 2, 3]`; its returned successes are `[1, 2, 3]`. Six unique 32-example updates
are counted. Both cold model downloads serve round one with valid, identical
snapshots; retained rounds two and three need no new global download. All counted
values, shared seeds and later config history agree.

Production DeComFL reconstruction matches every registered final-model float32 bit
across 10 tensors and 62,006 values. Independent NumPy aggregation arithmetic also
matches every round; it shares the production canonical perturbation generator.
A host replay with the actual initial snapshot, seeds, exported loss program and
same 32 inputs reproduces all ten phone scalars, including their float64 bits. That
is an observed case result, not a cross-hardware equivalence guarantee.

Thirteen targeted server regressions pass. Real loopback gRPC checks cover response
withholding, resumed completion, premature-repeat rejection before upstream forwarding,
and distinguishing native deadline expiry from the controlled crash boundary. Earlier
full mobile/Android build and test evidence is inherited through unchanged source
and runtime hashes; those full suites were not rerun for this case.

Setup diagnostics are retained. The first helper installation was interrupted by
USB disappearance before any project was created; unused host routes were removed.
After reconnection the existing helper, app and data were verified. A later sign-in
attempt failed because fast test text entry dropped username characters. The same
account authenticated through the API; restoring per-character entry and checking
the visible username fixed the test harness before training. Failed observations
and helper versions are retained. An initial server test invocation imported the
root checkout; the qualified invocation uses the feature checkout's explicit Python
path. The interim aggregate receipt predates replay and cleanup; the completed
record is `final-verification.json`.

The original account and API setting were restored through normal UI. Cleanup at
19:20 UTC removed the temporary UI helper, relay, desktop process, local guards and
owned ADB/SSH routes. The phone was left asleep, USB-powered at 100% and 27°C, without
a training service or debugger. The installed APK and all 36 saved dataset-file
hashes are unchanged. TCP ADB and Metro remain off; the API is healthy and the
isolated project is completed. All 2,483 earlier immutable bindings were rechecked.

Natural Android kills, SIGKILL, reboot, force-stop, automatic resume, checkpoint
durability, death during DeComFL computation or before complete request acceptance,
other model contracts, long offline recovery, production signing and deployed TLS
remain unqualified. The next targeted interruption should occur during DeComFL
computation, before any scalar update is accepted.
