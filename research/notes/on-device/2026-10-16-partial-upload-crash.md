# Manual recovery after an incomplete transport upload — 16 October 2026

The unchanged Android app completes manual recovery after one forced process crash
during an incomplete first upload frame. The server counts no update before the
crash; explicit rejoin retrains round one, contributes once and completes the same
three-round run. This qualifies a transport interruption before delivery of the
first complete application chunk. It does not exercise cleanup of a partially
populated Python upload buffer or interruption during model computation.

The [qualification receipt](../../results/on-device/android-partial-upload-crash-2026-10-16.json)
binds source, the installed standalone internal Debug APK, diagnostic gate, wire
captures, process/service observations, UI actions, recorder window, actual desktop
parent states and cleanup. Runtime source remains `609d2cf`; qualification checkout
`6ce952b` contains subsequent documentation. No product code changed, and the APK
was neither rebuilt nor reinstalled. Metro remained stopped.

The case uses CNN/FULL/LOCAL_SNAPSHOT/FedAvg, three rounds, two participants and
32 disjoint real CIFAR training images per participant. The canonical Linux client
holds its first actual fit until the phone's recovery check finishes. The phone
uses its existing dataset through normal UI Join → dataset selection → Start.

The diagnostic proxy forwards the upload's original HTTP/2 DATA frame header and
only 1,024 payload bytes. The header still declares 16,384 bytes, leaving that frame
incomplete upstream. The gRPC envelope declares a 248,975-byte protobuf message;
its first 935 model bytes reach the upstream socket, out of 248,804 model bytes.
The proxy validates the client, run and round from the prefix and permanently
withholds subsequent request bytes on that connection. It later discards 248,143
bytes on disconnect. Resuming allows new connections only; the old remainder is
never released.

A real local gRPC fixture verifies that this cut delivers zero complete messages
to its service, leaves the client call pending, and still permits a fresh complete
upload after cancellation and resume. On the actual server, source ordering blocks
on the first complete protobuf chunk before creating its application upload buffer.
The physical run independently shows zero accepted recorder rows and server round
one with zero counted updates before and after the crash, and at manual cold launch.

Android's `am crash` kills verified PID 29062 with a fatal VM exception. Process-death
events confirm termination. Nineteen samples over more than 15 seconds show no app
process or foreground service. Explicit launch creates PID 8093; Home eventually
shows Idle and no joined run or dataset selection. Authentication survives, and
the project's local contribution is absent. No automatic training starts.

Manual Join → select the existing dataset → Start recomputes round one. The proxy
captures both complete original and recovery payloads on its phone-facing side;
they match byte-for-byte, SHA-256
`ca80bcc7bec6a29e1ae0cd73aede7ce8e99868aeb3de02fcd0056a8f119c6b7c`.
This capture includes bytes deliberately withheld upstream, so it is evidence of
what the phone attempted to send, not server acceptance. The new connection's
successful RPC, server status and recorder establish the recovery contribution.
Its full state matches all 10 counted float32 tensors and 62,006 values exactly.

During a further 10.58-second observation the server remains on round one with
one of two required updates, and the recovered phone sends no redundant upload.
Its normal round-one UI log establishes that the checkpoint-save promise returned.
Initial checkpoint absence follows fresh-run identity and the blocked native
`Finish()` call; encrypted storage was not directly inspected or edited. Neither
the UI log nor this test qualifies fsync, torn writes or durability after saving.

Releasing the desktop completes all three rounds with six unique accepted updates,
three per client and 32 examples per update. Reaggregation with production FedAvg
matches every float32 bit of the final model; all three actual desktop parent
states match the reconstructed inputs. Final NPZ SHA-256 is
`fc9dc4849c9915b2cca11a4cca8713e1d81b1a899f729f80d8165c12e26b8d79`.
This verifies aggregation, not independent numerical replay of phone training.

Two diagnostic corrections are retained in the private logs. The first post-launch
UI snapshot arrived during an empty loading screen, so the driver was changed to
wait for Idle. A counter also initially included the old phone-to-proxy payload
as another completed recovery upload; it now separates the frozen and fresh
connections. No additional crash or product change was required. The local fixture's
earlier capture expectation and logging error are retained with its final pass.

All 36 saved phone dataset files retain their hashes. The original account and API
setting were restored through normal UI, the desktop and proxy were stopped, and
temporary UI instrumentation, local guards, ADB routes and SSH forwards were removed.
The API configuration is unchanged; vivo was left asleep on USB power. All 1,511
prior bindings pass preservation checks with the existing explicit handling of
previously authorized source revisions. Four coordinator checks and the actual
gRPC gate fixture pass. JavaScript, JVM and native suites were not rerun because
product source and APK were unchanged.

No support cells, automatic resume, low-memory/SIGKILL, reboot, interrupted compute,
partially populated application-buffer cleanup, checkpoint-write durability or
DeComFL process recovery are qualified. Internal Debug signing and development
USB/loopback routes also leave production Release/TLS, natural Doze, endurance and
clinical-quality claims open. Further testing must identify the exact interruption
boundary before deciding whether additional checkpoint storage is necessary.
