# Manual recovery after a forced exit in native backward computation — 16 October 2026

The unchanged Android app recovers manually after a debugger-directed process exit
while paused at the first convolution backward kernel in actual federated training.
No upload begins before the exit. Explicit cold launch, same-run rejoin, dataset
selection and Start recompute round one and finish all three rounds. The six accepted
updates reconstruct the registered final model bitwise.

The [qualification receipt](../../results/on-device/android-native-compute-exit-2026-10-16.json)
binds the native stop, process observations, wire payloads, recorder window, actual
desktop parents, diagnostics and cleanup. Runtime source remains `609d2cf`, installed
standalone internal Debug APK `03d3a21e`; the qualification checkout is `ecb8b0b`.
Product source and APK were not changed, rebuilt or reinstalled. Metro stayed stopped.

The case uses CNN/FULL/LOCAL_SNAPSHOT/FedAvg, three rounds, two participants and
32 disjoint real CIFAR training images per participant. The canonical Linux client
holds its first actual fit through the phone's recovery check. A passive proxy
observes the phone's protocol traffic without withholding or changing bytes.

The packaged native ELF and existing unstripped symbols share build ID
`55a36b6a47878015e9aa33d35bd35463629fb2dc`. Before setting the breakpoint, the debugger
checks the module identity, actual load mapping, ELF header and kernel instruction
bytes against the packaged library. The stop is inside the physical
`convolution_backward_out` symbol, with `TrainableExecutorchModel::forwardBackward`
and `FederatedLoop::firstOrderRound` on the logical stack. The native `currentRound`
scalar is readable as integer one. Inline tuple helpers above the physical kernel
are retained in the captured stack. This is a stop at kernel entry after its
prologue, not a timing estimate after tapping Start or an arbitrary point inside
a tensor operation.

While all threads are stopped, server status and the recorder show zero accepted
updates; the passive observer has seen no upload request headers or chunks. The
debugger verifies the exact mapped Bionic libc `_exit` instructions, writes and
reads back its entry address and argument 73 in the stopped worker's registers,
then continues. It reports exit 73 after 0.66 seconds. Android independently shows
the original PID gone and no foreground training service in 19 samples over
15.54 seconds. No app or service restarts during that observation. This uses the
app's existing debug permission; no root or security-setting changes are made.

A manual Activity launch creates a different PID and reaches idle Home without a
joined run. Normal UI Join → existing dataset → Start recomputes round one. Its
single completed upload is accepted once, with no additional upload during a
10-second same-round hold. The normal round-one activity log proves the submitted
round save promise returned; encrypted storage is not inspected. Releasing the
desktop completes the remaining rounds.

Each of the three completed phone RPCs carries the expected run, client and round.
Its captured payload matches all 62,006 float32 values in the corresponding counted
state. The six unique 32-example contributions reconstruct all three actual desktop
parent models and the final 10-tensor model bitwise using production FedAvg. An
independent NumPy weighted-mean reconstruction also matches every parent and final
value. This verifies aggregation, not independent replay of the phone's local math.

Earlier attempts remain diagnostic controls: a strict logical-top-frame guard
rejected inline helpers; a debugger kill reported exit while Android's PID survived;
selective-thread suspension failed kernel-state validation; and an asynchronous
stale-stop check detached before reaching the boundary. None qualifies process
recovery. An idle `_exit` trial established actual process disappearance but lost
its debugger connection without an exit code. The final case corrects the native
frame and asynchronous stop checks and independently verifies death.

After all three uploads completed, a later connection reset produced one diagnostic
error. That connection has no observed decoded RPCs; zero transport bytes are not
proven. The completed upload streams, counted states and aggregate remain verified.

The original account and API setting are restored through normal UI. The temporary
UI helper, debugger files, owned processes, ADB routes and SSH routes are removed.
The main APK and all 36 saved dataset files remain unchanged; the phone is asleep
on USB power without a training service. All 1,676 prior evidence bindings remain
valid. Concurrent changes in the original workspace are tracked separately from
immutable qualification evidence and are not restored by this work.

This qualifies bounded manual CNN FedAvg recovery from a debugger-directed `_exit`
at a paused native backward boundary. It does not qualify SIGKILL, natural low-memory
termination, arbitrary instruction interruption, automatic resume, checkpoint-write
durability, DeComFL state recovery, other models or strategies, production signing,
or deployed TLS. DeComFL needs its own process-death test before durable checkpoint
storage is designed.
