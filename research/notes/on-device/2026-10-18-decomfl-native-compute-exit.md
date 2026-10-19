# DeComFL recovery after a native forward-computation exit — 18 October 2026

The unchanged Android app completes the tested DeComFL run after a controlled
process exit inside its initial base-loss forward computation, before any scalar
request or update is accepted. Explicit cold launch, rejoin, dataset selection and
Start recompute round one. Three rounds finish with six unique contributions and
an exactly reconstructed final model. The
[evidence receipt](../../results/on-device/android-decomfl-native-compute-exit-2026-10-18.json)
binds the native stop, exit mechanism, Android observations, actual requests,
numerical checks and cleanup.

This extends the [lost-acknowledgement qualification](2026-10-18-decomfl-acknowledgement-crash.md)
to interruption before submission. It uses the same standalone internal Debug APK
and implementation `748ba8a`, with qualification checkout `311f799`. No product source,
main APK or checkpoint-storage changes were needed. Metro remained off. The isolated
case is CNN/FULL/LOCAL_SNAPSHOT/DeComFL with two clients and 32 disjoint real CIFAR
images per client. The canonical Linux client is held before each fit.

The installed APK's packaged native library and unstripped symbols match by ELF
build ID. The debugger checks the live mapping and instruction bytes, then arms a
breakpoint after the physical forward-convolution function's prologue. A qualifying
stop must contain `ExecutorchModel::loss`, `DeComFLClient::fit` and
`FederatedLoop::deComFLRound`, with a successful typed read of `cfg.currentRound=1`.
The first qualifying invocation resumes. The selected second stop has a fresh stop
ID and the exact same breakpoint PC. Both recorded stacks and code bindings agree.

The selected stack contains `etBaseLoss<ExecutorchModel>`. The stop is therefore in
the initial base-loss evaluation, before a completed perturbation or local update.
It is a debugger-controlled kernel-entry boundary, not arbitrary instruction
interruption within a tensor operation.

While the process is stopped, independent server status and the passive protobuf
observer show round one with zero counted updates and zero phone scalar requests.
Only after those checks does the controller confirm exit. The debugger validates
the live mapped Bionic `_exit` instructions, writes and reads back its entry address
and status argument, and continues. It reports process exit with status 73 after a
0.65-second pause. Nineteen independent Android samples over 15.90 seconds show no
app process or foreground training service. Debugger status alone is not used as
Android death proof. No scalar request or counted update appears through death.

Manual launch creates a new process and idle Home while the server remains at round
one with zero received updates. Explicit rejoin and Start produce one recomputed
round-one request and one counted contribution. There is no extra request during
the ten-second desktop hold. Releasing the desktop completes all three rounds with
six unique 32-example updates and phone request rounds `[1, 2, 3]`. The two cold
round-one model downloads have valid framing/hashes and identical bytes; retained
rounds two and three need no new global download.

Captured scalar request bytes, seeds, run/client/round identities and counted values
agree. Production DeComFL reconstruction matches all 62,006 float32 values across
the registered final model's 10 tensors bitwise. Independent NumPy aggregation
arithmetic matches every round, using the shared production canonical perturbation
generator. A host native replay with the actual snapshot, server seeds, exported
loss program and same 32 inputs also matches all ten phone scalar float64 values.
This is an observed case result, not a cross-hardware equivalence guarantee.

The normal recovered-round completion log appears after the submitted-round save
await returns. Source ordering places that save after the interrupted native call;
encrypted storage was not inspected and checkpoint-write durability is not qualified.
Previous build/regression evidence is inherited through unchanged source and runtime
hashes; full mobile and Android suites were not rerun for this case.

The original account and API setting were restored through normal UI. Cleanup at
19:42 UTC removed the app-UID debugger and host ADB process, both verified debugger
files, local debugger client, temporary UI helper, observer, desktop and local guards.
The debugger's ADB forward and SSH local route, plus all owned reverse routes, are
removed. The phone was left asleep and USB-powered at 100%, 27°C, without a debugger
or training service. The APK and all 36 saved dataset-file hashes are unchanged.
TCP ADB and Metro remain off, the API is healthy and the project is completed.
All 2,695 earlier immutable bindings were rechecked.

The temporary UI helper's optional security upload was declined without changing
settings. An initial finalizer check expected a non-template function name; inspection
of the captured stack showed the actual template instantiation. The check was
corrected to that exact name. Its earlier source and failure log are retained; no
physical rerun or product change was involved. The interim aggregate receipt
predates cleanup/replay reconciliation; the completed record is
`final-verification.json`.

This qualifies bounded manual recovery from this controlled native forward-loss
boundary. Natural Android low-memory termination, SIGKILL, reboot, force-stop,
automatic resume, arbitrary instruction interruption, completed perturbation/local
step boundaries, checkpoint durability, interrupted scalar request transport, other
model contracts, long offline recovery, production signing and deployed TLS remain
unqualified.
