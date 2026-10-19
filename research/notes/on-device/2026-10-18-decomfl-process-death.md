# DeComFL cannot cold-restart after round one — test 16 October, verified 18 October 2026

DeComFL process-death recovery is blocked on Android. In this test, the phone
contributes round one, crashes, and explicitly rejoins after the server advances
to round two. It refuses initialization instead of continuing. No phone round-two
scalar request or accepted round-two update is observed in the bounded test.
This is a confirmed recovery gap with safe refusal, not successful recovery.

The [evidence receipt](../../results/on-device/android-decomfl-process-death-2026-10-18.json)
binds source, process observations, normal UI actions, scalar/config RPCs, recorder
window, captured server weights and cleanup. The live test and account restoration
occurred on 16 October. Work was interrupted before final evidence collection and
cleanup; those steps were completed and verified on 18 October after fresh access.
The delayed cleanup and monitor/deadline logs are retained. Prompt cleanup is not
claimed.

Runtime source remains `609d2cf`, installed standalone internal Debug APK `03d3a21e`;
qualification checkout `151e1a8`. Product source and the main APK were not changed,
rebuilt, reinstalled or reset. Metro stayed stopped. The isolated run used
CNN/FULL/LOCAL_SNAPSHOT/DeComFL, two participants and 32 disjoint real CIFAR images
per participant. Three rounds were planned; only round one completed, after which
the project was explicitly stopped at round two.

The canonical Linux client was held before its first actual fit. The phone joined
through normal UI, selected its existing dataset and submitted its first scalar
grid. The server counted that contribution once. The phone's normal round-one log
appeared after its awaited submitted-round save returned. This does not prove that
the encrypted marker was read after restart or that a write survived an immediate
crash.

Android `am crash` targeted the verified phone process. The original PID disappeared,
and all 19 independent samples during a 15.7-second observation showed no app process
or foreground training service. While the phone remained dead, the desktop's first fit was
released, completing round one and advancing the server to round two. The desktop's
second fit was independently confirmed held before computation. A read-only stream
captured the server's current model at round two, with its framing and hash checked.

A manual Activity launch created a different PID and reached idle Home without a
joined run. Explicit rejoin, dataset selection and Start reached the native guard:

> this device can join a DeComFL run only at its first round

The UI labeled the refusal a transient error and retried it. Across the subsequent
ten-second observation, the server remained at round two with zero received updates;
no phone round-two scalar RPC request was observed. The run was then stopped and
the original account and API setting restored through normal UI. Neither all three
rounds nor a registered final artifact is claimed.

The complete scalar/config log binds the phone's one accepted request to its run,
client, round, seeds and counted scalar values. The two accepted round-one grids
reconstruct every float32 value of the captured server model bitwise using production
DeComFL. An independent NumPy update, using the canonical seed-generated directions,
also matches all 10 tensors and 62,006 values. The server's later config contains
round one's exact shared seeds and averaged scalars. Missing server history is not
the cause of this refusal.

The cause is the mobile bootstrap contract. `DeComFLState` holds the flat model and
`syncedThrough` marker only in memory. A fresh contracted trajectory requires
round one, downloads the initial model and checks its initial-state hash before
binding. Process death loses that trajectory. At round two the guard rejects the
fresh state before it can use the available rebuild history. Saving only the last
submitted round cannot restore the missing model or replay position.

The next corrective direction should evaluate bootstrap from the current server
snapshot, which the Python client already supports. It must validate run/contract
identity, exact tensor layout, finite values and stream integrity; bind the validated
weights and their replay marker together; and reconcile snapshot/config round races
before training. Replay must apply only a complete missing suffix. Upload deduplication
remains separate from the model's replay marker. Simply removing the first-round
guard or skipping the initial hash check is not a valid fix. A permanent bootstrap
refusal should also stop clearly instead of being labeled as a transient connection
failure. Durable phone trajectory storage is a separate option, not a prerequisite
established by this test.

An existing native regression for the first-round-only refusal passes. Initial
qualification setup waited for the temporary UI helper's optional security-check
dialog; the first UI attempt failed before training and the same run continued after
installation. An offline verifier initially assumed the contract seed generated the
server's perturbation grids; the server uses its DeComFL configuration seed. The
verifier was corrected to replay the observed server-issued grids, without changing
the product or rerunning the physical test. Failed diagnostic logs are retained.

Final cleanup confirms the main APK and all 36 saved dataset files unchanged. The
temporary UI helper and install screenshot are removed, owned desktop/proxy processes
and local guards are absent, ADB/SSH test routes are absent, and TCP ADB remains off.
The phone is asleep on USB power with no foreground training service. All 2,114 prior
immutable evidence bindings still verify.

This test does not qualify successful DeComFL cold recovery, automatic resume,
checkpoint durability, reboot, force-stop, natural low-memory eviction, other
models or strategies, or production signing and deployed TLS. Safe refusal is
verified only within the recorded observation window.
