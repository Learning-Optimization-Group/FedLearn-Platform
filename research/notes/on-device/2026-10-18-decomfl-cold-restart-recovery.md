# DeComFL manual cold recovery from the current snapshot — 18 October 2026

The corrected Android client completes a DeComFL run after actual app-process death
and explicit rejoin at round two. This closes the
[earlier first-round-only bootstrap gap](2026-10-18-decomfl-process-death.md)
for the tested CNN/FULL/LOCAL_SNAPSHOT case. The
[evidence receipt](../../results/on-device/android-decomfl-cold-restart-recovery-2026-10-18.json)
binds the implementation, installed runtime, process observations, RPCs, numerical
checks and cleanup. It qualifies manual recovery on a powered device; automatic
resume, natural Android eviction and checkpoint durability remain unverified.

Implementation commit `748ba8a` downloads the current global snapshot for a fresh
contracted DeComFL trajectory. The client validates the stream framing, full hash,
exact tensor names and shapes, and finite weights. The model's served round must
match the fetched configuration; a race retries configuration without binding that
snapshot. A round-one snapshot must still match the contract's initial-state hash.
Validated weights and their completed-round watermark bind together. History older
than that watermark is skipped, and retained in-process state replays only a complete
missing suffix without another global download. Submitted-round deduplication stays
separate. Deterministic training-state errors stop with a clear error instead of
being treated as transient connection failures. A validated cold snapshot may remain
bound if subsequent history validation fails; it is not used to train in that case.

The standalone internal Debug APK was rebuilt and installed as a same-package update.
All 36 existing dataset-file hashes were preserved. Metro stayed stopped. The physical
run used two clients, each with 32 disjoint real CIFAR images, and three rounds.
The canonical Linux client was held before each fit to control the boundaries.

The phone contributed round one exactly once. Its normal completion log appeared
after the awaited submitted-round save returned. Android `am crash` then targeted
the verified app PID. Independent samples for more than 15 seconds showed no app
process or training service. While the phone remained absent, the desktop completed
round one and the server advanced to round two. A read-only server download captured
the current state before the desktop's second fit was released.

Manual Activity launch created a new PID and idle Home without a training service.
Explicit rejoin, selection of the existing dataset and Start downloaded the round-two
snapshot. Its bytes exactly match the independent server capture. The phone then
contributed rounds two and three once each; each contribution was observed for ten
seconds with the desktop held and no duplicate request or count. Round three reused
the retained trajectory without another global download. The run completed with
exactly six unique 32-example updates and three phone scalar requests.

A host execution of the current native DeComFL trainer, using the captured round-two
weights, actual server seeds, same exported loss program and same 32-image inputs,
reproduces all ten uploaded scalars exactly, including their float64 representations.
This is an observed result for this case, not a cross-hardware equivalence guarantee.
Two counterfactual replays using the initial weights or applying round one's update
again both fail the preselected `atol=rtol=0.001` comparison; their largest scalar
differences are approximately 0.00286 and 0.00262 respectively.

The six counted grids reconstruct every float32 value in the registered final model
bitwise using production DeComFL. Independent NumPy aggregation arithmetic also
matches all three rounds and the final state: 10 tensors and 62,006 values. That
check shares the production canonical perturbation generator; it is not an independent
RNG implementation. Config history, phone requests and counted values agree on run,
client, round, seeds and scalar grids.

Verification passed: 175 native tests in 39 suites, 702 mobile tests in 45 suites,
TypeScript, lint, the standalone Android build, and all 68 JVM tests. Regressions cover
cold later-round bootstrap, skipping already represented history, snapshot/config
races, bad initial identity, invalid tensor shape and values, stream corruption and
framing, permanent-error propagation, and retained-state identity changes. Physical
malformed-stream or round-race injection was not performed.

The original account and API setting were restored through normal UI. Cleanup on
18 October removed the temporary UI helper, observer, desktop process, local guards,
and owned ADB/SSH routes. The phone was left asleep and USB-powered at 100%, 22°C,
without a training service or debugger. TCP ADB and Metro remained off. The candidate
APK and all saved dataset hashes were rechecked. The API stayed healthy and the
isolated project remained completed. All 2,272 earlier immutable bindings were
verified with the authorized source revision recorded separately; old evidence,
helpers and historical default build outputs were preserved.

Setup diagnostics are retained. The first preparation stopped before routes or
training when its SSH control socket vanished. A fresh connection resumed after
checking the installed candidate and empty route baseline. The temporary helper's
optional security-upload prompt was declined without changing security settings.
The interim aggregate receipt predates replay and cleanup; `final-verification.json`
is the completed verification record.

This test uses a forced VM crash after a completed contribution and an explicitly
restarted app. It does not qualify SIGKILL, natural low-memory termination, reboot,
force-stop, death during DeComFL compute or upload, immediate checkpoint-write loss,
long offline recovery, other model contracts, automatic resume, production signing
or deployed TLS. Durable trajectory storage was not needed for this bounded manual
rejoin; a broader storage design should follow explicit offline/resume requirements.
