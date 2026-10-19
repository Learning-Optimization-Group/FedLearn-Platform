# Bounded CNN DeComFL on local images

Source 63b2f45 adds CNN/DeComFL/FULL with LOCAL_SNAPSHOT data consistently across
Python, Java and TypeScript contracts. It uses the real server defaults: one local
step, ten perturbations, learning rate 0.001 and smoothing 0.001. All 62,006 CNN
parameters participate. There is no dropout or BatchNorm in this recipe.

Each client trains one sequential batch of 1–32 local images. Desktop now loads
this scalar-update image contract through the canonical client and refuses
oversized snapshots before accepting local data. The phone uses its existing
whole-batch native trainer and refuses larger snapshots. Fixture data remains
unsupported for this combination. No protobuf or native ABI change was required.

Review exposed a real export defect: edge conversion narrowed a nominal 1–32
CNN functional graph to an actual upper bound of 15 when traced with eight shape
examples. CNN functional exports now trace at the requested upper bound, while
retaining the original trainable export, fixture data and qualification probe.
Real portable runtime loss/inference checks pass at 1, 16 and 32 images against
eager PyTorch. A standalone native check uses the fresh run's actual loss program
and initial state; DeComFL produces ten finite, repeatable scalars at each count,
without modifying the input model. Existing staged programs retain their older
bound and must be replaced before reuse with this feature.

The live run used a fresh project and fresh exports. Vivo and the canonical
Linux CPU client each trained on 32 CIFAR-10 images selected from previously
verified disjoint training packages. Their original training indices and source
package digests are retained. The phone package was imported through the app's
normal system picker and produced the expected immutable snapshot identity.
No official test images were used for training.

The real mounted provider completed three rounds with a USB-powered phone.
After round one was accepted and checkpointed, the desktop peer was held while
three heartbeat calls were deliberately failed. The actual server accepted
re-registration and acknowledged a new heartbeat before the desktop was released.
Round two then encountered four rejected scalar uploads before upstream
submission. Exhausting the default retries caused another accepted rejoin. All
five round-two protobuf request payloads were identical, including the successful
fifth attempt. The initial model was downloaded only once.

The server recorded six unique client/round contributions. Accepted scalars and
actual server seeds reconstruct the registered final model bit for bit, and
subsequent delivered history matches the accepted averages. The phone reached
checkpoint three, the project completed, and its foreground training service
ended. The exercise lasted 98.93 seconds. Four active samples from the screen-off
interval show the same app process, USB charging, asleep/noninteractive display,
and battery temperature of 26°C. These samples do not establish natural deep
Doze, continuous observation or long-duration background survival.

All 33 original dataset files retain their hashes. The new phone snapshot adds
exactly three files. USB reverse/forward routes were restored to their original
empty state, TCP ADB stayed disabled, and all owned helpers and deadline/wake
guards ended. The root workspace edit, historical APK/core, previous candidate
and 770 earlier evidence bindings remain unchanged.

Validation passed: 1,175 backend tests and its coverage check; 1,364 framework
tests at 78.38% coverage; 692 mobile tests across 44 suites, lint and TypeScript;
265 focused export/contract cases; and the existing native suite. Runtime's first
full invocation passed 666 tests and failed one because its Qwen tokenizer needed
network access; that same test passed with network access. Sandbox socket
restrictions also affected the first framework run; the permitted full run passed.

The [receipt](../../results/on-device/android-cnn-decomfl-2026-10-15.json) binds the
source, existing installed native APK, current development JavaScript, new model
programs, dataset provenance, raw accepted-record window, host checks and cleanup.
The Debug APK is the prior cdab9ffb candidate; native code was unchanged and no new
APK was assembled for this extension. The revised JavaScript was served by Metro.

This establishes bounded execution and same-process recovery. It does not measure
CIFAR accuracy, independently replay every physical scalar calculation, qualify
ambiguous server acceptance with a lost response, or persist a trajectory across
app process death. The application-level fault proxy forwards successful raw
payloads and the connection token, drops other metadata/trailers, and caps upstream
deadlines at 60 seconds. Larger datasets, other real-model DeComFL recipes,
production release/TLS, server restart, battery-only endurance, mobile XGBoost,
text/LoRA, iOS and secure aggregation remain separate work.
