# Native heartbeat progress

Native source `c331e59` fixes the heartbeat that captured its registration round
and sent no step counts. A coherent snapshot now holds the latest confirmed
client round, phase and completed local steps. The independent channel and
five-second heartbeat cadence are unchanged; no protocol change is needed.

First-order training counts optimizer updates, including each minibatch and its
final partial batch. DeComFL counts completed local updates rather than
perturbations. Each attempt resets its counters, retains the last confirmed
round during download and replaces it with the newly served round. Initial
model download and missed-history rebuild are distinct from training. Upload
stays pending until its RPC returns successfully; exceptions report error and
clean aborts report stopped. Counts outside the existing int32 wire range are
unknown `0/0`, with phase and round retained.

Seven regressions were observed failing before their changes and passing after.
They cover full-batch steps, all minibatches, DeComFL with K=2/P=3, coherent
concurrent snapshots, failed-upload retry, early download failure and initial
DeComFL download phase. Review exposed the latter two inaccuracies before they
were corrected. All 154 native tests, 681 mobile tests and 61 Android unit tests
pass; the Debug APK builds. Code review reran seven focused and 154 native
checks with no unresolved blocking findings. Numeric training operations did
not change, and the existing numerical goldens pass.

A fresh two-round original-pneumonia FedAvg run uses the actual mounted vivo
provider and canonical Linux client. Each trains 16 local images with one Adam
update per round. With the Linux peer held, the real upstream server acknowledges
`IDLE round1 0/0`, `TRAINING round1 0/1`, then `WAITING round1 1/1` after successful
upload completion. The peer proceeds only after the phone's durable checkpoint.
The next round acknowledges `TRAINING round2 0/1` and `WAITING round2 1/1`.
This directly verifies round advancement, fresh counts and post-upload waiting.

Four unique 16-example updates complete the run. The unfiltered sealed recorder
window contains those four rows; they reconstruct the registered final model
bit-for-bit. Checkpoint two and the terminal provider state are retained. All
33 dataset files remain unchanged across installation and training; the
foreground service, owned peer and proxy are absent, and normal routes are
restored. The installed APK matches the rebuilt APK. The previous qualified
APK and core archive were copied and hash-checked before rebuilding. The
[receipt](../../results/on-device/android-heartbeat-progress-2026-10-11.json)
binds the source, installed runtime, wire observations and private evidence.

Two private harness failures are retained. Initial debugger attachment failed
before phone login/join/training; a bounded mounted-provider readiness check
allowed the same run and peer to continue. After successful training, the live
driver's copied proxy-cleanup shell string raised an embedded-NUL error. Its
exact running source and log remain preserved. A separate collector removed the
owned proxy and verified routes, helper absence and dataset preservation. The
future driver uses structured remote Python cleanup. Neither issue changed
product training or restarted the physical run.

This changes native heartbeat fields; it adds no dashboard or phone progress UI.
Provisioning/self-tests and post-round evaluation are outside these step counts.
Five-second sampling may skip short phases or intermediate steps. Multiple
minibatches, DeComFL, error/retry and unknown-count behavior are not freshly
qualified on hardware; the first three are covered by the relevant host checks,
while unknown-count wire degradation is code-inspected. Physical client
numerical replay was not repeated for this telemetry-only slice. Aggregate
reconstruction verifies the accepted states, not independent local-training
reproduction. Successful-round counters still omit failed-attempt work.

The case is foreground and USB-connected. Natural wireless Doze, longer
suspension, graph-boundary Stop, sustained thermal behavior, server restart,
client death, prolonged partitions, physical Release training and deployed TLS
remain open. Boundary temperatures are not peak measurements. Code review did
not independently repeat physical training, device attestation or preservation.

Read-only evidence review of `c331e59..1b27763` verified all 38 bindings, wire
request/response correspondence, upload-before-waiting order and the unfiltered
four-row recorder window. It independently reconstructed both parent hashes
and the final aggregate exactly. No material findings remain; a grammar edit
in the integration-readiness note was corrected. Physical actions were not
independently repeated.
