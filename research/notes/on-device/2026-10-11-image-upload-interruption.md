# Partial image-update upload qualification

A fresh original-pneumonia run uses the actual mounted vivo training provider and
canonical Linux CPU client: three FedAvg rounds, 16 balanced local images per
client, native application source `73ca4bf` and unchanged JavaScript/Linux source
`50a01ca`. The installed Debug APK is checked by its actual device hash. This
experiment adds evidence without changing application or training behavior.

An owned forwarding proxy cuts the phone connection during its first model
upload, after observing two complete one-MiB chunks of a declared 99-chunk update.
The final chunk has not arrived. The server accepted-update recorder contains no
phone contribution, and the durable submitted-round checkpoint is absent. The
proxy refuses connections for eight seconds. The provider reports the failed
upload, exhausts its short retry budget, and re-registers on the same run.

The retry finishes once: the server records one phone round-one contribution and
the phone saves checkpoint one before the held Linux peer is released. Both
attempts declare the same full-update SHA-256. That declaration does not prove
the interrupted attempt's complete contents, because they were not captured.
Exact replay verifies the accepted retry tensors. Server accepted records are
authoritative; a proxy connection count or a `received` response alone does not
establish that an update counted.

The completed run has six unique 16-example contributions, checkpoint three and
no foreground training service. Three original phone outputs replay bit-for-bit
on the original phone runtime, and all three Linux outputs replay exactly.
Changing one label changes each client's result. Reconstructing their aggregate
matches the registered final artifact. This pneumonia architecture has no integer
counters or BatchNorm variance tensors; those generic checks add no coverage.
All 14 transferred replay files match their hashes. Both replay jobs exit
successfully, owned helpers are absent, and all 33 dataset files remain unchanged.
The [receipt](../../results/on-device/android-image-upload-interruption-2026-10-11.json)
binds the experiment, runtime identities, private drivers and evidence.

The new passive observer fixes terminal response HEADERS/CONTINUATION completion
without modifying the historically bound observer. Three regressions fail before
the fragmentation/upload fix; a separate response-buffer regression fails before
its cap. Ten protocol checks pass on each host. A separate synthetic real-gRPC
smoke test preserves complete payload hashes through normal forwarding and retry,
and leaves the interrupted stream unaccepted. It is harness verification, not
physical federated training. No authentication headers or model payloads are
logged by the observer. The first preflight failed before phone training because
ADB routes were missing after host downtime; that failure is retained separately.

This qualifies one foreground, USB-connected, controlled partial-upload failure
on the Debug app over authenticated development plaintext. It does not qualify
loss of the acknowledgement after a complete accepted upload, long suspension,
natural wireless Doze, sustained thermal behavior, graph-boundary Stop, physical
Release training or deployed HTTPS/TLS. Provider counters cover successful reported rounds and omit failed-attempt
traffic and compute. Boundary battery readings are not peak measurements. Native heartbeat round/step fields remain fixed at registration.
The private datasets and physical harness are hash-bound; independent physical
repetition is not claimed. Earlier regression suites were not repeated for this
follow-up, which changes documentation and private qualification tools only.

Read-only review reran all ten protocol checks and the evidence verifier, checked
the bound hashes and independently reconstructed the aggregate. It found no
blocking issues. The zero-counter wording was corrected after checking both
initial and registered dtype inventories. No physical repetition or fresh
device attestation was performed during review.
