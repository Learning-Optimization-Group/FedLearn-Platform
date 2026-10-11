# Final-source image reconnect qualification

A fresh original-pneumonia run exercises the lifecycle correction in native
source `73ca4bf`, with JavaScript and Linux source `50a01ca`. It uses the actual
mounted phone training provider, the canonical Linux CPU client, three FedAvg
rounds and 16 balanced local images per client. This is a separate experiment
from the retained October 10 run; its evidence does not change that older run's
source or heartbeat limitation.

The first phone round ran with the screen asleep and forced device idle. This
uses simulated unplug and USB forwarding. Stop during pending native round-two
work completed in 1703 ms, kept the round-one submitted checkpoint, removed the
training service and produced no round-two upload. The same run was rejoined.
A pending native status request proves serialized native work, rather than
identifying whether a graph or model RPC is holding the lock.

After the resumed second round was checkpointed, an owned proxy cut active
sockets and refused connections for 20 seconds while the Linux peer was held.
The first rejoin failed. A passive decoder then observed the real upstream
server return accepted registration, followed by an acknowledged heartbeat,
before the peer was released. Another heartbeat was acknowledged while resumed
native work was pending. The proxy forwards raw HTTP/2 bytes unchanged and
requires successful upstream gRPC trailers; it records selected fields without
credentials. Five protocol checks cover fragmented frames/header continuation,
acknowledgement, failure responses, large model streams and missing trailers.

The [fresh receipt](../../results/on-device/android-image-reconnect-final-2026-10-11.json)
binds the completed run, source/APK/core hashes and private evidence. Six unique
16-example updates were accepted, the submitted checkpoint reached round three,
and the training service ended. All three original phone updates and all three
Linux updates replay bit-for-bit; changing one label changes each client's result.
Their aggregate reconstructs the registered checkpoint exactly. All 14 transferred
replay files match their prepared hashes, both replay jobs exit successfully,
owned temporary phone replay files are absent, and all 33 saved dataset files
remain hash-identical.

The initial attempt lost its development SSH tunnel before phone registration.
The canonical desktop then needed a normal enrollment refresh after its token
expired, retaining partition zero. Those attempts are retained separately from
the successful qualification; authentication policy and platform defaults were
unchanged. No application or training implementation changed in this follow-up.
The fresh checks are physical execution/replay, five protocol checks and receipt
verification; the previous complete regression suites were not repeated.

Native heartbeat round and step fields are fixed at registration; liveness does
not establish accurate per-round progress telemetry. Boundary battery readings
are not peak-temperature measurements. This experiment does not qualify natural
wireless Doze, hour-scale suspension, sustained thermal load, interrupted update
uploads, physical Release training or deployed HTTPS/TLS enrollment and training.
