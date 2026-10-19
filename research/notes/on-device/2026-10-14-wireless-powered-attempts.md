# Charger-powered Wi-Fi attempts

The ten-round vivo/Linux screen-off qualification **did not pass**. Two attempts
used the unchanged Debug app, the original pneumonia CNN, FedAvg FULL and one
Adam update on 16 images per client per round. Vivo stayed on charger power
without a host USB connection because its battery was reported to be weak.
Neither attempt qualifies battery-only reliability or natural deep Doze.

Only FedLearn was allowed in vivo's **High background battery usage** page.
Its checkbox changed from off to on and the allowed-app count from zero to one.
Leaving and reopening the page confirmed persistence. The same OEM toggle also
added FedLearn to Android's idle allowlist; no direct whitelist command was used.
Normal battery mode remained selected, with Low power and Super power-saving
off. No other app switch or global battery service was changed. This app setting
remains enabled, but these attempts do not prove that it alone prevents an OS kill.

The first attempt completed round one: the unfiltered recorder contains exactly
two accepted 16-example updates, one from each client, and the phone's durable
checkpoint is one. Complete phone upload metadata matches its accepted state.
While fetching round two, a Wi-Fi ADB dumpsys power command timed out after
15 seconds. The harness stopped the owned project. Subsequent collection found
the temporary reverse routes empty. The original app PID was still present and
the training-service record absent; all 33 dataset files were unchanged.
The initial readiness JSON was overwritten by the successful post-outage
readiness probe, so it is bound as recovery evidence rather than an immutable
initial preflight record. The failed attempt and its logs remain preserved.

After restoring the six known development routes and verifying mounted-provider
login and an authenticated API request, a separate fresh project was started.
Its monitor could record and retry one diagnostic timeout, while retaining the
same under-60-second sample/heartbeat freshness requirement. No retry was used:
the monitor stayed responsive, but acknowledged training heartbeats stalled
during the first model download. Ten IDLE heartbeats were acknowledged; the
last was at 18:05:36.527 UTC, and failure was detected at 18:06:46.259 UTC.
Three further DOWNLOADING heartbeat requests were sent without observed replies,
so the evidence shows an acknowledgment stall rather than proven failure to
generate heartbeats. The Linux client also reported "Stream removed (ping timeout)"
while downloading.
No update was accepted and the phone's checkpoint is null. This project is stopped.

The server sends approximately 98.36 MiB for each global-model stream. In the
second attempt, the Linux stream took over a minute between its early chunk
logs. After training stopped, a synthetic 10 MiB transfer from the Mac to g14
did not finish within 25 seconds. The reverse transfer took 17.56 seconds
(0.57 MiB/s), although small SSH commands took about 0.1 seconds. These results
measure the current shared connection; they do not establish the underlying
cause of either earlier failure. A subsequent same-direction comparison again timed out on the old connection
after 25 seconds, while a fresh SSH connection completed the same 10 MiB operation
in 1.75 seconds. The owned development tunnel was then replaced, preserving the
API and Metro processes and verifying their actual responses through the new
connection. An initial refresh-helper argument-order error rolled back all routes
before a corrected refresh passed. This restores development access; it does not
reclassify either failed training attempt or qualify the new path with a full run.

Both retained background event windows contain no matching app-kill event, and
the same app PID was observed after each attempt. Those facts distinguish these
failures from the earlier recorded com.vivo.abe kill, without proving continuous
process survival between samples or excluding every native fault. Charging,
the app setting and fresh model initialization differ from that earlier case;
this is not a causal A/B test of the setting alone.

The APK, g14 source, all 33 saved files, the root working-copy edit, 305 historical
evidence bindings and the preserved previous APK/core remain unchanged. Owned
Linux peers, observers, monitors and Mac guards are absent; development services
are retained. USB reconnection is still required to disable temporary Wi-Fi
debugging and restore the original empty routes. The
[receipt](../../results/on-device/android-wireless-powered-attempts-2026-10-14.json)
binds both attempts, policy evidence, accepted states, failure checks and transport
measurements. No successful final aggregate or independent numerical client
replay is claimed. Physical Release/deployed TLS, battery-only survival, sustained
thermal behavior, prolonged suspension and process-death recovery remain open.
