# Wireless battery background test failed: vivo killed the app

The unchanged Debug vivo app did not complete the first original-pneumonia
FedAvg round while physically unplugged, on Wi-Fi and with the display off.
Android's event log records `am_kill` for the active app process at
2026-10-13T20:05:43.862000+00:00, with reason `stop by com.vivo.abe` and OOM adjustment 200.
The next device sample showed both app process and training-service record
absent, before the harness stopped the owned project. This is recorded OS
termination, not evidence of a new native crash. The wireless background path
remains unqualified on this phone under the tested policy.

The intended run had ten rounds, with the actual mounted phone provider and
canonical Linux client each using 16 images and one local Adam update per round.
The phone's authenticated API path and Wi-Fi ADB identity were verified while
USB was connected; the user then unplugged it. All background samples showed
USB, AC and wireless charging absent and the display asleep. The screen-off
window lasted 8.2 minutes before failure detection. The installed APK
and native/JavaScript/Linux training source were unchanged.

There were 31 acknowledged native heartbeats
in that window. The phone reached `TRAINING round1 0/1`, but no update or
checkpoint was accepted. The unfiltered recorder interval is empty, no decoded
upload chunks or submission completions were observed, and the actual durable
run checkpoint remained absent after app relaunch. The app reported zero
completed rounds after relaunch; that reset state is not pre-kill error history.
No final aggregate or physical numerical client replay was verified.

Device samples show light idle before the kill and no sampled deep idle. Two
bounded CPU-time samples showed the same process advancing 1,781 ticks over
55.568337 seconds before it was killed. These counters establish some CPU work,
not full-rate execution or the cause of the slower update. Battery samples
remained 100% and 27°C. Active Wi-Fi ADB diagnostics can influence power management;
these observations do not qualify natural Doze or sustained suspension.
Samples may miss brief display events and thermal peaks. Service-record
presence and the recorded oom adjustment are not continuous foreground-flag
measurements.

The harness stopped only its project and restored its RPC route. Its first
cleanup assertion exposed a Linux peer still held in the test wrapper; that
identified peer was explicitly terminated, and fresh cleanup confirmed peer,
proxy and monitor absent. An initial post-failure Promise-result diagnostic
failed; the established asynchronous store read then verified no checkpoint.
The later 45-minute guard did not cause this failure. These harness issues and
the failed exercise remain retained with the evidence.

All 33 saved dataset files, installed APK/core, root user edit, previous runtime
and 243 historical bindings are preserved. After USB reconnection, temporary
Wi-Fi debugging was disabled and original empty ADB route lists restored. The
old development tunnel had already stopped before this case; a replacement
loopback tunnel restored access. That connection expired during the later
overnight gap; a new connection restored Metro/API health on 14 October. The
USB restoration and its later battery level are outside the test window. No
application or battery policy change was made.

Next, inspect the app-specific vivo background-power policy and retest the
same contract with any approved setting explicitly recorded. The kill reason
supports investigating that policy; it does not prove a particular setting or
wake lock would solve the problem. Immediate operator preemption, deep Doze,
prolonged suspension, thermal endurance and physical Release/deployed TLS
retain their existing qualification limits.

Read-only review independently verified all 62 new bindings, all 243 historical
bindings, the kill chronology, all 31 heartbeat request/acknowledgement pairs,
the empty recorder interval and initial model/data contract. The failure verifier
passed without changing retained evidence. Phone, USB and HTTP checks accept
the recorded session proofs; physical/network actions were not repeated. No
material findings remain. Sampling language in the matrix was clarified.

Receipt: [android-wireless-background-2026-10-13.json](../../results/on-device/android-wireless-background-2026-10-13.json).
