# Ten-round powered Wi-Fi qualification after heartbeat recovery correction

The updated Debug app completed ten original-pneumonia FedAvg FULL rounds with
vivo and the canonical Linux CPU peer. Each client used 16 images and one Adam
update per round. All twenty unique accepted states reconstruct the registered
final model bit for bit, and the phone's durable checkpoint is ten. The project
completed and the training service ended.

The display remained asleep in 142 samples over 36.6 minutes, from approximately
01:40:29 to 02:17:04 UTC on 15 October. The same app process was present throughout
those samples. All 139 active-service samples retained the service record. Each
round has an actual acknowledged TRAINING 0/1 heartbeat, a complete 99-chunk upload
and an acknowledged WAITING 1/1 heartbeat before the held Linux peer proceeds.
There were 412 acknowledged heartbeats in the background window; their maximum
gap was 15.11 seconds. Device samples were at most 16.09 seconds apart, with no
recorded monitoring timeout. The harness made no debugger evaluations during
that window.

USB remained physically connected and supplied charging power. App traffic and
device diagnostics used the explicit Wi-Fi ADB transport. Sampled battery level
was 100% and battery temperature was 24–26°C. No natural idle state was sampled.
These observations qualify this externally powered Debug configuration; they
do not establish battery-only survival, wall-charger-only operation, natural deep
Doze or sustained thermal endurance. Sampling can miss brief wakes and peaks.
Service-name observations do not independently inspect every foreground-service
flag.

The [receipt](../../results/on-device/android-heartbeat-recovery-longrun-2026-10-15.json)
binds the accepted states, protocol and device observations, final artifact,
verification, cleanup and exact drivers. Source is f497dd0, including native fix
f24e2cd; APK SHA256 begins e31d467c. All ten phone payload hashes match their
accepted states. Independent numerical replay of each phone training step was
not repeated; exact aggregate reconstruction checks the accepted contributions.
Sixteen images and one update per round do not qualify clinical model quality.

All 33 saved dataset files matched before and after the run and again after
USB-only debugging was restored at 16:48 UTC. The later collection time does not
extend the measured training interval. Temporary Wi-Fi ADB was disabled, the
original empty reverse/forward routes were restored, and all owned test helpers
and deadline/wake guards were absent. The existing development services and
tunnel were retained. The root workspace edit, historical runtime artifacts,
545 earlier evidence bindings and the 42-file short-recovery seal were verified
unchanged. Only rebuildable, untracked Debug native-library intermediates were
removed to recover disk space; bound APK/core artifacts were preserved.

The separate [controlled recovery test](2026-10-15-heartbeat-recovery-physical.md)
passed before this run. This ten-round test injected no network fault. It used
a fresh model initialization and different recorded transport/power/runtime
conditions from earlier failures; it does not isolate the fix's causal effect
on those failures. The earlier battery-kill and seven-round timeout cases retain
their failed verdicts. The OEM background-policy UI observation is historical;
current idle-allowlist membership was checked without a new policy change.

DeComFL native continuation after round one, process-death recovery, server
restart, longer partitions, battery-only survival, natural deep Doze, physical
production signing/deployed TLS, mobile XGBoost and secure aggregation remain
open.
