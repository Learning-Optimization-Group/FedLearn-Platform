# Ten-round screen-off USB qualification

The unchanged Debug app completed ten original-pneumonia FedAvg rounds with
the vivo display off for 32.8 minutes. The actual mounted phone provider
and canonical Linux client each trained 16 local images with one Adam update
per round. Native source is `c331e59`; JavaScript/Linux source is `50a01ca`.
The installed APK and g14 source stayed unchanged during this run.

The display was asleep before a single training trigger. After that trigger,
the debugger was detached and the harness made no CDP evaluations until the
server completed and the training-service record disappeared while the display was
still asleep. An independent monitor sampled power, battery, service presence
and app-process presence about every 15 seconds. A passive proxy observed real
upstream acknowledgements without adding faults or changing training methods.

For every round, `TRAINING 0/1`, a coherent complete upload and its successful
response preceded acknowledged `WAITING 1/1`. The Linux peer proceeded only
after that phone update was accepted and waiting observed. Each new round
reset its local steps. The unfiltered sealed recorder interval contains exactly
20 unique 16-example contributions. Their canonical FedAvg reconstruction matches
the registered final model bit-for-bit. After completion, the phone was woken
once to confirm the terminal provider state and durable checkpoint 10.

The active-window monitor retained 126 service-presence samples;
all background samples showed the display asleep, USB power and the app process
present. The largest device-sample gap was 15.38 seconds.
The retained 381 acknowledged background heartbeats had a largest
gap of 6.46 seconds. Sampled battery temperature ranged
from 25.0 to 26.0°C. Battery stayed at 100% on USB power.
Service observations check the named service record, not its `isForeground`
flag. The implementation separately calls `startForeground`.

All 33 saved dataset files remain hash-identical. The owned training peer,
proxy and device monitor are absent and the normal route is restored. The
root branch/user edit, 151 earlier evidence bindings and preserved previous
APK/core remain unchanged. Existing development services are retained.
The [receipt](../../results/on-device/android-screenoff-longrun-2026-10-11.json)
binds wire metadata, device samples, debugger audit, source/runtime, accepted
states, aggregate reconstruction and the exact private drivers.

This is USB-powered Debug screen-off operation. It does not qualify natural
wireless Doze, prolonged suspension, physical Release training or deployed
TLS. These samples can miss brief wake events and temperature peaks; battery
temperature is not processor temperature. Ten rounds are not a thermal
endurance study. Physical numerical client replay was not repeated; aggregate
reconstruction checks accepted states rather than independently reproducing
local training. Graph-boundary Stop, server restart, client death and prolonged
partitions remain separate cases. No application change or new progress UI
was added, and earlier physical receipts retain their original runtime scope.

Read-only evidence review of `a02e0ef..38c076b` verified all 42 new and 151
historical bindings, each complete upload/waiting sequence and the unfiltered
20-row recorder interval. Independent float32 arithmetic reconstructs all ten
parent hashes and the registered final tensors bit-for-bit; the deterministic
verifier passes. No material findings remain. The review did not repeat network
or physical actions, current process inspection or local numerical client replay.
