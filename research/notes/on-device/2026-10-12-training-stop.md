# Stop during native image training

Stop completed correctly in this bounded test, but it took **106.993 seconds**.
The app requests cancellation promptly and waits for active native work to settle.
The first-order loop checks cancellation between local updates and before upload;
it cannot preempt an in-flight model operator. The earlier 371ms blocked-registration
check therefore does not predict image-computation Stop latency.

The unchanged Debug app uses native `c331e59` and JavaScript/Linux `50a01ca`.
The actual mounted phone provider trains 16 original-pneumonia images with one
Adam update per round. The canonical Linux peer stays held. Two upstream
acknowledged `TRAINING round1 0/1` heartbeats five seconds apart precede Stop.
Pending snapshots retain the stopping flag and pinned dataset. Real Start and
Join calls during that interval are rejected by the existing single-flight guard.
After the Stop promise settles, the provider is not joined, its dataset is released,
its session has zero rounds and the training-service record is absent.

The cancelled recorder interval is empty without filtering, and the phone's
submitted-round checkpoint remains absent. The wire observer records no decoded
upload chunks or submission completions for that attempt. After same-run rejoin,
fresh registration and an acknowledged native idle heartbeat precede resumed
training. Both resumed rounds finish with one accepted 16-example update per
client and checkpoint two. Four accepted states reconstruct both round parents
and the registered final FedAvg model bit-for-bit. No physical numerical client
replay was repeated.

The independent monitor retains 42 USB-powered, awake, non-idle,
app-present samples, with a largest gap of 15.38 seconds.
Sampled battery temperature ranges from 23.0 to 24.0°C.
All 33 dataset files remain hash-identical; the installed APK and g14 source are
unchanged during the run. Owned peer/proxy/monitor cleanup and restored routes,
root/user-edit preservation, all 193 older bindings and retained development
services are verified after completion. The
[receipt](../../results/on-device/android-training-stop-2026-10-12.json) binds the
exact running drivers, phase/wire observations, pending/terminal provider states
and final proofs. No application change or new progress UI is introduced.

Initial harness review found two evidence gaps: the live exercise did not check
cancelled-window wire events or device-sample freshness. Its exact running source
remains preserved. The separate final verifier closes both gates before reporting
success. It enforces no cancelled-window decoded upload chunks/completions, no
observer errors, and a complete fresh sample sequence with bounded gaps.

This observes the local-training phase, not exact kernel entry or mid-operator
preemption. The observer does not prove absence of every sub-chunk byte or stream
header. Samples do not establish continuous activity ownership or temperature
peaks. The measured 107-second delay is one case, not a worst-case Stop bound.
Wireless Doze, prolonged suspension, server restart/client death, thermal
endurance, physical Release training and deployed TLS remain separate cases.
Successful-round counters also omit the cancelled computation.
