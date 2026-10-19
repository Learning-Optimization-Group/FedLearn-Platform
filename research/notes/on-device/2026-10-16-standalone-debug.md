# Standalone Android Debug qualification — 16 October 2026

The internal Android app now embeds its current Hermes JavaScript bundle and can
launch and train with Metro stopped. A powered vivo and the canonical Linux client
completed two fresh CNN/FULL/LOCAL_SNAPSHOT runs: FedAvg and DeComFL, ten rounds
each, using 32 disjoint CIFAR training images per participant. Each run accepted
exactly twenty unique client updates. Reconstructing those updates with the
production aggregation implementation reproduced every float32 bit of its
registered final model. This verifies the accepted aggregate, not an independent
numerical replay of either physical client's training.

The [receipt](../../results/on-device/android-standalone-2026-10-16.json) binds
source commit `1266d7a`, the installed APK and embedded bundle, model exports,
actual per-round desktop configurations, FedAvg parent states, accepted-update
windows, UI observations, build checks and cleanup. Dataset provenance remains
in the [previous CNN receipt](../../results/on-device/android-cnn-decomfl-2026-10-15.json).
These are packaging and execution checks; they establish no model-quality or
clinical-performance result and add no model/strategy support cells.

Login, enrollment, file import through Android's native picker, dataset selection
and Start used the normal app UI. A separate read-only accessibility test package
captured Home without the stock dumper's idle wait; it called no app training
functions and was removed afterward. The stale early `fedavg-home.xml` snapshot
is excluded. The first FedAvg attempt expired after 900 seconds with one desktop
update and no phone updates while UI capture was being prepared. That failed
attempt and an earlier refused cross-organization setup are retained separately;
neither counts as successful qualification.

Both the API and training traffic used Wi-Fi through a temporary peer-restricted
relay. USB stayed connected for the old phone's power, installation and test
controls. ADB reverse/forward lists were empty and TCP debugging remained disabled.
Metro was stopped before cold launch and absent throughout all retained device
samples. Five samples observed the active training service; each retained the
same phone process, USB power, 100% battery and a 27°C battery temperature.
These short foreground runs do not qualify background endurance or natural Doze.

The opt-in build uses `-PfedlearnStandalone=true`. Ordinary Debug retains Metro;
a subsequent ordinary Debug assembly contained no embedded JavaScript bundle.
Four actual Gradle configuration cases passed, along with the Debug assembly,
61 Android JVM tests and 692 JavaScript tests in 44 suites. Historical default
build outputs were restored. The installed candidate retains the existing
Debug identity, `__DEV__=true` and development transport policy. Release's
JavaScript mode, signing gate and transport restrictions remain unchanged.
See the [build instructions](../../../wikis/mobile/14-release-readiness.md#internal-standalone-debug-build).

All 36 existing phone dataset files retained their hashes. Both training services,
desktop participants, relay, temporary port forwards and local guards stopped.
The original saved account and server URL were restored through the app UI;
a final Metro-off cold launch displayed them on Sign in. The development API's
loopback advertisement was restored. Vivo was put to sleep with USB power
connected. All 876 previous source/evidence bindings and the root working-tree
edit were rechecked unchanged. The standalone APK remains installed.

Production signing/deployed TLS and process-death recovery remain open. Restoring
the original loopback URL also removes this temporary Wi-Fi connection; another
live session needs a reachable endpoint or explicitly configured test route.
