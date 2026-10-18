# Android signing and TLS qualification

Release builds now require an explicit signing identity. They refuse missing or
partial credentials, missing stores, invalid passwords or aliases, expired
certificates and the Android debug identity. Debug builds remain usable.
The mandatory `preReleaseBuild` gate runs even when Gradle omits its optional
`validateSigningRelease` task.

Supply these environment variables from the build environment:

- `FEDLEARN_RELEASE_STORE_FILE`: existing keystore path.
- `FEDLEARN_RELEASE_STORE_PASSWORD`: store password.
- `FEDLEARN_RELEASE_KEY_ALIAS`: private-key alias.
- `FEDLEARN_RELEASE_KEY_PASSWORD`: key password.
- `FEDLEARN_RELEASE_STORE_TYPE`: optional, defaults to `PKCS12`; use `JKS` for a JKS store.

Keep the key and passwords outside version control. Build with Java21, Node24,
the configured Android SDK and the pinned ExecuTorch/gRPC artifacts, then run
`./gradlew :app:assembleRelease` from `mobile_client/android`. The
[signing check](../../mobile_client/scripts/check_release_signing.py) exercises
actual Gradle validation with temporary stores and removes its keys afterwards.
Run it with the same Java/SDK/native-artifact build configuration.

Release enrollment refuses a plaintext transport, and the native gRPC client
also enforces TLS. Only the Debug native configuration permits plaintext.
Android's HTTP network-security XML does not enforce the separately linked C++
gRPC channel; the native guard is required.

The actual vivo TLS test initially crashed when BoringSSL's weak `sdallocx`
function bound to the function-pointer variable exported by React Native's Folly.
The Android build now copies the crypto archive into its build directory and
renames only that optional reference. Folly's variable and the original cached
archive retain their identities. The unresolved private weak hook falls back to
BoringSSL's normal `free` path. The
[ELF check](../../mobile_client/scripts/check_android_allocator_symbols.py)
verifies the isolated weak reference in `libappmodules.so`; pass the NDK's
`llvm-readelf` with `--readelf`.

The [qualification receipt](../../research/results/on-device/android-release-tls-2026-10-11.json)
binds the signed APK, Debug APK, certificate fingerprints and native cases.
A temporary local release key was used and discarded. The installed Debug app
rejected the differently signed release update, then accepted a same-key Debug
update. All33 existing dataset files retained their hashes. No uninstall or
app-data reset was performed.

The real phone's native gRPC transport accepted the trusted localhost certificate
and completed a status RPC. Wrong root, wrong hostname, default untrusted roots
and plaintext sent to the TLS listener were refused before registration reached
the server. This used a fixed-response local server over USB/SSH routes, with a
server key held only in memory. It qualifies native transport behavior, not
HTTPS login/enrollment or a complete federated run over deployed TLS.

The final [native lifecycle check](../../research/results/on-device/android-lifecycle-review-fixes-2026-10-11.json)
also cancels a deliberately blocked registration through the real phone's Stop
method. Stop completion was observed after371ms; the cancelled registration
started no heartbeat. A fresh accepted registration then produced a heartbeat
observed by the server. This is a bounded native fixture measurement, not a
worst-case Stop bound during model computation.

Production release work still needs a persistent backed-up signing key, an
HTTPS API endpoint and reachable TLS gRPC service, followed by full authenticated
training and a same-release-key install/update check. The local qualification
certificate cannot serve as the future update identity.

The [bounded CNN DeComFL qualification](../../research/notes/on-device/2026-10-15-cnn-decomfl.md)
adds local-image training and short powered screen-off recovery with the existing
Debug native APK and updated Metro JavaScript. It changes neither release signing
nor the production TLS status above. CNN DeComFL requires fresh functional model
exports and one sequential batch of at most 32 images per client.

## Internal standalone Debug build

Use `./gradlew :app:assembleDebug -PfedlearnStandalone=true` from
`mobile_client/android`, with the same SDK and native-artifact arguments used for
an ordinary Debug build. This embeds the Hermes JavaScript bundle and disables
React Native developer support, so the installed app can launch without Metro.
The property accepts only `true` or `false`; ordinary Debug builds continue to
use Metro when the property is absent or explicitly `false`.

This is an internal Debug package. Its embedded JavaScript retains `__DEV__=true`
and the existing development transport policy. It uses the Debug signing identity
and does not satisfy the production signing or deployed TLS requirements above.
Release bundling and transport restrictions remain unchanged.

Run `python ../scripts/check_standalone_build.py` from `mobile_client/android`
with the normal build environment to check the configured Gradle tasks and flags.
The check covers standalone Debug, ordinary Debug, explicit `false`, rejection of
an invalid value, and Release's production JavaScript mode. It does not prove a
physical training run. Qualification also needs a cold launch with Metro stopped,
normal UI login/import/Start, and completed training with the packaged bundle.

The [16 October standalone qualification](../../research/notes/on-device/2026-10-16-standalone-debug.md)
completed two Metro-off, normal-UI runs on vivo and Linux: CNN FedAvg and CNN
DeComFL, ten rounds each, with 32 local images per client. Each run's twenty
accepted updates reconstructed its registered final model bit-for-bit. All 36
saved dataset files remained unchanged. This qualifies the internal package and
short foreground execution over development Wi-Fi routes; production signing,
deployed TLS and process-death recovery remain open.

The [sign-out lifecycle correction](../../research/notes/on-device/2026-10-16-signout-lifecycle.md)
is packaged in a new standalone internal Debug APK. It stops participation when
authentication unmounts the training provider and prevents a new provider from
racing unfinished cleanup. The powered-phone qualification covers sign-out and
explicit same-process rejoin. Process-death recovery and production release
requirements remain open.

## Process-death recovery boundary — 16 October

The [forced-crash qualification](../../research/notes/on-device/2026-10-16-process-death.md)
adds bounded manual recovery evidence for CNN FedAvg after an accepted upload.
The training service no longer restarts without its training loop; null and
retry/redelivery requests stop safely. The corrected standalone Debug app completes
the same run after explicit rejoin and dataset selection. The foreground-service
fix is covered by seven JVM regressions and a powered-vivo crash test. Automatic
resume, low-memory termination, interruption during compute/upload, DeComFL durable
state, production signing and deployed TLS remain separate requirements.

The [lost-completion crash test](../../research/notes/on-device/2026-10-16-acknowledgement-crash.md)
also qualifies manual same-round CNN FedAvg rejoin when the server accepts an update
but the app dies before receiving its completion. Existing server deduplication
counts the repeated upload once; all three rounds finish with an exact reconstructed
aggregate. No product changes or new checkpoint storage were required. The test
does not inspect encrypted storage or qualify durability after saving, interruption
during computation/partial upload, automatic resume or DeComFL process recovery.
The same internal Debug package and development USB/loopback transport are used;
production signing and deployed TLS requirements remain unchanged.

The [incomplete-transport crash qualification](../../research/notes/on-device/2026-10-16-partial-upload-crash.md)
adds manual CNN FedAvg recovery before the first complete application upload chunk
reaches the server. Zero updates are counted before death; explicit rejoin completes
the run with six unique updates and an exact final aggregate. The old connection's
tail is discarded permanently. This does not qualify cleanup of an application
buffer containing chunks, interrupted computation, checkpoint-write durability,
automatic resume or DeComFL process recovery. The unchanged internal Debug package
and development routes add no production signing or deployed TLS evidence.

The [native backward forced-exit test](../../research/notes/on-device/2026-10-16-native-compute-exit.md)
adds bounded manual CNN FedAvg recovery after actual process disappearance during
a debugger-paused first-round backward kernel, before any upload. Explicit rejoin
completes all three rounds with six unique accepted updates and an exact final
aggregate. No product change or new checkpoint storage was needed. This does not
qualify natural Android low-memory kills, SIGKILL, arbitrary instruction interruption,
automatic resume, storage durability or DeComFL recovery. The internal Debug package
and development transport add no production signing or deployed TLS evidence.

The [DeComFL cold-restart qualification](../../research/notes/on-device/2026-10-18-decomfl-process-death.md)
is negative: an app crash loses the in-memory trajectory, and rejoin after round one
refuses to train despite an available current server model and rebuild history.
No stale round-two scalars are observed in the bounded test. The UI retries this
permanent state refusal as a transient error. Recovery requires a validated model
bootstrap/replay position and explicit error handling; it is not solved by storing
only a submitted-round number. No successful cold recovery, automatic resume or
checkpoint durability is qualified. Final cleanup was verified on 18 October after
an interrupted continuation; product source and the installed APK remain unchanged.

The [current-snapshot recovery fix and qualification](../../research/notes/on-device/2026-10-18-decomfl-cold-restart-recovery.md)
supersede the preceding DeComFL first-round-only gap for the tested manual rejoin
path. A rebuilt standalone internal Debug app completes all three rounds after
verified process death and explicit round-two rejoin. Snapshot integrity, exact
native scalar replay, six unique contributions and the registered final model are
checked. Permanent state errors now stop clearly; snapshot/config round races retry.
The original account/settings and all saved data are preserved, and temporary test
processes, routes and helper are removed. This does not qualify automatic resume,
natural Android eviction, interrupted DeComFL compute/upload, durable local state,
other model contracts, production signing or deployed TLS. No checkpoint storage
was needed for this bounded recovery case.

The [DeComFL lost-acknowledgement test](../../research/notes/on-device/2026-10-18-decomfl-acknowledgement-crash.md)
qualifies manual same-round rejoin when the server counts a scalar update but the
app dies before receiving success. A fresh deadline-margin check and shared-host
cancellation timing establish the pending-RPC crash boundary. Two identical round-one
requests produce one counted contribution; all six unique updates reconstruct the
completed three-round model exactly. No product or APK change was needed. Account,
settings and data are preserved and temporary test resources removed. This adds no
automatic resume, natural Android kill, interrupted computation/request transport,
checkpoint durability, production signing or deployed TLS qualification.
