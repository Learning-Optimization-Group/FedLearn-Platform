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
