# Android Release and TLS qualification procedure

This is the next qualification procedure, not a completed release claim. Existing
evidence is in [release readiness](14-release-readiness.md). The first target is
CNN/FULL/LOCAL_SNAPSHOT/FedAvg with one powered phone and one Linux client, three
rounds and 32 disjoint CIFAR images per participant. Exercise DeComFL only after
this first path works.

## Trust and identity prerequisites

There are three different identities:

| Identity | Provisioning | Acceptance condition |
| --- | --- | --- |
| HTTPS API | Server certificate and full chain for the exact API hostname | The unchanged Release app trusts the chain through Android system roots |
| gRPC server | Certificate/key served by the real training process | Enrollment supplies the trust certificate; the advertised hostname matches its SAN |
| Android signer | Private release keystore with a retained, backed-up key | Release validation passes and two successive APKs share the same signer |

A self-signed local HTTPS certificate is useful for host-side preparation, but
the current Release network policy trusts system roots only. Host-side `--cacert`
success does not prove the phone will trust it. Do not weaken that policy, disable
hostname verification, install a system root, or use Debug to claim Release TLS.
Use an operator-controlled endpoint with an Android-trusted chain for the real
phone test. No public deployment or DNS change is required for local preparation.

Use server TLS plus the platform's connection-token authentication first. Leave
`FEATURE_FL_CLIENT_CERT=false` and gRPC client-certificate enforcement off: mobile
client-certificate provisioning is not implemented. This does not qualify mTLS.

## Isolate the staging stack

Use a fresh private output directory inside the checkout, an isolated PostgreSQL
database/container, unused API/gRPC ports and separate bundle/artifact directories.
Do not reuse or reconcile the historical experiment database. Bind the API/proxy
to loopback; use only explicitly owned routes for remote participants. The current
neural FL server binds `[::]`: `FL_SERVER_GRPC_HOST` changes its advertised address,
not its bind address. Before starting a run, verify firewall or network-namespace
isolation of the gRPC ports. The generated files alone do not provide that isolation.

The existing deployment layout terminates HTTPS at nginx and sends requests to
the API over host loopback; callbacks also use loopback. See
[deploy/TLS.md](../../deploy/TLS.md). Retain that layout to avoid introducing a
second HTTPS callback trust mechanism. Set these in the isolated backend's
private environment, using actual paths/ports:

```bash
APP_FL_REQUIRE_TLS=true
APP_FL_REQUIRE_CLIENT_AUTH=true
FEATURE_FL_CLIENT_CERT=false
FEDLEARN_GRPC_REQUIRE_CLIENT_AUTH=0
FEDLEARN_GRPC_SERVER_CERT=/path/to/staging/grpc-server.crt
FEDLEARN_GRPC_SERVER_KEY=/path/to/staging/grpc-server.key
FL_SERVER_GRPC_HOST=hostname-covered-by-grpc-certificate
APP_BACKEND_INTERNAL_URL=http://127.0.0.1:isolated-api-port
APP_AUTH_COOKIE_SECURE=true
```

Supply distinct generated JWT, FL-token and internal-API secrets, isolated
datasource credentials, explicit CORS origins, Python paths and writable storage
paths. Keep the environment and private keys out of Git and logs. Certificate
fingerprints and public configuration may be retained in a sanitized report.
The backend reads the gRPC certificate at startup and publishes its trust material
at enrollment. It also passes fail-closed TLS/token enforcement to its child.

From the repository root, generate local preparation material in a fresh directory:

```bash
mkdir -p .temp
python mobile_client/scripts/prepare_tls_staging.py \
  --out .temp/release-tls-staging --python /path/to/pinned-python/bin/python
python -m unittest discover -s mobile_client/scripts/tests \
  -p test_prepare_tls_staging.py -v
```

The helper starts no services and changes no phone settings. It writes separate
seven-day API/gRPC test certificates, private keys, a private backend environment,
a PostgreSQL environment, a loopback nginx configuration and an honest preparation
manifest. It refuses existing directories and outputs outside the checkout's
ignored `.temp` directory. The
tests perform actual loopback TLS handshakes and check hostname/root refusals,
private-file permissions and overwrite protection.

The generated ports are API8086, HTTPS8443, PostgreSQL5546 and gRPC55100–55103.
Check they are unused before starting any services. For a fresh local PostgreSQL:

```bash
docker run --name fedlearn-release-tls-pg --detach \
  --env-file .temp/release-tls-staging/postgres.env \
  --publish 127.0.0.1:5546:5432 postgres:16.6-alpine
```

Load `backend.env` in a private shell, use Java21, and start the API's existing
`./gradlew bootRun` from `backend/fl-platform-api`. The generated environment uses
the local dev profile with explicit fresh secrets, required TLS/token authentication
and separate storage/ports. Production bootstrap correctly rejects a loopback
advertised gRPC hostname. A production-profile run therefore needs a real reachable
hostname and matching certificate SAN; do not bypass that validator.
Start a host nginx with the absolute generated `nginx.conf` after `nginx -t`
passes. Its listener is loopback-only, forwarding HTTPS8443 to API8086 and leaving
internal callbacks on host loopback. For example:

```bash
nginx -t -c /absolute/path/to/.temp/release-tls-staging/nginx.conf
nginx -c /absolute/path/to/.temp/release-tls-staging/nginx.conf
curl --cacert .temp/release-tls-staging/api.crt https://localhost:8443/actuator/health
```

Use an available host nginx installation; the preparation helper does not install
one or publish the endpoint. The container image can syntax-check the configuration
with its generated directory mounted at the same absolute path, but that alone
does not start a working host proxy. The image version above is the repository's
existing local PostgreSQL fixture, not a production deployment recommendation.

Stop only the nginx/database/API instances started for this case; retain their
logs and private outputs for verification. Keep self-signed local results labelled
as preparation. The phone Release test still needs the trusted HTTPS endpoint above.

## Build and update without losing the installed app

Configure the existing signing gate with `FEDLEARN_RELEASE_STORE_FILE`,
`FEDLEARN_RELEASE_STORE_PASSWORD`, `FEDLEARN_RELEASE_KEY_ALIAS`,
`FEDLEARN_RELEASE_KEY_PASSWORD` and optionally `FEDLEARN_RELEASE_STORE_TYPE`.
Build `:app:assembleRelease` with the native arguments in the
[current run guide](15-current-android-runbook.md#build-the-internal-standalone-apk).
Do not add `fedlearnStandalone` to a Release build; Release already bundles
production JavaScript and enforces TLS in both the JS and native paths.

The current phone installation is Debug-signed. A differently signed Release
cannot replace it in place. Preserve it and all saved datasets. Select a separate
test installation/device before installing Release; record package identity and
any qualification-only packaging difference. Never uninstall or clear the current
app to work around the signature mismatch. A separate staging package cannot
qualify migration of the original Debug app's data.

Keep the signing key between the first install and the update test. A disposable
test signer can qualify the update mechanism, but is not the future production
identity. Establish and back up the production signer before distributing an APK.

## Release workflow configuration

The Android workflow reads these repository secrets:

- `FEDLEARN_RELEASE_KEYSTORE_BASE64`: the base64-encoded retained keystore.
- `FEDLEARN_RELEASE_STORE_PASSWORD` and `FEDLEARN_RELEASE_KEY_PASSWORD`.
- `FEDLEARN_RELEASE_KEY_ALIAS`.

`FEDLEARN_RELEASE_STORE_TYPE` is an optional repository variable (default PKCS12).
The workflow refuses absent secrets before native compilation, imports the store
into a private temporary directory, passes credentials only to the signing/build
steps, and removes the temporary store even after a failure. Its host preparation
tests run before native compilation. Configure secrets through the repository's
normal secret-management UI; never put a key/password in an issue, commit or log.
Base64 is encoding, not encryption. The configured identity and its backup remain
operator responsibilities. A local test of the workflow's shell steps is not a
successful GitHub Actions run or release publication.

## Acceptance sequence

1. Confirm the signed APK's identity, embedded production bundle, native library,
   disabled Metro/developer support and unchanged TLS restrictions. Record hashes.
2. Cold-launch Release with Metro stopped. Through normal UI, set the trusted
   HTTPS origin, log in, join the project, import/select local data and Start.
3. Confirm the real backend enrolls both clients, advertises TLS and issues
   distinct scoped connection tokens. Confirm the real training server accepts
   registration/heartbeats and all model/update traffic uses TLS.
4. Complete three rounds with six unique contributions. Verify the registered
   final artifact against the actual accepted updates; record time/bytes and
   held-out metrics separately from execution correctness.
5. Exercise one bounded connection interruption and explicit rejoin. Verify no
   duplicate counted contribution. Repeating the whole crash matrix is unnecessary.
6. Refuse wrong gRPC root, wrong hostname, plaintext and missing/invalid connection
   tokens without counting a contribution. Refuse untrusted HTTPS without sending
   login credentials. Use disposable accounts/tokens; do not retain secrets.
7. Install a second Release APK with a higher version code and the same signing
   key. Verify a cold launch, saved account/settings/datasets and a short run.
8. Restore the original phone setup and remove only owned staging processes,
   routes and helpers. Recheck the preserved Debug app and historical bindings.

Report host preparation, physical Release execution, update preservation and
production deployment as separate outcomes. Passing a fixture or building an APK
must not turn an unperformed step into a pass.
