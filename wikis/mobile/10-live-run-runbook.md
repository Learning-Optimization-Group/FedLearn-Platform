# Live Mixed-Device Run — Runbook

How to run one TinyNet federation with the Android phone and three laptop CLI clients on one published execution
contract, then check the phone's training against the contract. This is the procedure behind the live results in
[06](06-execution-contract-v1-implementation-plan.md) (Stage 2I), [07](07-fedopt-robust-contract-plan.md),
[08](08-decomfl-contract-design.md) and [09](09-fedprox-contract-plan.md). Every step here has failed at least once
when skipped; the [troubleshooting table](#troubleshooting) lists how.

## Topology

```
phone ──USB── relay host ──SSH reverse tunnels── workstation (backend :8084, Metro :8088, FL servers :50000-50010,
      adb reverse                                                laptop clients on 127.0.0.1)
```

The phone reaches everything through `127.0.0.1` on its own side:

| Phone port | Relay port | Workstation | What |
| --- | --- | --- | --- |
| 8081 | 8081 | 8088 | Metro (a debug APK loads its JS from `localhost:8081`) |
| 8082 | 8082 | 8084 | Backend REST |
| 50000–50010 | same | same | FL servers (one per running project) |

Metro is moved off 8081 because the debug APK and a default backend both want that port.

The phone can also be USB-attached to the workstation directly. Then skip the SSH tunnels and point `adb reverse`
at the workstation ports.

Set these for the commands below:

```bash
REPO=~/code/FedLearn-Platform
RELAY=relay-host                         # ssh alias of the host the phone is plugged into
ADB=adb                                  # path to adb on that host; it may not be on PATH there
WORK=$(mktemp -d)                        # run artifacts: logs, cookies, contracts
```

## 1. Build and install the APK

The native build needs the cross-compiled artifacts under `mobile_client/.artifacts/` (gitignored):

```bash
A=$REPO/mobile_client/.artifacts
cd $REPO/mobile_client/android && ./gradlew assembleDebug -PreactNativeArchitectures=arm64-v8a \
  -PET_SRC=$A/executorch-android-v1.3.1-arm64-v8a/executorch \
  -PET_BUILD=$A/executorch-android-v1.3.1-arm64-v8a/executorch/cmake-out \
  -PTORCH_INCLUDE=$A/torch-headeronly-shim-2.12.0 \
  -PGRPC_DIR=$A/grpc-android-v1.67.1-arm64-v8a \
  -PGENERATED_PROTO_DIR=$REPO/proto/gen/cpp
```

A build with no `-P` flags fails at CMake configure (`Could not find ET_LIB_executorch`). Record the APK's SHA-256
and the commit it was built from. When the run exercises new native code, also check that the change made it in:
`unzip` `lib/arm64-v8a/libappmodules.so` from the APK and grep it for a string the change added. Then install:

```bash
scp app/build/outputs/apk/debug/app-debug.apk $RELAY:/tmp/app-debug.apk
ssh $RELAY "$ADB devices -l; $ADB install -r /tmp/app-debug.apk"
```

`adb devices` must list the phone as `device`. `unauthorized` means the phone is waiting on its "Allow USB
debugging?" prompt. An empty list means the host sees no USB data device at all: check the cable, the phone's USB
mode ("File transfer", not "Charging only") and, on a Mac, the "Allow accessory to connect?" prompt.

## 2. Metro

```bash
cd $REPO/mobile_client && npx react-native start --port 8088
curl -s localhost:8088/status      # packager-status:running
```

## 3. Backend

A throwaway PostgreSQL, then the backend on :8084:

```bash
docker run -d --name fedlearn-live-pg -e POSTGRES_USER=federance -e POSTGRES_PASSWORD=federance \
  -e POSTGRES_DB=federance -p 127.0.0.1:5438:5432 postgres:16.6-alpine
mkdir -p $WORK/models
cd $REPO/backend/fl-platform-api && \
SPRING_PROFILES_ACTIVE=dev SERVER_PORT=8084 \
SPRING_DATASOURCE_URL=jdbc:postgresql://localhost:5438/federance \
SPRING_DATASOURCE_USERNAME=federance SPRING_DATASOURCE_PASSWORD=federance \
FEDLEARN_PYTHON=$REPO/.venv/bin/python \
FEDLEARN_ROUND_TIMEOUT_S=900 \
APP_MODEL_BUNDLE_DIR=$WORK/models \
FL_SERVER_GRPC_HOST=127.0.0.1 \
nohup ./gradlew bootRun > $WORK/backend.log 2>&1 &
```

Every variable is load-bearing:

| Variable | Why |
| --- | --- |
| `FEDLEARN_PYTHON` | the spawned FL server and scripts otherwise use the global `python3` |
| `FEDLEARN_ROUND_TIMEOUT_S=900` | the default 120 s is too short to join a phone by hand |
| `APP_MODEL_BUNDLE_DIR` | the default `/var/models` is not writable on macOS; staging fails and the run gets no contract |
| `FL_SERVER_GRPC_HOST=127.0.0.1` | on the dev profile the backend auto-advertises a Tailscale or LAN IP instead of `localhost`, and the phone then bypasses the USB tunnel |

Ready when `curl -s -o /dev/null -w "%{http_code}" localhost:8084/api/auth/me` prints `401`.

## 4. Tunnels

Run the SSH tunnel under **bash**. In zsh, `"$p:localhost"` applies the `:l` modifier and silently mangles the
FL-port forwards, while 8081/8082 still work:

```bash
bash -c 'args=(-N -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 -R 8081:localhost:8088 -R 8082:localhost:8084)
for p in $(seq 50000 50010); do args+=(-R "${p}:localhost:${p}"); done
exec ssh "${args[@]}" '"$RELAY" &
ssh $RELAY "for p in 8081 8082 \$(seq 50000 50010); do $ADB reverse tcp:\$p tcp:\$p; done"
```

Check every hop from the relay before touching the phone. An FL port only answers once a project is running, so
repeat the last check after step 5:

```bash
ssh $RELAY 'curl -s -o /dev/null -w "%{http_code}\n" localhost:8082/api/auth/me; curl -s localhost:8081/status'
# with a run started: the FL server must answer an HTTP/2 preface with a SETTINGS frame (00 00 .. 04 ..)
ssh $RELAY "printf 'PRI * HTTP/2.0\r\n\r\nSM\r\n\r\n\x00\x00\x00\x04\x00\x00\x00\x00\x00' | nc -w 3 127.0.0.1 50000 | head -c 12 | od -An -tx1"
```

An empty reply on the last check means the FL forward is broken even when the REST forward works.

## 5. Create the run and start the laptops

```bash
B=localhost:8084; C=$WORK/cookies.txt; STRAT=FedProx   # FedAvg | FedOpt | Robust | FedProx | DeComFL
curl -s -X POST $B/api/auth/register -H 'Content-Type: application/json' \
  -d '{"username":"mobiledemo","email":"mobiledemo@example.com","password":"demopass123"}'
docker exec fedlearn-live-pg psql -U federance -d federance -tAc \
  "UPDATE users SET platform_role='PLATFORM_ADMIN' WHERE username='mobiledemo';"
curl -s -c $C -X POST $B/api/auth/login -H 'Content-Type: application/json' \
  -d '{"username":"mobiledemo","password":"demopass123"}'
P=$(curl -s -b $C -X POST $B/api/projects -H 'Content-Type: application/json' \
  -d "{\"name\":\"$STRAT live run\",\"modelType\":\"TINYNET_GOLDEN\",\"modelName\":\"tinynet-golden\",\"optimizer\":\"sgd\",\"pretrainEpochs\":0}" \
  | python3 -c "import json,sys;print(json.load(sys.stdin)['id'])")
# wait until the project leaves INITIALIZING (GET $B/api/projects/$P -> status), then:
curl -s -b $C -X POST $B/api/projects/$P/start -H 'Content-Type: application/json' \
  -d "{\"strategy\":\"$STRAT\",\"numRounds\":3,\"minClients\":4,\"clientsPerRound\":4,\"secureAggregation\":false}"
RUN=$(docker exec fedlearn-live-pg psql -U federance -d federance -tAc \
  "select id from runs where project_id='$P' order by created_at desc limit 1;")
docker exec fedlearn-live-pg psql -U federance -d federance -tAc \
  "select state, contract_id, unavailable_reason from run_execution_contracts where run_id='$RUN';"
```

The contract must be `READY`. `UNAVAILABLE` carries its reason, and the backend log has the script output. Fetch it
the way a client does, check it, and start the laptops:

```bash
curl -s -b $C -X POST $B/api/runs/$RUN/enroll > $WORK/enroll.json
python3 -c "import json;json.dump(json.load(open('$WORK/enroll.json'))['manifest']['executionContract'],open('$WORK/contract.json','w'))"
cd $REPO && PYTHONPATH=framework/src:fl-runtime .venv/bin/python -c "
import json, execution_plan
from google.protobuf import json_format
from fedlearn.communication.generated import execution_contract_pb2 as pb
c = json_format.ParseDict(json.load(open('$WORK/contract.json')), pb.ExecutionContract())
print(execution_plan.check_contract(c, 'TINYNET_GOLDEN', '$STRAT', 'FULL', project_id='$P', run_id='$RUN'))"  # expect []
cd $REPO/fl-runtime && for part in 1 2 3; do PYTHONUNBUFFERED=1 nohup ../.venv/bin/python client.py \
  --project-id $P --server-address 127.0.0.1:50000 --partition-id $part --model-type TINYNET_GOLDEN \
  --strategy $STRAT --execution-contract $WORK/contract.json --run-id $RUN > $WORK/laptop_$part.log 2>&1 & done
grep -h "\[contract\]\|Refusing" $WORK/laptop_*.log     # three "execution contract accepted" lines
```

The round-deadline clock is already running. Join the phone within `FEDLEARN_ROUND_TIMEOUT_S`.

## 6. Join from the phone

In the app: sign in against server `http://127.0.0.1:8082` → **Projects** → the running project → **Join training
run** → **Home** → **Start training**. Joining only registers the phone. Nothing trains until Start training.

The phone can be driven remotely:

- Screenshots are reliable: `ssh $RELAY "$ADB exec-out screencap -p > /tmp/s.png"`. `uiautomator dump` returns
  stale trees on some devices.
- Scale taps from a screenshot by the ratio of `adb shell wm size` to the screenshot width.
- `adb shell input text` drops characters in React Native inputs; type one character at a time.
- A debug build shows an "Open debugger to view warnings" toast that can cover buttons. Dismiss it first.
- JS logs go to Metro, not logcat. Errors surface as the red banner on Home and in the Activity log.

## 7. Verify

The backend relays the FL server's output into its own log:

```bash
sed 's/\x1b\[[0-9;]*m//g' $WORK/backend.log | grep "FL_SERVER $P" | sed 's/.*\[FL_SERVER [^]]*\] //' > $WORK/flserver.log
grep -c "Coordinator accepted update" $WORK/flserver.log          # laptop (unary) updates
grep -c "Receiving streamed model update" $WORK/flserver.log      # phone (streamed) updates
grep -o "All [0-9]* clients reported for round [0-9]*" $WORK/flserver.log
```

Acceptance for a three-round, four-client run:

- the run reaches `COMPLETED`;
- `All 4 clients reported` for rounds 1–3, with 9 laptop and 3 phone updates;
- the phone's Activity log shows `Execution contract <id prefix>… accepted.`, matching the published `contract_id`;
- replaying the run under the contract predicts the phone's per-round losses and the saved final model
  (`backend/fl-platform-api/models/$P.npz`).

For the replay, vary only the phone's hypothesis, because the laptops were checked separately. Include at least
one wrong hypothesis that the run should reject. Otherwise a replay that "matches" proves nothing. The saved model
is the stronger evidence: the phone's 4-dp losses can miss a small deviation that the aggregate reveals (see
[09](09-fedprox-contract-plan.md)).

## 8. Tear down

```bash
pkill -f "client.py --project-id"; pkill -f fl_server.py; pkill -f bootRun
pkill -f "ssh -N -o ExitOnForwardFailure=yes"
docker rm -f fedlearn-live-pg
ssh $RELAY "$ADB reverse --remove-all"
```

## Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| Gradle: `Could not find ET_LIB_executorch` | the native `-P` paths are missing | step 1 |
| `adb devices` empty | no USB data connection | cable, USB mode, the host's accessory prompt |
| Contract `UNAVAILABLE`, `STAGING_FAILED`; laptops refuse with "contract ProtoJSON does not parse" | bundle staging could not write `/var/models` | `APP_MODEL_BUNDLE_DIR` |
| Phone banner: `RegisterClient failed … <tailnet IP>:50000 … timed out before receiving SETTINGS frame` | the backend advertised a Tailscale IP; gRPC to it from the app fails (plain HTTP works; cause not diagnosed) | `FL_SERVER_GRPC_HOST=127.0.0.1` |
| Phone banner: `UNAVAILABLE ipv4:127.0.0.1:50000: Socket closed` | the FL-port tunnel is broken (e.g. started from zsh) | step 4 under bash; rerun the SETTINGS check |
| Round never closes; `Clients reported 3/4` | the phone joined but never started training | Home → Start training |
| Run marked FAILED after training | FL-server callbacks went to another backend | fixed in `ca18c08`; with an older backend set `app.backend.internal-url` |
| Transient errors, then "Reconnecting to the run" on the phone | seen once before round 1 in the FedProx run; cause unknown | the resilient loop recovers; record it |
