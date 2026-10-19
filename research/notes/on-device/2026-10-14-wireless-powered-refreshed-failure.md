# Powered Wi-Fi upload timeout and premature clean stop

The planned ten-round qualification **failed after seven federated rounds**.
Fourteen unique 16-example updates were accepted, seven from each client, and
the phone's durable checkpoint remains seven. Round eight's local computation
finished, but its upload was partial and was not accepted. No successful final
aggregate or independent numerical client replay is claimed.

The unchanged Debug vivo app and canonical Linux client used the original
pneumonia CNN, FedAvg FULL and one Adam update per client/round. The display was
asleep for 28.2 minutes before failure detection, with external charger
power and the same app PID in all 110 background samples.
Sampled battery temperature was 24.0–26.0°C.
The named training-service record disappeared; the app process survived.
No matching app-kill event appears in the retained background event window.
The harness made no debugger evaluations during that window.

The persistent forwarding master had expired during the preceding idle gap.
A reopened master passed an explicitly verified 103 MB transfer with a matching
hash in 9.53 seconds, plus actual API/Metro responses and mounted-provider login.
A duplicate project-name refusal was resolved before creation. All previous
failed attempts remain unchanged; these setup checks do not guarantee sustained
throughput during later training.

The passive observer decoded 62 of the declared 99 round-eight upload chunks;
the server logged receiving 61. No final chunk, successful upload response,
accepted round-eight state or checkpoint eight was recorded. The server log
shows a roughly 30-second gap mid-upload. The phone's scrubbed diagnostic journal
then records gRPC code 4, Deadline Exceeded, at 21:01:54.384 UTC and a planned
one-second retry. At 21:01:55.530 UTC it records "Server ended this client’s
participation." The notification cancellation follows at 21:01:55.581 UTC,
service absence is sampled at 21:02:04.676 UTC, and the harness detects failure
at 21:02:08.213 UTC before requesting the owned project stop. These observations
distinguish the native clean-stop path from a subsequent harness intervention.

Inspection identifies a recovery defect. Three failed native heartbeats set the
same abort flag used for explicit/server Stop. The native round reads that flag,
the bridge emits a STOP-prefixed error, and the resilient JavaScript loop treats
it as terminal instead of rejoining. The incident matches this path, but the
internal flag reason was not directly sampled. The upload timeout itself is
confirmed by the retained on-device journal. The original network stall's cause
is not established; neither memory exhaustion nor an OEM process kill is proved.

The proposed bounded correction is to distinguish recoverable heartbeat loss
from genuine user/server Stop and send it through the existing bounded rejoin
path. Keep the RPC deadline and checkpoint/deduplication rules. Regressions must
prove retry after heartbeat loss, preservation of explicit Stop, no duplicate
accepted update and re-registration before resumed training. No product fix,
new build or changed app was included in this qualification.

All 33 saved files, installed APK/native core, g14 source, root working-copy edit,
305 older and 135 recent evidence bindings are preserved. Owned test helpers and
Mac guards are absent; development services remain available. Cleanup restored
the original empty reverse routes, confirmed empty forwards, and verified all
33 saved files and the installed APK before switching ADB to USB mode over the
existing Wi-Fi connection. The command acknowledged the switch; three subsequent
connections to the old debugging port were refused and its ADB transport was
absent. Physical USB identity, TCP-property and route postchecks remain pending;
this remote shutdown is distinct from a completed USB restoration check. The [receipt](../../results/on-device/android-wireless-powered-refreshed-failure-2026-10-14.json)
binds wire metadata, accepted states, journal, Android logs, runtime/source and
the exact private drivers.

Charging, the OEM app background opt-in with its derived Android idle allowlist
effect, fresh initialization and active Wi-Fi ADB diagnostics are recorded
conditions. Battery-only survival, natural deep Doze, sustained thermal behavior,
physical Release/deployed TLS and process-death recovery remain unqualified.
The partial-upload safety boundary held; prolonged connection recovery did not.
