# Manual recovery with a lost upload completion — 16 October 2026

The existing first-order recovery path passes one stronger interruption boundary:
the server accepts the phone's complete round-one update, its successful response
is withheld, and Android crashes the phone process before the native call can
return. Explicit cold launch and rejoin retry the same round; the server counts
the phone once. No product code or new checkpoint storage was needed for this case.

The [qualification receipt](../../results/on-device/android-acknowledgement-crash-2026-10-16.json)
binds the unchanged installed standalone internal Debug APK, relevant source,
wire payloads, process/service observations, UI actions, recorder window, actual
desktop parent models, final model and cleanup. Runtime source is `609d2cf`;
qualification checkout `e254afd` adds documentation only. The APK was neither
rebuilt nor reinstalled. Its SHA-256 is
`03d3a21e0b2d3021a84ea9d7a8254b249e82ef6f33e730aa7ea4c9b13b6c7d4d`.

The case uses CNN/FULL/LOCAL_SNAPSHOT/FedAvg, two participants, three rounds and
32 disjoint real CIFAR training images per participant. The canonical Linux
client holds its first actual fit until the phone's duplicate check finishes,
keeping the server on round one throughout restart and retry.

The diagnostic proxy gates server responses synchronously before forwarding the
phone's complete upload chunk. It observes upstream gRPC status zero and
`received=true`, while the server recorder independently shows one accepted phone
update. All 53 server-response bytes on that connection remain withheld and are
discarded when the old connection closes. Resuming the proxy allows fresh
connections; it never releases that old completion. A real local gRPC fixture
also verifies that upstream success leaves the client call pending, and that a
fresh connection succeeds after resume.

Android's `am crash` terminates verified phone PID 24207 with a fatal VM exception.
Process-death events confirm termination. Nineteen samples over 15.97 seconds show
no phone process or foreground service. Explicit Activity launch creates PID
29062 and opens Idle, without an orphan training service. Authentication survives;
the joined run and dataset selection do not. The new project's local contribution
is absent at this cold launch. This is an actual process crash, but does not
qualify low-memory termination, SIGKILL, reboot or a partially transmitted update.

Normal UI Join → select the existing dataset → Start retrains and uploads round
one. The two complete 248,804-byte wire payloads match byte-for-byte, SHA-256
`f578ddfe8e2b748f84f6eb9b144c683fd7342d7c05b9949c38859a5f05532890`.
Their full tensor state also matches the first counted recorder state across all
10 float32 tensors and 62,006 values. Wire and recorder serialization metadata
differ, so that latter comparison checks tensor keys, dtypes, shapes and bits.

The retry receives a successful RPC response while the server remains at one
counted round-one update. `received=true` acknowledges the RPC; it does not mean
the coordinator counted another contribution. The coordinator rejects a duplicate
from the same client under its lock. Four existing coordinator tests pass for
duplicate suppression, distinct-client aggregation, stale and future rounds.
No third phone round-one upload appears during another 10.38 seconds; every
sampled server status still reports round one and one of two required updates.

Checkpoint claims are deliberately limited. The encrypted checkpoint value was
not directly inspected or edited. Its initial absence follows from the fresh run
identity and source ordering: native `writer->Finish()` cannot return while the
completion is withheld; JavaScript saves the submitted-round marker only after
the native round returns. After the retry, the ordinary UI's round-one log proves
the awaited save promise returned. This does not test fsync, torn writes or
durability after a second crash. A local marker is not authoritative evidence
that the server counted an update.

Releasing the desktop completes all three rounds. The phone makes four upload
RPCs but contributes three counted updates; the desktop contributes three more.
All six unique accepted updates contain 32 examples. Reaggregation using production
FedAvg matches every float32 bit of the registered final model, and all three
actual desktop parent states match the reconstructed round inputs. Final NPZ
SHA-256 is `62e99fe17a24eff06938b9a652e1cd9ab3f9ec47b3d1fab34a3880342847439f`.
This verifies the aggregate, not independent numerical replay of phone training.

All 36 saved dataset files retain their hashes; no data import, app uninstall or
app-data reset was performed. The original saved account and API setting were
restored through the normal UI. The desktop, proxy, temporary UI helper, local
guards, ADB routes and SSH forwards were stopped or removed. Metro remains off,
the API configuration is unchanged, and vivo was left asleep on USB power at
100% charge and 26°C. The 1,365 prior bindings pass the preservation audit, including
the existing explicit handling of previously authorized source revisions.

The JavaScript, JVM and native suites were not rerun because product source and
APK were unchanged; their prior results remain attached to the previous receipt.
This test adds no support-matrix cells, automatic resume, DeComFL durable recovery,
production signing, deployed TLS, natural Doze, endurance or clinical-quality
claim. Further first-order interruption tests should target computation, partial
upload and checkpoint-write boundaries. DeComFL still needs its reconstructed
model and history cursor to survive together and reconcile with server history.
