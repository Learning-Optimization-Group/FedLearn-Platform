# Sign-out cancellation before restart recovery — 16 October 2026

The immediate reliability priority was ownership of a running training session.
Signing out changes authentication state and unmounts the authenticated navigator,
including `TrainingProvider`. The previous provider cleaned up its status polling
and log subscription but never set its training stop flag, aborted the native
client or stopped the foreground service on unmount. Its pending training closure
could keep running after sign-out. A new provider also received a separate
participation guard despite sharing the same native core.

Source `07f2d44` makes unmount cancel local participation, stop its service and
polling, and drain pending work through the existing native abort path. The
participation guard now spans provider lifetimes and remains closed through the
final abort of any registration created while enrollment unwinds. Detached
callbacks cannot record later contributions or dispatch into the old provider.
A queued log callback now uses the subscription's existing liveness guard.
Ordinary provider rerenders keep training alive; explicit Stop remains available.

Seven regressions cover unmount during training, late enrollment, a final abort
still pending, ordinary rerenders and Stop, repeated unmount/stale screen actions,
queued logs, and unmount during explicit Stop. The initial two cases failed on
the old provider; the queued-log case also failed before its guard was added.
The final mobile suite passed 699 tests in 45 suites, plus lint and type checking.
The provider tests drive its real callbacks/effect cleanup with separate hook
cells and mocked native/network boundaries; they do not substitute for actual
React lifecycle or native cancellation qualification. Three existing native
DeComFL guard/reconnect cases also passed. The new standalone Debug assembly
passed; its JVM test task was up to date, so no fresh JVM execution is claimed.

The [physical receipt](../../results/on-device/android-signout-lifecycle-2026-10-16.json)
binds the installed standalone APK, source, accepted-update recorder window,
actual desktop parents/configurations, UI observations and cleanup. A fresh
CNN/FedAvg/FULL/LOCAL_SNAPSHOT run used 32 disjoint CIFAR training images per
participant. The canonical Linux client registered to meet the two-client quorum
but held its actual first fit behind a bounded test gate. Vivo submitted round
one, then the normal Settings → Sign out action removed its training service
without changing the phone process.

Releasing the desktop advanced the server to round two and accepted the desktop's
round-two update. During a further ten-second observation, the phone remained
signed out with no foreground service and no round-two update. The recorded
prefix was exactly phone round one, desktop round one and desktop round two.
After explicit Sign in → Join → dataset selection → Start, the phone submitted
rounds two and three. The run completed with six unique accepted contributions,
three per participant. Those updates reconstructed every float32 bit of the
registered final model. This checks the accepted aggregate with the production
aggregation implementation; it is not independent numerical replay of the phone.

This test used temporary USB reverse/SSH loopback routes, with USB power retained
for the old battery and Metro stopped throughout. It adds no Wi-Fi, deployed TLS,
endurance or natural-Doze qualification. The physical cancellation point was the
wait after an accepted round; the unit cases cover pending enrollment/in-flight
work. No worst-case bound for interrupting an expensive model step is claimed.
The normal relogin/rejoin occurred within the same phone process, so this is not
active process-death recovery.

All 36 saved phone dataset files retained their hashes. The original saved account
and server setting were restored, and an idle Metro-off cold launch succeeded.
The temporary test package, USB routes, SSH forwards, desktop process and local
guards were removed or stopped. Vivo was left asleep on USB power. All 1,013
previous source/evidence/program bindings, historical build outputs and the root
working-tree edit were verified unchanged. Previous APK/evidence bindings remain
intact; the newly installed package contains this fix.

The next recovery work should be split by algorithm. First-order rounds already
reload the server's current model, and their submitted-round checkpoint persists;
a controlled process-kill qualification should establish what actually fails
before adding model-state storage. A small explicit recovery offer could then
retain the run, contract, account/server and dataset identity without auto-starting
training. DeComFL also needs its reconstructed model and applied-round cursor;
saving the last uploaded round alone cannot restore it. Its server history is
pruned using server-side client progress, so a durable trajectory design must
also reconcile that progress and define a safe fallback for unavailable history.
The current late-start refusal must remain until that behavior is implemented
and qualified. Production signing and deployed TLS remain separate work.
