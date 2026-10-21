# Simulation Lab: live demonstration and saved-results replay

Verified 21 October 2026. The dashboard runs the existing NIH No Finding versus
any-finding workload on frozen pretrained ResNet-18 features. Clients are simulated
patient groups on one host. The disease-specific follow-up has not been selected
or trained. This demonstration does not establish clinical usefulness or physical
multi-device performance.

## What was actually checked

The [aggregate verification receipt](simulation-lab-browser-verification-2026-10-21.json)
binds the execution source, dashboard source, feature pin and retained record hashes.
Actual browser actions used cookie-authenticated HTTP against PostgreSQL and the real
Python numerical worker. No canned responses or intercepted training results were used.

| Demonstration | Seed | Schedule | Outcome | Final validation AUROC | Final validation accuracy |
| --- | ---: | --- | --- | ---: | ---: |
| No DP, linear head, ten clients | 20261021 | E1/R3 | Completed, 3/3 | 0.635325 | 61.7351% |
| Central DP, epsilon 4, linear head, ten clients | 20261022 | E1/R3 | Completed, 3/3 | 0.610683 | 58.1152% |
| Stop control, MLP512, eighty clients | 20261023 | E7/R20 | Stopped, 2/20 | 0.653902 | 60.4684% |

These are short exploratory validation runs, with different seeds and schedules.
Their gaps are not estimates of a privacy effect or a model ranking. A stopped run
is partial and unqualified. The historical study and its matched-seed comparisons
remain separate; neither table establishes a clinically useful classifier.

The first enabled browser pass created exactly three jobs. A later responsive check
reopened the same identifiers without retraining. The browser observed a running
completed round, then STOPPING, then STOPPED with two retained rounds. The owned
worker was absent afterwards. Reload did not create a new run; the downloaded JSON
matched the owned API record; source/configuration identities matched the worker.
Anonymous API access returned 401, ordinary-user access 403, and another owner's
read/stop/export requests 404. The foreign-owner UI showed no download control.

At desktop and 390px width, controls remained usable. The phone-width menu was
keyboard-operable and restored focus after Escape. The initial document-width-only
check missed clipped controls; screenshot inspection found the fixed-sidebar defect,
which was corrected and covered by stronger main/control-width assertions. A
separate disabled-workload phase refused launches with 503 and retained all three
jobs. None of these checks used the vivo, g14 or Jetson.

## Dashboard captures

[Desktop results and settings](figures/simulation-lab-desktop-2026-10-21.png) and
[390px configuration view](figures/simulation-lab-mobile-2026-10-21.png) are actual
browser captures from the responsive recheck. Their hashes are in the receipt.
The narrow capture shows configuration; results continue below it.

## Prepare a live local demonstration

1. Use the capstone checkout, Node 24, Java 21, the prepared Python environment and
   PostgreSQL. Follow the [API configuration contract](simulation-lab-api-contract.md)
   to enable the worker with private interpreter/module/cache/storage settings.
2. Supply the independently verified feature-report pin:
   `6c9e6969c6fd762156ef2246323fbdbc5e9cb272c9910a8b1c9b691ab33783fd`.
   Do not point at a different or unverified cache merely because it has the same name.
3. Start the API and frontend from this checkout on explicitly chosen local ports.
   Point `VITE_FEDLEARN_API_URL` and `VITE_SERVER_ROOT_URL` at that API. Keep the
   demonstration database and job directory separate from other running stacks.
4. Sign in as PROJECT_OWNER or PLATFORM_ADMIN. The workload must show ready, with
   68,650 training and 17,763 validation images. A normal USER cannot use the Lab.

Simulation execution is disabled by default. No browser field accepts a server path,
command, interpreter or official-test evaluation choice. The 180-second deadline
and observed-memory qualification remain applicable; a valid recipe may time out.

## Live walkthrough

1. Open **Simulation Lab** and explain that the clients are simulated patient groups.
   Use **Three-round demo**, review its explicit settings, then **Review recipe**.
2. Show the resolved configuration identity. Choose **Start simulation** once. Point
   to actual completed rounds, status and validation measurements. No elapsed-time
   estimate is used as training progress.
3. Once completed, reload the page. The run stays selected through its URL; training
   does not restart. Open **Recipe and provenance**, load the execution details and
   distinguish measured local CPU/wall time from modeled network/deployment costs.
4. Download the research record and show its configuration, source, seeds and outcome.
   It is a diagnostic record, not a DP-safe release or a deployable clinical model.
5. Select two compatible terminal runs in history and choose **Compare selected**.
   Discuss differing settings alongside final observed metrics. Avoid best-round
   selection or treating these exploratory runs as a statistical ranking.
6. For the stop demonstration, use the already verified retained stop record when
   rehearsal time is limited. If demonstrating a new stop live, use the bounded
   reference schedule, wait for an actual round, then stop. Show STOPPING until
   verified termination, retained partial rounds and absent qualification.

## Run the external browser suite

The suite is maintained in homelab's `fedlearn` directory, outside this repository.
Its dedicated `playwright.simulation.config.ts` requires explicit loopback origins
and starts no servers. It does not reuse the default suite's sessions or attach
implicitly to another checkout's API. Set these variables in the test terminal:

- `FEDLEARN_REPO`: capstone checkout.
- `FEDLEARN_SIMULATION_FRONTEND_URL`: chosen loopback frontend origin and port.
- `FEDLEARN_SIMULATION_API_URL`: chosen loopback API origin and port, without `/api`.
- `FEDLEARN_SIMULATION_EVIDENCE_DIR`: a fresh private directory for this qualification.
- `FEDLEARN_SIMULATION_FEATURE_PIN`: the independently checked report pin above.
- `FEDLEARN_SIMULATION_SOURCE_COMMIT`: expected execution-source commit.
- `FEDLEARN_ADMIN_USER` and `FEDLEARN_ADMIN_PASSWORD`: the isolated stack's admin.

From the external suite directory, run:

```sh
npx playwright test --config playwright.simulation.config.ts \
  --project core --grep 'real training'
```

Expected: one passing actual-stack test. Automatic retries are disabled. Accepted
run IDs are retained immediately; rechecks reopen those IDs instead of replacing
training. Keep failures and outcomes. A source/identity mismatch requires investigation,
not deletion of the checkpoint to obtain another favorable result.

For the unavailable-workload phase, first establish that the isolated stack's jobs
are terminal, restart only that API with `app.simulation.enabled=false`, then run:

```sh
FEDLEARN_SIMULATION_EXPECT_UNAVAILABLE=1 npx playwright test \
  --config playwright.simulation.config.ts --project core \
  --grep 'unavailable workload'
```

Expected: one pass, a disabled UI, a 503 launch refusal and unchanged retained-run
count. This phase creates no additional training run. The private evidence directory
contains credentials, diagnostic records and screenshots; only the curated aggregate
receipt belongs with the public capstone material.

## Saved-results replay fallback

Announce **“This is a saved-results replay; no training is running.”** Open a retained
terminal run from history, or follow **View completed reference study** to the explicit
`/capstone/study-v1/index.html` page. It shows saved historical outcomes, all 64 recipe
groups, audited figures and the source/verification manifest. It never starts training.
If the API is unavailable, the static historical page remains the replay entry.

The original Markdown report download deliberately preserves its historical bytes.
Its supplementary links require the repository layout; the public page discloses that
limitation. A self-contained Markdown dependency package remains deferred.

## Remaining capstone work

Select the new condition and evaluation route, qualify the stronger centralized and
matched federated baselines, then bound the supported privacy and attack studies.
Final report, presentation, advisor scope/title confirmation and full rehearsal are
still open. The current dashboard/browser evidence is preserved locally; its
new off-device backup has not been verified. The historical study is unchanged.
