# Simulation Lab worker contract

20 October 2026. Version 1 worker foundation is implemented and verified. The
persistent API, dashboard and browser demonstration are separate remaining work.
These are exploratory validation runs; this worker never launches the frozen study.

## Entry and recipe boundary

Use the lightweight `python -m fedlearn.lab_worker` entrypoint. It establishes
deadline and parent-pipe monitoring before importing Torch or the simulation
package. Historical simulation package imports and numerical code are unchanged.

The installed library owns `fedlearn.simulation.lab_contract.resolve_choices` and
`write_research_record`; it does not import the uninstalled benchmark package.
`resolve_choices` returns the actual resolved `config`, its `config_hash` and a
path-free `display_recipe`. Identical choices and server workload paths retain the
same hash; a run UUID is separate from the fixed config name `simulation-lab`.

```sh
python -m fedlearn.lab_worker resolve \
  --choices "$SIMULATION_CHOICES" --cache "$CAPSTONE_CACHE_DIR" \
  --report-sha256 "$CAPSTONE_FEATURE_PIN"

python -m fedlearn.lab_worker run \
  --config "$SIMULATION_JOB_ROOT/config.json" \
  --output "$SIMULATION_JOB_ROOT/execution" \
  --report-sha256 "$CAPSTONE_FEATURE_PIN" --run-id "$SIMULATION_JOB_ID" \
  --deadline-seconds 180
```

The service writes only the resolver's `config` object to `config.json`, not the
whole response envelope. The job directory's basename must be its canonical UUID;
config and a fresh `execution` directory share that parent. Existing/symlink outputs
and foreign parents are refused. Server configuration supplies the cache and pin;
neither comes from browser choices. Keep the child's stdin pipe open while it is
owned; EOF means that its owner disconnected.

Supported choices follow the [design table](../../superpowers/specs/2026-10-20-simulation-lab-design.md):
four heads, 10/80/1,000 full-cohort clients, 1–20 rounds, 1–7 local epochs, mean or
20% trimmed mean, clipping 0.05/0.4, no DP or central epsilon 1/4/8, clean/own-update
sign flip/own-data label flip, and an integer seed. The worker reconstructs the
complete permitted recipe and compares its canonical JSON before loading data.
Validation-only evaluation, natural prevalence, pretrained cache, learning rate
0.003, batch size 256, derived seeds and four CPU threads are enforced.

Input JSON files are bounded to 64 KiB and reject duplicate fields/nonfinite JSON
constants. Cache integrity and pretrained checks precede core entry. The worker
uses the existing `run_core` qualification and completed-round callbacks.

## Evidence and lifecycle

| File | Meaning |
| --- | --- |
| `progress.json` | Individually atomic latest snapshot: version/run/hash/state, actual completed rounds, requested rounds, record encoding/availability and controlled error |
| `execution.json` | Individually atomic process evidence: PID, state/exit, start time, elapsed worker time, deadline and CPU threads |
| `record.json` | Retained core research record, published only from the selected staged candidate |
| `diagnostics.txt` | Bounded private traceback when execution fails; never a browser error response |

Starting snapshots may have a null config hash until recipe validation succeeds.
Round entries come only from actual callbacks; refused/invalid metrics cannot make
the final run qualified. Nonfinite failure values use the existing
`nonfinite_tagged_v1` representation instead of invalid JSON. Partial records omit
the core qualification marker. Record filenames in progress are fixed relative
names, not instructions for a file-serving endpoint.

SIGTERM/SIGINT request a round-boundary stop. Parent EOF requests interruption.
An unresponsive stopped core terminates after two seconds. An independent deadline
also covers cache loading and final record staging; a killed worker retains its last
atomic completed-round snapshot. Partial and qualified record candidates are staged
outside the watchdog lock, then selected at the terminal decision. Signals arriving
after that decision remain harmless through process shutdown.

| Exit | Terminal outcome |
| --- | --- |
| 0 | Completed and core-qualified, with retained record |
| 2 | Failed/refused, with available partial evidence retained |
| 3 | Stopped by request or interrupted by parent disconnection |
| 124 | Worker deadline reached |

The API must additionally enforce its budget from process launch, including
interpreter startup, and drain the owned process/tree before freeing admission.
Never infer completion from round count alone: require the actual process exit and
consistent run/config-bound terminal snapshots and record. External SIGKILL can
leave the two individually atomic snapshots at different phases.

Observed engine-completion high-water RSS must be positive and at most 2 GiB for
qualification. This is an observation gate, not an enforced process-tree or
late-artifact memory cap. Normal trusted-local filesystem assumptions apply;
power-loss durability or a hostile host is not established.

Records are **diagnostic research data, not DP-safe exports**. Private records retain
configuration paths and internal metadata. The API must authenticate ownership and
project/redact browser responses and downloads; this worker does not provide those
future endpoint guarantees.

## Verification and remaining limitation

The 107 focused checks include real synthetic-cache subprocesses, actual first/final
round cancellation, parent loss, abrupt kill, postflight/nonfinite failures, stalled
record serialization, source-only CLI execution, slow startup and late terminal
signals. Both Important review findings were reproduced before correction. The full
framework gate passed **2,183 tests**, with 14 slow tests deselected, 79.82% coverage
and thread exceptions treated as errors.

An offline-built wheel completed two actual synthetic-cache rounds from an isolated
installation. All 45 loaded FedLearn modules originated there, with no benchmark
imports. Synthetic controls establish execution behavior, not NIH learning quality.

One minor input-formatting issue remains: an extremely large integer in a numeric
choice can produce a private CLI traceback/exit 1 instead of the ordinary safe
exit-2 refusal. It still refuses before scoring. The future API must map any failed
resolver exit to a controlled validation response; browser disclosure is not claimed
or demonstrated. This is recorded separately from the completed lifecycle fixes.
