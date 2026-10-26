# Clean Capstone Reproduction Implementation Plan

**Goal:** Verify generated-data qualification from fresh, self-contained CPU environments on all three hosts and replace stale delivery status with current evidence.

**Architecture:** Add a standard-library entry point for fresh environment installation and environment checks; load numerical dependencies only for generated qualification. Reuse the frozen manifest, canonical client/server and existing analysis/audit. Preserve completed medical artifacts and fitting source. The current request to continue authorizes this bounded reproduction work; no new medical fitting, evaluation cohort, service change or external messaging is included.

**Tech stack:** Python3.12, uv0.11.9, platform-specific hashed CPU locks, existing Torch/gRPC runtime.

## Global constraints

- Reuse the isolated capstone worktree; preserve existing reports, models, raw studies and source/input checkpoints.
- New environments must not inherit another environment or user site packages. Reuse base Python interpreters and download caches only.
- Only generated features/labels and their metadata may be used for fresh qualification. No holdout/test reads or medical fits.
- CPU Torch2.12.0, NumPy2.1.2, SciPy1.17.1; respect current repository dependency floors for cryptography rather than copying an older research environment.
- Mac coordinates; G14/Orin receive generated training packs only. Existing private SSH routes/configuration remain private; no shared service changes.
- Keep endpoints and absolute workstation paths out of tracked artifacts. Preserve the branch/worktree locally; no push/merge.

## Review focus

Environment leakage through `.pth` or user packages; platform-specific Torch wheel selection; hidden simulator imports; stale outputs falsely marked complete; generated qualification accidentally accepting a medical bundle.

## Task1 — Self-contained environment and dependency locks

Files: add `repro/requirements-capstone-cpu.in`, three platform hash locks, `repro/reproduce_capstone.py`, `repro/test_reproduce_capstone.py`.

Interfaces: `select_lock(system, machine)->Path`; `install_environment(destination, uv)->dict`; `environment_report()->dict`. Environment CLI uses standard library only before installation. Fresh directories only; Python3.12 and supported platforms required. Verify uv version, install hash-locked CPU packages, refuse outside-site imports and report full installed package versions plus lock hash.

- [x] Write failing tests for unknown platforms, reused output, environment leakage and generated bundle creation/refusal.
- [x] Resolve and inspect platform hash locks using official package indexes; install a genuinely fresh Mac environment.
- [x] Verify core imports, source resolution and environment isolation without the parent environment.

## Task2 — Generated qualification command

Interfaces: `write_generated_bundle(output)->sha256`; `qualify(output, launch=None)->dict`. Deterministic NumPy seed911,1040 training/80 selection rows,512 features, groups of4 generated patients, disjoint train/selection IDs, labels `x[:,0]>0`. Local mode runs both methods Mac-only; three-host mode runs both methods and placements using existing qualification mode. Both modes compare against the actual simulator HeadClient and require max parameter error <=1e-5, then run existing selection rescore and raw-journal audit for the selected cells. Fresh outputs only; partial failures retained and no success receipt on failure.

- [x] Implement generated builder, simulator comparison and command using existing execution lifecycle; no second FL client or new aggregation protocol.
- [x] Verify local qualification from the clean Mac environment, including existing real-transport failure checks.
- [x] Provision fresh remote environments from the same locks, create a private launch config pointing to them, and run the four generated three-host qualification cells.
- [x] Preserve a path-free receipt binding source, locks, runtime versions, comparisons and complete outcomes.

## Task3 — Delivery documentation, review and checkpoint

- [x] Document exact fresh-install/doctor/qualification commands and private launch template.
- [x] Update current checklist to v5/completed CPU evidence; keep actual advisor/presentation/submission decisions open.
- [x] Fresh whole-change review; meaningful focused tests, fresh environment checks and generated real-transport gates. Production fitting source is unchanged, so retain its valid full-suite gate rather than rerunning unrelated slow tests without a code change.
- [x] Save reviewed source and aggregate evidence locally. Report exact verified scope and remaining delivery work.

## Ledger

Starting source checkpoint21de32b. Previous recommendation accepted by continuation: prioritize independent reproduction and delivery over a new experiment matrix. Estimates are not completion evidence. The Mac overlay currently inherits a shared environment; clean reproduction has not yet been demonstrated.


Verified implementation ledger: freshMac lock installs25distributions with system/user-site inheritance disabled; fresh G14/Orin environments independently pass exact Linux CPU locks. Final generated v3 qualification completed4cells/20rounds/60fits; simulator errors0.0 on Mac-only and7.450580596923828e-9/1.862645149230957e-9 on three hosts. The full bounded profile passed164tests (101existing timestamp warnings) in26.18s. No production fitting source or medical artifacts changed.

Review findings corrected with observed failing-then-passing regressions: externally loaded typing_extensions, remote-stdin working-directory/sitecustomize inheritance, and a preexisting external search root supplying lazy optimizer dependencies. Current checker verifies the complete package inventory, search roots and loaded dependency origins. Review fix inspection reported no remaining material finding within trusted-machine scope. This is an isolation/reproducibility check, not a malicious-code sandbox.

Additional final verification: export only tracked files after the source checkpoint and run generated local qualification from that export using the clean environment. Outcome pending the final verification receipt. Existing source/data/report checkpoints remain immutable; no new medical evaluation or external communication is authorized by this phase.
