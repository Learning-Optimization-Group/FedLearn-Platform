# NIH diagnostics and adaptive CNN qualification

User continuation follows the completed four-slice plan at 1a966db. Keep the
existing isolated feature checkout, preserve imported datasets and user edits,
keep new g14 files inside its requested codebase, and do not push/merge.

Bounded design: use existing CNN FULL contracts/production phone functions for
sequential FedProx then FedOpt runs, 320 disjoint CIFAR training images/client,
two rounds and ten32-image batches/round. Record current sources, contract,
parent/program hashes, all accepted updates and telemetry. Native/g14 replay
must match float32 bits; wrong labels/rates/zero proximal coefficient must
change phone updates. Aggregate replay must reproduce registered artifacts.
Use the existing same-signed APK; verify all27 imported file hashes unchanged.

NIH has139 positive validation patients and279 positive patients/client.
Prepare256 balanced unique-patient training images/client (512 pooled) and256
balanced unique-patient validation images (128 positives) from existing verified
manifests. Validate source hashes/patient/content disjointness. The official test
set stays untouched. Balanced validation estimates discrimination in this
sample; its precision/accuracy are not prevalence estimates. Retain the existing
proportional512-image validation sample as a separate reference evaluation.

Extend the existing research control driver with bounded configurable epochs/
rates while preserving its original three-control defaults. Pin defaults and
invalid/duplicate/empty settings with regressions before edits. Run10 epochs at
1e-4 and1e-5 from the same retained initial state/seed, persistent Adam, batch16;
record every epoch and reload final checkpoints. Do not tune against the
reference or official test set. This is a research configuration, not a change
to published NIH platform contracts.

- [x] Driver regression tests RED, implementation, focused suite; local commit.
- [x] Sequential CNN/FedProx and CNN/FedOpt mixed qualification and exact replays.
- [x] NIH source-bound packages,10-epoch controls, checkpoint reload, reference.
- [x] Curated reports/guides, final fresh whole-branch review, fixes/tests as
  appropriate, local commits, g14 fast-forward and root status context.

Ruling: use enriched patient-unique validation because six proportional positives
cannot robustly distinguish configurations. Cost: balanced-sample precision and
accuracy cannot represent the source population; report that limitation and
separate proportional reference metrics explicitly.
Ruling: keep default FedOpt server settings in the mixed CNN execution proof.
Cost: loss growth may prevent useful learning; preserve failures rather than
claim convergence or change the contract silently.


Driver:9 new regressions failed first and then passed; focused15 passed,
full runtime611 passed with30 slow deselected. Commit322c7e5 synchronized to
clean g14. Defaults preserve the original three controls.

CNN: all8 contributions and both aggregates replay bitwise. Original native
changed-label/rate/zero-mu controls differ. Registry reload on10,000 official
CIFAR test images gives1509 correct (FedProx) and1000 (FedOpt). Phone compute
about2.7 seconds/round; uploads248805 bytes. All27 dataset files preserved.

Ruling: reproduce the actual one-thread desktop execution environment, not the
shell default. Default-thread replay differed by1.32e-6; pinned replay is bitwise
exact. Cost: bitwise research replay must retain numerical thread configuration.


Ruling: a bounded timing probe measured mean NIH batch1.604 seconds with1 CPU
thread and0.578 with4 (2.77x). Preserve the incomplete one-thread pilot and
restart both fixed10-epoch configurations with4 threads, recorded in results.
Cost: comparisons/replays must use their own retained thread configuration;
old/new experiment numerical equivalence is not claimed. No configuration was
selected from reference/test performance. Driver defaults still pin1 thread.


NIH completed: both fixed10-epoch rates1e-4/1e-5, persistent Adam,512 training/
256 balanced validation rows,4 threads. Final balanced AUROC.640991/.628601;
training AUROC.831314/.710876. Checkpoint/hash/state and reloaded balanced
metrics match exactly. Initial balanced AUROC.460571. Separate proportional
reference AUROC.410738/.388340, only6 positives; poor performance. Held-out
samples share169 patients/134 images; neither overlaps training. All20 epochs,
source/package/checkpoint receipts and incomplete pilot retained. Full runtime
616 passed (30 slow deselected); focused20 passed. Official test untouched.


Final review: no blocking correctness findings. Public CNN updates/replay/cards
and NIH all20 sequential epochs/common seed/initial/hash/exactreload/reference
match retained receipts. Focused20 tests independently passed. Final reports
strip transient pilot PID, identify timing as synthetic16-image batches with
one warmup/two measurements and concurrent pilot, and use observational wording
for collapse avoidance rather than attribute causality to changed factors.
No code changes needed after final review; no deferred minor findings.

Final: Ruling: reviewer did not rerun physical hardware, full raw NIH inventory
or model execution on g14; it checked code and retained receipts. These live
checks were performed during execution. Clinical utility and general efficacy
are outside proof. Cost: further independent/clinical experiments remain needed.
