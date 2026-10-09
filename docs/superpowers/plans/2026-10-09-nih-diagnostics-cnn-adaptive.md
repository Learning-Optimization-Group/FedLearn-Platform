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

- [ ] Driver regression tests RED, implementation, focused suite; local commit.
- [ ] Sequential CNN/FedProx and CNN/FedOpt mixed qualification and exact replays.
- [ ] NIH source-bound packages,10-epoch controls, checkpoint reload, reference.
- [ ] Curated reports/guides, final fresh whole-branch review, fixes/tests as
  appropriate, local commits, g14 fast-forward and root status context.

Ruling: use enriched patient-unique validation because six proportional positives
cannot robustly distinguish configurations. Cost: balanced-sample precision and
accuracy cannot represent the source population; report that limitation and
separate proportional reference metrics explicitly.
Ruling: keep default FedOpt server settings in the mixed CNN execution proof.
Cost: loss growth may prevent useful learning; preserve failures rather than
claim convergence or change the contract silently.
