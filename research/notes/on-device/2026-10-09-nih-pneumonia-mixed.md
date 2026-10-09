# NIH pneumonia finding: Android/Linux qualification — 9 October 2026

The platform's separate `NIH_PNEUMONIA_CNN` recipe completed two FedAvg rounds
on vivo Android and g14 Linux CPU, using Adam, FULL training and local snapshots.
Run `adf7860c-9566-4777-9fc3-fdacc851ca83` registered artifact
`6c31bec3-c0d8-4a49-80cd-023a803a9ec2`, SHA-256
`8b49463eaa4d87ae9f9d742b427de59c97ed070ab4e69d07ee1f9e61a8742f45`.
Training/export source was `0b279a7`; the installed native APK remained the
verified `bdb8a9e` build, with current JavaScript delivered through the development
session. Its installed SHA-256 matches the retained build.

Both participants trained 32 explicitly recorded patient-unique images,
16 pneumonia-present and 16 pneumonia-absent, from distinct client manifests.
Their patients and image contents are disjoint. Labels record the presence of
`Pneumonia` in report-derived findings; absence does not imply clinically normal.
The full training/validation/test manifests remain intact. The selected record
lists and tensor hashes identify the qualification samples exactly; these small
balanced samples do not represent full NIH training or natural prevalence.

Four contributions were accepted. Each original g14 replay matches every
float32 bit, and their reconstructed aggregates match every float32 bit of the
registered checkpoint. Both original vivo replays also match every float32 bit. Changed-label controls produce
different contributions. Cross-runtime numerical identity is not claimed.

The phone qualified the 25,783,554-parameter model with a 137,189 ms probe.
Its two training rounds took 276,219 and 277,243 ms. Each uplink was 103,135,097
bytes; downlinks were 103,135,096 and 103,135,134 bytes. All 21 previous dataset
files retain their hashes; the NIH import added exactly three snapshot files.

Round evaluation used a fixed proportional 512-image validation sample,
including six pneumonia positives, from 17,565 validation images. No official
test image was used. Independent registry reload reproduces every reported
binary metric and the exact evaluation provenance; validation images have no
patient or content overlap with the client samples.

Final model quality is poor: it predicts pneumonia-present for all 512 images.
Accuracy is 1.171875%, balanced accuracy 50%, AUROC 0.4384057971014493 and
average precision 0.011809725900320495. The majority baseline is 98.828125%.
The first round's 98.83% accuracy matched that baseline. Persisted benchmarks
have no generic accuracy target, and registered-model inference returns the
explicit NIH labels correctly. These numbers demonstrate execution and expose
model failure; they support no diagnostic-performance claim.

An initial staging attempt was stopped before any participant joined: the
export helper requested NIH's unsupported FIXTURE plan, fell back to maxBatch8,
and publication correctly refused it against batch16. The corrected export
uses LOCAL_SNAPSHOT and published a batch16 contract. Independent review also
found and verified fixes for case aliases bypassing project guards, a misleading
generic accuracy target, and feasible patient-unique samples rejected by greedy
assignment.

Final checks: 569 runtime tests, 1,345 framework tests with 78.52% coverage,
1,159 backend tests with required coverage verification, 647 mobile tests and
137 frontend tests. Mobile/frontend lint and mobile typecheck pass. The frontend
build used an explicit validation origin. All five proto mirrors match;
regenerating pinned Python/TypeScript readers produces no diff. The NIH focused
suite passes all 32 tests on g14.

This qualification used the installed debug APK and production import/join/
training functions through its debugger. It does not qualify the full UI flow,
release signing, deployed TLS, packaged desktop or iOS. Transport was
authenticated development plaintext. Long runs, thermal/background behavior,
other image strategies and training arms remain separate work. Full-scale NIH
training needs a fixed experimental protocol and useful comparison baselines;
the official test patients remain reserved for that evaluation.

The NIH catalog now clears Android participation following these device checks;
the runtime still requires each device to pass its own program probe.
