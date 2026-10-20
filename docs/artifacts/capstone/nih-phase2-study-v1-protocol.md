# NIH phase-2 matched study v1

20 October 2026. **Prospectively frozen:** the authoritative version is the first
commit containing this document. An independent implementability check found no
important unanswered questions. Commit and verify the source-bound runner before
executing the fixed cells; no main-study run or official-test classifier scoring
has occurred. Earlier pilots and qualification outcomes remain exploratory,
including the twelve-cell validation study.

## Question, primary endpoint and contrast

Does central DP change the incremental attack resistance of clipped coordinate-wise
trimmed mean relative to clipped mean on the qualified NIH No Finding/any-finding
task? The practical outputs are a privacy/utility frontier, matched clean and
attacked comparisons, and bounded head/cohort/clipping cost evidence.

The primary endpoint is **round-20 official-test image-level AUROC**, scored by
float64 class-1 minus class-0 logit margins (`margin_v2`), at official prevalence.
For each seed s, A(r,p,a) denotes this endpoint under rule r, privacy p and attack a.
Define the attack-adjusted trimmed-mean advantage:

```
B(p) = [A(trimmed, p, sign_flip) - A(mean, p, sign_flip)]
       - [A(trimmed, p, clean) - A(mean, p, clean)]
D(s) = B(epsilon_4) - B(no_DP)
```

The **single primary contrast** is the mean of D(s) over the ten fixed new seeds.
The question is two-sided. Positive D means DP increases this attack-adjusted
trimmed-mean advantage. It does not mean either classifier improves: report all
eight absolute components and their paired differences. The exploratory validation
example already shows why a positive interaction can coexist with lower absolute
private utility. No directional success threshold or best-round selection.

## Dataset, representation and tuning history

Use only the qualified frozen ImageNet ResNet-18, 512-feature cache:

- Feature-report SHA-256:
  `6c9e6969c6fd762156ef2246323fbdbc5e9cb272c9910a8b1c9b691ab33783fd`.
- Cohort-report SHA-256:
  `287dca8359300f08cea79098d367793a73e43205bcfe3213e9d22115e3bb7267`.
- Train/validation/test: 68,650/17,763/25,596 images and
  22,399/5,600/2,797 patients respectively.
- Label 0 is exactly `No Finding`; recognized finding combinations are label 1.
  These are report-derived labels, not verified clinical truth. No balancing.
- Development quarantine excludes 111 images from nine non-test patients.
  Official-test membership remains unchanged. No near-duplicate certificate.

Earlier validation pilots selected the linear E7/R20, LR0.003 recipe after weak
initial calibration/optimization and a versioned AUROC correction. Three declared
seeds passed the no-DP learning gate; these are exploratory, not independent test
replications. A four-head short cost probe and a single-seed twelve-cell full-
schedule mechanism qualification followed. C0.05 was chosen before the latter
outcomes; it is informed by observed clean update norms, not an optimized clipping
quantile. A private draft of this contrast/cell list was hashed before that
qualification's numerical table was examined, but this prospective commit is the
binding protocol. No official-test classifier performance has been examined.

## Fixed common recipe

The primary recipe is linear head (1,026 parameters), all 80 clients accepted in
each of 20 rounds, whole-patient IID allocation, seven local epochs, SGD LR0.003,
batch256, clipping C0.05, no proximal term or server optimizer, CPU four threads.
Each client's complete patient groups remain local throughout its run. Preserve
natural training/evaluation prevalence and bind the actual assignment digest.

Rules are uniform mean and coordinate-wise trimmed mean with trim ratio0.2,
removing floor(0.2*n) values per coordinate from each tail. Privacy modes are
matched clipped no DP and central DP epsilon1/4/8, delta0.00001, replace-one
simulated-client adjacency, full cohort q=1 and no amplification credit. Calibrate
the Gaussian multiplier for all twenty rounds with the declared existing Rényi
order grid/classic conversion. Rule-specific sensitivity changes absolute noise;
identical epsilon does not imply identical noise SD.

Conditions, in order:

1. Clean: no malicious slots.
2. Sign flip: sixteen fixed attacker slots (fraction0.2), each multiplying its own
   update by -4 before common clipping.
3. Label flip: sixteen fixed slots reversing binary labels on their own data.

Use the same attacker identities across conditions where applicable, fixed across
rounds. Neither attack reads another client's raw current update. IPM/ALIE, local
DP, non-IID allocation, dropout, stragglers, deadlines and early stopping are
excluded from this study; historical implementations/records remain unchanged.

Ten new base seeds, in order: **20261101 through 20261110 inclusive**. For base s,
partition/init/client seeds equal s, attack=s+1000, noise=s+2000. Match all streams
and settings except declared treatment factors. This defines their joint scheme;
it does not assert independence merely from stream names. Shared Gaussian standard
draws couple matched private cells while epsilon/rule change their amplitude;
trajectories diverge after treatment. Earlier pilot/qualification seeds never
enter the confirmatory estimate.

## Exact bounded cell families: 360 total

Execute these families in the order listed. Within each, iterate seeds ascending,
then the remaining axes in their stated order. Generate and retain all resolved
configurations/hashes and a 360-entry pending index before the first child.
No unrestricted matrix expansion or outcome-dependent family/cell selection.

| Family | Fixed axes and order within each seed | Seeds | Cells |
| --- | --- | --- | ---: |
| Primary | linear, n80, C0.05; conditions clean/sign_flip/label_flip; rules mean/trimmed_mean; privacy none/epsilon1/epsilon4/epsilon8 | All ten |240|
| Head | heads MLP8/MLP128/MLP512; n80, C0.05; conditions clean/sign_flip; rules mean/trimmed_mean; privacy none/epsilon4 | First three |72|
| Cohort | linear; n10/n1000, C0.05; clean; rules mean/trimmed_mean; privacy none/epsilon4 | First three |24|
| Clipping | linear, n80, C0.4; conditions clean/sign_flip; rules mean/trimmed_mean; privacy none/epsilon4 | First three |24|

MLP8/128/512 contain 4,122/65,922/263,682 trainable parameters on the same frozen
features. The linear, n80 and C0.05 controls are reused exactly from the primary
family, not rerun or selected. No head-specific tuning or head selection. Compare
each head against its matched no-DP control; weak baseline training prevents a
claim of useful private learning. Width changes capacity/optimization as well as
dimension. Cohort size changes data allocation, minibatch trajectories and local
step counts as well as sensitivity; do not attribute all effects to the noise
denominator. No full-backbone/LoRA or representation change.

## Final artifacts and secondary operating points

The runner must retain a source/config/cache-bound final model identity and aligned
final validation/test margins, labels, patient/image identities, with a verifiable
checkpoint or equivalent final-state binding. This is required before launch;
aggregate diagnostic JSON alone cannot support independent AUROC, threshold or
patient analyses. Independently recompute the primary AUROC from retained scores.

Report final default-argmax confusion counts, balanced accuracy, finding recall,
specificity, accuracy, loss and ten-bin ECE alongside AUROC. Binary argmax ties
predict class0 (positive iff margin>0). Retain full curves, but use only round20
for endpoints; never select rounds/seeds/thresholds from test performance.

For every final model, choose the **highest validation margin threshold attaining
at least 80% observed finding recall**. Sort positive validation margins descending;
select rank ceil(0.8*N_positive), one-based. Predict positive for score>=threshold,
including ties. Apply this same threshold to official test without recalibration.
Report realized validation/test recall, specificity and confusion counts. The 80%
target is a research operating point, not clinical acceptance.

Preserve the full official test as primary. Two descriptive sensitivity analyses
use the same final scores without retraining:

1. Give each image weight inverse to its patient's image count within that split;
   report patient-weighted AUROC and confusion-based metrics. This examines uneven
   image contribution, not a patient diagnostic classifier or population interval.
2. Remove only from secondary test metrics the nine images in the bound ambiguous-
   content mask. The mask considers all six ambiguous original-audit RGB groups
   intersecting test, including single test images whose other patient's matching
   image was quarantined. Three groups remain multi-member ambiguous groups in the
   curated cohort; these are different scopes. Test training/membership is unchanged.

The private mask SHA-256 is
`352b10c16342e4160d06a1e536ea82d201ea739eec17ed67fc9801141e60470d`;
original audit report SHA-256
`7851361e9fa6f72d24745b781f10d5453800a16bcb3148abed775a3cd2bb2b8d`;
original per-image audit SHA-256
`bb267d0a96d8b19b4da47113c8b76ec05d51e1f39b7a6753f475c2a555c48829`;
test manifest SHA-256
`6d7c1eceda4a175c771acf0bbf480a25aaf7eaa2fbdbcea6744397cb889ef1fe`.
The nine exclusions contain seven patients and three positive images, leaving
25,587 images for the secondary calculation. Bind and validate exact IDs before
launch; do not select exclusions from prediction errors. Raw identities stay in
private reproducibility evidence.

## Inference, missingness and multiplicity

Compute D within each complete matched seed first. Report all ten effects,
mean, sample standard deviation, range and approximate two-sided95% Student-t
interval: mean ± t(0.975,9)*SD/sqrt(10). The
[NIST paired-interval formula](https://www.itl.nist.gov/div898/handbook/prc/section3/prc312.htm)
provides arithmetic, not study-specific validity. Interpretation is conditional
Monte Carlo seed variation on fixed cohorts, not repeated clinical sampling.
Independence/approximately normal seed-effect assumptions and small-n limitations
must be stated. No posthoc normality test switches methods. An interval containing
zero is not evidence of equivalence. If SD=0, report zero observed seed variation
without a universal certainty claim.

Only this contrast receives primary95% interval interpretation. All other effects,
curves and head/cohort/clipping/threshold/patient comparisons are descriptive
secondary results, with seed values/mean/range and explicit counts. No unadjusted
multiple-comparison winner or familywise significance claim. Show absolute utility,
matched controls and noise/clipping/cost diagnostics beside interaction effects.

All eighty cells defining the eight primary-contrast components across ten seeds
must qualify for its planned analysis. If any component is missing/failed, mark
the primary contrast incomplete; show available pairs descriptively with missing
identities, not a selected-survivor formal interval. Retain every other planned
failure and poor finite/chance result. No imputation, retries, replacement seeds,
adaptive cells or best-round selection after outcomes.

## Resource, execution and release contract

One sequential isolated CPU child per cell; per-child180-second timeout;
combined child cap10,800seconds (three hours). Parent checks/control are separate.
Engine-completion observed child process RSS must be at most2GiB: a post-run
observation gate, not an OS-enforced/process-tree cap. Bound final-prediction/
checkpoint work belongs inside child elapsed time; report it separately from
federation-loop and full core-call timings. No concurrent regressions during
measurement. Report ordinary background-load/fixed-order limits, not replicated
hardware performance claims.

Preserve every planned index/config entry when budget stops execution. Ordinary
cell failure continues the fixed list; source/protocol/cache-integrity failure
stops remaining launches. Persist atomic start proof before training (survives
interruption), completion/error proof, raw failed/nonfinite returned records,
traceback logs and artifact hashes. Refuse existing outputs/overwrites. Verify
source/runtime/config/cache/model/score identity, complete requested cohorts and
all20 rounds, actual attacker IDs, calibrated sensitivity/noise and prefix epsilon
before interpreting numerical results. No scoring until the runner and its
regressions, full suite, provenance and artifact checks pass and are committed.

The server is trusted with raw updates. The conditional mechanism protects one
honest simulated client's replacement, under fixed public settings/history,
accepted set/cardinality and permitted attacker information. Patient disjointness
does not establish patient-level DP. Twenty-round calibration includes all model
releases in that run. Epsilon is **per run**: reusing private data across the360
runs or tuning would require additional composition; this public-data benchmark
does not claim an overall epsilon4 research workflow.

Published randomness, pre-noise diagnostics and identities remain private research
records, not DP-safe deployment exports. A deployment needs separately secret
randomness and a reviewed release boundary. Preserve all945 historical Milestone2
records and every NIH pilot/version without relabelling them confirmatory. No
clinical efficacy, new DP theorem, protected whole research log or physical-device
scaling claim is authorized by this protocol.
