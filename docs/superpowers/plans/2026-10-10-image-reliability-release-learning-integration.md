# Image reliability, release readiness, learning and integration

Goal: carry the four next steps authorized by “continue” through measured
qualification and a concrete integration preparation. Base8c91746; existing
isolated feature branch and inline execution. No push, main merge or deployment.
Preserve root edits,33 phone dataset files, app data and unrelated services.
All new g14 files stay inside its project. No release key or HTTPS endpoint
exists; prepare signing and qualify local TLS without a production claim.

## 1. Image-duration reliability

- [x] Create one isolated3round original-pneumonia FedAvg run,16 local images
  per client, with gated canonical Linux participation and the actual mounted
  phone TrainingProvider. Record accepted-update windows and exact run identity.
- [x] Verify screen-off/forced-idle state during the longer native workload;
  distinguish simulated battery/USB-assisted idle from natural wireless Doze.
  Restore battery/deviceidle state in finally, even if the probe fails.
- [x] Exercise Stop after round1 acceptance while round2 has not uploaded;
  measure latency, checkpoint preservation and same-run rejoin without a
  duplicate accepted round. Exercise a real20second socket interruption through
  an owned proxy, not ADB mapping removal alone.
- [x] Verify6unique accepted contributions, terminal status/service cleanup,
  bounded compute/battery temperature observations, finite artifact and original
  contribution/aggregate replay. Retain failed probes; fix actual defects by
  regression RED→GREEN before rerunning affected qualification.

## 2. Release readiness

- [x] Demonstrate current release validation accepts the debug key, then make
  release signing fail closed without explicit environment-supplied credentials.
  Keep debug and native unit checks usable; never print signing passwords.
- [x] Verify missing/partial/wrong/valid signing configuration through real Gradle
  tasks. Any generated key is temporary qualification material, not a production
  identity; do not replace the installed Debug app with a different signing
  identity or uninstall preserved data. Same-key Debug updates are permitted for
  native qualification fixes, with dataset hashes checked afterwards.
- [x] Exercise existing native gRPC TLS against local trusted/wrong-host/wrong-root
  fixtures where tooling permits, retaining negative controls. HTTPS deployment
  and real signed-app update remain explicit prerequisites for production.

## 3. Learning diagnosis

- [x] Inspect weak longer ResNet/NIH traces, data/labels/transforms, optimizer and
  initialization. Choose bounded fixed comparisons on validation only; official
  test/reference sets do not choose settings.
- [x] Run justified controls with shared data/seed/budgets, class-wise metrics,
  complete trajectories and checkpoint reload. Report improvements or failures
  without promising useful classification or clinical performance.

## 4. Integration preparation

- [x] Verify branch range and root user edits; prepare a reproducible integration
  candidate in an isolated checkout if needed, leaving mobSupp/main untouched.
- [x] Run appropriate checks for changes, retain curated evidence and source
  identities, update status/remaining work, and perform one fresh final review.
- [x] Keep local commits and clean g14 synchronization. Publishing/merging remains
  a separate decision after a concrete reviewable result exists.
