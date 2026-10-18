# Android research client — current run guide

Current as of 18 October 2026. Start here for a new experiment. The
[qualification matrix](13-qualification-matrix.md) retains the evidence and limits
of each older build; an older run does not qualify a newly built APK automatically.

## Supported scope

The implemented target is Android arm64, portable CPU execution, FULL training arm.
The resolver declares 24 recipe/strategy plans: TinyNet (five strategies), ECG MLP
(four), CIFAR CNN (five), CIFAR ResNet18 (four), original pneumonia CNN (four),
and NIH pneumonia-finding CNN (FedAvg and Robust). See the matrix for exact plans,
dataset sizes, original-runtime replay and historical-only evidence.

CNN DeComFL accepts one sequential batch of 1–32 local images per client. Other
real-model DeComFL recipes, alternate arms, Android XGBoost, secure aggregation,
iOS training, text/LoRA and accelerated training remain outside this scope.
Mac/Linux collaborative XGBoost is a separate, working tabular path; see its
[current result](../../research/notes/2026-10-10-breast-cancer-xgboost.md).

The qualified standalone package is an **internal Debug APK**. It launches without
Metro but retains development security settings. Keep an old phone externally
powered. Powered screen-off and bounded manual process-restart recovery have
evidence; battery-only survival, automatic resume and natural Android eviction
do not have successful qualification. Exact model reconstruction establishes
execution correctness; it does not establish useful model accuracy.

## Prepare a clean experiment

1. Record the source commit, APK SHA-256, signing-certificate fingerprint and
   toolchain versions. Use Node 24, Python 3.12, Java 21, torch 2.12.0,
   ExecuTorch 1.3.1 and gRPC C++ v1.67.1. Keep generated artifacts outside Git.
2. Check `adb devices`: the intended phone must be `device`, rather than
   `unauthorized`. Record charging, temperature and existing forward/reverse
   routes. Do not uninstall the app or clear its data to resolve an update error.
3. Prepare disjoint client datasets and reserve evaluation data before training.
   For NIH, use its patient/content-disjoint preparation and NIH-specific class
   mapping. Different sampling seeds alone do not prove disjointness.
4. Use fresh output directories. Preserve earlier receipts, datasets, exported
   programs and installed APK copies; never rerun a historical helper that
   overwrites evidence bound by a receipt.

For image packages, `scripts/make_image_package.py` supports CNN, CIFAR_RESNET18
and PNEUMONIA_CNN. It writes a ZIP and provenance sidecar. For example, from the
repository root, with the matching Python environment:

```bash
PYTHONPATH=framework/src python scripts/make_image_package.py \
  --recipe CNN --split train --count 32 --seed 7 --out /path/to/new/phone.zip
```

Prepare the peer package separately and compare source indices/content hashes
to prove separation. Use `scripts/prepare_nih.py` and
`scripts/make_nih_image_package.py --help` for NIH; the generic pneumonia source
and class mapping are not interchangeable with NIH.

## Build the internal standalone APK

Build the native dependencies with the scripts described in the
[mobile build guide](../../mobile_client/README.md#build). Generate the canonical
proto stubs, and provide a header-only torch shim so torch's bundled protobuf
headers do not shadow gRPC's headers. From `mobile_client/android`:

```bash
./gradlew :app:assembleDebug -PfedlearnStandalone=true \
  -PET_SRC="$ET_SRC" -PET_BUILD="$ET_BUILD" \
  -PTORCH_INCLUDE="$TORCH_INCLUDE" -PGRPC_DIR="$GRPC_DIR" \
  -PGENERATED_PROTO_DIR="$GENERATED_PROTO_DIR"
```

These variables must point to the actual compatible native builds and generated
stubs. The output is `app/build/outputs/apk/debug/app-debug.apk`; copy and hash it
before a later build overwrites that path. Inspect the packaged
`lib/arm64-v8a/libappmodules.so` and embedded Hermes bundle. A successful build
alone is not device qualification. Install with `adb install -r` only when the
package and signing identity match the installed app. Ordinary Debug needs Metro;
the standalone command above does not. Release is a separate qualification.

## Run through the platform

1. Start PostgreSQL and the Spring API using the repository's development setup.
   Point `FEDLEARN_PYTHON` and `MODEL_BUNDLE_PYTHON` at the matching Python
   environment. Set `APP_MODEL_BUNDLE_DIR` to a writable experiment directory.
   The real export entry point is repository-root `scripts/export_model.py`;
   model construction imports `fl-runtime/recipes.py`.
2. Choose a reachable API origin in the phone's **Settings → Backend server URL**.
   Confirm the reachability check, then log in normally. For development USB
   routes, the API and the advertised gRPC endpoint must both map through the
   owned tunnels. The [older topology guide](10-live-run-runbook.md#topology)
   explains the forwarding pattern; its Metro routes are unnecessary for
   standalone Debug. On a relay host, keep all project files inside its checkout.
3. As project owner, create the matching recipe/optimizer with FULL arm. Start
   the selected strategy with two clients, a bounded round count and
   `dataSource=LOCAL_SNAPSHOT`; leave secure aggregation off. Wait for the run's
   execution contract to become `READY`. A staging failure or unsupported plan
   is a refusal to fix, not a reason to bypass the contract.
4. Grant the phone user client membership. On the phone, select the project and
   join its active run, import/select the matching dataset, then press **Start
   training**. The client verifies programs, checks dataset compatibility and
   probes the native runtime before training.
5. Enroll the Linux/desktop participant separately. Launch the canonical
   `fl-runtime/client.py` with its own connection token, partition ID, run ID,
   exact execution contract and local `--dataset-path`. Follow the enrollment's
   gRPC endpoint and TLS settings; never reuse another participant's enrollment.
   Keep tokens and enrollment responses private.
6. Observe both participants contributing. Finish only when the platform reports
   `COMPLETED` and the final artifact is registered. Retain the contract,
   datasets/provenance, accepted contributions, model hashes and held-out metrics.
   Reconstruct/replay contributions before claiming numerical correctness.

## Stop, recover and clean up

Stop is cooperative. A measured image case took about 107 seconds while native
work settled. Wait for stopping to finish; another Start/Join must remain blocked
during that interval. Sign-out also drains the old participation.

After actual process death, recovery is manual: launch the app, rejoin the same
active run, select the saved dataset and press Start. CNN FedAvg and bounded CNN
DeComFL have process-death recovery evidence, including lost acknowledgements and
controlled computation exits. Do not infer automatic resume or every recipe's
recovery from those cases. DeComFL restores a validated current server snapshot;
new durable trajectory storage is not required for the tested manual path.

Stop owned unfinished projects and clients. Remove only routes/helpers created by
this experiment, restore the original account/API setting, disable temporary
wireless debugging, and recheck APK/dataset hashes. Leave the phone asleep on
power with no orphan training service. Keep historical evidence immutable.

## Before distributing a Release APK

Use the [release/TLS preparation guide](16-release-tls-qualification.md). Required
evidence is a real signed Release APK using HTTPS login/enrollment, authenticated
gRPC TLS training, a registered verified final model, negative transport/auth
controls and a same-release-key update preserving data. Local status fixtures and
standalone Debug do not close those requirements. See
[release readiness](14-release-readiness.md) for retained measurements.
