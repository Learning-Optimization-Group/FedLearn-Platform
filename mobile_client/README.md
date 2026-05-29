# FedLearn Mobile Client (v2)

Native React Native + C++ (libtorch) federated-learning client. Built fresh per
[`docs/v2/build/15-LLD-mobile.md`](../docs/v2/build/15-LLD-mobile.md); see also the
build order in [`docs/v2/build/90-BUILD-SEQUENCE.md`](../docs/v2/build/90-BUILD-SEQUENCE.md).

> **RN** = React Native · **FL** = Federated Learning · **RNG** = Random Number Generator ·
> **ZO** = Zeroth-Order optimization · **gRPC** = Google Remote Procedure Call ·
> **JNI** = Java Native Interface · **NDK** = Native Development Kit · **CI** = Continuous Integration.

---

## Status — what is built so far

This is **increment 1: the cross-language determinism contract + C++ core foundation**
(15-LLD §13 tasks 2–5, plus the framework prerequisite). It is the load-bearing
foundation everything else depends on: the DeComFL protocol only works if the Python
server and this C++ client regenerate **identical** perturbation vectors from the same
seed.

| Built | File |
|---|---|
| Python source of truth | `../framework/src/fedlearn/estimators/perturbation.py` (`canonical_perturbation`) |
| Frozen golden vectors | `../framework/tests/fixtures/decomfl_golden/` (`*.f32` + `manifest.json`, torch 2.12.0) |
| Python parity test | `../framework/tests/test_perturbation.py` |
| C++ canonical RNG | `shared/src/Perturbation.cpp` + `include/fedlearn/Perturbation.h` |
| C++ dtype whitelist | `shared/src/DtypeMap.cpp` + `include/fedlearn/DtypeMap.h` |
| C++ parity gate (gtest) | `shared/tests/rng_parity_test.cpp` (**release blocker**, 15-LLD §13.4) |
| C++ dtype test (gtest) | `shared/tests/dtype_map_test.cpp` |
| Host build | `CMakeLists.txt`, `shared/CMakeLists.txt`, `shared/tests/CMakeLists.txt` |

**Not yet built** (subsequent increments, 15-LLD §13 tasks 6–19): `ModelManager`,
`ZerothOrderEstimator`, `DeComFLClient`, `FederatedLoop`, `FedLearnClient` (gRPC), the
TurboModule bridge, the Android/iOS app projects, the RN TypeScript app layer and
screens, the foreground-service lifecycle, telemetry wiring, the native-dep prebuild
scripts, and `mobile.yml` CI.

---

## The determinism contract (read this first)

`canonical_perturbation(seed, n)` generates `z ~ N(0, I_d)` **on the CPU** with a *local*
generator, then callers move it to their device. This is the only way to get the same `z`
on a CPU server and an Apple/Android device. The Python implementation is the **source of
truth**; the golden vectors in `framework/tests/fixtures/decomfl_golden/` are frozen from
it; this C++ core must reproduce them.

- **Re-freeze the fixture only on an intentional torch bump:**
  `cd framework && PYTHONPATH=src python tests/fixtures/decomfl_golden/generate.py`
  (then re-run the C++ parity test and review any change — never hand-edit the vectors).
- If `rng_parity_test` fails, the mobile build **must not ship** — a divergent `z`
  silently corrupts DeComFL aggregation between the server and this client.

---

## Build & run the C++ core tests in isolation (the parity gate)

Requires a host libtorch matching the pinned torch version (2.12.0). CMake ≥ 3.22, a
C++17 compiler, and network access for the gtest fetch.

```bash
# 1. Get libtorch (CPU build is fine for the host parity gate). Example:
#    download + unzip libtorch 2.12.0 from https://pytorch.org/get-started/locally/
export LIBTORCH_DIR=/path/to/libtorch

# 2. Configure + build
cmake -S mobile_client -B mobile_client/build -DLIBTORCH_DIR="$LIBTORCH_DIR"
cmake --build mobile_client/build -j

# 3. Run the parity + dtype tests
ctest --test-dir mobile_client/build --output-on-failure
```

Python side (same contract, bit-exact):

```bash
cd framework && PYTHONPATH=src pytest tests/test_perturbation.py -v
```

---

## Toolchain for the full app (later increments, 15-LLD §11.1)

Node 24 LTS, JDK 21, the Android SDK + NDK (for `arm64-v8a`), Xcode (iOS), CMake ≥ 3.22,
and the CI-prebuilt libtorch/gRPC ARM64 artifacts (`scripts/build_libtorch_arm64.sh`,
`scripts/build_grpc_arm64.sh` — not yet written). The Android app targets `arm64-v8a`
only; the C++ core is consumed through the JNI TurboModule bridge.

---

## Conventions

- One product name: **FedLearn** (the v1 "FedMob"/`com.mobileclientnew` names are retired).
- One canonical proto (`buf`-generated); no drifted copies (the v1 third proto is deleted).
- No committed dataset blobs; demo data is fetched (`scripts/fetch_demo_data.sh`, later).
- The C++ `SAFE_DTYPES` whitelist stays in lockstep with the Python serializer whitelist.
