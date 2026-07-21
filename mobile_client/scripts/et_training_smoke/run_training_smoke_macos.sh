#!/usr/bin/env bash
# run_training_smoke_macos.sh — end-to-end on-device forward+backward de-risk on a macOS host.
#
# Proves ExecuTorch can do REAL on-device backprop (it is otherwise inference-only), which is the
# precondition for "mobile supports all algorithms" — a native first-order FedAvg/FedProx round
# instead of the current zeroth-order-only (DeComFL) path that MO-4 fail-closed-restricts us to.
#
# Steps:
#   1. Require a host ET build WITH the training extension (build_executorch_host_macos.sh, which now
#      passes -DEXECUTORCH_BUILD_EXTENSION_TRAINING=ON).
#   2. Provision an ISOLATED venv and `pip install executorch` (it pulls its own pinned torch — kept
#      out of the framework's torch to avoid disturbing it), then export a TRAINABLE .pte via
#      export_xor_trainable.py (torch.export + _export_forward_backward captures the backward graph).
#   3. Compile train_smoke.cpp against the host ET training libs and run it: it trains XOR with
#      TrainingModule.execute_forward_backward + optimizer/sgd and asserts the loss collapses.
#
# Exit 0 == on-device backprop works. Measured on Apple M4 Max (ET 1.3.1): XOR loss 0.406 -> 0.002.
#
# The two macOS link details this needs (Linux CI does not):
#   * -I <ET>/runtime/core/portable_type/c10  — resolves ET's vendored <c10/util/irange.h>.
#   * -Wl,-force_load <portable_ops_lib.a>     — ET ops self-register at static init; without the
#     force-load the linker drops the archive and execute_forward_backward finds no operators.
set -euo pipefail

ET_VERSION="${ET_VERSION:-1.3.1}"
ET_SRC="${ET_SRC:-$HOME/executorch-host/executorch}"
CO="$ET_SRC/cmake-out"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="${WORK:-$HOME/executorch-host}"
VENV="$WORK/etpy-venv"
PTE="$WORK/xor_trainable.pte"
BIN="$WORK/train_smoke"

# 1. host ET build with the training extension present.
if [[ ! -f "$CO/extension/training/libextension_training.a" ]]; then
  echo "ERROR: $CO/extension/training/libextension_training.a not found." >&2
  echo "       Run mobile_client/scripts/build_executorch_host_macos.sh first (it enables" >&2
  echo "       EXECUTORCH_BUILD_EXTENSION_TRAINING=ON). If you built ET before that flag was added," >&2
  echo "       delete $CO and rebuild." >&2
  exit 1
fi

# 2. isolated venv -> export a trainable .pte.
if [[ ! -f "$PTE" ]]; then
  echo "[1/3] provision isolated venv + install executorch==$ET_VERSION (pulls its own torch)"
  python3 -m venv "$VENV"
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
  python -m pip install --upgrade pip -q
  pip install -q "executorch==${ET_VERSION}"
  echo "[2/3] export XOR trainable .pte (captures the backward graph)"
  python "$HERE/export_xor_trainable.py" "$PTE"
  deactivate
else
  echo "[1-2/3] trainable .pte cache hit: $PTE"
fi

# 3. compile + run the on-device trainer.
echo "[3/3] compile train_smoke.cpp against host ET training libs + run"
clang++ -std=c++17 -O1 -Wno-deprecated-declarations \
  -I "$(dirname "$ET_SRC")" \
  -I "$ET_SRC/runtime/core/portable_type/c10" \
  -DC10_USING_CUSTOM_GENERATED_MACROS \
  "$HERE/train_smoke.cpp" -o "$BIN" \
  "$CO/extension/training/libextension_training.a" \
  "$CO/extension/tensor/libextension_tensor.a" \
  "$CO/extension/data_loader/libextension_data_loader.a" \
  "$CO/extension/flat_tensor/libextension_flat_tensor.a" \
  "$CO/extension/named_data_map/libextension_named_data_map.a" \
  "$CO/extension/module/libextension_module.a" \
  -Wl,-force_load,"$CO/kernels/portable/libportable_ops_lib.a" \
  "$CO/kernels/portable/libportable_kernels.a" \
  "$CO/libexecutorch.a" \
  "$CO/libexecutorch_core.a" \
  "$CO/third-party/flatcc_ep/lib/libflatccrt.a"

"$BIN" "$PTE"
echo "OK: on-device forward+backward smoke passed."
