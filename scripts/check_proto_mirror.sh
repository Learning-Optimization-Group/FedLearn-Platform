#!/usr/bin/env bash
# check_proto_mirror.sh — fail if any in-tree proto mirror diverges from its canonical proto.
#
# The canonical gRPC contract for the gradient path lives at proto/fedlearn/v2/fedlearn.proto (the
# single source of truth, buf-governed). Two other units keep a byte-identical copy in-tree for
# their own builds:
#   1. mobile_client/proto/fedlearn/v2/fedlearn.proto        — the native mobile core's CMake build
#   2. framework/src/fedlearn/communication/protos/fedlearn.proto — the running Python framework
# The FoT (Federation over Text) contract has its own canonical at proto/fedlearn/fot/v1/fot.proto,
# byte-mirrored into framework/src/fedlearn/communication/protos/fot.proto for the same reason.
# All copies are regenerated/synced, never hand-edited. This gate (wired into CI as proto.yml,
# 15-LLD-mobile.md §13 task 2) guarantees no copy ever drifts from its canonical — one stray edit
# to a mirror fails the build with a clear cp fix.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CANON="$ROOT/proto/fedlearn/v2/fedlearn.proto"

if [[ ! -f "$CANON" ]]; then
  echo "ERROR: canonical proto not found at $CANON" >&2
  exit 2
fi

# check_mirror <label> <canonical> <mirror> — byte-compare a mirror against its canonical; exit
# non-zero on drift.
check_mirror() {
  local label="$1" canon="$2" mirror="$3"
  if [[ ! -f "$canon" ]]; then
    echo "ERROR: $label canonical proto not found at $canon" >&2
    exit 2
  fi
  if [[ ! -f "$mirror" ]]; then
    echo "ERROR: $label proto mirror not found at $mirror" >&2
    echo "Fix: cp '$canon' '$mirror'" >&2
    exit 2
  fi
  if ! diff -u "$canon" "$mirror"; then
    echo "" >&2
    echo "ERROR: the $label proto mirror diverges from the canonical proto." >&2
    echo "The canonical proto is the single source of truth. Fix with:" >&2
    echo "  cp '$canon' '$mirror'" >&2
    exit 1
  fi
  echo "OK: $label proto mirror matches canonical."
}

check_mirror "mobile" "$CANON" "$ROOT/mobile_client/proto/fedlearn/v2/fedlearn.proto"
check_mirror "framework" "$CANON" "$ROOT/framework/src/fedlearn/communication/protos/fedlearn.proto"
check_mirror "framework-fot" "$ROOT/proto/fedlearn/fot/v1/fot.proto" "$ROOT/framework/src/fedlearn/communication/protos/fot.proto"

# Portable sha256 (Linux CI has sha256sum; macOS has shasum -a 256). Informational only —
# the diff checks above are the gate, so a missing hash tool must not fail the script.
if command -v sha256sum >/dev/null 2>&1; then
  echo "  sha256 (fedlearn.v2): $(sha256sum "$CANON" | cut -d' ' -f1)"
  echo "  sha256 (fot.v1):      $(sha256sum "$ROOT/proto/fedlearn/fot/v1/fot.proto" | cut -d' ' -f1)"
elif command -v shasum >/dev/null 2>&1; then
  echo "  sha256 (fedlearn.v2): $(shasum -a 256 "$CANON" | cut -d' ' -f1)"
  echo "  sha256 (fot.v1):      $(shasum -a 256 "$ROOT/proto/fedlearn/fot/v1/fot.proto" | cut -d' ' -f1)"
fi
