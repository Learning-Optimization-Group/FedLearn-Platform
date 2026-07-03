#!/usr/bin/env bash
# check_proto_mirror.sh — fail if any in-tree proto mirror diverges from the canonical proto.
#
# The canonical gRPC contract lives at proto/fedlearn/v2/fedlearn.proto (the single source of
# truth, buf-governed). Two other units keep a byte-identical copy in-tree for their own builds:
#   1. mobile_client/proto/fedlearn/v2/fedlearn.proto        — the native mobile core's CMake build
#   2. framework/src/fedlearn/communication/protos/fedlearn.proto — the running Python framework
# Both copies are regenerated/synced, never hand-edited. This gate (wired into CI as proto.yml,
# 15-LLD-mobile.md §13 task 2) guarantees neither copy ever drifts from canonical — one stray
# edit to a mirror fails the build with a clear cp fix.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CANON="$ROOT/proto/fedlearn/v2/fedlearn.proto"

if [[ ! -f "$CANON" ]]; then
  echo "ERROR: canonical proto not found at $CANON" >&2
  exit 2
fi

# check_mirror <label> <mirror-path> — byte-compare a mirror against $CANON; exit non-zero on drift.
check_mirror() {
  local label="$1" mirror="$2"
  if [[ ! -f "$mirror" ]]; then
    echo "ERROR: $label proto mirror not found at $mirror" >&2
    echo "Fix: cp '$CANON' '$mirror'" >&2
    exit 2
  fi
  if ! diff -u "$CANON" "$mirror"; then
    echo "" >&2
    echo "ERROR: the $label proto mirror diverges from the canonical proto." >&2
    echo "The canonical proto is the single source of truth. Fix with:" >&2
    echo "  cp '$CANON' '$mirror'" >&2
    exit 1
  fi
  echo "OK: $label proto mirror matches canonical."
}

check_mirror "mobile" "$ROOT/mobile_client/proto/fedlearn/v2/fedlearn.proto"
check_mirror "framework" "$ROOT/framework/src/fedlearn/communication/protos/fedlearn.proto"

# Portable sha256 (Linux CI has sha256sum; macOS has shasum -a 256). Informational only —
# the diff checks above are the gate, so a missing hash tool must not fail the script.
if command -v sha256sum >/dev/null 2>&1; then
  echo "  sha256: $(sha256sum "$CANON" | cut -d' ' -f1)"
elif command -v shasum >/dev/null 2>&1; then
  echo "  sha256: $(shasum -a 256 "$CANON" | cut -d' ' -f1)"
fi
