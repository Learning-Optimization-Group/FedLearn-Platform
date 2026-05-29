#!/usr/bin/env bash
# check_proto_mirror.sh — fail if the mobile proto mirror diverges from the canonical proto.
#
# The canonical gRPC contract lives at proto/fedlearn/v2/fedlearn.proto (the single source of
# truth, buf-governed). The mobile core needs an in-tree copy for its CMake build; this gate
# (wired into CI as proto.yml, 15-LLD-mobile.md §13 task 2) guarantees the copy never drifts.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CANON="$ROOT/proto/fedlearn/v2/fedlearn.proto"
MIRROR="$ROOT/mobile_client/proto/fedlearn/v2/fedlearn.proto"

if [[ ! -f "$CANON" ]]; then
  echo "ERROR: canonical proto not found at $CANON" >&2
  exit 2
fi
if [[ ! -f "$MIRROR" ]]; then
  echo "ERROR: mobile proto mirror not found at $MIRROR" >&2
  echo "Fix: cp '$CANON' '$MIRROR'" >&2
  exit 2
fi

if ! diff -u "$CANON" "$MIRROR"; then
  echo "" >&2
  echo "ERROR: the mobile proto mirror diverges from the canonical proto." >&2
  echo "The canonical proto is the single source of truth. Fix with:" >&2
  echo "  cp '$CANON' '$MIRROR'" >&2
  exit 1
fi

echo "OK: mobile proto mirror matches canonical."
echo "  sha256: $(sha256sum "$CANON" | cut -d' ' -f1)"
