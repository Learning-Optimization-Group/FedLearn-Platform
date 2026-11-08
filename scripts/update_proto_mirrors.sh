#!/usr/bin/env bash
# update_proto_mirrors.sh — copy canonical protos to in-tree mirrors and verify integrity.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

CANON_FEDLEARN="$ROOT/proto/fedlearn/v2/fedlearn.proto"
CANON_FOT="$ROOT/proto/fedlearn/fot/v1/fot.proto"
CANON_CONTRACT="$ROOT/proto/fedlearn/contract/v1/execution_contract.proto"
CANON_TREE="$ROOT/proto/fedlearn/tree/v1/tree.proto"

echo "Syncing proto mirrors from canonical sources..."
cp "$CANON_FEDLEARN" "$ROOT/mobile_client/proto/fedlearn/v2/fedlearn.proto"
cp "$CANON_FEDLEARN" "$ROOT/framework/src/fedlearn/communication/protos/fedlearn.proto"
cp "$CANON_FOT" "$ROOT/framework/src/fedlearn/communication/protos/fot.proto"
cp "$CANON_CONTRACT" "$ROOT/framework/src/fedlearn/communication/protos/execution_contract.proto"
cp "$CANON_TREE" "$ROOT/framework/src/fedlearn/communication/protos/tree.proto"

echo "Running verification..."
"$ROOT/scripts/check_proto_mirror.sh"
echo "Proto mirrors updated and verified successfully."
