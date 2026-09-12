#!/usr/bin/env bash
# Regenerate the framework's committed Python gRPC stubs, reproducibly.
#
# WHY THIS EXISTS
#   The committed stubs under framework/src/fedlearn/communication/generated/ were generator
#   output PLUS an undocumented hand edit: grpcio-tools emits `import fedlearn_pb2`, while the
#   committed file carries `from . import fedlearn_pb2` so the package imports cleanly. That
#   rewrite lived nowhere in the repo, so regeneration was never reproducible — and because it
#   was not reproducible, no freshness gate could work.
#
#   That is the root cause of the drift this script fixes: proto.yml's "buf generate must be a
#   no-op" step points at proto/gen/, which .gitignore:179 ignores, so the check passes whatever
#   the state of the real stubs. fedlearn_pb2.py sat at gencode 4.25.1 while fot_pb2.py had moved
#   to 5.29.0 and the protobuf pins had been raised to >=5.29.0 to match the latter.
#
# WHAT IT GUARANTEES
#   Running this on a clean tree produces no diff. CI runs exactly this and fails if it does, so
#   a stale stub is caught instead of discovered months later.
#
# SCOPE
#   Python only. The Java, TypeScript and C++ consumers generate their own stubs from the same
#   .proto via buf (proto/buf.gen.yaml) and are unaffected by this script.
#
#   The generator version is asserted rather than assumed: different grpcio-tools releases emit
#   different gencode headers, which is exactly how the drift went unnoticed.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PROTO_DIR="$ROOT/framework/src/fedlearn/communication/protos"
OUT_DIR="$ROOT/framework/src/fedlearn/communication/generated"

# Pinned so the output is deterministic. Raising this is a deliberate act that will show up as a
# stub diff in the same commit -- which is the point.
EXPECTED_GRPCIO_TOOLS="1.71.2"

actual="$(python3 -c 'from importlib.metadata import version; print(version("grpcio-tools"))')"
if [[ "$actual" != "$EXPECTED_GRPCIO_TOOLS" ]]; then
  echo "ERROR: grpcio-tools $actual installed, but the committed stubs were generated with" >&2
  echo "       $EXPECTED_GRPCIO_TOOLS. Different releases emit different gencode, so this would" >&2
  echo "       produce a spurious diff." >&2
  echo "Fix:   pip install 'grpcio-tools==$EXPECTED_GRPCIO_TOOLS'" >&2
  echo "       (or, if the bump is intentional, update EXPECTED_GRPCIO_TOOLS here and commit the" >&2
  echo "        regenerated stubs in the same change)" >&2
  exit 1
fi

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

# Flat -I, matching how the committed stubs were produced: the modules are imported as
# `fedlearn_pb2`, not `fedlearn.v2.fedlearn_pb2`.
python3 -m grpc_tools.protoc \
  -I "$PROTO_DIR" \
  --python_out="$tmp" --pyi_out="$tmp" --grpc_python_out="$tmp" \
  "$PROTO_DIR/fedlearn.proto" "$PROTO_DIR/fot.proto"

# The post-processing that was previously done by hand: make the generated cross-imports relative
# so `from fedlearn.communication.generated import fedlearn_pb2_grpc` resolves inside the package
# instead of requiring the generated directory on sys.path.
for f in "$tmp"/*_pb2_grpc.py; do
  [[ -e "$f" ]] || continue
  perl -pi -e 's/^import (\w+_pb2) as /from . import $1 as /' "$f"
done

mkdir -p "$OUT_DIR"
for f in "$tmp"/*.py "$tmp"/*.pyi; do
  [[ -e "$f" ]] || continue
  cp "$f" "$OUT_DIR/$(basename "$f")"
done

echo "Regenerated Python stubs -> $OUT_DIR"
echo "  grpcio-tools $actual"
ls -1 "$OUT_DIR" | grep -E '_pb2(_grpc)?\.pyi?$' | sed 's/^/    /'
