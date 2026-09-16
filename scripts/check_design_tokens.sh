#!/usr/bin/env bash
# check_design_tokens.sh — the design-system analogue of check_proto_mirror.sh.
#
# design/tokens.json is the single source of truth; design/build-tokens.mjs generates the per-platform
# token outputs. Nothing enforced that the committed outputs stay in sync, so a hand-edit of a generated
# file (or a stray hardcoded color that "fixes" a drift by hand) could slip through. This regenerates
# from tokens.json and FAILS if any committed output has drifted. The fix is a literal command.
#
# Assumes a clean working tree for the generated files (as in CI). Runs on plain Node — no deps.
set -euo pipefail
cd "$(dirname "$0")/.."  # repo root

OUTPUTS=(
  frontend/src/styles/tokens.css
  fedlearn-desktop/src/renderer/tokens.css
  mobile_client/src/theme/tokens.generated.ts
  mobile_client/src/theme/global.css
)

node design/build-tokens.mjs >/dev/null

if git diff --quiet -- "${OUTPUTS[@]}"; then
  echo "OK: design token outputs are in sync with design/tokens.json."
  exit 0
fi

echo "ERROR: design token outputs drifted from design/tokens.json (the source of truth)."
echo "       A generated file was hand-edited, or tokens.json changed without regenerating."
echo "Fix:   node design/build-tokens.mjs   (then commit the regenerated files)"
echo
git --no-pager diff --stat -- "${OUTPUTS[@]}"
exit 1
