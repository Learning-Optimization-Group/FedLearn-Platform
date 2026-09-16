#!/usr/bin/env bash
# check_no_skipped_tests.sh — fail if any jest test file skips or focuses a test (TE-10).
#
# A skipped test is a false green: the suite reports success while the behaviour under test never
# ran. A focused test is the same defect inverted — every OTHER test silently stops running. Jest
# has no built-in "forbid skipped" switch, so this gate statically scans the given directories'
# test files for the skip/focus forms:
#   it.skip / test.skip / describe.skip / it.only / test.only / describe.only (incl. .skip.each)
#   xit / xtest / xdescribe / fit / fdescribe
# and exits non-zero listing every offender with file:line. Wired into ci.yml (mobile-js, desktop).
#
# Usage: scripts/check_no_skipped_tests.sh <dir> [<dir>...]
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <dir> [<dir>...]" >&2
  exit 2
fi

# BSD/GNU-portable ERE (no \b, no -P):
#   .skip( / .only( and the chained .skip.each(...) form
SKIP_ONLY_PATTERN='\.(skip|only)[[:space:]]*[(.]'
#   xit(/xtest(/xdescribe(/fit(/fdescribe( as standalone identifiers — the leading
#   [^.[:alnum:]_] (or line start) keeps method calls like model.fit( and names like
#   outfit( from matching.
FOCUS_ALIAS_PATTERN='(^|[^.[:alnum:]_])(xit|xtest|xdescribe|fit|fdescribe)[[:space:]]*\('

status=0
scanned=0

for dir in "$@"; do
  if [[ ! -d "$dir" ]]; then
    echo "ERROR: not a directory: $dir" >&2
    exit 2
  fi
  while IFS= read -r -d '' file; do
    scanned=$((scanned + 1))
    if hits="$(grep -nE -e "$SKIP_ONLY_PATTERN" -e "$FOCUS_ALIAS_PATTERN" "$file")"; then
      echo "FAIL: skipped/focused test in $file"
      echo "$hits" | sed 's/^/  /'
      status=1
    fi
  done < <(
    find "$dir" -type d -name node_modules -prune -o -type f \
      \( -name '*.test.ts' -o -name '*.test.tsx' -o -name '*.test.js' -o -name '*.test.jsx' \
         -o -name '*.spec.ts' -o -name '*.spec.tsx' -o -name '*.spec.js' -o -name '*.spec.jsx' \
         -o -path '*/__tests__/*.ts' -o -path '*/__tests__/*.tsx' \
         -o -path '*/__tests__/*.js' -o -path '*/__tests__/*.jsx' \) \
      -print0
  )
done

if [[ $scanned -eq 0 ]]; then
  echo "ERROR: no test files found under: $*" >&2
  exit 2
fi

if [[ $status -ne 0 ]]; then
  echo ""
  echo "A skipped test is a false green and a focused test silences the rest of the suite."
  echo "Un-skip/un-focus the tests above (or delete them) before merging."
else
  echo "OK: no skipped/focused jest tests in $scanned test file(s)."
fi
exit $status
