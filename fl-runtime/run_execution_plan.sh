#!/bin/bash
set -e # Exit immediately if a command fails.

# Wrapper for execution_plan.py — mirrors run_recipes.sh so the backend's ProcessBuilder pattern works the same
# way. execution_plan.py prints one JSON line (the resolved plan, or why the run is not representable).

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

cd "$SCRIPT_DIR"
PYTHON="${FEDLEARN_PYTHON:-python3}"
"$PYTHON" execution_plan.py "$@"

EXIT_CODE=$?
exit $EXIT_CODE
