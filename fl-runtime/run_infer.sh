#!/bin/bash
set -e # Exit immediately if a command fails.

# Wrapper for infer.py — mirrors run_init_model.sh so the same backend
# ProcessBuilder pattern works across Mac/Linux/Windows (.bat companion).
# infer.py writes its result to the --out file; this wrapper's stdout is
# diagnostic only.

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Activate the virtual environment if one is configured (disabled for local Mac).
# source /home/ec2-user/app/venv/bin/activate

cd "$SCRIPT_DIR"
PYTHON="${FEDLEARN_PYTHON:-python3}"
"$PYTHON" infer.py "$@"

EXIT_CODE=$?
exit $EXIT_CODE
