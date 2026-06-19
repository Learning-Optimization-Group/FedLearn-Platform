#!/bin/bash
set -e # Exit immediately if a command fails.

# Wrapper for recipes.py — mirrors run_infer.sh so the same backend
# ProcessBuilder pattern works across Mac/Linux/Windows. recipes.py prints
# its JSON catalog to stdout (e.g. `bash run_recipes.sh --describe`).

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Activate the virtual environment if one is configured (disabled for local Mac).
# source /home/ec2-user/app/venv/bin/activate

cd "$SCRIPT_DIR"
PYTHON="${FEDLEARN_PYTHON:-python3}"
"$PYTHON" recipes.py "$@"

EXIT_CODE=$?
exit $EXIT_CODE
