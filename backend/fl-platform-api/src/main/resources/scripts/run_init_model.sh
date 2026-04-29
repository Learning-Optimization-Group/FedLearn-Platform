#!/bin/bash
set -e # Exit immediately if a command fails.

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "[WRAPPER-SH] Activating Python virtual environment..."

# Activate the virtual environment (Disabled for local Mac testing)
# source /home/ec2-user/app/venv/bin/activate

echo "[WRAPPER-SH] Environment activated. Executing init_model.py..."

# Change to script directory and execute init_model.py with absolute path
cd "$SCRIPT_DIR"
PYTHON="${FEDLEARN_PYTHON:-python3}"
"$PYTHON" init_model.py "$@"

EXIT_CODE=$?
echo "[WRAPPER-SH] Python script finished with exit code: $EXIT_CODE"

# Deactivation happens automatically when the script exits
# deactivate
exit $EXIT_CODE