#!/bin/bash
set -e # Exit immediately if a command fails.

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "[WRAPPER-FL-SH] Script started."

# Define a log file path in the script directory
LOG_FILE="${SCRIPT_DIR}/fl_server_deep_debug.log"

# Clear the log file at the start of each run
echo "--- New server run starting at $(date) ---" > "$LOG_FILE"

echo "[WRAPPER-FL-SH] Activating Python virtual environment..."
# source /home/ec2-user/app/venv/bin/activate
echo "[WRAPPER-FL-SH] Environment activated."

echo "[WRAPPER-FL-SH] Executing fl_server.py with output to both console and log file"

# Change to script directory and execute fl_server.py
cd "$SCRIPT_DIR"
/home/anurag/codebase/Projects/FedLearn-Platform/venv/bin/python3 fl_server.py "$@" 2>&1 | tee -a "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}  # Get exit code of python3, not tee
echo "[WRAPPER-FL-SH] Python script finished with exit code: $EXIT_CODE" | tee -a "$LOG_FILE"

# deactivate
exit $EXIT_CODE