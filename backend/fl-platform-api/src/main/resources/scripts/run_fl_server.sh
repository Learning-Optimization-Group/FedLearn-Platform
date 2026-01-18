#!/bin/bash
set -e # Exit immediately if a command fails.

echo "[WRAPPER-FL-SH] Script started."

# Define a log file path.
SCRIPT_DIR=$(dirname "$0")
LOG_FILE="${SCRIPT_DIR}/fl_server_deep_debug.log"

# Clear the log file at the start of each run.
echo "--- New server run starting at $(date) ---" > "$LOG_FILE"

echo "[WRAPPER-FL-SH] Activating Python virtual environment..."
source /home/ec2-user/app/venv/bin/activate
echo "[WRAPPER-FL-SH] Environment activated."

echo "[WRAPPER-FL-SH] Executing fl_server.py with output to both console and log file"

# Execute python script and use 'tee' to send output to BOTH stdout AND log file
# This allows Java backend to capture logs via stdout while also saving to file
python3 "${SCRIPT_DIR}/fl_server.py" "$@" 2>&1 | tee -a "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}  # Get exit code of python3, not tee
echo "[WRAPPER-FL-SH] Python script finished with exit code: $EXIT_CODE" | tee -a "$LOG_FILE"

deactivate
exit $EXIT_CODE