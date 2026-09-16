#!/bin/bash
set -e # Exit immediately if a command fails.

# Wrapper for the standalone FoT (Federation over Text) server, spawned by the control plane the
# same way run_fl_server.sh spawns the gradient FL server. Stdout (JSON event lines) is streamed
# to the dashboard via STOMP.

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "[WRAPPER-FOT-SH] Script started."

LOG_FILE="${SCRIPT_DIR}/fot_server_debug.log"
echo "--- New FoT server run starting at $(date) ---" > "$LOG_FILE"

cd "$SCRIPT_DIR"
PYTHON="${FEDLEARN_PYTHON:-python3}"
"$PYTHON" fl_fot_server.py "$@" 2>&1 | tee -a "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}  # exit code of python, not tee
echo "[WRAPPER-FOT-SH] Python script finished with exit code: $EXIT_CODE" | tee -a "$LOG_FILE"

exit $EXIT_CODE
