#!/bin/bash
set -e # Exit immediately if a command fails.

echo "[WRAPPER-SH] Activating Python virtual environment..."

# Activate the virtual environment that we created on the server.
source /home/ec2-user/app/venv/bin/activate

echo "[WRAPPER-SH] Environment activated. Executing init_model.py..."

# Execute the python script, passing along all arguments ("$@").
# The path is relative to the project root directory, which is set by the Java ProcessBuilder.
python3 src/main/resources/scripts/init_model.py "$@"

EXIT_CODE=$?
echo "[WRAPPER-SH] Python script finished with exit code: $EXIT_CODE"

# Deactivation happens automatically when the script exits, but this is good practice.
deactivate
exit $EXIT_CODE