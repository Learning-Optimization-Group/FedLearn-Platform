#!/bin/bash

# Configuration
PROJECT_ID="c5e230a0-d71d-41c6-934a-6726c914ddec"
# Cascade: Check for FEDLEARN_SERVER_IP (LAN) -> AWS_HOST (Cloud) -> Default to localhost
TARGET_IP="${FEDLEARN_SERVER_IP:-${AWS_HOST:-localhost}}"
SERVER_ADDRESS="${TARGET_IP}:54832"

echo "[NETWORK] Clients configured to connect to: $SERVER_ADDRESS"
MODEL_TYPE="CNN"
MODEL_NAME="cnn"
DATASET="cb"
STRATEGY="FedAvg"

# Number of clients to launch
NUM_CLIENTS=1

# Dynamically resolve the path relative to where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
CLIENT_SCRIPT_DIR="${SCRIPT_DIR}"

# Assume the venv is at the root of the project (6 levels up from this script)
# scripts/ -> resources/ -> main/ -> src/ -> fl-platform-api/ -> backend/ -> PROJECT_ROOT
PROJECT_ROOT="$( cd "${SCRIPT_DIR}/../../../../../.." && pwd )"
VENV_PATH="${PROJECT_ROOT}/venv"

# Terminal emulator (will auto-detect)
TERMINAL=""

# Detect available terminal emulator (macOS support)
if [[ "$OSTYPE" == "darwin"* ]]; then
    if [ -d "/Applications/iTerm.app" ] || [ -d "$HOME/Applications/iTerm.app" ]; then
        TERMINAL="iterm"
    else
        TERMINAL="macos-terminal"
    fi
elif command -v gnome-terminal &> /dev/null; then
    TERMINAL="gnome-terminal"
elif command -v xfce4-terminal &> /dev/null; then
    TERMINAL="xfce4-terminal"
elif command -v konsole &> /dev/null; then
    TERMINAL="konsole"
elif command -v xterm &> /dev/null; then
    TERMINAL="xterm"
else
    echo "Error: No supported terminal emulator found"
    echo "Please install iTerm, gnome-terminal, xfce4-terminal, konsole, or xterm"
    exit 1
fi

echo "=================================================="
echo "Launching $NUM_CLIENTS Federated Learning Clients"
echo "=================================================="
echo "Project ID: $PROJECT_ID"
echo "Server: $SERVER_ADDRESS"
echo "Model: $MODEL_TYPE/$MODEL_NAME"
echo "Dataset: $DATASET"
echo "Strategy: $STRATEGY"
echo "Terminal: $TERMINAL"
echo "=================================================="
echo ""

# Function to launch a client
launch_client() {
    local partition_id=$1
    local title="FL Client $partition_id"

    echo "Launching client with partition ID: $partition_id"

    case $TERMINAL in
        iterm)
            osascript -e "tell application \"iTerm\"" \
                      -e "tell current window" \
                      -e "create tab with default profile" \
                      -e "tell current session of current tab" \
                      -e "write text \"source '$VENV_PATH/bin/activate' && cd '$CLIENT_SCRIPT_DIR' && python client.py --project-id $PROJECT_ID --server-address $SERVER_ADDRESS --partition-id $partition_id --model-type $MODEL_TYPE --model-name $MODEL_NAME --dataset $DATASET --strategy $STRATEGY; echo ''; echo 'Client $partition_id finished. Press Enter to close...'; read\"" \
                      -e "end tell" \
                      -e "end tell" \
                      -e "end tell" &
            ;;
        macos-terminal)
            osascript -e "tell application \"Terminal\"" \
                      -e "tell application \"System Events\" to keystroke \"t\" using command down" \
                      -e "do script \"source '$VENV_PATH/bin/activate' && cd '$CLIENT_SCRIPT_DIR' && python client.py --project-id $PROJECT_ID --server-address $SERVER_ADDRESS --partition-id $partition_id --model-type $MODEL_TYPE --model-name $MODEL_NAME --dataset $DATASET --strategy $STRATEGY; echo ''; echo 'Client $partition_id finished. Press Enter to close...'; read\" in front window" \
                      -e "end tell" &
            ;;
        gnome-terminal)
            gnome-terminal --tab --title="$title" -- bash -c "
                source '$VENV_PATH/bin/activate' && \
                cd '$CLIENT_SCRIPT_DIR' && \
                python client.py \
                    --project-id $PROJECT_ID \
                    --server-address $SERVER_ADDRESS \
                    --partition-id $partition_id \
                    --model-type $MODEL_TYPE \
                    --model-name $MODEL_NAME \
                    --dataset $DATASET \
                    --strategy $STRATEGY; \
                echo ''; \
                echo 'Client $partition_id finished. Press Enter to close...'; \
                read
            " &
            ;;
        xfce4-terminal)
            xfce4-terminal --tab --title="$title" -e "bash -c \"
                source '$VENV_PATH/bin/activate' && \
                cd '$CLIENT_SCRIPT_DIR' && \
                python client.py \
                    --project-id $PROJECT_ID \
                    --server-address $SERVER_ADDRESS \
                    --partition-id $partition_id \
                    --model-type $MODEL_TYPE \
                    --model-name $MODEL_NAME \
                    --dataset $DATASET \
                    --strategy $STRATEGY; \
                echo ''; \
                echo 'Client $partition_id finished. Press Enter to close...'; \
                read
            \"" &
            ;;
        konsole)
            konsole --new-tab --title="$title" -e bash -c "
                source '$VENV_PATH/bin/activate' && \
                cd '$CLIENT_SCRIPT_DIR' && \
                python client.py \
                    --project-id $PROJECT_ID \
                    --server-address $SERVER_ADDRESS \
                    --partition-id $partition_id \
                    --model-type $MODEL_TYPE \
                    --model-name $MODEL_NAME \
                    --dataset $DATASET \
                    --strategy $STRATEGY; \
                echo ''; \
                echo 'Client $partition_id finished. Press Enter to close...'; \
                read
            " &
            ;;
        xterm)
            xterm -title "$title" -e bash -c "
                source '$VENV_PATH/bin/activate' && \
                cd '$CLIENT_SCRIPT_DIR' && \
                python client.py \
                    --project-id $PROJECT_ID \
                    --server-address $SERVER_ADDRESS \
                    --partition-id $partition_id \
                    --model-type $MODEL_TYPE \
                    --model-name $MODEL_NAME \
                    --dataset $DATASET \
                    --strategy $STRATEGY; \
                echo ''; \
                echo 'Client $partition_id finished. Press Enter to close...'; \
                read
            " &
            ;;
    esac

    # Small delay between launches to avoid overwhelming the server
    sleep 2
}

# Launch all clients
for i in $(seq 0 $((NUM_CLIENTS - 1))); do
    launch_client $i
done

echo ""
echo "All clients launched successfully!"
echo "Each client is running in a separate tab."
echo ""