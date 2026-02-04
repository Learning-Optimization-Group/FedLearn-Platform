#!/bin/bash

# Configuration
PROJECT_ID="702f07e6-c6b6-43f1-9c6a-94c6a4c84b71"
SERVER_ADDRESS="18.116.165.47:37191"
MODEL_TYPE="TRANSFORMER"
MODEL_NAME="opt-125m"
DATASET="cb"
STRATEGY="DeComFL"

# Number of clients to launch
NUM_CLIENTS=2

# Path to your client script
CLIENT_SCRIPT_DIR="$HOME/Desktop/FedLearn-Platform/backend/fl-platform-api/src/main/resources/scripts"
VENV_PATH="$HOME/Desktop/FedLearn-Platform/backend/fl-platform-api/venv"

# Terminal emulator (will auto-detect)
TERMINAL=""

# Detect available terminal emulator
if command -v gnome-terminal &> /dev/null; then
    TERMINAL="gnome-terminal"
elif command -v xfce4-terminal &> /dev/null; then
    TERMINAL="xfce4-terminal"
elif command -v konsole &> /dev/null; then
    TERMINAL="konsole"
elif command -v xterm &> /dev/null; then
    TERMINAL="xterm"
else
    echo "Error: No supported terminal emulator found"
    echo "Please install gnome-terminal, xfce4-terminal, konsole, or xterm"
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
        gnome-terminal)
            gnome-terminal --title="$title" -- bash -c "
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
            xfce4-terminal --title="$title" -e "bash -c \"
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
echo "Each client is running in a separate terminal window."
echo ""