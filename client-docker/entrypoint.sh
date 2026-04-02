#!/bin/bash
set -e

echo "[entrypoint] Container started successfully."
echo "[entrypoint] PROJECT_ID=$PROJECT_ID"
echo "[entrypoint] SERVER_ADDRESS=$SERVER_ADDRESS"
echo "[entrypoint] PARTITION_ID=$PARTITION_ID"

if [ -z "$PROJECT_ID" ] || [ -z "$SERVER_ADDRESS" ] || [ -z "$PARTITION_ID" ]; then
    echo "[entrypoint] Error: Missing required environment variables"
    echo "Usage: docker run -e PROJECT_ID=<id> -e SERVER_ADDRESS=<address> -e PARTITION_ID=<id> <image> [--use-llm] [--dataset cb|sst2]"
    exit 1
fi

echo "[entrypoint] Launching python3 client.py ..."
exec python3 -u client.py \
    --project-id "$PROJECT_ID" \
    --server-address "$SERVER_ADDRESS" \
    --partition-id "$PARTITION_ID" \
    "$@"
