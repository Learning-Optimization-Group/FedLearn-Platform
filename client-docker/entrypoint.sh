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

# Disable all proxies (fixes 'Failed parsing HTTP/2' when VPN/lab proxies intercept gRPC)
unset http_proxy HTTP_PROXY https_proxy HTTPS_PROXY
export no_proxy="*"
export GRPC_ENABLE_FORK_SUPPORT=0

# Forward the recipe key + aggregation strategy from the environment when set (the desktop's
# buildContainerEnv sets MODEL_TYPE/STRATEGY). MODEL_TYPE was previously dropped here — the container
# client then silently defaulted to CNN; and without STRATEGY a non-MLP DeComFL project ran the
# FedAvg client path against a DeComFL server (a silent mismatch). An array keeps empty/spaced values
# safe, and an explicit --model-type/--strategy in "$@" still wins (argparse takes the last).
EXTRA_ARGS=()
if [ -n "$MODEL_TYPE" ]; then
    echo "[entrypoint] MODEL_TYPE=$MODEL_TYPE"
    EXTRA_ARGS+=(--model-type "$MODEL_TYPE")
fi
if [ -n "$STRATEGY" ]; then
    echo "[entrypoint] STRATEGY=$STRATEGY"
    EXTRA_ARGS+=(--strategy "$STRATEGY")
fi

echo "[entrypoint] Launching python3 client.py ..."
exec python3 -u client.py \
    --project-id "$PROJECT_ID" \
    --server-address "$SERVER_ADDRESS" \
    --partition-id "$PARTITION_ID" \
    "${EXTRA_ARGS[@]}" \
    "$@"
