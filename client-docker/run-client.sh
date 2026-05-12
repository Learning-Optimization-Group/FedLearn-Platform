#!/bin/bash
# =============================================================================
# FedLearn Platform — Smart Client Launcher
# =============================================================================
# Detects the hardware platform and launches the training client with the
# correct GPU flags automatically.
#
# Usage:
#   ./run-client.sh <PROJECT_ID> <SERVER_ADDRESS> <PARTITION_ID> [extra args...]
#
# Examples:
#   # Docker (NVIDIA Desktop GPU):
#   ./run-client.sh "abc-123" "192.168.0.7:50181" 0 --use-llm --dataset sst2
#
#   # Docker (Jetson SoC):
#   ./run-client.sh "abc-123" "192.168.0.7:50181" 0
#
#   # Native (macOS with MPS — auto-detected):
#   ./run-client.sh "abc-123" "127.0.0.1:50181" 1 --use-llm
# =============================================================================

set -e

# ANSI Color Codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

# Resolve project root (this script lives in client-docker/)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "${SCRIPT_DIR}/.." && pwd )"

DOCKER_IMAGE="${FEDLEARN_CLIENT_IMAGE:-fedlearn-client:latest}"

# ── Argument Parsing ─────────────────────────────────────────────────────────
if [ "$#" -lt 3 ]; then
    echo -e "${RED}Usage: $0 <PROJECT_ID> <SERVER_ADDRESS> <PARTITION_ID> [extra args...]${NC}"
    echo ""
    echo "  PROJECT_ID      UUID of the FedLearn project"
    echo "  SERVER_ADDRESS   gRPC server address (e.g. 192.168.0.7:50181)"
    echo "  PARTITION_ID     Client partition index (0-9)"
    echo ""
    echo "Extra args are forwarded to client.py (e.g. --use-llm --dataset sst2)"
    exit 1
fi

PROJECT_ID="$1"
SERVER_ADDRESS="$2"
PARTITION_ID="$3"
shift 3
EXTRA_ARGS="$@"

# ── Platform Detection ───────────────────────────────────────────────────────
detect_platform() {
    local os_name
    os_name="$(uname -s)"

    if [ "$os_name" = "Darwin" ]; then
        echo "macos"
        return
    fi

    # Linux: check for Jetson vs discrete NVIDIA GPU
    if [ -f /etc/nv_tegra_release ] || [ -d /sys/devices/platform/gpu.0 ]; then
        echo "jetson"
        return
    fi

    if command -v nvidia-smi &> /dev/null; then
        echo "nvidia-desktop"
        return
    fi

    echo "cpu"
}

PLATFORM=$(detect_platform)

echo -e "${CYAN}======================================================${NC}"
echo -e "${CYAN} FedLearn Client Launcher${NC}"
echo -e "${CYAN}======================================================${NC}"
echo -e "  Platform:       ${GREEN}${PLATFORM}${NC}"
echo -e "  Project ID:     ${PROJECT_ID}"
echo -e "  Server:         ${SERVER_ADDRESS}"
echo -e "  Partition ID:   ${PARTITION_ID}"
echo -e "  Extra args:     ${EXTRA_ARGS:-<none>}"
echo -e "${CYAN}======================================================${NC}"
echo ""

# ── macOS / Apple Silicon: Run Natively ───────────────────────────────────────
if [ "$PLATFORM" = "macos" ]; then
    echo -e "${YELLOW}[macOS] Running client natively for MPS (Metal) GPU acceleration.${NC}"
    echo -e "${YELLOW}[macOS] Docker on Mac does NOT support Metal GPU pass-through.${NC}"
    echo ""

    # Activate venv if it exists
    if [ -f "${SCRIPT_DIR}/venv/bin/activate" ]; then
        echo -e "[macOS] Activating virtual environment..."
        source "${SCRIPT_DIR}/venv/bin/activate"
    fi

    export PYTHONPATH="${PROJECT_ROOT}/framework/src:${PYTHONPATH}"

    exec python3 "${SCRIPT_DIR}/scripts/client.py" \
        --project-id "$PROJECT_ID" \
        --server-address "$SERVER_ADDRESS" \
        --partition-id "$PARTITION_ID" \
        $EXTRA_ARGS
fi

# ── Docker: Ensure daemon is running ─────────────────────────────────────────
if ! command -v docker &> /dev/null; then
    echo -e "${RED}[ERROR] Docker is not installed or not in PATH.${NC}"
    exit 1
fi

if ! docker info &> /dev/null; then
    echo -e "${RED}[ERROR] Docker daemon is not running.${NC}"
    exit 1
fi

# ── Build GPU flags based on platform ────────────────────────────────────────
GPU_FLAGS=""

case "$PLATFORM" in
    nvidia-desktop)
        GPU_FLAGS="--gpus all"
        echo -e "${GREEN}[GPU] Using NVIDIA Desktop GPU (--gpus all)${NC}"
        ;;
    jetson)
        GPU_FLAGS="--device /dev/nvhost-ctrl --device /dev/nvhost-ctrl-gpu --device /dev/nvhost-prof-gpu --device /dev/nvmap --device /dev/nvhost-gpu"
        echo -e "${GREEN}[GPU] Using Jetson SoC (direct device mounts)${NC}"
        ;;
    cpu)
        echo -e "${YELLOW}[GPU] No GPU detected — running on CPU only.${NC}"
        ;;
esac

echo ""

# ── Launch Docker Container ──────────────────────────────────────────────────
echo -e "${GREEN}[Docker] Launching container: ${DOCKER_IMAGE}${NC}"

exec docker run --rm -it \
    $GPU_FLAGS \
    -e PROJECT_ID="$PROJECT_ID" \
    -e SERVER_ADDRESS="$SERVER_ADDRESS" \
    -e PARTITION_ID="$PARTITION_ID" \
    "$DOCKER_IMAGE" \
    $EXTRA_ARGS
