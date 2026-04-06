#!/bin/bash
# =============================================================================
# FedLearn Platform — Docker Build & Validation Test Suite
# Tests the full client-docker image with the heavy PyTorch runtime.
# =============================================================================

# Exit immediately if a command exits with a non-zero status
set -e

# ANSI Color Codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Determine project root (script is in client-docker/)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "${SCRIPT_DIR}/.." && pwd )"

IMAGE_NAME="fedlearn-client:integration-test"

echo -e "${YELLOW}======================================================${NC}"
echo -e "${YELLOW} Docker Client Validation Suite (${IMAGE_NAME})${NC}"
echo -e "${YELLOW}======================================================${NC}"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}[ERROR] Docker is not installed or not in PATH.${NC}"
    exit 1
fi

echo -e "\n${YELLOW}[1/4] Building Docker Image from Project Root...${NC}"
echo "Context: $PROJECT_ROOT"
cd "$PROJECT_ROOT" || exit 1
docker build -t "$IMAGE_NAME" -f client-docker/Dockerfile .
echo -e "${GREEN}[✔] Build Succeeded.${NC}"

echo -e "\n${YELLOW}[2/4] Verifying Framework Pip Installation...${NC}"
# We override the entrypoint to run python and import the library natively installed via pip
docker run --rm --entrypoint python3 "$IMAGE_NAME" -c "
import sys
try:
    import fedlearn
    print('FedLearn framework successfully imported from:', fedlearn.__file__)
except ImportError as e:
    print('Failed to import fedlearn framework!', file=sys.stderr)
    sys.exit(1)
"
echo -e "${GREEN}[✔] Framework Installed and Importable.${NC}"

echo -e "\n${YELLOW}[3/4] Verifying File Structure & Dependencies...${NC}"
# Check if entrypoint and scripts were successfully created/copied
docker run --rm --entrypoint /bin/bash "$IMAGE_NAME" -c "
if [ ! -f /app/entrypoint.sh ]; then echo 'Missing entrypoint.sh!'; exit 1; fi
if [ ! -d /app/scripts ]; then echo 'Missing scripts directory!'; exit 1; fi
echo 'All critical paths exist.'
"
echo -e "${GREEN}[✔] Directory Structure Valid.${NC}"

echo -e "\n${YELLOW}[4/4] Verifying PyTorch Environment...${NC}"
docker run --rm --entrypoint python3 "$IMAGE_NAME" -c "
import torch
print(f'PyTorch Version: {torch.__version__}')
"
echo -e "${GREEN}[✔] PyTorch Functional.${NC}"

echo -e "\n${GREEN}======================================================${NC}"
echo -e "${GREEN} All Tests Passed! The refactored Dockerfile is solid.${NC}"
echo -e "${GREEN}======================================================${NC}"

# Optionally clean up the tagging (we leave the layers cached)
docker rmi "$IMAGE_NAME" >/dev/null 2>&1 || true
