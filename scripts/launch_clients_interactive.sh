#!/bin/bash
# =============================================================================
#  FedLearn — Interactive FL Client Launcher  (runs in Window 4)
# =============================================================================
#  1. Waits until the Spring Boot backend is responding
#  2. Prompts for Project UUID, number of clients, and model config
#  3. Shows a confirmation summary
#  4. Launches each client in its own Terminal window
# =============================================================================

# ── Colors ────────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; DIM='\033[2m'; NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CLIENT_SCRIPT_DIR="$PROJECT_ROOT/backend/fl-platform-api/src/main/resources/scripts"
VENV_PATH="$PROJECT_ROOT/venv"
BACKEND_URL="http://localhost:8081"
HEALTH_URL="$BACKEND_URL/actuator/health"

clear
printf "${CYAN}${BOLD}"
printf "╔══════════════════════════════════════════════════╗\n"
printf "║       FedLearn — Interactive Client Launcher     ║\n"
printf "╚══════════════════════════════════════════════════╝\n"
printf "${NC}\n"

# ── 1. Wait for Spring Boot to be responding ──────────────────────────────────
printf "${YELLOW}⏳ Waiting for backend at $BACKEND_URL ...${NC}\n"
printf "   ${DIM}(Spring Boot typically takes 15–30 seconds)${NC}\n\n"

MAX_WAIT=180
WAITED=0
while true; do
  # Any HTTP response (even 401/403) means Spring Boot is up.
  HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$HEALTH_URL" 2>/dev/null || echo "000")
  if [ "$HTTP_CODE" != "000" ]; then
    printf "\r${GREEN}✓ Backend is UP! (HTTP $HTTP_CODE)                    ${NC}\n\n"
    break
  fi
  if [ $WAITED -ge $MAX_WAIT ]; then
    printf "\n${RED}✗ Backend did not respond within ${MAX_WAIT}s.${NC}\n"
    printf "  Is ${YELLOW}./gradlew bootRun${NC} still running in Window 1?\n"
    read -r -p "  Press Enter to keep waiting, or Ctrl+C to quit..." _
    WAITED=0
  fi
  printf "\r  ${DIM}Elapsed: ${WAITED}s — retrying...${NC}   "
  sleep 3
  WAITED=$((WAITED + 3))
done

# ── 2. Prompt for configuration ───────────────────────────────────────────────
printf "${BOLD}Configuration${NC}\n"
printf "${DIM}─────────────────────────────────────────────────${NC}\n"

# Server address
DEFAULT_SERVER="localhost:54832"
read -r -p "  gRPC Server address      [$DEFAULT_SERVER]: " SERVER_ADDRESS
SERVER_ADDRESS="${SERVER_ADDRESS:-$DEFAULT_SERVER}"

# Project UUID
printf "\n  ${DIM}Tip: copy the UUID from the browser URL after creating a project,${NC}\n"
printf "  ${DIM}or from http://localhost:8081/h2-console → SELECT id FROM project;${NC}\n"
read -r -p "  Project UUID             : " PROJECT_ID
while [[ -z "$PROJECT_ID" ]]; do
  printf "  ${RED}✗ Project UUID cannot be empty.${NC}\n"
  read -r -p "  Project UUID             : " PROJECT_ID
done

# Number of clients
read -r -p "  Number of clients        [3]: " NUM_CLIENTS
NUM_CLIENTS="${NUM_CLIENTS:-3}"
if ! [[ "$NUM_CLIENTS" =~ ^[1-9]$ ]]; then
  printf "  ${YELLOW}⚠ Invalid number — defaulting to 3.${NC}\n"
  NUM_CLIENTS=3
fi

# Model config (with defaults matching run_clients.sh)
printf "\n  ${DIM}Model defaults match the ECG/CB example. Press Enter to accept.${NC}\n"
read -r -p "  Model type               [CNN]: "   MODEL_TYPE;  MODEL_TYPE="${MODEL_TYPE:-CNN}"
read -r -p "  Model name               [cnn]: "   MODEL_NAME;  MODEL_NAME="${MODEL_NAME:-cnn}"
read -r -p "  Dataset                  [cb]:  "   DATASET;     DATASET="${DATASET:-cb}"
read -r -p "  Strategy                 [FedAvg]: " STRATEGY;  STRATEGY="${STRATEGY:-FedAvg}"

# ── 3. Confirmation summary ───────────────────────────────────────────────────
printf "\n${CYAN}${BOLD}"
printf "╔══════════════════════════════════════════════════╗\n"
printf "║                   Ready to Launch                ║\n"
printf "╚══════════════════════════════════════════════════╝\n"
printf "${NC}"
printf "  gRPC Server  →  ${CYAN}$SERVER_ADDRESS${NC}\n"
printf "  Project ID   →  ${CYAN}$PROJECT_ID${NC}\n"
printf "  Clients      →  ${CYAN}$NUM_CLIENTS${NC}\n"
printf "  Model        →  ${CYAN}$MODEL_TYPE / $MODEL_NAME${NC}\n"
printf "  Dataset      →  ${CYAN}$DATASET${NC}\n"
printf "  Strategy     →  ${CYAN}$STRATEGY${NC}\n"
printf "  Venv         →  ${DIM}$VENV_PATH${NC}\n\n"

read -r -p "  ▶ Press Enter to launch clients (Ctrl+C to abort)..." _
printf "\n"

# ── 4. Launch function ────────────────────────────────────────────────────────
launch_client() {
  local pid=$1
  local CLIENT_CMD="source '${VENV_PATH}/bin/activate' && cd '${CLIENT_SCRIPT_DIR}' && python client.py --project-id ${PROJECT_ID} --server-address ${SERVER_ADDRESS} --partition-id ${pid} --model-type ${MODEL_TYPE} --model-name ${MODEL_NAME} --dataset ${DATASET} --strategy ${STRATEGY}; echo ''; echo '--- Client ${pid} finished. Press Enter to close ---'; read"

  osascript -e 'tell application "Terminal"' \
            -e 'activate' \
            -e "do script \"$CLIENT_CMD\"" \
            -e 'end tell' &

  sleep 2
}

# ── 5. Launch all clients ─────────────────────────────────────────────────────
for i in $(seq 0 $((NUM_CLIENTS - 1))); do
  printf "  ${GREEN}▶ Launching Client $i (partition-id=$i)...${NC}\n"
  launch_client "$i"
done

printf "\n${GREEN}${BOLD}✓ All $NUM_CLIENTS clients launched!${NC}\n"
printf "\n${BOLD}What to watch:${NC}\n"
printf "  • Client windows — should print 'Connected to server' then start training\n"
printf "  • Window 1       — backend logs show rounds completing\n"
printf "  • Dashboard      — live log stream updates per round\n"
printf "\n${DIM}The FL server waits for all $NUM_CLIENTS clients before starting Round 1.\n"
printf "Training completes automatically after all rounds finish.${NC}\n\n"
