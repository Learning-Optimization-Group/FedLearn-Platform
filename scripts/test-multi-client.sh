#!/usr/bin/env bash
# =============================================================================
# FedLearn — Multi-Client End-to-End Test
# =============================================================================
# Connects 3 clients to an AWS FL project simultaneously:
#
#   Client 0  →  Desktop App (you click "Start Training" manually)
#   Client 1  →  Python CLI  (this script, partition 1)
#   Client 2  →  Docker      (this script, partition 2)
#
# Usage:
#   ./scripts/test-multi-client.sh <PROJECT_ID> [SERVER_PORT]
#
#   PROJECT_ID  — Copy from the web dashboard after creating a project with
#                 numClients=3, model=CNN, strategy=FedAvg, rounds=3.
#   SERVER_PORT — (Optional) The port the server is listening on. Defaults to 50000.
#                 Check the web dashboard to see what port your project was assigned.
# =============================================================================

set -euo pipefail

# ── Colour helpers ─────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; CYAN='\033[0;36m'; NC='\033[0m'
info()    { echo -e "${CYAN}[TEST]${NC}  $*"; }
success() { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# ── Config ─────────────────────────────────────────────────────────────────────
API_BASE="http://3.137.147.240:8081/api"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CLIENT_SCRIPT="$REPO_ROOT/fl-runtime/client.py"
FRAMEWORK_SRC="$REPO_ROOT/framework/src"
DOCKER_IMAGE="fedlearn-client:latest"

# ── Argument ───────────────────────────────────────────────────────────────────
PROJECT_ID="${1:-}"
PORT="${2:-50000}"
SERVER_ADDRESS="3.137.147.240:$PORT"

if [[ -z "$PROJECT_ID" ]]; then
  echo ""
  echo -e "${YELLOW}Usage:${NC}  ./scripts/test-multi-client.sh <PROJECT_ID> [SERVER_PORT]"
  echo ""
  echo "  Steps to get a Project ID:"
  echo "  1. Open http://3.137.147.240:8081  (or the frontend at http://localhost:5173)"
  echo "  2. Create a new project with:"
  echo "       Architecture : CNN"
  echo "       Model        : net"
  echo "       Clients      : 3"
  echo "       Rounds       : 3"
  echo "       Strategy     : FedAvg"
  echo "       Dataset      : (leave blank)"
  echo "  3. Copy the UUID from the URL or project card"
  echo "  4. Re-run:  ./scripts/test-multi-client.sh <that-UUID>"
  echo ""
  exit 1
fi

echo ""
echo "=================================================================="
echo "  FedLearn Multi-Client Test"
echo "  Project  : $PROJECT_ID"
echo "  Server   : $SERVER_ADDRESS"
echo "=================================================================="
echo ""

# ── Pre-flight checks ──────────────────────────────────────────────────────────
info "Checking pre-flight requirements..."

# 1. Python
python3 --version &>/dev/null || error "python3 not found"
success "python3 available"

# 2. Framework importable
PYTHONPATH="$FRAMEWORK_SRC" python3 -c "import fedlearn" 2>/dev/null \
  || error "Cannot import 'fedlearn'. Check that framework/src is correct."
success "fedlearn importable via PYTHONPATH"

# 3. Client script
[[ -f "$CLIENT_SCRIPT" ]] || error "client.py not found at $CLIENT_SCRIPT"
success "client.py found"

# 4. Docker image
if docker image inspect "$DOCKER_IMAGE" &>/dev/null; then
  success "Docker image '$DOCKER_IMAGE' found"
else
  warn "Docker image '$DOCKER_IMAGE' not found (or Docker is not running). Client 2 may fail."
fi

# 5. AWS port reachable
nc -z -w3 3.137.147.240 "$PORT" 2>/dev/null \
  && success "Port $PORT reachable on AWS" \
  || warn "Port $PORT not yet open — the server opens it when Training is activated from the dashboard."

echo ""
echo "------------------------------------------------------------------"
echo -e "  ${YELLOW}ACTION REQUIRED — Desktop App (Client 0)${NC}"
echo "------------------------------------------------------------------"
echo "  In the FedLearn Desktop App:"
echo "    1. Select the project: $PROJECT_ID"
echo "    2. Hardware profile  : MPS (or CPU)"
echo "    3. Click  Start Training"
echo ""
echo "  This laptop will connect as partition 0."
echo "  Come back here and press ENTER when you've clicked Start Training."
echo "------------------------------------------------------------------"
read -r -p "Press ENTER once Client 0 is started in the Desktop App..."
echo ""

# ── Launch Client 1 — Python CLI ───────────────────────────────────────────────
info "Launching Client 1 (Python CLI, partition 1)..."

LOG_CLI="$REPO_ROOT/logs/client1_$(date +%H%M%S).log"
mkdir -p "$REPO_ROOT/logs"

PYTHONPATH="$FRAMEWORK_SRC" python3 -u "$CLIENT_SCRIPT" \
  --project-id   "$PROJECT_ID" \
  --server-address "$SERVER_ADDRESS" \
  --partition-id 1 \
  --model-type   CNN \
  --strategy     FedAvg \
  2>&1 | tee "$LOG_CLI" &

CLI_PID=$!
success "Client 1 started (PID $CLI_PID) — log: logs/$(basename "$LOG_CLI")"

# Small delay so Client 1 finishes its import phase before Docker starts
sleep 3

# ── Launch Client 2 — Docker ───────────────────────────────────────────────────
info "Launching Client 2 (Docker container, partition 2)..."

LOG_DOCKER="$REPO_ROOT/logs/client2_$(date +%H%M%S).log"

docker run --rm \
  --name "fedlearn-test-client2" \
  -e PROJECT_ID="$PROJECT_ID" \
  -e SERVER_ADDRESS="$SERVER_ADDRESS" \
  -e PARTITION_ID=2 \
  -e MODEL_TYPE=CNN \
  -e STRATEGY=FedAvg \
  "$DOCKER_IMAGE" \
  2>&1 | tee "$LOG_DOCKER" &

DOCKER_PID=$!
success "Client 2 started (Docker PID $DOCKER_PID) — log: logs/$(basename "$LOG_DOCKER")"

echo ""
echo "=================================================================="
echo -e "  ${GREEN}All 3 clients are running!${NC}"
echo "=================================================================="
echo "  Client 0  Desktop App  (partition 0) — watch the app UI"
echo "  Client 1  Python CLI   (partition 1) — PID $CLI_PID"
echo "  Client 2  Docker       (partition 2) — PID $DOCKER_PID"
echo ""
echo "  Logs saved to:  $REPO_ROOT/logs/"
echo "  Press Ctrl+C to abort all background clients."
echo "=================================================================="
echo ""

# ── Wait for background processes and surface exit codes ──────────────────────
wait $CLI_PID
CLI_EXIT=$?

wait $DOCKER_PID
DOCKER_EXIT=$?

echo ""
echo "=================================================================="
echo "  Test Complete"
echo "=================================================================="
[[ $CLI_EXIT    -eq 0 ]] && success "Client 1 (CLI)    exited cleanly" \
                          || warn    "Client 1 (CLI)    exited with code $CLI_EXIT"
[[ $DOCKER_EXIT -eq 0 ]] && success "Client 2 (Docker) exited cleanly" \
                          || warn    "Client 2 (Docker) exited with code $DOCKER_EXIT"
echo ""
echo "  Check the web dashboard for final round results:"
echo "  http://3.137.147.240:8081  or  http://localhost:5173"
echo "=================================================================="
