#!/bin/bash
# =============================================================================
#  FedLearn Platform — Local Development Launcher
# =============================================================================
#  Opens separate Terminal windows for each service:
#    Window 1: Spring Boot Backend API        :8081
#    Window 2: React/Vite Frontend Dashboard  :5173
#    Window 3: Electron Desktop App           :9000
#    Window 4: Interactive FL Client Launcher
# =============================================================================

set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; DIM='\033[2m'; NC='\033[0m'

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INTERACTIVE_LAUNCHER="$PROJECT_ROOT/scripts/launch_clients_interactive.sh"

# Environment defaults
export APP_JWT_SECRET="${APP_JWT_SECRET:-MTIzNDU2Nzg5MDEyMzQ1Njc4OTAxMjM0NTY3ODkwMTI=}"
export APP_INTERNAL_API_KEY="${APP_INTERNAL_API_KEY:-local_default_internal_api_key_12345}"

BACKEND_PORT=8081
FRONTEND_PORT=5173
ELECTRON_PORT=9000

printf "\n${CYAN}${BOLD}"
printf "╔═══════════════════════════════════════════════════╗\n"
printf "║    🚀  FedLearn Ecosystem — Local Dev Launcher    ║\n"
printf "╚═══════════════════════════════════════════════════╝\n"
printf "${NC}\n"

# ── 1. Port conflict detection ────────────────────────────────────────────────
printf "${BOLD}Checking ports...${NC}\n"
PORTS_OK=true
check_port() {
  local port=$1; local label=$2
  if lsof -i ":$port" &>/dev/null; then
    printf "  ${RED}✗ Port $port ($label) already in use:${NC}\n"
    lsof -i ":$port" | awk 'NR>1 {printf "    PID %-6s  %s\n", $2, $1}' | sort -u
    printf "    → Run: ${YELLOW}kill -9 \$(lsof -ti :$port)${NC}\n"
    PORTS_OK=false
  else
    printf "  ${GREEN}✓ Port $port ($label) is free${NC}\n"
  fi
}
check_port $BACKEND_PORT  "Spring Boot API"
check_port $FRONTEND_PORT "React Dashboard"
check_port $ELECTRON_PORT "Electron Dev"

if [ "$PORTS_OK" = false ]; then
  printf "\n${RED}${BOLD}✗ Free the ports above, then re-run this script.${NC}\n\n"
  exit 1
fi
printf "\n"

# ── 2. Helper: open a new Terminal window with a command ──────────────────────
# Uses a simple osascript 'do script' which opens a new window. No keystroke
# simulation needed, so no Accessibility permission issues.
open_window() {
  local cmd="$1"
  osascript -e 'tell application "Terminal"' \
            -e 'activate' \
            -e "do script \"$cmd\"" \
            -e 'end tell'
}

# ── 3. Window 1 — Spring Boot Backend ────────────────────────────────────────
printf "${YELLOW}▶ Window 1: Spring Boot Backend${NC}\n"
BACKEND_CMD="cd '${PROJECT_ROOT}/backend/fl-platform-api' && export APP_JWT_SECRET='${APP_JWT_SECRET}' && export APP_INTERNAL_API_KEY='${APP_INTERNAL_API_KEY}' && echo '--- [1] Spring Boot Backend ---' && ./gradlew bootRun"
open_window "$BACKEND_CMD"
printf "  ${GREEN}✓ Launched${NC}\n"
sleep 1

# ── 4. Window 2 — React Dashboard ────────────────────────────────────────────
printf "${YELLOW}▶ Window 2: React Dashboard${NC}\n"
FRONTEND_CMD="cd '${PROJECT_ROOT}/frontend' && echo '--- [2] React Dashboard ---' && npm run dev"
open_window "$FRONTEND_CMD"
printf "  ${GREEN}✓ Launched${NC}\n"
sleep 0.5

# ── 5. Window 3 — Electron Desktop ───────────────────────────────────────────
printf "${YELLOW}▶ Window 3: Electron Desktop${NC}\n"
ELECTRON_CMD="cd '${PROJECT_ROOT}/fedlearn-desktop' && echo '--- [3] Electron Desktop ---' && npm run dev"
open_window "$ELECTRON_CMD"
printf "  ${GREEN}✓ Launched${NC}\n"
sleep 0.5

# ── 6. Window 4 — Interactive Client Launcher ─────────────────────────────────
printf "${YELLOW}▶ Window 4: Interactive Client Launcher${NC}\n"
CLIENT_CMD="echo '--- [4] FL Client Launcher ---' && bash '${INTERACTIVE_LAUNCHER}'"
open_window "$CLIENT_CMD"
printf "  ${GREEN}✓ Launched${NC}\n"

# ── 7. Summary ────────────────────────────────────────────────────────────────
printf "\n${CYAN}${BOLD}"
printf "╔═══════════════════════════════════════════════════╗\n"
printf "║              ALL SERVICES LAUNCHED                ║\n"
printf "╚═══════════════════════════════════════════════════╝\n"
printf "${NC}"
printf "  Backend API   →  ${CYAN}http://localhost:$BACKEND_PORT${NC}\n"
printf "  H2 Console    →  ${CYAN}http://localhost:$BACKEND_PORT/h2-console${NC}\n"
printf "  Dashboard     →  ${CYAN}http://localhost:$FRONTEND_PORT${NC}\n"
printf "  Electron Dev  →  ${CYAN}http://localhost:$ELECTRON_PORT${NC}\n"
printf "\n${BOLD}Next steps:${NC}\n"
printf "  1. Wait ~15s for Spring Boot to start (Window 1)\n"
printf "  2. Register a user on the dashboard (first time only)\n"
printf "  3. Log in on both the dashboard and the Electron app\n"
printf "  4. Create a project → click Start Training\n"
printf "  5. Switch to Window 4 — it will wait for the backend\n"
printf "     then prompt you for your Project UUID and launch clients\n\n"
