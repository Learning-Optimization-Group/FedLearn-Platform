#!/bin/bash
# =============================================================================
# FedLearn Platform — Local Build → EC2 Deploy Script
# =============================================================================
# Runs on your Mac. Builds the fat JAR locally, then SCPs the artifacts
# to the EC2 instance. Does NOT require Gradle, Node, or Python on EC2.
#
# Prerequisites:
#   - AWS key pair (.pem file)
#   - EC2 instance running (already bootstrapped with ec2-bootstrap.sh)
#   - SSH agent or key explicitly passed via EC2_KEY_PATH
#
# Usage:
#   export EC2_HOST=<your-ec2-public-ip>
#   export EC2_KEY_PATH=~/.ssh/your-key.pem
#   ./scripts/deploy-to-aws.sh
#
#   Or pass flags directly:
#   ./scripts/deploy-to-aws.sh --host 54.123.45.67 --key ~/.ssh/my-key.pem
#
# Optional flags:
#   --skip-build     Skip Gradle build (re-use last built JAR)
#   --bootstrap      Also run ec2-bootstrap.sh on the remote (first deploy only)
#   --restart        Restart the systemd service after deploy
# =============================================================================
set -euo pipefail

# ── Default config (override with env vars or flags) ─────────────────────────
EC2_HOST="${EC2_HOST:-}"
EC2_USER="${EC2_USER:-ubuntu}"
EC2_KEY_PATH="${EC2_KEY_PATH:-}"
SKIP_BUILD=false
RUN_BOOTSTRAP=false
RESTART_SERVICE=false

# ── Repo root (this script lives in scripts/) ─────────────────────────────────
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND_DIR="$REPO_ROOT/backend/fl-platform-api"
SCRIPTS_SRC="$BACKEND_DIR/src/main/resources/scripts"
REQUIREMENTS_SRC="$REPO_ROOT/framework/requirements.txt"
BOOTSTRAP_SCRIPT="$REPO_ROOT/scripts/ec2-bootstrap.sh"

# ── Parse CLI flags ───────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --host)        EC2_HOST="$2"; shift 2 ;;
    --key)         EC2_KEY_PATH="$2"; shift 2 ;;
    --user)        EC2_USER="$2"; shift 2 ;;
    --skip-build)  SKIP_BUILD=true; shift ;;
    --bootstrap)   RUN_BOOTSTRAP=true; shift ;;
    --restart)     RESTART_SERVICE=true; shift ;;
    *) echo "[ERROR] Unknown flag: $1"; exit 1 ;;
  esac
done

# ── Validate required inputs ──────────────────────────────────────────────────
if [[ -z "$EC2_HOST" ]]; then
  echo "[ERROR] EC2_HOST is not set."
  echo "        Export it: export EC2_HOST=<public-ip>"
  echo "        Or pass:   --host <public-ip>"
  exit 1
fi

if [[ -z "$EC2_KEY_PATH" ]]; then
  echo "[ERROR] EC2_KEY_PATH is not set."
  echo "        Export it: export EC2_KEY_PATH=~/.ssh/your-key.pem"
  echo "        Or pass:   --key ~/.ssh/your-key.pem"
  exit 1
fi

EC2_KEY_PATH="${EC2_KEY_PATH/#\~/$HOME}"  # expand ~ manually

if [[ ! -f "$EC2_KEY_PATH" ]]; then
  echo "[ERROR] Key file not found: $EC2_KEY_PATH"
  exit 1
fi

# ── SSH / SCP helpers ──────────────────────────────────────────────────────────
SSH_OPTS="-i $EC2_KEY_PATH -o StrictHostKeyChecking=no -o ConnectTimeout=10"
EC2_TARGET="$EC2_USER@$EC2_HOST"
REMOTE_APP_DIR="~/app"
REMOTE_SCRIPTS_DIR="$REMOTE_APP_DIR/scripts"

ssh_cmd() {
  ssh $SSH_OPTS "$EC2_TARGET" "$@"
}

scp_cmd() {
  scp $SSH_OPTS "$@"
}

# ── Banner ────────────────────────────────────────────────────────────────────
echo ""
echo "=============================================="
echo " FedLearn — Build & Deploy to EC2"
echo " Target: $EC2_TARGET"
echo "=============================================="
echo ""

# ── Step 1: Build fat JAR ─────────────────────────────────────────────────────
if [[ "$SKIP_BUILD" == false ]]; then
  echo "[1/5] Building fat JAR locally (tests skipped)..."
  cd "$BACKEND_DIR"
  ./gradlew bootJar -x test --quiet
  echo "      ✓ Build complete"
else
  echo "[1/5] Skipping build (--skip-build)"
fi

JAR_PATH=$(find "$BACKEND_DIR/build/libs" -name "*.jar" | grep -v plain | head -1)
if [[ -z "$JAR_PATH" ]]; then
  echo "[ERROR] No JAR found in $BACKEND_DIR/build/libs. Run without --skip-build."
  exit 1
fi
echo "      JAR: $JAR_PATH"

# ── Step 2: Upload Python requirements ────────────────────────────────────────
# Must happen BEFORE bootstrap so ec2-bootstrap.sh's pip-install step finds
# requirements.txt on first run. (Previously, bootstrap warned + skipped, then
# requirements landed too late and were never installed.)
echo ""
echo "[2/5] Uploading Python requirements..."
scp_cmd "$REQUIREMENTS_SRC" "$EC2_TARGET:~/requirements.txt"
echo "      ✓ requirements.txt uploaded"

# ── Step 3 (optional): Bootstrap the EC2 instance ─────────────────────────────
if [[ "$RUN_BOOTSTRAP" == true ]]; then
  echo ""
  echo "[3/5] Running ec2-bootstrap.sh on remote (one-time setup)..."
  scp_cmd "$BOOTSTRAP_SCRIPT" "$EC2_TARGET:~/ec2-bootstrap.sh"
  ssh_cmd "chmod +x ~/ec2-bootstrap.sh && sudo ~/ec2-bootstrap.sh"
  echo "      ✓ Bootstrap complete"
else
  echo ""
  echo "[3/5] Skipping bootstrap (use --bootstrap on first deploy)"
fi

# ── Step 4: SCP JAR and scripts ───────────────────────────────────────────────
echo ""
echo "[4/5] Uploading app artifacts..."

# Make sure remote directories exist
ssh_cmd "mkdir -p $REMOTE_APP_DIR $REMOTE_SCRIPTS_DIR $REMOTE_APP_DIR/models $REMOTE_APP_DIR/data"

# Upload JAR
scp_cmd "$JAR_PATH" "$EC2_TARGET:$REMOTE_APP_DIR/app.jar"
echo "      ✓ app.jar uploaded"

# Upload scripts directory — use rsync to skip __pycache__, logs, and debug files
# (rsync is pre-installed on macOS; if missing: brew install rsync)
if command -v rsync &>/dev/null; then
  rsync -az --quiet \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    --exclude='*.log' \
    --exclude='*.npz' \
    -e "ssh -i $EC2_KEY_PATH -o StrictHostKeyChecking=no" \
    "$SCRIPTS_SRC/" "$EC2_TARGET:$REMOTE_SCRIPTS_DIR/"
else
  # Fallback to scp if rsync is unavailable
  scp_cmd -r "$SCRIPTS_SRC"/* "$EC2_TARGET:$REMOTE_SCRIPTS_DIR/"
fi
echo "      ✓ Python scripts uploaded"

# Make shell scripts executable on the remote
ssh_cmd "chmod +x $REMOTE_SCRIPTS_DIR/*.sh 2>/dev/null || true"
echo "      ✓ Shell scripts marked executable"

# ── Step 5: Restart service (optional) ────────────────────────────────────────
echo ""
if [[ "$RESTART_SERVICE" == true ]]; then
  echo "[5/5] Restarting fedlearn systemd service..."
  ssh_cmd "sudo systemctl restart fedlearn"
  sleep 3
  echo "      Service status:"
  ssh_cmd "sudo systemctl status fedlearn --no-pager -l" || true
  echo "      ✓ Service restarted"
else
  echo "[5/5] Skipping automatic restart (use --restart to enable)"
  echo ""
  echo "      To start manually, SSH in and run:"
  echo "      ─────────────────────────────────────────────────────"
  echo "      ssh $SSH_OPTS $EC2_TARGET"
  echo ""
  echo "      # One-time: configure environment variables"
  echo "      sudo nano /etc/systemd/system/fedlearn.service"
  echo "      # Uncomment and fill in the Environment= lines, then:"
  echo "      sudo systemctl daemon-reload"
  echo "      sudo systemctl enable fedlearn"
  echo "      sudo systemctl start fedlearn"
  echo ""
  echo "      # OR run directly in the foreground for debugging:"
  echo "      export SPRING_PROFILES_ACTIVE=ec2demo"
  echo "      export APP_JWT_SECRET=\"\$(openssl rand -hex 64)\""
  echo "      export APP_INTERNAL_API_KEY=\"\$(openssl rand -hex 32)\""
  echo "      export CORS_ALLOWED_ORIGINS=\"http://localhost:5173\""
  echo "      export FEDLEARN_PYTHON=python3"
  echo "      export PYTHON_EXECUTABLE_PATH=~/app/scripts/run_init_model.sh"
  echo "      export PYTHON_SCRIPT_FL_SERVER_PATH=~/app/scripts/run_fl_server.sh"
  echo "      export FEATURE_LOG_PERSISTENCE=false"
  echo "      java -jar ~/app/app.jar"
  echo "      ─────────────────────────────────────────────────────"
fi

echo ""
echo "=============================================="
echo " Deploy complete! ✓"
echo ""
echo " Verify the backend is healthy:"
echo "   curl http://$EC2_HOST:8081/actuator/health"
echo ""
echo " Tail live logs:"
echo "   ssh $SSH_OPTS $EC2_TARGET 'sudo journalctl -u fedlearn -f'"
echo "=============================================="
