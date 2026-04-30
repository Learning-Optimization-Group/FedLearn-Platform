#!/bin/bash
# =============================================================================
# FedLearn Platform — EC2 Bootstrap Script
# =============================================================================
# Run this ONCE on a fresh Ubuntu 22.04 EC2 instance.
# It installs all runtime dependencies and creates the app directory layout.
#
# Usage:
#   chmod +x ec2-bootstrap.sh
#   ./ec2-bootstrap.sh
# =============================================================================
set -euo pipefail

echo ""
echo "=============================================="
echo " FedLearn EC2 Bootstrap — Starting"
echo "=============================================="
echo ""

# ── Runtime check ────────────────────────────────────────────────────────────
if [[ "$EUID" -ne 0 ]]; then
  echo "[ERROR] Please run this script as root: sudo ./ec2-bootstrap.sh"
  exit 1
fi

ACTUAL_USER="${SUDO_USER:-ubuntu}"
HOME_DIR=$(eval echo "~$ACTUAL_USER")
APP_DIR="$HOME_DIR/app"
SCRIPTS_DIR="$APP_DIR/scripts"

echo "[1/6] Updating system packages..."
apt-get update -qq

apt-get install -y --no-install-recommends \
  openjdk-21-jre-headless \
  python3 \
  python3-venv \
  python3-dev \
  python3-pip \
  build-essential \
  cmake \
  curl \
  unzip \
  htop \
  > /dev/null
echo "      ✓ System packages installed"

echo ""
echo "[2/6] Installing CPU-only PyTorch (saves ~2GB vs CUDA build)..."
# Install CPU wheels explicitly BEFORE requirements.txt to prevent pip from
# pulling down the enormous CUDA-enabled torch build.
sudo -u "$ACTUAL_USER" pip3 install --break-system-packages \
  torch \
  torchvision \
  torchaudio \
  --index-url https://download.pytorch.org/whl/cpu
echo "      ✓ PyTorch CPU wheels installed"

echo ""
echo "[3/6] Installing FedLearn Python dependencies (this may take a while)..."
if [[ -f "$HOME_DIR/requirements.txt" ]]; then
  sudo -u "$ACTUAL_USER" pip3 install --break-system-packages -r "$HOME_DIR/requirements.txt"
  echo "      ✓ Python dependencies installed"
else
  echo "      ⚠ WARNING: $HOME_DIR/requirements.txt not found."
  echo "        SCP it first: scp framework/requirements.txt ec2-user@<IP>:~/"
fi

echo ""
echo "[4/6] Creating app directory layout..."
mkdir -p "$APP_DIR"
mkdir -p "$SCRIPTS_DIR"
mkdir -p "$APP_DIR/models"
mkdir -p "$APP_DIR/logs"
mkdir -p "$APP_DIR/data"
chown -R "$ACTUAL_USER:$ACTUAL_USER" "$APP_DIR"
echo "      ✓ Directory layout created at $APP_DIR"

echo ""
echo "[5/6] Creating systemd service for FedLearn backend..."
# Create a systemd unit so the backend auto-restarts on crash/reboot
cat > /etc/systemd/system/fedlearn.service <<EOF
[Unit]
Description=FedLearn Platform Backend
After=network.target
Wants=network-online.target

[Service]
Type=simple
User=$ACTUAL_USER
WorkingDirectory=$APP_DIR
ExecStart=/usr/bin/java -Xmx4g -jar $APP_DIR/app.jar
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=fedlearn

# ── Non-secret settings (pre-configured, no changes needed) ─────────────────
Environment="SPRING_PROFILES_ACTIVE=ec2demo"
Environment="FEDLEARN_PYTHON=python3"
Environment="PYTHON_EXECUTABLE_PATH=$SCRIPTS_DIR/run_init_model.sh"
Environment="PYTHON_SCRIPT_FL_SERVER_PATH=$SCRIPTS_DIR/run_fl_server.sh"
Environment="FEATURE_LOG_PERSISTENCE=false"
Environment="FEATURE_ROUND_RESULTS=true"
# ── Secrets — YOU MUST FILL THESE IN before starting ─────────────────────────
# Generate APP_JWT_SECRET:     openssl rand -base64 64
# Generate APP_INTERNAL_API_KEY: openssl rand -hex 32
# Environment="APP_JWT_SECRET=CHANGE_ME"
# Environment="APP_INTERNAL_API_KEY=CHANGE_ME"
# Environment="CORS_ALLOWED_ORIGINS=http://localhost:5173"

[Install]
WantedBy=multi-user.target
EOF
systemctl daemon-reload
echo "      ✓ systemd service created (not started yet)"
echo "        Edit /etc/systemd/system/fedlearn.service to set env vars, then:"
echo "        sudo systemctl enable fedlearn && sudo systemctl start fedlearn"

echo ""
echo "[6/6] Verifying Java version..."
java -version 2>&1 | head -1
echo "      ✓ Java OK"

echo ""
echo "=============================================="
echo " Bootstrap complete!"
echo ""
echo " Next steps:"
echo "   1. SCP the JAR and scripts from your Mac (run deploy-to-aws.sh)"
echo "   2. Edit /etc/systemd/system/fedlearn.service — fill in env vars"
echo "   3. sudo systemctl enable fedlearn && sudo systemctl start fedlearn"
echo "   4. sudo journalctl -u fedlearn -f   (tail logs)"
echo "=============================================="
