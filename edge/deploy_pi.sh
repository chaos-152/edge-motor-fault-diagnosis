#!/usr/bin/env bash
# ==============================================================================
# Turnkey Deployment Script for Raspberry Pi Model 4 Edge Node
# Project: Edge-Based Fault Diagnosis of Induction Motors
# Author: Sai Samanyu K (231CS152) - NITK
# Ref: "Edge vs. Cloud: Empirical Insights into Data-Driven Condition Monitoring"
# ==============================================================================

set -e

echo "================================================================="
echo "   Setting up Induction Motor Fault Diagnosis on Raspberry Pi   "
echo "================================================================="

# 1. System package dependencies for Raspberry Pi OS (Debian-based)
echo "[1/4] Checking and installing system packages..."
sudo apt-get update -y
sudo apt-get install -y python3-pip python3-venv python3-numpy python3-scipy libatlas-base-dev

# 2. Setup isolated Virtual Environment on the Pi
echo "[2/4] Setting up Python virtual environment..."
PI_PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PI_PROJECT_DIR"

if [ ! -d ".venv_pi" ]; then
    python3 -m venv --system-site-packages .venv_pi
    echo "Virtual environment .venv_pi created."
fi

source .venv_pi/bin/activate

# 3. Install lightweight inference requirements
echo "[3/4] Installing Python edge inference libraries..."
pip install --upgrade pip
pip install scikit-learn pandas psutil joblib matplotlib

# 4. Create systemd Service for headless industrial monitoring
SERVICE_FILE="/etc/systemd/system/induction-motor-monitor.service"
echo "[4/4] Configuring headless background service at $SERVICE_FILE..."

sudo bash -c "cat <<EOF > $SERVICE_FILE
[Unit]
Description=Edge-Based Induction Motor Fault Monitor (Raspberry Pi)
After=network.target

[Service]
Type=simple
User=$(whoami)
WorkingDirectory=$PI_PROJECT_DIR
ExecStart=$PI_PROJECT_DIR/.venv_pi/bin/python $PI_PROJECT_DIR/edge/edge_runtime.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF"

echo "Reloading systemd daemon..."
sudo systemctl daemon-reload
echo "Service configured. To enable and start automatically on boot:"
echo "  sudo systemctl enable induction-motor-monitor.service"
echo "  sudo systemctl start induction-motor-monitor.service"

echo ""
echo "================================================================="
echo "  Raspberry Pi Edge Node Deployment Complete!                    "
echo "  To run interactive diagnosis now:                              "
echo "    source .venv_pi/bin/activate                                 "
echo "    python edge/edge_runtime.py                                  "
echo "================================================================="
