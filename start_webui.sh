#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# start_webui.sh – Launch the cuML WebUI training system
#
# Usage:
#   chmod +x start_webui.sh
#   ./start_webui.sh          # default port 7860
#   WEBUI_PORT=8080 ./start_webui.sh
#
# The WebUI automatically connects to the cuML bridge server at :19876
# if it is already running.  Otherwise it falls back to scikit-learn.
# ─────────────────────────────────────────────────────────────────────────────
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PORT="${WEBUI_PORT:-7860}"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  cuML WebUI – GPU Training System"
echo "  http://0.0.0.0:${PORT}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Install Python deps if needed
if ! python -c "import flask_socketio" 2>/dev/null; then
    echo "[setup] Installing webui requirements…"
    pip install -r webui/requirements.txt -q
fi

# Create required directories
mkdir -p logs models uploads

# Start
WEBUI_PORT="$PORT" python webui/app.py
