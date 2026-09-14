#!/usr/bin/env bash
# Runs inside WSL2 – starts the GPU sklearn bridge server.
# Called by Windows Task Scheduler at logon.
# Optional overrides (defaults keep the original behaviour):
#   SKLEARN_BRIDGE_HOME    WSL2-side clone of this repo   (default: $HOME/gpu-sklearn-bridge)
#   SKLEARN_BRIDGE_PYTHON  interpreter with cuML + flask  (default: $HOME/envs/gpu-sklearn/bin/python)

BRIDGE_HOME="${SKLEARN_BRIDGE_HOME:-$HOME/gpu-sklearn-bridge}"
LOGFILE="$BRIDGE_HOME/server.log"
PIDFILE="$BRIDGE_HOME/server.pid"
PYTHON="${SKLEARN_BRIDGE_PYTHON:-$HOME/envs/gpu-sklearn/bin/python}"
SERVER="$BRIDGE_HOME/server.py"

# Kill any previous instance
if [ -f "$PIDFILE" ]; then
    old_pid=$(cat "$PIDFILE")
    kill "$old_pid" 2>/dev/null
fi

echo "[$(date)] Starting gpu-sklearn-bridge..." >> "$LOGFILE"
nohup "$PYTHON" "$SERVER" >> "$LOGFILE" 2>&1 &
echo $! > "$PIDFILE"
echo "[$(date)] PID=$(cat $PIDFILE)" >> "$LOGFILE"
