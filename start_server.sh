#!/usr/bin/env bash
# Runs inside WSL2 – starts the GPU sklearn bridge server.
# Called by Windows Task Scheduler at logon.

LOGFILE="$HOME/gpu-sklearn-bridge/server.log"
PIDFILE="$HOME/gpu-sklearn-bridge/server.pid"
PYTHON="$HOME/envs/gpu-sklearn/bin/python"
SERVER="$HOME/gpu-sklearn-bridge/server.py"

# Kill any previous instance
if [ -f "$PIDFILE" ]; then
    old_pid=$(cat "$PIDFILE")
    kill "$old_pid" 2>/dev/null
fi

echo "[$(date)] Starting gpu-sklearn-bridge..." >> "$LOGFILE"
nohup "$PYTHON" "$SERVER" >> "$LOGFILE" 2>&1 &
echo $! > "$PIDFILE"
echo "[$(date)] PID=$(cat $PIDFILE)" >> "$LOGFILE"
