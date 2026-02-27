@echo off
REM Wrapper called by Task Scheduler.
REM Starts WSL2 Ubuntu and launches the GPU bridge server in background.
wsl -d Ubuntu -- bash /mnt/c/Users/nicho/gpu-sklearn-bridge/start_server.sh
