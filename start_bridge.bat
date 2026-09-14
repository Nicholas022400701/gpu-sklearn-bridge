@echo off
REM Wrapper called by Task Scheduler / HKCU Run.
REM Starts WSL2 and launches the GPU bridge server in background.
REM SKLEARN_BRIDGE_HOME defaults to the directory of this script (Windows-side clone).
REM SKLEARN_BRIDGE_WSL_DISTRO defaults to Ubuntu.
if not defined SKLEARN_BRIDGE_HOME set "SKLEARN_BRIDGE_HOME=%~dp0"
if "%SKLEARN_BRIDGE_HOME:~-1%"=="\" set "SKLEARN_BRIDGE_HOME=%SKLEARN_BRIDGE_HOME:~0,-1%"
if not defined SKLEARN_BRIDGE_WSL_DISTRO set "SKLEARN_BRIDGE_WSL_DISTRO=Ubuntu"
wsl -d %SKLEARN_BRIDGE_WSL_DISTRO% --cd "%SKLEARN_BRIDGE_HOME%" -- bash ./start_server.sh
