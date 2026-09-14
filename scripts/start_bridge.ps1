# GPU sklearn bridge - Windows 启动脚本
# 触发: 用户登录时由 Task Scheduler 自动调用
# 功能: 启动 WSL2 并在其中运行 cuML rpyc 服务端
# 环境变量（可选，未设置时与原脚本行为一致）：
#   SKLEARN_BRIDGE_HOME        仓库根目录，默认为本脚本所在目录的上一级
#   SKLEARN_BRIDGE_WSL_DISTRO  WSL2 发行版，默认 Ubuntu
#   SKLEARN_BRIDGE_WSL_USER    WSL2 用户名，默认与 Windows 用户名相同

param(
    [switch]$Stop,
    [switch]$Status,
    [switch]$Restart
)

$LogFile = "$env:TEMP\gpu_sklearn_bridge.log"
$PidFile = "$env:TEMP\gpu_sklearn_bridge_wsl.pid"
$ServerPort = 18861
$BridgeRoot = if ($env:SKLEARN_BRIDGE_HOME) { $env:SKLEARN_BRIDGE_HOME } else { Split-Path -Parent $PSScriptRoot }
$WslDistro  = if ($env:SKLEARN_BRIDGE_WSL_DISTRO) { $env:SKLEARN_BRIDGE_WSL_DISTRO } else { "Ubuntu" }
$WslUser    = if ($env:SKLEARN_BRIDGE_WSL_USER) { $env:SKLEARN_BRIDGE_WSL_USER } else { $env:USERNAME }

function Write-Log {
    param($Message)
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $line = "$ts  $Message"
    Add-Content -Path $LogFile -Value $line
    Write-Host $line
}

function Test-ServerRunning {
    try {
        $tcp = New-Object System.Net.Sockets.TcpClient
        $tcp.ConnectAsync("localhost", $ServerPort).Wait(2000) | Out-Null
        $running = $tcp.Connected
        $tcp.Close()
        return $running
    } catch {
        return $false
    }
}

function Start-Bridge {
    if (Test-ServerRunning) {
        Write-Log "GPU bridge 已在运行 (端口 $ServerPort)"
        return
    }

    Write-Log "正在启动 GPU sklearn bridge..."
    Write-Log "  → 启动 WSL2 $WslDistro 中的 cuML 服务端..."

    # 以后台方式启动 WSL2 服务端，不弹窗口；'$HOME' 保持字面，在 WSL2 内展开为该用户的 home
    $proc = Start-Process -FilePath "wsl.exe" -ArgumentList @(
        "-d", $WslDistro,
        "-u", $WslUser,
        "--",
        "bash", "-lc",
        '$HOME/gpu-sklearn-bridge/start.sh'
    ) -WindowStyle Hidden -PassThru

    # 等待服务就绪（最多 30 秒）
    $waited = 0
    $ready = $false
    while ($waited -lt 30) {
        Start-Sleep -Seconds 2
        $waited += 2
        if (Test-ServerRunning) {
            $ready = $true
            break
        }
        Write-Log "  等待服务就绪... ($waited/30 秒)"
    }

    if ($ready) {
        Write-Log "✅ GPU sklearn bridge 启动成功！(端口 $ServerPort)"
        $proc.Id | Out-File -FilePath $PidFile -Force
    } else {
        Write-Log "⚠️  服务在 30 秒内未就绪，请检查日志："
        Write-Log "    WSL2 日志: \\wsl.localhost\$WslDistro\tmp\cuml_server.log"
        Write-Log "    Windows 日志: $LogFile"
    }
}

function Stop-Bridge {
    Write-Log "正在停止 GPU sklearn bridge..."
    wsl.exe -d $WslDistro -u $WslUser -- bash -lc '$HOME/gpu-sklearn-bridge/stop.sh'
    Remove-Item -Path $PidFile -Force -ErrorAction SilentlyContinue
    Write-Log "服务已停止"
}

function Get-Status {
    if (Test-ServerRunning) {
        Write-Host "✅ GPU bridge: 在线 (localhost:$ServerPort)"
    } else {
        Write-Host "❌ GPU bridge: 离线"
        Write-Host "   手动启动: & '$BridgeRoot\scripts\start_bridge.ps1'"
    }
}

# 主逻辑
if ($Stop) {
    Stop-Bridge
} elseif ($Status) {
    Get-Status
} elseif ($Restart) {
    Stop-Bridge
    Start-Sleep -Seconds 2
    Start-Bridge
} else {
    Start-Bridge
}
