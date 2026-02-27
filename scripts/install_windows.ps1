# GPU sklearn bridge - Windows 端一键安装脚本
# 执行内容：
#   1. 在 C:\Users\nicho\envs\gpu-sklearn (已有 uv venv) 中安装 rpyc
#   2. 将 windows_bridge 包复制到该 venv 的 site-packages
#   3. 创建 .pth 文件实现 Python 启动时自动加载 hook
#   4. 注册 Task Scheduler 开机自启任务

$ErrorActionPreference = "Stop"
$BridgeRoot    = "C:\Users\nicho\gpu-sklearn-bridge"
$VenvPython    = "C:\Users\nicho\envs\gpu-sklearn\Scripts\python.exe"
$VenvSitePkg   = "C:\Users\nicho\envs\gpu-sklearn\Lib\site-packages"
$TaskXml       = "$BridgeRoot\scripts\GPU_sklearn_bridge.xml"
$TaskName      = "GPU_sklearn_bridge"

Write-Host "========================================"
Write-Host "  GPU sklearn bridge - Windows 安装"
Write-Host "========================================"

# ── 1. 验证 uv venv 存在 ──────────────────────────
if (-not (Test-Path $VenvPython)) {
    Write-Host "[1/4] 创建 uv 虚拟环境..."
    uv venv "C:\Users\nicho\envs\gpu-sklearn" --python 3.11
} else {
    Write-Host "[1/4] uv 虚拟环境已存在: C:\Users\nicho\envs\gpu-sklearn"
}

# ── 2. 安装 rpyc ──────────────────────────────────
Write-Host "[2/4] 安装 rpyc..."
uv pip install --python $VenvPython rpyc scikit-learn
Write-Host "  ✅ rpyc + scikit-learn (CPU fallback) 安装完成"

# ── 3. 复制 bridge 包并创建 .pth 自动加载 ─────────
Write-Host "[3/4] 部署 bridge 包到 site-packages..."

$BridgeSrc = "$BridgeRoot\windows_bridge"
$BridgeDst = "$VenvSitePkg\gpu_sklearn_bridge"

if (Test-Path $BridgeDst) {
    Remove-Item $BridgeDst -Recurse -Force
}
Copy-Item $BridgeSrc $BridgeDst -Recurse

# .pth 文件：Python 启动时自动执行，安装 import hook
$PthContent = "import gpu_sklearn_bridge; gpu_sklearn_bridge.install()"
$PthContent | Out-File -FilePath "$VenvSitePkg\gpu_sklearn_bridge.pth" -Encoding UTF8 -NoNewline

Write-Host "  ✅ bridge 包部署完成"
Write-Host "  ✅ .pth 自动加载 hook 已配置"

# ── 4. 注册 Task Scheduler 开机自启任务 ───────────
Write-Host "[4/4] 注册 Task Scheduler 任务..."

# 先删除旧任务（如果存在）
schtasks /Delete /TN $TaskName /F 2>$null | Out-Null

# 注册新任务
schtasks /Create /XML $TaskXml /TN $TaskName /F
if ($LASTEXITCODE -eq 0) {
    Write-Host "  ✅ Task Scheduler 任务 '$TaskName' 注册成功"
    Write-Host "  → 下次登录时将自动启动 GPU bridge"
} else {
    Write-Host "  ⚠️  Task Scheduler 注册失败，请手动运行:"
    Write-Host "     schtasks /Create /XML '$TaskXml' /TN $TaskName /F"
}

Write-Host ""
Write-Host "========================================"
Write-Host "  ✅ Windows 端安装完成！"
Write-Host "========================================"
Write-Host ""
Write-Host "下一步: 运行 WSL2 端安装脚本"
Write-Host "  wsl -d Ubuntu -u nicho -- bash /mnt/c/Users/nicho/gpu-sklearn-bridge/wsl_server/setup.sh"
Write-Host ""
Write-Host "安装完成后立即启动服务（无需重启）:"
Write-Host "  & '$BridgeRoot\scripts\start_bridge.ps1'"
Write-Host ""
Write-Host "验证 GPU bridge 状态:"
Write-Host "  & '$VenvPython' -c `"import gpu_sklearn_bridge; gpu_sklearn_bridge.status()`""
