#!/usr/bin/env bash
# WSL2 端安装脚本：安装 uv、创建 GPU Python 环境、安装 cuML + rpyc
set -e

VENV_PATH="$HOME/.venv/gpu-sklearn"
SERVER_DIR="$HOME/gpu-sklearn-bridge"
LOG_FILE="/tmp/gpu_sklearn_setup.log"

echo "========================================"
echo "  GPU sklearn bridge - WSL2 安装脚本"
echo "========================================"

# 1. 安装 uv（如果未安装）
if ! command -v uv &> /dev/null; then
    echo "[1/5] 安装 uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
    # 持久化到 .bashrc
    if ! grep -q 'export PATH=.*\.local/bin' ~/.bashrc; then
        echo 'export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"' >> ~/.bashrc
    fi
    echo "✅ uv 安装完成: $(uv --version)"
else
    echo "[1/5] uv 已安装: $(uv --version)"
fi

export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"

# 2. 创建 Python 3.11 虚拟环境（cuML 支持最好）
echo "[2/5] 创建 Python 3.11 虚拟环境: $VENV_PATH"
uv venv "$VENV_PATH" --python 3.11
echo "✅ 虚拟环境创建完成"

# 3. 安装 cuML（CUDA 12.x）和 rpyc
echo "[3/5] 安装 cuML (CUDA 12) + rpyc，这可能需要几分钟..."
uv pip install --python "$VENV_PATH/bin/python" \
    "cuml-cu12" \
    "rpyc" \
    "numpy" \
    --extra-index-url https://pypi.nvidia.com \
    --index-strategy unsafe-best-match
echo "✅ cuML + rpyc 安装完成"

# 4. 部署服务端脚本
echo "[4/5] 部署服务端脚本..."
mkdir -p "$SERVER_DIR"

# 从 Windows 挂载路径复制（或内联写入）
WINDOWS_SERVER="/mnt/c/Users/nicho/gpu-sklearn-bridge/wsl_server/server.py"
if [ -f "$WINDOWS_SERVER" ]; then
    cp "$WINDOWS_SERVER" "$SERVER_DIR/server.py"
    echo "✅ 从 Windows 复制 server.py"
else
    echo "⚠️  未找到 Windows 侧 server.py，请手动复制"
fi

# 5. 创建快速启动脚本
echo "[5/5] 创建启动脚本..."
cat > "$SERVER_DIR/start.sh" << 'EOF'
#!/usr/bin/env bash
VENV_PATH="$HOME/.venv/gpu-sklearn"
SERVER_DIR="$HOME/gpu-sklearn-bridge"

# 检查是否已在运行
if [ -f /tmp/cuml_server.pid ]; then
    OLD_PID=$(cat /tmp/cuml_server.pid)
    if kill -0 "$OLD_PID" 2>/dev/null; then
        echo "服务已在运行 (PID: $OLD_PID)"
        exit 0
    fi
fi

export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
nohup "$VENV_PATH/bin/python" "$SERVER_DIR/server.py" \
    >> /tmp/cuml_server.log 2>&1 &

echo "GPU sklearn bridge 已启动 (PID: $!)"
echo "日志: /tmp/cuml_server.log"
EOF
chmod +x "$SERVER_DIR/start.sh"

cat > "$SERVER_DIR/stop.sh" << 'EOF'
#!/usr/bin/env bash
if [ -f /tmp/cuml_server.pid ]; then
    PID=$(cat /tmp/cuml_server.pid)
    kill "$PID" 2>/dev/null && echo "服务已停止 (PID: $PID)" || echo "进程不存在"
    rm -f /tmp/cuml_server.pid
else
    pkill -f "cuml_server" && echo "服务已停止" || echo "没有运行中的服务"
fi
EOF
chmod +x "$SERVER_DIR/stop.sh"

echo ""
echo "========================================"
echo "  ✅ 安装完成！"
echo "========================================"
echo ""
echo "手动启动:  $SERVER_DIR/start.sh"
echo "手动停止:  $SERVER_DIR/stop.sh"
echo "日志位置:  /tmp/cuml_server.log"
echo ""
echo "Windows 开机自启已通过 Task Scheduler 配置，"
echo "登录后约 15 秒服务自动可用。"
