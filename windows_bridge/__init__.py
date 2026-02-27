"""
gpu_sklearn_bridge - Windows 端 import hook
将 sklearn 的导入透明重定向到 WSL2 中运行的 cuML GPU 服务。

工作原理：
  1. 本包通过 .pth 文件在 Python 启动时自动加载
  2. 安装 sys.meta_path hook，拦截所有 sklearn.* 导入
  3. 通过 rpyc 连接 WSL2 localhost:18861 取得 cuML 的远程引用
  4. 将 cuML 模块注入为 sklearn.* —— 用户代码无需任何改动

用法：
  正常使用即可，无需任何修改：
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression

查看状态：
    import gpu_sklearn_bridge
    gpu_sklearn_bridge.status()
"""

from .hook import install, _get_connection, SKLEARN_TO_CUML

__version__ = "1.0.0"
__all__ = ["status", "is_server_running", "install"]


def is_server_running() -> bool:
    """检查 WSL2 GPU bridge 服务是否在线"""
    conn = _get_connection()
    if conn is None:
        return False
    try:
        return conn.root.ping() == "pong"
    except Exception:
        return False


def status():
    """打印 GPU bridge 当前状态"""
    if is_server_running():
        try:
            conn = _get_connection()
            version = conn.root.get_cuml_version()
            print(f"✅ GPU bridge: 在线 (cuML {version})")
            print(f"   所有 sklearn.* 导入将自动使用 GPU 加速 (WSL2)")
            print(f"   已映射模块:")
            for win_mod, gpu_mod in SKLEARN_TO_CUML.items():
                print(f"     {win_mod}  →  {gpu_mod}")
        except Exception as e:
            print(f"✅ GPU bridge: 在线 (无法获取版本: {e})")
    else:
        print(f"❌ GPU bridge: 离线")
        print(f"   sklearn 导入将回退到 CPU 版本（标准 scikit-learn）")
        print(f"   请检查 WSL2 服务是否启动：wsl -d Ubuntu -- ~/gpu-sklearn-bridge/start.sh")
