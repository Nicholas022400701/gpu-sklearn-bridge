#!/usr/bin/env python3
"""
GPU sklearn bridge server - 运行在 WSL2 中，将 cuML 暴露给 Windows Python
端口: 18861
"""

import sys
import importlib
import logging
import signal
import os

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("/tmp/cuml_server.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

PORT = 18861


def main():
    # 先验证 cuML 可以导入
    logger.info("正在加载 cuML...")
    try:
        import cuml
        logger.info(f"✅ cuML {cuml.__version__} 加载成功，GPU 可用")
    except ImportError as e:
        logger.error(f"❌ cuML 导入失败: {e}")
        sys.exit(1)

    try:
        import rpyc
        from rpyc.utils.server import ThreadedServer
    except ImportError:
        logger.error("❌ rpyc 未安装，请运行: pip install rpyc")
        sys.exit(1)

    class GPUSklearnService(rpyc.Service):
        ALIASES = ["GPUSklearn"]

        def on_connect(self, conn):
            client_addr = getattr(conn._channel.stream.sock, "getpeername", lambda: "?")()
            logger.info(f"客户端连接: {client_addr}")

        def on_disconnect(self, conn):
            logger.info("客户端断开连接")

        def exposed_ping(self):
            """心跳检测"""
            return "pong"

        def exposed_get_cuml_version(self):
            import cuml
            return cuml.__version__

        def exposed_get_module(self, module_name: str):
            """
            返回 cuML 模块的远程引用。
            Windows 端通过此接口获取任意 cuml 子模块。
            """
            return importlib.import_module(module_name)

        def exposed_list_cuml_modules(self):
            """列出所有可用的 cuML 子模块"""
            import cuml
            import pkgutil
            return [
                name for _, name, _ in pkgutil.walk_packages(
                    cuml.__path__, prefix="cuml."
                )
            ]

    # 优雅退出
    def _shutdown(sig, frame):
        logger.info("收到退出信号，关闭服务...")
        sys.exit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    server = ThreadedServer(
        GPUSklearnService,
        hostname="0.0.0.0",
        port=PORT,
        protocol_config={
            "allow_all_attrs": True,
            "allow_setattr": True,
            "allow_delattr": True,
            "allow_pickle": True,
            "sync_request_timeout": 300,
        },
    )

    logger.info(f"🚀 GPU sklearn bridge 已启动，监听端口 {PORT}")
    logger.info(f"   Windows 端可通过 localhost:{PORT} 连接")

    # 写入 PID 文件供管理脚本使用
    pid_file = "/tmp/cuml_server.pid"
    with open(pid_file, "w") as f:
        f.write(str(os.getpid()))

    server.start()


if __name__ == "__main__":
    main()
