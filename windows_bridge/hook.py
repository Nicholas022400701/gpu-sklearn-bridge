"""
import hook - 拦截 sklearn.* 导入，透明重定向到 WSL2 cuML GPU 服务
"""

import sys
import logging

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
#  sklearn → cuML 模块映射表
#  cuML 不支持的模块保持原样（走 CPU sklearn）
# ──────────────────────────────────────────────
SKLEARN_TO_CUML = {
    "sklearn":                      "cuml",
    "sklearn.cluster":              "cuml.cluster",
    "sklearn.datasets":             "cuml.datasets",
    "sklearn.decomposition":        "cuml.decomposition",
    "sklearn.ensemble":             "cuml.ensemble",
    "sklearn.feature_extraction":   "cuml.feature_extraction",
    "sklearn.linear_model":         "cuml.linear_model",
    "sklearn.manifold":             "cuml.manifold",
    "sklearn.metrics":              "cuml.metrics",
    "sklearn.model_selection":      "cuml.model_selection",
    "sklearn.neighbors":            "cuml.neighbors",
    "sklearn.pipeline":             "cuml.pipeline",
    "sklearn.preprocessing":        "cuml.preprocessing",
    "sklearn.svm":                  "cuml.svm",
    "sklearn.random_projection":    "cuml.random_projection",
    "sklearn.solvers":              "cuml.solvers",
}

# ──────────────────────────────────────────────
#  连接状态
# ──────────────────────────────────────────────
_connection = None
_server_available = None   # None=未知  True=在线  False=离线

BRIDGE_HOST = "localhost"
BRIDGE_PORT = 18861


def _get_connection():
    """
    获取到 WSL2 rpyc 服务端的连接（懒初始化，自动重连）。
    若服务不在线则返回 None，不抛出异常。
    """
    global _connection, _server_available

    # 已知不可用 → 快速失败，避免每次 import 都 timeout
    if _server_available is False:
        return None

    try:
        import rpyc

        # 连接已建立且未关闭
        if _connection is not None:
            try:
                _connection.root.ping()          # 心跳
                return _connection
            except Exception:
                _connection = None               # 连接已断开，重新建立

        _connection = rpyc.connect(
            BRIDGE_HOST,
            BRIDGE_PORT,
            config={
                "allow_all_attrs":    True,
                "allow_setattr":      True,
                "allow_delattr":      True,
                "allow_pickle":       True,
                "sync_request_timeout": 120,
            },
        )
        _server_available = True
        logger.debug("🔗 已连接到 GPU sklearn bridge (WSL2:%d)", BRIDGE_PORT)
        return _connection

    except Exception as e:
        _server_available = False
        logger.debug("GPU bridge 不可用，回退到 CPU sklearn: %s", e)
        return None


class _GPUSklearnFinder:
    """
    sys.meta_path finder：拦截 sklearn.* 的 import 请求，
    若 GPU bridge 在线则返回 cuML 的远程模块引用。
    """

    def find_module(self, fullname, path=None):
        if fullname in SKLEARN_TO_CUML:
            return self
        return None

    def load_module(self, fullname):
        # 已缓存则直接返回
        if fullname in sys.modules:
            return sys.modules[fullname]

        conn = _get_connection()
        if conn is None:
            # 服务离线 → 不拦截，让 Python 继续正常导入 CPU sklearn
            return None

        cuml_name = SKLEARN_TO_CUML[fullname]
        try:
            remote_mod = conn.root.get_module(cuml_name)
            sys.modules[fullname] = remote_mod
            logger.debug("🚀 %s → GPU (%s via WSL2)", fullname, cuml_name)
            return remote_mod
        except Exception as e:
            logger.debug("无法从 bridge 获取 %s: %s，回退到 CPU", cuml_name, e)
            return None


_finder_instance = _GPUSklearnFinder()
_hook_installed = False


def install():
    """安装 import hook（幂等，多次调用无副作用）"""
    global _hook_installed
    if _hook_installed:
        return
    # 插到 meta_path 最前面，优先拦截
    sys.meta_path.insert(0, _finder_instance)
    _hook_installed = True
    logger.debug("GPU sklearn import hook 已安装")
