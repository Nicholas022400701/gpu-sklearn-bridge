"""
cuml (Windows shim) — 将 `import cuml` 透明地转发到 cuml_proxy。

在 Windows 上 RAPIDS cuML 没有原生 wheel，本包通过 WSL2 桥接
提供与原版 cuml 相同的导入路径，计算仍在 GPU 上完成。

用法与真正的 cuml 完全一致：
    from cuml.svm import SVC
    from cuml.preprocessing import StandardScaler
    import cuml.linear_model
"""
import sys
from cuml_proxy import (
    linear_model,
    cluster,
    decomposition,
    neighbors,
    ensemble,
    svm,
    preprocessing,
    manifold,
)

# 将子模块注册到 sys.modules，这样 `from cuml.svm import SVC` 才能正确解析
_submodules = {
    "cuml.linear_model":  linear_model,
    "cuml.cluster":       cluster,
    "cuml.decomposition": decomposition,
    "cuml.neighbors":     neighbors,
    "cuml.ensemble":      ensemble,
    "cuml.svm":           svm,
    "cuml.preprocessing": preprocessing,
    "cuml.manifold":      manifold,
}
for _name, _mod in _submodules.items():
    sys.modules.setdefault(_name, _mod)

# 让 cuml.__version__ 可用（从桥接服务查询）
try:
    from cuml_proxy.proxy import _session, _BRIDGE_URL
    _r = _session.get(f"{_BRIDGE_URL}/health", timeout=2)
    __version__ = _r.json().get("cuml_version", "unknown")
except Exception:
    __version__ = "unknown (bridge not running)"
