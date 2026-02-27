"""
Core proxy machinery – handles HTTP communication with the WSL2 bridge server.
"""
import os
import sys
import time
import base64
import uuid
import importlib.util
from pathlib import Path

import numpy as np
import requests

# ── 加载 mmap 传输层 ────────────────────────────────────────────────────────────────
_shm_file = Path(__file__).parent.parent / "shm_transport.py"
_shm_spec = importlib.util.spec_from_file_location("shm_transport", _shm_file)
_shm_mod  = importlib.util.module_from_spec(_shm_spec)
_shm_spec.loader.exec_module(_shm_mod)
ShmTransport  = _shm_mod.ShmTransport
SLOT_INPUT_START = _shm_mod.SLOT_INPUT_START
MMAP_THRESHOLD = _shm_mod.MMAP_THRESHOLD

_BRIDGE_PORT = int(os.environ.get("SKLEARN_BRIDGE_PORT", 19876))
_BRIDGE_URL = f"http://127.0.0.1:{_BRIDGE_PORT}"
_TIMEOUT = 30  # seconds per request

# 共享文件传输路径（Windows 视角）
_SHARED_DIR_WIN = Path(os.environ.get("SKLEARN_BRIDGE_SHARED",
    r"C:\Users\nicho\gpu-sklearn-bridge\shm"))
_MODELS_DIR_WIN = Path(os.environ.get("SKLEARN_BRIDGE_MODELS",
    r"C:\Users\nicho\gpu-sklearn-bridge\models"))
_ARRAY_FILE_THRESHOLD = 10 * 1024  # 10 KB：超过此大小写共享文件

# 专用 Session，强制绕过系统代理（HTTP_PROXY / HTTPS_PROXY 等环境变量）
# 避免 Clash / V2Ray 等代理软件拦截发往 localhost 的桥接请求
_session = requests.Session()
_session.trust_env = False          # 不读取环境变量里的代理
_session.proxies = {"http": None, "https": None}  # 显式清空


def wait_for_server(timeout: int = 30) -> bool:
    """Block until the bridge server responds or timeout."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = _session.get(f"{_BRIDGE_URL}/health", timeout=2)
            if r.status_code == 200:
                return True
        except Exception:
            time.sleep(1)
    return False


# ── Array serialisation ───────────────────────────────────────────────────────

def _encode_array(arr):
    if isinstance(arr, np.ndarray):
        if arr.nbytes >= MMAP_THRESHOLD:
            # 自动分配输入 slot（轮转）
            meta = ShmTransport.get().write(arr)
            return meta
        # 小数组 inline base64
        return {
            "__ndarray__": True,
            "data": base64.b64encode(arr.tobytes()).decode(),
            "dtype": str(arr.dtype),
            "shape": list(arr.shape),
        }
    return arr


def _decode_array(obj):
    if isinstance(obj, list):
        items = [_decode_array(item) for item in obj]
        # 若列表元素全部是 ndarray（来自 tuple 结果如 kneighbors），还原为 tuple
        if items and all(isinstance(i, np.ndarray) for i in items):
            return tuple(items)
        return items
    if isinstance(obj, dict):
        if obj.get("__ndarray__"):
            data = base64.b64decode(obj["data"])
            return np.frombuffer(data, dtype=obj["dtype"]).reshape(obj["shape"])
        if obj.get("__mmap__"):
            return ShmTransport.get().read(obj)
        if obj.get("__file__"):
            # 备用分支：老服务端可能仍然返回文件引用
            path = _MODELS_DIR_WIN.parent / "shm" / obj["name"]
            arr = np.load(path)
            try:
                path.unlink()
            except OSError:
                pass
            return arr
    return obj


def _encode_args(args, kwargs):
    return (
        [_encode_array(a) for a in args],
        {k: _encode_array(v) for k, v in kwargs.items()},
    )


# ── ProxyEstimator ────────────────────────────────────────────────────────────

class ProxyEstimator:
    """
    A scikit-learn–compatible estimator that delegates all computation
    to the cuML GPU server running inside WSL2.
    """

    def __init__(self, _class_name: str, _preloaded_id: str = None, **params):
        self._class_name = _class_name
        self._params = params
        self._model_id: str | None = _preloaded_id
        if self._model_id is None:
            self._ensure_created()

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def _ensure_created(self):
        if self._model_id is not None:
            return
        resp = _session.post(
            f"{_BRIDGE_URL}/create",
            json={"class": self._class_name, "params": self._params},
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        if "error" in data:
            raise RuntimeError(f"[cuml_proxy] Server error: {data['error']}\n{data.get('traceback','')}")
        self._model_id = data["model_id"]

    def _call(self, method: str, *args, **kwargs):
        self._ensure_created()
        enc_args, enc_kwargs = _encode_args(args, kwargs)
        resp = _session.post(
            f"{_BRIDGE_URL}/call/{self._model_id}/{method}",
            json={"args": enc_args, "kwargs": enc_kwargs},
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        if "error" in data:
            raise RuntimeError(f"[cuml_proxy] {data['error']}\n{data.get('traceback','')}")
        return _decode_array(data["result"])

    def __del__(self):
        if self._model_id:
            try:
                _session.delete(f"{_BRIDGE_URL}/delete/{self._model_id}", timeout=2)
            except Exception:
                pass

    # ── sklearn API ───────────────────────────────────────────────────────────

    def fit(self, X, y=None, **kwargs):
        self._call("fit", X, y, **kwargs) if y is not None else self._call("fit", X, **kwargs)
        return self

    def predict(self, X, **kwargs):
        return self._call("predict", X, **kwargs)

    def transform(self, X, **kwargs):
        return self._call("transform", X, **kwargs)

    def fit_transform(self, X, y=None, **kwargs):
        return self._call("fit_transform", X, y, **kwargs) if y is not None else self._call("fit_transform", X, **kwargs)

    def fit_predict(self, X, y=None, **kwargs):
        return self._call("fit_predict", X, y, **kwargs) if y is not None else self._call("fit_predict", X, **kwargs)

    def inverse_transform(self, X, **kwargs):
        return self._call("inverse_transform", X, **kwargs)

    def score(self, X, y, **kwargs):
        return self._call("score", X, y, **kwargs)

    def get_params(self, deep=True):
        return dict(self._params)

    def set_params(self, **params):
        self._params.update(params)
        # recreate model with new params
        if self._model_id:
            try:
                _session.delete(f"{_BRIDGE_URL}/delete/{self._model_id}", timeout=2)
            except Exception:
                pass
            self._model_id = None
        self._ensure_created()
        return self

    # ── 持久化 ────────────────────────────────────────────────────────────────

    def save(self, name: str) -> str:
        """将模型 pickle 到共享目录，之后可用 ProxyEstimator.load(name) 恢复。
        name：不含扩展名的文件名，例如 'my_svm_model'。
        """
        self._ensure_created()
        resp = _session.post(
            f"{_BRIDGE_URL}/save/{self._model_id}",
            json={"name": name},
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        if "error" in data:
            raise RuntimeError(f"[cuml_proxy] save failed: {data['error']}")
        win_path = str(_MODELS_DIR_WIN / f"{name}.pkl")
        print(f"[cuml_proxy] 模型已保存 → {win_path}")
        return win_path

    @classmethod
    def load(cls, name: str) -> "ProxyEstimator":
        """从共享目录加载已保存的模型，返回可立即使用的 ProxyEstimator 实例。
        name：与 save() 时相同的名称，不含扩展名。
        """
        resp = _session.post(
            f"{_BRIDGE_URL}/load",
            json={"name": name},
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        if "error" in data:
            raise RuntimeError(f"[cuml_proxy] load failed: {data['error']}")
        obj = object.__new__(cls)
        obj._class_name = data["class"]
        obj._params     = data.get("params", {})
        obj._model_id   = data["model_id"]
        print(f"[cuml_proxy] 模型已加载: {obj._class_name} (id={obj._model_id})")
        return obj

    @staticmethod
    def list_saved() -> list:
        """列出共享目录中已保存的所有模型名称。"""
        resp = _session.get(f"{_BRIDGE_URL}/list_models", timeout=_TIMEOUT)
        resp.raise_for_status()
        return resp.json().get("models", [])

    def __repr__(self):
        params_str = ", ".join(f"{k}={v!r}" for k, v in self._params.items())
        return f"{self._class_name}({params_str}) [GPU via WSL2]"
