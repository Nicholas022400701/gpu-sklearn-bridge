#!/usr/bin/env python3
"""
GPU sklearn bridge server - runs in WSL2, exposes cuML via HTTP JSON-RPC.
Supports: fit, predict, transform, fit_predict, fit_transform, score, get_params
"""
import json
import pickle
import base64
import traceback
import os
import sys
import uuid

from flask import Flask, request, jsonify

# ── mmap 共享内存传输 ────────────────────────────────────────────────────────────────
import importlib, importlib.util
_shm_spec = importlib.util.spec_from_file_location(
    "shm_transport",
    os.path.join(os.path.dirname(__file__), "shm_transport.py")
)
_shm_mod = importlib.util.module_from_spec(_shm_spec)
_shm_spec.loader.exec_module(_shm_mod)
ShmTransport = _shm_mod.ShmTransport
SLOT_INPUT_START   = _shm_mod.SLOT_INPUT_START
SLOT_OUTPUT_START  = _shm_mod.SLOT_OUTPUT_START
MMAP_THRESHOLD = _shm_mod.MMAP_THRESHOLD

# ── cuML imports ──────────────────────────────────────────────────────────────
import cuml
import cuml.linear_model
import cuml.cluster
import cuml.decomposition
import cuml.neighbors
import cuml.ensemble
import cuml.svm
import cuml.preprocessing
import cuml.manifold
import numpy as np

app = Flask(__name__)

# ── 共享文件系统路径（Windows 和 WSL2 均可访问相同物理位置）────────────────────
SHARED_DIR = os.environ.get("SKLEARN_BRIDGE_SHARED",
    "/mnt/c/Users/nicho/gpu-sklearn-bridge/shm")
MODELS_DIR = os.environ.get("SKLEARN_BRIDGE_MODELS",
    "/mnt/c/Users/nicho/gpu-sklearn-bridge/models")
ARRAY_FILE_THRESHOLD = 10 * 1024  # 10 KB：超过此大小写共享文件，避免 base64 膨胀
os.makedirs(SHARED_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# In-memory model store: {model_id: model_instance}
_models: dict = {}
_counter = 0

# Mapping of class names to cuML classes
_CLASS_MAP = {
    # Linear models
    "LinearRegression":          cuml.linear_model.LinearRegression,
    "LogisticRegression":        cuml.linear_model.LogisticRegression,
    "Ridge":                     cuml.linear_model.Ridge,
    "Lasso":                     cuml.linear_model.Lasso,
    "ElasticNet":                cuml.linear_model.ElasticNet,
    "MBSGDClassifier":           cuml.linear_model.MBSGDClassifier,
    "MBSGDRegressor":            cuml.linear_model.MBSGDRegressor,
    # Clustering
    "KMeans":                    cuml.cluster.KMeans,
    "DBSCAN":                    cuml.cluster.DBSCAN,
    # Decomposition
    "PCA":                       cuml.decomposition.PCA,
    "TruncatedSVD":              cuml.decomposition.TruncatedSVD,
    # Neighbors
    "KNeighborsClassifier":      cuml.neighbors.KNeighborsClassifier,
    "KNeighborsRegressor":       cuml.neighbors.KNeighborsRegressor,
    "NearestNeighbors":          cuml.neighbors.NearestNeighbors,
    # Ensemble
    "RandomForestClassifier":    cuml.ensemble.RandomForestClassifier,
    "RandomForestRegressor":     cuml.ensemble.RandomForestRegressor,
    # SVM
    "SVC":                       cuml.svm.SVC,
    "SVR":                       cuml.svm.SVR,
    # Preprocessing
    "StandardScaler":            cuml.preprocessing.StandardScaler,
    "MinMaxScaler":              cuml.preprocessing.MinMaxScaler,
    "LabelEncoder":              cuml.preprocessing.LabelEncoder,
    # Manifold
    "TSNE":                      cuml.manifold.TSNE,
    "UMAP":                      cuml.manifold.UMAP,
}


def _decode_arg(obj):
    """Decode argument: inline base64 array, shared-file reference, or mmap slot."""
    if isinstance(obj, dict):
        if obj.get("__ndarray__"):
            data = base64.b64decode(obj["data"])
            return np.frombuffer(data, dtype=obj["dtype"]).reshape(obj["shape"])
        if obj.get("__file__"):
            path = os.path.join(SHARED_DIR, obj["name"])
            arr = np.load(path)
            try:
                os.unlink(path)
            except OSError:
                pass
            return arr
        if obj.get("__mmap__"):
            return ShmTransport.get().read(obj)
    return obj


def _encode_result(obj):
    """Encode result array:
    - >= MMAP_THRESHOLD              : mmap OUTPUT slot（轮转分配，最多 256 MB/slot）
    - < MMAP_THRESHOLD               : inline base64
    - scalar / None                  : pass through
    
    注意：扩展 mmap 架构消除了 .npy fallback，所有数据均走 mmap 或 base64。
    """
    if hasattr(obj, "to_numpy"):
        obj = obj.to_numpy()
    if isinstance(obj, np.ndarray):
        if obj.nbytes >= MMAP_THRESHOLD:
            # 自动分配输出 slot（轮转）
            meta = ShmTransport.get().write(obj, is_output=True)
            return meta
        # 小数组：inline base64
        return {
            "__ndarray__": True,
            "data": base64.b64encode(obj.tobytes()).decode(),
            "dtype": str(obj.dtype),
            "shape": list(obj.shape),
        }
    if isinstance(obj, (tuple, list)):
        return [_encode_result(item) for item in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


# ── API Endpoints ─────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "cuml_version": cuml.__version__})


@app.route("/create", methods=["POST"])
def create():
    """Create a model instance. Returns model_id."""
    global _counter
    body = request.get_json()
    cls_name = body["class"]
    params = body.get("params", {})

    if cls_name not in _CLASS_MAP:
        return jsonify({"error": f"Unknown class: {cls_name}"}), 400
    try:
        model = _CLASS_MAP[cls_name](**params)
        _counter += 1
        model_id = f"{cls_name}_{_counter}"
        _models[model_id] = model
        return jsonify({"model_id": model_id})
    except Exception as e:
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


@app.route("/call/<model_id>/<method>", methods=["POST"])
def call_method(model_id, method):
    """Call a method on an existing model."""
    if model_id not in _models:
        return jsonify({"error": f"model_id not found: {model_id}"}), 404

    model = _models[model_id]
    body = request.get_json()

    # Decode args / kwargs（支持 inline base64 和共享文件两种格式）
    args = [_decode_arg(a) for a in body.get("args", [])]
    kwargs = {k: _decode_arg(v) for k, v in body.get("kwargs", {}).items()}

    try:
        fn = getattr(model, method)
        result = fn(*args, **kwargs)

        # fit* methods return self — just confirm ok
        if result is model or result is None:
            return jsonify({"result": None})

        encoded = _encode_result(result)
        return jsonify({"result": encoded})
    except Exception as e:
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


@app.route("/get_params/<model_id>", methods=["GET"])
def get_params(model_id):
    if model_id not in _models:
        return jsonify({"error": "not found"}), 404
    return jsonify({"params": _models[model_id].get_params()})


@app.route("/delete/<model_id>", methods=["DELETE"])
def delete_model(model_id):
    _models.pop(model_id, None)
    return jsonify({"ok": True})


@app.route("/save/<model_id>", methods=["POST"])
def save_model(model_id):
    """Pickle model to MODELS_DIR/{name}.pkl for later reload."""
    if model_id not in _models:
        return jsonify({"error": f"model_id not found: {model_id}"}), 404
    body = request.get_json()
    name = body.get("name", model_id)
    path = os.path.join(MODELS_DIR, f"{name}.pkl")
    try:
        with open(path, "wb") as f:
            pickle.dump(_models[model_id], f, protocol=5)
        return jsonify({"ok": True, "name": name})
    except Exception as e:
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


@app.route("/load", methods=["POST"])
def load_model_endpoint():
    """Unpickle a saved model and register it in _models."""
    global _counter
    body = request.get_json()
    name = body.get("name")
    path = os.path.join(MODELS_DIR, f"{name}.pkl")
    if not os.path.exists(path):
        return jsonify({"error": f"Model file not found: {path}"}), 404
    try:
        with open(path, "rb") as f:
            model = pickle.load(f)
        _counter += 1
        cls_name = type(model).__name__
        model_id = f"{cls_name}_{_counter}"
        _models[model_id] = model
        try:
            params = model.get_params()
        except Exception:
            params = {}
        return jsonify({"model_id": model_id, "class": cls_name, "params": params})
    except Exception as e:
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


@app.route("/list_models", methods=["GET"])
def list_saved_models():
    """List all saved model files in MODELS_DIR."""
    try:
        files = [f[:-4] for f in os.listdir(MODELS_DIR) if f.endswith(".pkl")]
        return jsonify({"models": sorted(files)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("SKLEARN_BRIDGE_PORT", 19876))
    print(f"[gpu-sklearn-bridge] cuML {cuml.__version__} ready on port {port}", flush=True)
    app.run(host="0.0.0.0", port=port, threaded=True)
