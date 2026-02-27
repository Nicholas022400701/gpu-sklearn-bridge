"""
WebUI main application – NVIDIA-themed cuML training system.

Run:
    cd gpu-sklearn-bridge
    python -m webui.app          # or: python webui/app.py

Then open http://localhost:7860 in your browser.
"""
import json
import os
import pickle
import sys
import uuid
import traceback
from pathlib import Path

from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_socketio import SocketIO, emit

# Allow running as `python webui/app.py` directly
sys.path.insert(0, str(Path(__file__).parent.parent))

from webui.model_configs import MODEL_CONFIGS, TASK_METRICS, METRIC_HIGHER_BETTER
from webui import data_manager as dm
from webui import training_manager as tm
from webui import log_manager as log_mgr

# ── App setup ─────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).parent
TEMPLATE_DIR = BASE_DIR / "templates"
STATIC_DIR   = BASE_DIR / "static"

app = Flask(__name__, template_folder=str(TEMPLATE_DIR), static_folder=str(STATIC_DIR))
app.config["SECRET_KEY"] = os.environ.get("WEBUI_SECRET", "nvidia-cuml-webui-2026")

socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading",
                    logger=False, engineio_logger=False)

# In-memory model registry: { model_id: sklearn/cuml model instance }
_model_registry: dict = {}

# ── Health ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/health")
def health():
    bridge_ok = False
    try:
        import requests as _req
        r = _req.get("http://127.0.0.1:19876/health", timeout=1)
        bridge_ok = r.status_code == 200
    except Exception:
        pass
    return jsonify({"status": "ok", "bridge_connected": bridge_ok})


# ── Model configs API ─────────────────────────────────────────────────────────

@app.route("/api/models/configs")
def model_configs_api():
    """Return all model configs (without numpy arrays)."""
    return jsonify(MODEL_CONFIGS)


@app.route("/api/models/list")
def model_list_api():
    """Return high-level list of available models."""
    result = []
    for name, cfg in MODEL_CONFIGS.items():
        result.append({
            "name": name,
            "task": cfg["task"],
            "module": cfg["module"],
            "desc": cfg["desc"],
        })
    return jsonify(result)


# ── Data API ──────────────────────────────────────────────────────────────────

@app.route("/api/data/upload", methods=["POST"])
def data_upload():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    f = request.files["file"]
    session_id = request.form.get("session_id") or str(uuid.uuid4())
    result = dm.load_dataframe(session_id, f.read(), f.filename)
    if "error" in result:
        return jsonify(result), 400
    result["session_id"] = session_id
    return jsonify(result)


@app.route("/api/data/describe")
def data_describe():
    session_id = request.args.get("session_id")
    if not session_id:
        return jsonify({"error": "session_id required"}), 400
    return jsonify(dm.describe(session_id))


@app.route("/api/data/clean", methods=["POST"])
def data_clean():
    body = request.get_json()
    session_id = body.get("session_id")
    ops = body.get("ops", [])
    if not session_id:
        return jsonify({"error": "session_id required"}), 400
    result = dm.clean(session_id, ops)
    if "error" in result:
        return jsonify(result), 400
    return jsonify(result)


@app.route("/api/data/split", methods=["POST"])
def data_split():
    body = request.get_json()
    session_id  = body.get("session_id")
    feature_cols = body.get("feature_cols", [])
    target_col  = body.get("target_col")
    test_size   = float(body.get("test_size", 0.2))
    random_state = int(body.get("random_state", 42))
    scaler      = body.get("scaler")
    encode_target = bool(body.get("encode_target", False))
    if not session_id:
        return jsonify({"error": "session_id required"}), 400
    result = dm.prepare_split(session_id, feature_cols, target_col,
                               test_size, random_state, scaler, encode_target)
    if "error" in result:
        return jsonify(result), 400
    return jsonify(result)


# ── Training API ──────────────────────────────────────────────────────────────

@app.route("/api/train/start", methods=["POST"])
def train_start():
    body = request.get_json()
    session_id    = body.get("session_id") or str(uuid.uuid4())
    data_session  = body.get("data_session")
    model_cls     = body.get("model")
    model_params  = body.get("params", {})
    mode          = body.get("mode", "unlimited")          # timed | target | unlimited
    time_limit    = body.get("time_limit", 60)
    target_metric = body.get("target_metric", "val_accuracy")
    target_value  = body.get("target_value", 0.95)
    n_steps       = body.get("n_steps", 10)
    model_name    = body.get("model_name", f"{model_cls}_{session_id[:8]}")

    if model_cls not in MODEL_CONFIGS:
        return jsonify({"error": f"Unknown model: {model_cls}"}), 400

    split = dm.get_split(data_session) if data_session else None
    if split is None:
        return jsonify({"error": "No data split found. Run /api/data/split first."}), 400

    task = MODEL_CONFIGS[model_cls]["task"]

    # Cast param values to correct types
    typed_params = _cast_params(model_cls, model_params)

    model_id = model_name
    sess = tm.create_session(
        session_id=session_id,
        model_id=model_id,
        model_cls_name=model_cls,
        model_params=typed_params,
        X_train=split["X_train"], X_test=split["X_test"],
        y_train=split["y_train"], y_test=split["y_test"],
        task=task,
        mode=mode,
        time_limit=time_limit,
        target_metric=target_metric,
        target_value=target_value,
        n_steps=n_steps,
        socketio=socketio,
    )
    log_mgr.log(model_id, f"Training session created | mode={mode} | model={model_cls}")
    sess.start()
    return jsonify({"ok": True, "session_id": session_id, "model_id": model_id})


@app.route("/api/train/stop", methods=["POST"])
def train_stop():
    body = request.get_json()
    session_id = body.get("session_id")
    ok = tm.stop_session(session_id)
    return jsonify({"ok": ok})


@app.route("/api/train/status/<session_id>")
def train_status(session_id):
    sess = tm.get_session(session_id)
    if sess is None:
        return jsonify({"error": "session not found"}), 404
    return jsonify(sess.get_state())


@app.route("/api/train/sessions")
def train_sessions():
    return jsonify(tm.list_sessions())


# ── Model weights API ─────────────────────────────────────────────────────────

@app.route("/api/weights/list")
def weights_list():
    return jsonify({"models": tm.list_saved_models()})


@app.route("/api/weights/delete", methods=["POST"])
def weights_delete():
    body = request.get_json()
    name = body.get("name")
    ok = tm.delete_model_file(name)
    return jsonify({"ok": ok})


@app.route("/api/weights/load", methods=["POST"])
def weights_load():
    body = request.get_json()
    name = body.get("name")
    model, err = tm.load_model(name)
    if err:
        return jsonify({"error": err}), 404
    model_id = name
    _model_registry[model_id] = model
    return jsonify({"ok": True, "model_id": model_id,
                    "class": type(model).__name__})


@app.route("/api/weights/save", methods=["POST"])
def weights_save():
    body = request.get_json()
    model_id = body.get("model_id")
    name = body.get("name", model_id)
    model = _model_registry.get(model_id)
    if model is None:
        # Try to find in completed training sessions
        sess = None
        for s in tm._sessions.values():
            if s.model_id == model_id and s.best_model is not None:
                model = s.best_model
                break
    if model is None:
        return jsonify({"error": f"Model '{model_id}' not found in registry"}), 404
    path, err = tm.save_model(model, name)
    if err:
        return jsonify({"error": err}), 500
    return jsonify({"ok": True, "path": path})


# ── Prediction API ────────────────────────────────────────────────────────────

@app.route("/api/predict", methods=["POST"])
def predict():
    body = request.get_json()
    model_id     = body.get("model_id")
    data_session = body.get("data_session")
    input_data   = body.get("data")  # [[f1, f2, ...], ...]
    feature_cols = body.get("feature_cols", [])

    # Find model
    model = _get_active_model(model_id)
    if model is None:
        return jsonify({"error": f"Model '{model_id}' not available"}), 404

    try:
        import numpy as np
        if input_data is not None:
            X = dm.predict_on_input(data_session or "", input_data, feature_cols)
        else:
            split = dm.get_split(data_session) if data_session else None
            if split is None:
                return jsonify({"error": "No data provided"}), 400
            X = split["X_test"]

        preds = model.predict(X)
        preds_list = preds.tolist() if hasattr(preds, "tolist") else list(preds)

        # Map labels back if available
        split = dm.get_split(data_session) if data_session else None
        label_map = split.get("label_map") if split else None

        if label_map:
            preds_list = [label_map.get(int(p), p) for p in preds_list]

        return jsonify({"predictions": preds_list, "n": len(preds_list)})
    except Exception as e:
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


@app.route("/api/evaluate", methods=["POST"])
def evaluate():
    body = request.get_json()
    model_id     = body.get("model_id")
    data_session = body.get("data_session")

    model = _get_active_model(model_id)
    if model is None:
        return jsonify({"error": f"Model '{model_id}' not available"}), 404

    split = dm.get_split(data_session) if data_session else None
    if split is None:
        return jsonify({"error": "No data split found"}), 400

    # Detect task
    model_class = type(model).__name__
    task = MODEL_CONFIGS.get(model_class, {}).get("task", "classification")

    try:
        metrics = tm._compute_metrics(
            model,
            split["X_train"], split["X_test"],
            split["y_train"], split["y_test"],
            task,
        )
        return jsonify({"metrics": metrics, "model_id": model_id})
    except Exception as e:
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


# ── Stacking API ──────────────────────────────────────────────────────────────

@app.route("/api/stacking/create", methods=["POST"])
def stacking_create():
    """Create a Stacking ensemble from existing trained models."""
    body = request.get_json()
    base_model_ids  = body.get("base_models", [])
    meta_model_name = body.get("meta_model", "LogisticRegression")
    meta_params     = body.get("meta_params", {})
    stack_name      = body.get("name", f"stack_{uuid.uuid4().hex[:8]}")
    data_session    = body.get("data_session")

    base_models = []
    for mid in base_model_ids:
        m = _get_active_model(mid)
        if m is None:
            return jsonify({"error": f"Base model '{mid}' not found"}), 404
        base_models.append((mid, m))

    split = dm.get_split(data_session) if data_session else None
    if split is None:
        return jsonify({"error": "No data split found"}), 400

    try:
        from sklearn.ensemble import StackingClassifier, StackingRegressor
        import numpy as np

        # Detect task from first base model
        first_cls = type(base_models[0][1]).__name__
        task = MODEL_CONFIGS.get(first_cls, {}).get("task", "classification")

        # Build meta model
        meta_cfg = MODEL_CONFIGS.get(meta_model_name, {})
        meta_task = meta_cfg.get("task", task)
        typed_meta = _cast_params(meta_model_name, meta_params) if meta_model_name in MODEL_CONFIGS else {}

        from sklearn import linear_model as sk_lm
        meta_cls_map = {
            "LogisticRegression": sk_lm.LogisticRegression,
            "LinearRegression":   sk_lm.LinearRegression,
            "Ridge":              sk_lm.Ridge,
        }
        meta_cls = meta_cls_map.get(meta_model_name, sk_lm.LogisticRegression)
        meta_model = meta_cls(**typed_meta)

        estimators = [(mid, m) for mid, m in base_models]
        if task == "regression":
            stack = StackingRegressor(estimators=estimators, final_estimator=meta_model, cv=3)
        else:
            stack = StackingClassifier(estimators=estimators, final_estimator=meta_model, cv=3)

        X_train, y_train = split["X_train"], split["y_train"]
        if y_train is None:
            return jsonify({"error": "Stacking requires labeled data"}), 400

        stack.fit(X_train, y_train)

        _model_registry[stack_name] = stack
        log_mgr.log(stack_name, f"Stacking model created: base={base_model_ids} meta={meta_model_name}")

        metrics = tm._compute_metrics(stack, X_train, split["X_test"], y_train, split["y_test"], task)
        path, _ = tm.save_model(stack, stack_name)

        return jsonify({"ok": True, "model_id": stack_name, "metrics": metrics, "path": path})
    except Exception as e:
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


# ── Log API ───────────────────────────────────────────────────────────────────

@app.route("/api/logs/list")
def logs_list():
    return jsonify({"logs": log_mgr.list_log_files()})


@app.route("/api/logs/<model_id>")
def logs_get(model_id):
    entries = log_mgr.get_logs(model_id)
    raw     = log_mgr.read_log_file(model_id)
    return jsonify({"model_id": model_id, "entries": entries, "raw": raw})


@app.route("/api/logs/<model_id>/clear", methods=["DELETE"])
def logs_clear(model_id):
    log_mgr.clear_log(model_id)
    return jsonify({"ok": True})


# ── SocketIO events ───────────────────────────────────────────────────────────

@socketio.on("connect")
def on_connect():
    emit("connected", {"status": "ok"})


@socketio.on("join_session")
def on_join(data):
    session_id = data.get("session_id")
    sess = tm.get_session(session_id)
    if sess:
        emit("session_state", sess.get_state())


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_active_model(model_id: str):
    """Look up model in registry → then in completed training sessions."""
    if model_id in _model_registry:
        return _model_registry[model_id]
    # Search training sessions for best model
    for sess in tm._sessions.values():
        if sess.model_id == model_id and sess.best_model is not None:
            return sess.best_model
    # Try loading from saved files
    model, err = tm.load_model(model_id)
    if model:
        _model_registry[model_id] = model
        return model
    return None


def _cast_params(model_cls: str, raw_params: dict) -> dict:
    """Cast raw form values to correct Python types based on MODEL_CONFIGS."""
    cfg = MODEL_CONFIGS.get(model_cls, {})
    param_defs = cfg.get("params", {})
    result = {}
    for k, v in raw_params.items():
        if k not in param_defs:
            result[k] = v
            continue
        pdef = param_defs[k]
        ptype = pdef.get("type", "str")
        nullable = pdef.get("nullable", False)

        if v is None or v == "" or v == "null" or v == "None":
            if nullable:
                result[k] = None
            # Skip non-nullable None values
            continue

        try:
            if ptype == "int":
                result[k] = int(v)
            elif ptype == "float":
                result[k] = float(v)
            elif ptype == "bool":
                if isinstance(v, bool):
                    result[k] = v
                else:
                    result[k] = str(v).lower() in ("true", "1", "yes")
            elif ptype == "select":
                result[k] = str(v)
            elif ptype == "tuple_float":
                if isinstance(v, (list, tuple)):
                    result[k] = tuple(float(x) for x in v)
                else:
                    result[k] = v
            else:
                result[k] = v
        except (ValueError, TypeError):
            result[k] = v

    return result


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("WEBUI_PORT", 7860))
    print(f"[cuML WebUI] Starting on http://0.0.0.0:{port}", flush=True)
    socketio.run(app, host="0.0.0.0", port=port, debug=False,
                 allow_unsafe_werkzeug=True)
