"""
Training manager: background thread-per-session training with three modes.

Modes:
  - "timed"     : stop after `time_limit` seconds, auto-save best
  - "target"    : stop when `target_metric` >= `target_value` (or <= for loss metrics), auto-save best
  - "unlimited" : run until user calls stop(), auto-save best on stop

For iterative models (MBSGDClassifier/Regressor) real partial_fit is used.
For all other models a "learning curve" simulation is used:
  train on increasingly large subsets → emit metrics after each step.
"""
import time
import threading
import traceback
import uuid
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import log_manager as log_mgr
from .model_configs import METRIC_HIGHER_BETTER

_MODELS_DIR = Path(__file__).parent.parent / "models"
_MODELS_DIR.mkdir(exist_ok=True)

# Active sessions: { session_id: TrainingSession }
_sessions: dict = {}
_sessions_lock = threading.Lock()


# ── helpers ───────────────────────────────────────────────────────────────────

def _compute_metrics(model, X_train, X_test, y_train, y_test, task):
    """Return a dict of metric_name→float for a fitted model."""
    metrics = {}
    try:
        if task in ("classification",):
            from sklearn.metrics import (accuracy_score, f1_score,
                                          precision_score, recall_score)
            y_pred_train = model.predict(X_train)
            y_pred_test  = model.predict(X_test)
            metrics["train_accuracy"] = float(accuracy_score(y_train, y_pred_train))
            metrics["val_accuracy"]   = float(accuracy_score(y_test,  y_pred_test))
            metrics["train_f1"]       = float(f1_score(y_train, y_pred_train, average="weighted", zero_division=0))
            metrics["val_f1"]         = float(f1_score(y_test,  y_pred_test,  average="weighted", zero_division=0))

        elif task in ("regression",):
            from sklearn.metrics import mean_squared_error, r2_score
            y_pred_train = model.predict(X_train)
            y_pred_test  = model.predict(X_test)
            metrics["train_r2"]      = float(r2_score(y_train, y_pred_train))
            metrics["val_r2"]        = float(r2_score(y_test,  y_pred_test))
            metrics["train_rmse"]    = float(np.sqrt(mean_squared_error(y_train, y_pred_train)))
            metrics["val_rmse"]      = float(np.sqrt(mean_squared_error(y_test,  y_pred_test)))

        elif task == "clustering":
            from sklearn.metrics import silhouette_score, calinski_harabasz_score
            labels = model.labels_ if hasattr(model, "labels_") else model.predict(X_train)
            labels = np.asarray(labels)
            n_unique = len(np.unique(labels[labels >= 0]))
            if n_unique >= 2:
                metrics["silhouette"]         = float(silhouette_score(X_train, labels))
                metrics["calinski_harabasz"]  = float(calinski_harabasz_score(X_train, labels))
            if hasattr(model, "inertia_"):
                metrics["inertia"] = float(model.inertia_)

    except Exception as e:
        metrics["_error"] = str(e)
    return metrics


def _is_iterative(cls_name):
    return cls_name in ("MBSGDClassifier", "MBSGDRegressor")


# ── TrainingSession ───────────────────────────────────────────────────────────

class TrainingSession:
    def __init__(self, session_id, model_id, model_cls_name, model_params,
                 X_train, X_test, y_train, y_test, task,
                 mode, time_limit, target_metric, target_value,
                 n_steps, socketio, namespace="/"):
        self.session_id   = session_id
        self.model_id     = model_id
        self.model_cls_name = model_cls_name
        self.model_params = dict(model_params)
        self.X_train      = X_train
        self.X_test       = X_test
        self.y_train      = y_train
        self.y_test       = y_test
        self.task         = task
        self.mode         = mode
        self.time_limit   = float(time_limit or 60)
        self.target_metric = target_metric or "val_accuracy"
        self.target_value  = float(target_value or 0.95)
        self.n_steps       = max(int(n_steps or 10), 2)
        self.socketio      = socketio
        self.namespace     = namespace

        self.stop_event    = threading.Event()
        self.status        = "idle"       # idle / running / stopped / completed / error
        self.history       = []           # list of {step, metrics, elapsed}
        self.best_metrics  = {}
        self.best_model    = None
        self.best_step     = 0
        self.start_time    = None
        self._thread       = None

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self):
        if self.status == "running":
            return
        self.status = "running"
        self.start_time = time.time()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self.stop_event.set()

    def get_state(self):
        return {
            "session_id":   self.session_id,
            "model_id":     self.model_id,
            "model":        self.model_cls_name,
            "status":       self.status,
            "mode":         self.mode,
            "history":      self.history,
            "best_metrics": self.best_metrics,
            "best_step":    self.best_step,
            "elapsed":      round(time.time() - self.start_time, 2) if self.start_time else 0,
        }

    # ── Internal ──────────────────────────────────────────────────────────────

    def _emit(self, event, data):
        try:
            self.socketio.emit(event, data, namespace=self.namespace)
        except Exception:
            pass

    def _log(self, msg, level="INFO"):
        entry = log_mgr.log(self.model_id, msg, level)
        self._emit("log_entry", {"model_id": self.model_id, "entry": entry})

    def _build_model(self):
        """Instantiate the model.
        Uses real RAPIDS cuML when the bridge server is reachable;
        falls back to scikit-learn otherwise.
        """
        # ── sklearn fallback map ─────────────────────────────────────────────
        from sklearn import (linear_model, cluster, decomposition,
                              neighbors, ensemble, svm, preprocessing,
                              manifold)
        _sklearn_map = {
            "LinearRegression":      linear_model.LinearRegression,
            "Ridge":                 linear_model.Ridge,
            "Lasso":                 linear_model.Lasso,
            "ElasticNet":            linear_model.ElasticNet,
            "LogisticRegression":    linear_model.LogisticRegression,
            "MBSGDClassifier":       linear_model.SGDClassifier,
            "MBSGDRegressor":        linear_model.SGDRegressor,
            "SVC":                   svm.SVC,
            "SVR":                   svm.SVR,
            "KMeans":                cluster.KMeans,
            "DBSCAN":                cluster.DBSCAN,
            "PCA":                   decomposition.PCA,
            "TruncatedSVD":          decomposition.TruncatedSVD,
            "KNeighborsClassifier":  neighbors.KNeighborsClassifier,
            "KNeighborsRegressor":   neighbors.KNeighborsRegressor,
            "NearestNeighbors":      neighbors.NearestNeighbors,
            "RandomForestClassifier":ensemble.RandomForestClassifier,
            "RandomForestRegressor": ensemble.RandomForestRegressor,
            "StandardScaler":        preprocessing.StandardScaler,
            "MinMaxScaler":          preprocessing.MinMaxScaler,
            "TSNE":                  manifold.TSNE,
            "UMAP":                  None,
        }

        cls = None

        # ── Try real RAPIDS cuML via bridge (only if bridge server reachable) ─
        try:
            import requests as _req
            from cuml_proxy.proxy import _BRIDGE_URL
            r = _req.get(f"{_BRIDGE_URL}/health", timeout=1)
            if r.status_code == 200:
                # Real cuML available through the bridge proxy
                from cuml_proxy.proxy import ProxyEstimator
                from .model_configs import MODEL_CONFIGS
                module_name = MODEL_CONFIGS[self.model_cls_name]["module"]
                # Filter params
                params = {k: v for k, v in self.model_params.items()}
                return ProxyEstimator(self.model_cls_name, **params)
        except Exception:
            pass  # bridge not available → fall through to sklearn

        # ── scikit-learn fallback ────────────────────────────────────────────
        cls = _sklearn_map.get(self.model_cls_name)
        if cls is None:
            raise RuntimeError(
                f"Model '{self.model_cls_name}' requires cuML (bridge not running) "
                "and has no scikit-learn equivalent."
            )

        # Filter params to only those accepted by the class
        try:
            import inspect
            sig = inspect.signature(cls.__init__)
            valid_params = set(sig.parameters.keys()) - {"self"}
            params = {k: v for k, v in self.model_params.items() if k in valid_params}
        except Exception:
            params = self.model_params

        return cls(**params)

    def _is_better(self, new_metrics, old_metrics):
        """Return True if new_metrics represents a better model."""
        if not old_metrics:
            return True
        primary_keys = [k for k in new_metrics if not k.startswith("_")]
        for k in ("val_accuracy", "val_r2", "silhouette"):
            if k in primary_keys:
                higher_better = METRIC_HIGHER_BETTER.get(k, True)
                new_val = new_metrics.get(k)
                old_val = old_metrics.get(k)
                if new_val is None or old_val is None:
                    continue
                return new_val > old_val if higher_better else new_val < old_val
        return True

    def _check_stop_conditions(self, metrics, elapsed):
        """Return True if we should stop training."""
        if self.stop_event.is_set():
            return True
        if self.mode == "timed" and elapsed >= self.time_limit:
            return True
        if self.mode == "target":
            val = metrics.get(self.target_metric)
            if val is not None:
                higher = METRIC_HIGHER_BETTER.get(self.target_metric, True)
                if higher and val >= self.target_value:
                    return True
                if not higher and val <= self.target_value:
                    return True
        return False

    def _run(self):
        self._log(f"Training started: {self.model_cls_name} | mode={self.mode} | "
                  f"X_train={self.X_train.shape}")
        try:
            if _is_iterative(self.model_cls_name):
                self._run_iterative()
            else:
                self._run_learning_curve()
        except Exception as e:
            self.status = "error"
            self._log(f"Training error: {e}\n{traceback.format_exc()}", "ERROR")
            self._emit("training_error", {"session_id": self.session_id, "error": str(e)})
            return

        # Auto-save best model
        if self.best_model is not None:
            save_name = f"{self.model_id}_best"
            save_path = _MODELS_DIR / f"{save_name}.pkl"
            try:
                with open(save_path, "wb") as f:
                    pickle.dump(self.best_model, f, protocol=5)
                self._log(f"Best model saved → models/{save_name}.pkl")
            except Exception as e:
                self._log(f"Save error: {e}", "WARNING")

        self.status = "completed" if not self.stop_event.is_set() else "stopped"
        self._log(f"Training {self.status}. Best metrics: {self.best_metrics}")
        self._emit("training_done", self.get_state())

    # ── Learning curve (non-iterative) ────────────────────────────────────────

    def _run_learning_curve(self):
        n = len(self.X_train)
        y_avail = self.y_train is not None

        # Steps: train on 10%..100% of training data
        fractions = np.linspace(0.1, 1.0, self.n_steps)
        fractions[-1] = 1.0

        self._emit("training_start", self.get_state())

        for step_idx, frac in enumerate(fractions, 1):
            if self.stop_event.is_set():
                break

            n_sub = max(int(n * frac), 2)
            X_sub = self.X_train[:n_sub]
            y_sub = self.y_train[:n_sub] if y_avail else None

            # Build fresh model each step
            model = self._build_model()

            if y_sub is not None:
                model.fit(X_sub, y_sub)
            else:
                model.fit(X_sub)

            metrics = _compute_metrics(model, X_sub, self.X_test,
                                        y_sub, self.y_test, self.task)
            elapsed = round(time.time() - self.start_time, 2)
            entry = {"step": step_idx, "frac": round(frac, 3),
                     "metrics": metrics, "elapsed": elapsed,
                     "n_samples": n_sub}
            self.history.append(entry)

            if self._is_better(metrics, self.best_metrics):
                self.best_metrics = metrics
                self.best_model   = model
                self.best_step    = step_idx
                self._log(f"Step {step_idx}/{len(fractions)} | New best: {metrics}")

            self._emit("training_step", {"session_id": self.session_id, **entry,
                                          "best_metrics": self.best_metrics})

            if self._check_stop_conditions(metrics, elapsed):
                break

    # ── Iterative (SGD partial_fit) ───────────────────────────────────────────

    def _run_iterative(self):
        epochs    = int(self.model_params.get("epochs", 100))
        batch_sz  = int(self.model_params.get("batch_size", 512))
        n_train   = len(self.X_train)
        y_avail   = self.y_train is not None

        # Build model once, then call partial_fit repeatedly
        model = self._build_model()

        # Determine classes for partial_fit
        classes = None
        if y_avail and self.task == "classification":
            classes = np.unique(self.y_train)

        self._emit("training_start", self.get_state())

        global_step = 0
        for epoch in range(1, epochs + 1):
            if self.stop_event.is_set():
                break

            # Shuffle
            idx = np.random.permutation(n_train)
            X_shuf = self.X_train[idx]
            y_shuf = self.y_train[idx] if y_avail else None

            for batch_start in range(0, n_train, batch_sz):
                if self.stop_event.is_set():
                    break
                X_batch = X_shuf[batch_start:batch_start + batch_sz]
                y_batch = y_shuf[batch_start:batch_start + batch_sz] if y_avail else None

                if y_batch is not None:
                    if classes is not None:
                        model.partial_fit(X_batch, y_batch, classes=classes)
                    else:
                        model.partial_fit(X_batch, y_batch)
                else:
                    model.partial_fit(X_batch)

                global_step += 1

            # Emit once per epoch
            metrics = _compute_metrics(model, self.X_train, self.X_test,
                                        self.y_train, self.y_test, self.task)
            elapsed = round(time.time() - self.start_time, 2)
            entry = {"step": epoch, "frac": epoch / epochs,
                     "metrics": metrics, "elapsed": elapsed,
                     "n_samples": n_train}
            self.history.append(entry)

            if self._is_better(metrics, self.best_metrics):
                self.best_metrics = metrics
                self.best_model   = pickle.loads(pickle.dumps(model))  # snapshot
                self.best_step    = epoch
                self._log(f"Epoch {epoch}/{epochs} | New best: {metrics}")

            self._emit("training_step", {"session_id": self.session_id, **entry,
                                          "best_metrics": self.best_metrics})

            if self._check_stop_conditions(metrics, elapsed):
                break


# ── Public functions ──────────────────────────────────────────────────────────

def create_session(session_id, model_id, model_cls_name, model_params,
                   X_train, X_test, y_train, y_test, task,
                   mode, time_limit, target_metric, target_value,
                   n_steps, socketio):
    sess = TrainingSession(
        session_id, model_id, model_cls_name, model_params,
        X_train, X_test, y_train, y_test, task,
        mode, time_limit, target_metric, target_value,
        n_steps, socketio,
    )
    with _sessions_lock:
        _sessions[session_id] = sess
    return sess


def get_session(session_id):
    with _sessions_lock:
        return _sessions.get(session_id)


def stop_session(session_id):
    with _sessions_lock:
        sess = _sessions.get(session_id)
    if sess:
        sess.stop()
    return sess is not None


def list_sessions():
    with _sessions_lock:
        return [
            {
                "session_id": s.session_id,
                "model_id": s.model_id,
                "model": s.model_cls_name,
                "status": s.status,
                "mode": s.mode,
                "best_metrics": s.best_metrics,
            }
            for s in _sessions.values()
        ]


def list_saved_models():
    try:
        return sorted(p.stem for p in _MODELS_DIR.glob("*.pkl"))
    except OSError:
        return []


def load_model(name):
    path = _MODELS_DIR / f"{name}.pkl"
    if not path.exists():
        return None, f"Model file not found: {path}"
    try:
        with open(path, "rb") as f:
            return pickle.load(f), None
    except Exception as e:
        return None, str(e)


def save_model(model, name):
    path = _MODELS_DIR / f"{name}.pkl"
    try:
        with open(path, "wb") as f:
            pickle.dump(model, f, protocol=5)
        return str(path), None
    except Exception as e:
        return None, str(e)


def delete_model_file(name):
    path = _MODELS_DIR / f"{name}.pkl"
    try:
        path.unlink(missing_ok=True)
        return True
    except OSError:
        return False
