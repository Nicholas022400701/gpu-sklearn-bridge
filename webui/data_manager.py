"""
Data manager: in-memory storage for uploaded datasets, cleaning/preprocessing
state, and train/test splits.
"""
import io
import json
import threading
import numpy as np
import pandas as pd
from pathlib import Path

_lock = threading.Lock()

# Uploaded raw DataFrames: { session_id: DataFrame }
_raw: dict = {}
# Cleaned DataFrames: { session_id: DataFrame }
_cleaned: dict = {}
# Preprocessing pipeline descriptions: { session_id: [step_dict, ...] }
_pipeline: dict = {}
# Active splits: { session_id: {X_train, X_test, y_train, y_test, feature_cols, target_col} }
_splits: dict = {}

_UPLOAD_DIR = Path(__file__).parent.parent / "uploads"
_UPLOAD_DIR.mkdir(exist_ok=True)


# ── Upload ─────────────────────────────────────────────────────────────────────

def load_dataframe(session_id: str, file_bytes: bytes, filename: str) -> dict:
    """Parse uploaded file into a DataFrame. Returns summary dict."""
    fname = filename.lower()
    try:
        if fname.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(file_bytes))
        elif fname.endswith(".tsv"):
            df = pd.read_csv(io.BytesIO(file_bytes), sep="\t")
        elif fname.endswith((".xls", ".xlsx")):
            df = pd.read_excel(io.BytesIO(file_bytes))
        elif fname.endswith(".json"):
            df = pd.read_json(io.BytesIO(file_bytes))
        elif fname.endswith(".parquet"):
            df = pd.read_parquet(io.BytesIO(file_bytes))
        else:
            return {"error": f"Unsupported file format: {filename}"}
    except Exception as e:
        return {"error": str(e)}

    with _lock:
        _raw[session_id] = df
        _cleaned[session_id] = df.copy()
        _pipeline[session_id] = []

    return _describe(df)


def get_raw(session_id: str) -> pd.DataFrame | None:
    with _lock:
        return _raw.get(session_id)


def get_cleaned(session_id: str) -> pd.DataFrame | None:
    with _lock:
        return _cleaned.get(session_id)


def _describe(df: pd.DataFrame) -> dict:
    """Return a JSON-serialisable description of a DataFrame."""
    info = []
    for col in df.columns:
        s = df[col]
        info.append({
            "name": col,
            "dtype": str(s.dtype),
            "n_missing": int(s.isna().sum()),
            "pct_missing": round(100 * s.isna().mean(), 2),
            "n_unique": int(s.nunique()),
            "sample": s.dropna().head(5).tolist(),
        })
    return {
        "n_rows": len(df),
        "n_cols": len(df.columns),
        "columns": info,
        "preview": df.head(20).fillna("").astype(str).values.tolist(),
        "col_names": list(df.columns),
    }


def describe(session_id: str, which: str = "cleaned") -> dict:
    with _lock:
        df = _cleaned.get(session_id) if which == "cleaned" else _raw.get(session_id)
    if df is None:
        return {"error": "No data loaded"}
    return _describe(df)


# ── Cleaning ──────────────────────────────────────────────────────────────────

def clean(session_id: str, ops: list) -> dict:
    """Apply a list of cleaning operations to the cleaned DataFrame.

    Each op is a dict with at least a "type" key:
      {"type": "drop_missing"}
      {"type": "fill_missing", "strategy": "mean"|"median"|"mode"|"value", "value": ...}
      {"type": "drop_duplicates"}
      {"type": "drop_columns", "columns": [...]}
      {"type": "rename_column", "old": "...", "new": "..."}
      {"type": "cast_column", "column": "...", "dtype": "float32"|"int32"|"str"|...}
      {"type": "clip_outliers", "method": "iqr"|"zscore", "threshold": 1.5}
      {"type": "reset"}
    """
    with _lock:
        raw = _raw.get(session_id)
        df  = _cleaned.get(session_id)
    if df is None:
        return {"error": "No data loaded"}

    df = df.copy()
    applied = []

    for op in ops:
        t = op.get("type", "")
        try:
            if t == "reset":
                df = raw.copy()
                applied.append("reset to raw")

            elif t == "drop_missing":
                cols = op.get("columns", None)
                before = len(df)
                df = df.dropna(subset=cols)
                applied.append(f"drop_missing({cols}): {before}→{len(df)} rows")

            elif t == "fill_missing":
                cols    = op.get("columns") or list(df.columns)
                strategy = op.get("strategy", "mean")
                fill_val = op.get("value", 0)
                for col in cols:
                    if col not in df.columns:
                        continue
                    is_numeric = pd.api.types.is_numeric_dtype(df[col])
                    if strategy in ("mean", "median") and not is_numeric:
                        # Fall back to mode for non-numeric columns
                        mode = df[col].mode()
                        df[col] = df[col].fillna(mode.iloc[0] if len(mode) else "")
                    elif strategy == "mean":
                        df[col] = df[col].fillna(df[col].mean())
                    elif strategy == "median":
                        df[col] = df[col].fillna(df[col].median())
                    elif strategy == "mode":
                        mode = df[col].mode()
                        df[col] = df[col].fillna(mode.iloc[0] if len(mode) else (0 if is_numeric else ""))
                    else:
                        df[col] = df[col].fillna(fill_val)
                applied.append(f"fill_missing(strategy={strategy}, cols={cols})")

            elif t == "drop_duplicates":
                before = len(df)
                df = df.drop_duplicates()
                applied.append(f"drop_duplicates: {before}→{len(df)} rows")

            elif t == "drop_columns":
                cols = op.get("columns", [])
                df = df.drop(columns=[c for c in cols if c in df.columns], errors="ignore")
                applied.append(f"drop_columns({cols})")

            elif t == "rename_column":
                df = df.rename(columns={op["old"]: op["new"]})
                applied.append(f"rename_column({op['old']}→{op['new']})")

            elif t == "cast_column":
                col = op["column"]
                dtype = op["dtype"]
                if col in df.columns:
                    df[col] = df[col].astype(dtype)
                applied.append(f"cast_column({col}→{dtype})")

            elif t == "clip_outliers":
                method    = op.get("method", "iqr")
                threshold = float(op.get("threshold", 1.5))
                num_cols  = df.select_dtypes(include="number").columns.tolist()
                if method == "iqr":
                    for col in num_cols:
                        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
                        iqr = q3 - q1
                        df[col] = df[col].clip(q1 - threshold * iqr, q3 + threshold * iqr)
                elif method == "zscore":
                    for col in num_cols:
                        mean, std = df[col].mean(), df[col].std()
                        if std > 0:
                            df[col] = df[col].clip(mean - threshold * std, mean + threshold * std)
                applied.append(f"clip_outliers(method={method}, thresh={threshold})")

        except Exception as e:
            applied.append(f"ERROR in {t}: {e}")

    with _lock:
        _cleaned[session_id] = df

    return {"ok": True, "applied": applied, **_describe(df)}


# ── Preprocessing / split ─────────────────────────────────────────────────────

def prepare_split(
    session_id: str,
    feature_cols: list,
    target_col: str | None,
    test_size: float = 0.2,
    random_state: int = 42,
    scaler: str | None = None,
    encode_target: bool = False,
) -> dict:
    """Extract X, y from cleaned DF and create train/test split."""
    with _lock:
        df = _cleaned.get(session_id)
    if df is None:
        return {"error": "No data loaded"}

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        return {"error": f"Feature columns not found: {missing}"}

    X = df[feature_cols].copy()

    # Encode categorical columns
    for col in X.select_dtypes(include="object").columns:
        X[col] = X[col].astype("category").cat.codes.astype("float32")
    X = X.fillna(0).astype("float32").values

    y = None
    label_map = None
    if target_col:
        if target_col not in df.columns:
            return {"error": f"Target column not found: {target_col}"}
        y_series = df[target_col].copy()
        if encode_target or y_series.dtype == object:
            y_series = y_series.astype("category")
            label_map = dict(enumerate(y_series.cat.categories))
            y = y_series.cat.codes.astype("float32").values
        else:
            y = y_series.fillna(0).astype("float32").values

    # Optional scaling
    scaler_obj = None
    if scaler == "standard":
        from sklearn.preprocessing import StandardScaler as SK_SS
        scaler_obj = SK_SS()
        X = scaler_obj.fit_transform(X)
    elif scaler == "minmax":
        from sklearn.preprocessing import MinMaxScaler as SK_MM
        scaler_obj = SK_MM()
        X = scaler_obj.fit_transform(X)
    elif scaler == "robust":
        from sklearn.preprocessing import RobustScaler
        scaler_obj = RobustScaler()
        X = scaler_obj.fit_transform(X)

    # Train/test split
    from sklearn.model_selection import train_test_split
    if y is not None:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
    else:
        X_train, X_test = train_test_split(
            X, test_size=test_size, random_state=random_state
        )
        y_train = y_test = None

    with _lock:
        _splits[session_id] = {
            "X_train": X_train, "X_test": X_test,
            "y_train": y_train, "y_test": y_test,
            "feature_cols": feature_cols,
            "target_col": target_col,
            "scaler": scaler_obj,
            "label_map": label_map,
        }

    return {
        "ok": True,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_features": X_train.shape[1],
        "feature_cols": feature_cols,
        "target_col": target_col,
        "label_map": {str(k): v for k, v in label_map.items()} if label_map else None,
    }


def get_split(session_id: str) -> dict | None:
    with _lock:
        return _splits.get(session_id)


def predict_on_input(session_id: str, data: list, feature_cols: list) -> np.ndarray:
    """Transform raw dict/list input using the stored scaler and return X array."""
    with _lock:
        split = _splits.get(session_id)
    X = np.array(data, dtype="float32")
    if split and split.get("scaler"):
        X = split["scaler"].transform(X)
    return X
