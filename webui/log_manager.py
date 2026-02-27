"""
Per-model logging system.
Each model gets its own log file under logs/<model_id>.log
Logs are also kept in memory for live display in the WebUI.
"""
import threading
from datetime import datetime, timezone
from pathlib import Path

_LOG_DIR = Path(__file__).parent.parent / "logs"
_LOG_DIR.mkdir(exist_ok=True)

_lock = threading.Lock()
# In-memory: { model_id: [{"ts": ..., "level": ..., "msg": ...}, ...] }
_memory_logs: dict = {}


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def log(model_id: str, message: str, level: str = "INFO") -> dict:
    """Append a log entry for *model_id*. Returns the entry dict."""
    entry = {"ts": _ts(), "level": level, "msg": message}
    line  = f"[{entry['ts']}] [{level}] {message}\n"

    with _lock:
        # In-memory
        if model_id not in _memory_logs:
            _memory_logs[model_id] = []
        _memory_logs[model_id].append(entry)

        # File
        log_path = _LOG_DIR / f"{model_id}.log"
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(line)
        except OSError:
            pass

    return entry


def get_logs(model_id: str) -> list:
    """Return all in-memory log entries for *model_id*."""
    with _lock:
        return list(_memory_logs.get(model_id, []))


def list_log_files() -> list:
    """Return list of model IDs that have log files."""
    try:
        return sorted(p.stem for p in _LOG_DIR.glob("*.log"))
    except OSError:
        return []


def read_log_file(model_id: str) -> str:
    """Return full text of the log file for *model_id*."""
    log_path = _LOG_DIR / f"{model_id}.log"
    try:
        return log_path.read_text(encoding="utf-8")
    except OSError:
        return ""


def clear_log(model_id: str) -> None:
    """Clear in-memory + file log for *model_id*."""
    with _lock:
        _memory_logs.pop(model_id, None)
        log_path = _LOG_DIR / f"{model_id}.log"
        try:
            log_path.unlink(missing_ok=True)
        except OSError:
            pass
