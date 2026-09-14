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

# ── 加载 mmap 传输层 ────────────────────────────────────────────────────
_shm_file = Path(__file__).parent.parent / "shm_transport.py"
_shm_spec = importlib.util.spec_from_file_location("shm_transport", _shm_file)
_shm_mod  = importlib.util.module_from_spec(_shm_spec)
_shm_spec.loader.exec_module(_shm_mod)
ShmTransport  = _shm_mod.ShmTransport
SLOT_INPUT_START = _shm_mod.SLOT_INPUT_START
MMAP_THRESHOLD = _shm_mod.MMAP_THRESHOLD

_BRIDGE_PORT = int(os.environ