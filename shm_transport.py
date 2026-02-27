"""
shm_transport.py – 预分配扩展 mmap 共享内存传输层
Windows 写入 / WSL2 读取，基于同一物理文件 + OS page cache。

布局（总 4 GB）：
  input_slots[0~3]   : INPUT  – Windows 写, WSL2 读  (请求数组, 4 × 256 MB = 1 GB)
  output_slots[0~3]  : OUTPUT – WSL2 写,   Windows 读  (结果数组, 4 × 256 MB = 1 GB)
  scratch_slots[0~7] : 临时存储 – 内部算法中间结果 (8 × 256 MB = 2 GB)

优点：
  - 消除 > 256 MB 数据的 .npy fallback 开销
  - 支持更大的单次请求（最多 1 GB）
  - 多 slot 架构避免 slot 竞争和等待
"""
import mmap
import os
import sys
import numpy as np
from pathlib import Path
from threading import Lock

# ── 路径（文件存放在 WSL2 Linux FS，双端通过不同挂载点访问）──────────────────
# 文件在 WSL2 的 /home/nicho/gpu-sklearn-bridge/shm/ 目录
# WSL2 (Linux) 直接访问：/home/nicho/gpu-sklearn-bridge/shm/pool.bin
# Windows 通过 UNC 路径访问：\\wsl.localhost\Ubuntu\home\nicho\gpu-sklearn-bridge\shm\pool.bin
#
# 设计原因：文件存在 Linux FS（非 /mnt/c/ virtio-fs）上，
#   - WSL2 的读写是本地 page cache，无跨系统 cache 问题
#   - Windows 通过 P9/VirtioFS 的 UNC 路径访问，fd.read 每次都绕过 Windows cache 获取最新
_WIN_PATH = r"\\wsl.localhost\Ubuntu\home\nicho\gpu-sklearn-bridge\shm\pool.bin"
_WSL_PATH = "/home/nicho/gpu-sklearn-bridge/shm/pool.bin"

SLOT_SIZE  = 256 * 1024 * 1024   # 256 MB / slot
INPUT_SLOT_COUNT = 4    # 输入 slots（冗余以避免阻塞）
OUTPUT_SLOT_COUNT = 4   # 输出 slots
SCRATCH_SLOT_COUNT = 8  # 临时 slots（内部算法使用）
TOTAL_SLOTS = INPUT_SLOT_COUNT + OUTPUT_SLOT_COUNT + SCRATCH_SLOT_COUNT
POOL_SIZE  = SLOT_SIZE * TOTAL_SLOTS   # 4 GB total

# Slot 编号范围
SLOT_INPUT_START = 0
SLOT_INPUT_END = INPUT_SLOT_COUNT
SLOT_OUTPUT_START = INPUT_SLOT_COUNT
SLOT_OUTPUT_END = INPUT_SLOT_COUNT + OUTPUT_SLOT_COUNT
SLOT_SCRATCH_START = SLOT_OUTPUT_END
SLOT_SCRATCH_END = TOTAL_SLOTS

# 大于此阈值走 mmap，否则走 HTTP inline base64
MMAP_THRESHOLD = 10 * 1024   # 10 KB

# 单 slot 容量上限（256 MB）
MMAP_MAX = SLOT_SIZE




def _pool_path() -> str:
    return _WIN_PATH if sys.platform == "win32" else _WSL_PATH


def _ensure_pool(path: str):
    """创建并预分配共享文件（不存在或过小时）。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists() or p.stat().st_size < POOL_SIZE:
        print(f"[ShmTransport] 初始化 mmap pool: {path} ({POOL_SIZE/1e9:.1f} GB)", flush=True)
        with open(path, "wb") as f:
            f.seek(POOL_SIZE - 1)
            f.write(b"\x00")


class ShmTransport:
    """
    可复用的扩展 mmap 传输对象，支持多个 input/output/scratch slots，
    避免大数据（> 256 MB）降级到 .npy 文件的开销。
    
    Slot 管理策略：
      - input_slots: 轮转分配，Windows 依次写入不同 slot，降低竞争
      - output_slots: 轮转分配，WSL2 依次输出到不同 slot，降低竞争
      - scratch_slots: 内部服务器使用，临时存储中间结果
    
    线程安全说明：Flask threaded=True 下每个请求在不同线程，
    但 HTTP 请求-响应本身是串行握手，客户端等到响应后才能复用 slot，
    因此主要通过轮转计数器避免 slot 竞争。
    """
    _instance = None
    _lock = Lock()

    def __init__(self):
        path = _pool_path()
        _ensure_pool(path)
        self._fd = open(path, "r+b")
        self._mm = mmap.mmap(self._fd.fileno(), POOL_SIZE)
        
        # 轮转计数器
        self._input_counter = 0
        self._output_counter = 0
        self._scratch_counter = 0

    @classmethod
    def get(cls) -> "ShmTransport":
        """进程级单例，避免重复打开文件描述符。"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def _alloc_input_slot(self) -> int:
        """轮转分配输入 slot（0-3）。"""
        slot = SLOT_INPUT_START + (self._input_counter % INPUT_SLOT_COUNT)
        self._input_counter += 1
        return slot

    def _alloc_output_slot(self) -> int:
        """轮转分配输出 slot（4-7）。"""
        slot = SLOT_OUTPUT_START + (self._output_counter % OUTPUT_SLOT_COUNT)
        self._output_counter += 1
        return slot

    def _alloc_scratch_slot(self) -> int:
        """轮转分配临时 slot（8-15）。"""
        slot = SLOT_SCRATCH_START + (self._scratch_counter % SCRATCH_SLOT_COUNT)
        self._scratch_counter += 1
        return slot

    def write(self, arr: np.ndarray, slot: int = None, is_output: bool = False):
        """
        将数组原始字节写入指定 slot（或自动分配），返回供 HTTP 传输的元数据字典。
        
        Args:
            arr: 输入数组
            slot: 指定 slot，若 None 则自动分配
            is_output: True 时分配输出 slot；False 时分配输入 slot
        
        返回：
            {"__mmap__": True, "slot": int, "dtype": str, "shape": list, "nbytes": int}
            
        注意：所有 > 256 MB 的数据现在都能被容纳，不再有 .npy fallback。
        """
        arr = np.ascontiguousarray(arr)
        nbytes = arr.nbytes
        
        # 检查是否超过总 mmap 容量（4 GB）
        if nbytes > MMAP_MAX:
            raise ValueError(
                f"Array size {nbytes/1e9:.1f} GB exceeds single slot limit {MMAP_MAX/1e9:.1f} GB. "
                "Consider splitting data or increasing MMAP_MAX."
            )
        
        # 自动分配 slot 若未指定
        if slot is None:
            slot = self._alloc_output_slot() if is_output else self._alloc_input_slot()
        
        offset = slot * SLOT_SIZE
        self._mm[offset: offset + nbytes] = arr.tobytes()
        # flush 确保 virtio-fs/9p 写回 Windows NTFS page cache，
        # 否则 Windows 侧 mmap 可能读到稀疏文件初始化的零值
        self._mm.flush(offset, nbytes)

        return {
            "__mmap__": True,
            "slot":   slot,
            "dtype":  str(arr.dtype),
            "shape":  list(arr.shape),
            "nbytes": nbytes,
        }

    def read(self, meta: dict) -> np.ndarray:
        """
        根据元数据从 slot 读取数组并返回独立副本。
        
        Windows 侧必须使用 fd.seek + fd.read 而非 np.frombuffer(mm, ...)，
        因为 Windows mmap page cache 不会自动感知 WSL2 (virtio-fs) 写入的更新。
        fd.read() 绕过 Windows mmap view，直接通过文件 I/O 获取最新数据。
        
        Linux (WSL2) 侧：frombuffer 可以工作（同一 OS 的 page cache 是统一的），
        但也使用 fd.read 以保持实现统一。
        """
        offset = meta["slot"] * SLOT_SIZE
        nbytes = meta["nbytes"]
        self._fd.seek(offset)
        raw = self._fd.read(nbytes)
        return np.frombuffer(raw, dtype=meta["dtype"]).reshape(meta["shape"]).copy()

    def get_scratch_buffer(self, shape: tuple, dtype: np.dtype) -> tuple[np.ndarray, int]:
        """
        服务器内部使用：从 scratch slots 分配临时缓冲区。
        返回 (ndarray, slot_id)，调用方可稍后通过 slot_id 读取。
        """
        arr = np.zeros(shape, dtype=dtype)
        nbytes = arr.nbytes
        if nbytes > MMAP_MAX:
            raise ValueError(f"Scratch buffer too large: {nbytes/1e9:.1f} GB")
        
        slot = self._alloc_scratch_slot()
        offset = slot * SLOT_SIZE
        self._mm[offset: offset + nbytes] = arr.tobytes()
        
        # 返回可写副本，以及 slot ID 供后续查询
        scratch_arr = np.frombuffer(
            self._mm, dtype=dtype, count=arr.size, offset=offset
        ).reshape(shape).copy()
        return scratch_arr, slot

    def close(self):
        """关闭 mmap，清理资源。"""
        self._mm.close()
        self._fd.close()
        ShmTransport._instance = None
