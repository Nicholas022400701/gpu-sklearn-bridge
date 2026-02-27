"""
_local_test.py - 本地单元测试（不需要 WSL2/GPU，仅测试 shm_transport 层）
"""
import sys, time
import numpy as np

sys.path.insert(0, r"C:\Users\nicho\gpu-sklearn-bridge")

import importlib.util
spec = importlib.util.spec_from_file_location(
    "shm_transport",
    r"C:\Users\nicho\gpu-sklearn-bridge\shm_transport.py"
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

PASS = "[OK]"
FAIL = "[FAIL]"

def check(cond, msg):
    if cond:
        print(f"  {PASS}  {msg}")
    else:
        raise AssertionError(f"{FAIL}  {msg}")

print("=" * 60)
print(" shm_transport local unit tests")
print("=" * 60)

# ── 常量检查 ──────────────────────────────────────────────────
print("\n[1] 常量检查")
check(mod.SLOT_SIZE == 256 * 1024 * 1024,       f"SLOT_SIZE = 256 MB")
check(mod.INPUT_SLOT_COUNT == 4,                 f"INPUT_SLOT_COUNT = 4")
check(mod.OUTPUT_SLOT_COUNT == 4,                f"OUTPUT_SLOT_COUNT = 4")
check(mod.SCRATCH_SLOT_COUNT == 8,               f"SCRATCH_SLOT_COUNT = 8")
check(mod.TOTAL_SLOTS == 16,                     f"TOTAL_SLOTS = 16")
check(mod.POOL_SIZE == 4 * 1024 ** 3,            f"POOL_SIZE = 4 GB")
check(mod.SLOT_INPUT_START == 0,                 f"SLOT_INPUT_START = 0")
check(mod.SLOT_OUTPUT_START == 4,                f"SLOT_OUTPUT_START = 4")
check(mod.SLOT_SCRATCH_START == 8,               f"SLOT_SCRATCH_START = 8")
check(mod.MMAP_THRESHOLD == 10 * 1024,           f"MMAP_THRESHOLD = 10 KB")

# ── ShmTransport 初始化 ────────────────────────────────────────
print("\n[2] ShmTransport 初始化")
t = mod.ShmTransport.get()
check(t is not None, "单例创建成功")
check(mod.ShmTransport.get() is t, "单例复用（is 判断）")

import os
pool_path = r"C:\Users\nicho\gpu-sklearn-bridge\shm\pool.bin"
pool_size = os.path.getsize(pool_path)
check(pool_size == mod.POOL_SIZE, f"pool.bin 大小 = {pool_size/1e9:.1f} GB")

# ── write / read 往返 ──────────────────────────────────────────
print("\n[3] write / read 往返")

arr_s = np.arange(100, dtype=np.float32).reshape(10, 10)
meta = t.write(arr_s)
arr_back = t.read(meta)
check(arr_back.shape == arr_s.shape, "小数组形状一致")
check(np.allclose(arr_s, arr_back),  "小数组数值一致")

# 约 40 MB
arr_m = np.random.rand(10000, 1000).astype(np.float32)
t0 = time.perf_counter()
meta_m = t.write(arr_m)
write_ms = (time.perf_counter() - t0) * 1000
t0 = time.perf_counter()
arr_back_m = t.read(meta_m)
read_ms = (time.perf_counter() - t0) * 1000
mb = arr_m.nbytes / 1e6
check(np.allclose(arr_m, arr_back_m), f"中等数组 {mb:.0f} MB 往返数值一致")
print(f"       write={write_ms:.1f} ms  read={read_ms:.1f} ms  "
      f"吞吐≈{mb/(write_ms/1000)/1e3:.0f} GB/s")

# ── 输入 slot 轮转 ─────────────────────────────────────────────
print("\n[4] 输入 slot 轮转（is_output=False）")
# 重置计数器以便测试
t._input_counter = 0
slots = [t.write(np.zeros(1, dtype=np.float32))["slot"] for _ in range(8)]
check(slots == [0, 1, 2, 3, 0, 1, 2, 3], f"序列 {slots}")

# ── 输出 slot 轮转 ─────────────────────────────────────────────
print("\n[5] 输出 slot 轮转（is_output=True）")
t._output_counter = 0
slots_out = [t.write(np.zeros(1, dtype=np.float32), is_output=True)["slot"] for _ in range(8)]
check(slots_out == [4, 5, 6, 7, 4, 5, 6, 7], f"序列 {slots_out}")

# ── 不同 dtype 和形状 ──────────────────────────────────────────
print("\n[6] 多种 dtype 和形状")
for dtype in [np.float32, np.float64, np.int32, np.int64]:
    arr = np.random.rand(100, 50).astype(dtype)
    meta = t.write(arr)
    back = t.read(meta)
    check(back.shape == arr.shape and back.dtype == arr.dtype and np.allclose(arr, back),
          f"dtype={dtype.__name__}  shape=(100,50)")

# ── proxy._encode_array 行为 ──────────────────────────────────
print("\n[7] proxy._encode_array 行为")
from cuml_proxy.proxy import _encode_array, _decode_array, MMAP_THRESHOLD

small = np.ones((10,), dtype=np.float32)   # < 10 KB → base64
enc_small = _encode_array(small)
check(enc_small.get("__ndarray__") is True, "小数组走 __ndarray__ (base64)")
check("__mmap__" not in enc_small,           "小数组不走 mmap")

big = np.ones((100, 1000), dtype=np.float32)  # 400 KB → mmap
enc_big = _encode_array(big)
check(enc_big.get("__mmap__") is True,       "大数组走 __mmap__")
check("__ndarray__" not in enc_big,           "大数组不走 base64")

# ── _decode_array 往返 ────────────────────────────────────────
print("\n[8] proxy._decode_array 往返")
arr_orig = np.arange(200, dtype=np.float64).reshape(20, 10)
enc = _encode_array(arr_orig)
dec = _decode_array(enc)
check(np.allclose(arr_orig, dec), f"encode→decode 数值一致（dtype={arr_orig.dtype}）")

# ── server._encode_result 行为 ────────────────────────────────
print("\n[9] server._encode_result（读取 server.py 函数）")
import importlib.util as ilu, types

# 仅加载 _encode_result 函数逻辑，跳过 cuml import
src = open(r"C:\Users\nicho\gpu-sklearn-bridge\server.py", encoding="utf-8").read()
has_npy_fallback = "uuid.uuid4().hex" in src and "np.save" in src and "__file__" in src
check(not has_npy_fallback, "server.py 中无 .npy fallback（uuid + np.save）")
has_is_output = "is_output=True" in src
check(has_is_output, "server.py 使用 is_output=True 轮转分配")

# ── pool.bin 文件检查 ──────────────────────────────────────────
print("\n[10] shm/ 目录检查（无残留 .npy 文件）")
import glob
npy_files = glob.glob(r"C:\Users\nicho\gpu-sklearn-bridge\shm\????????????????????????????????????????.npy")
check(len(npy_files) == 0,
      f"无 UUID.npy 残留文件（当前 {len(npy_files)} 个）")

print("\n" + "=" * 60)
print(" All 10 test groups PASSED")
print("=" * 60)
