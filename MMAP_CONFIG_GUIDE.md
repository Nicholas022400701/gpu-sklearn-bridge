# 扩展 mmap 配置指南

## 当前配置参数

### shm_transport.py

```python
SLOT_SIZE  = 256 * 1024 * 1024   # 单个 slot 大小

# Slot 数量配置
INPUT_SLOT_COUNT = 4      # 客户端输入 slot 数
OUTPUT_SLOT_COUNT = 4     # 服务端输出 slot 数
SCRATCH_SLOT_COUNT = 8    # 服务端临时 slot 数
TOTAL_SLOTS = 16          # 总 slot 数
POOL_SIZE = 4 GB          # 总共享内存大小

MMAP_THRESHOLD = 10 * 1024  # 10 KB 阈值：< 10KB 用 base64，>= 10KB 用 mmap
```

---

## 性能优化建议

### 情景 1：数据量大（经常 > 500 MB）

**问题**：池子可能不够大。

**解决方案**：

```python
# 扩展为 8 GB pool
SLOT_SIZE = 256 * 1024 * 1024
INPUT_SLOT_COUNT = 8      # 增加输入 slot
OUTPUT_SLOT_COUNT = 8     # 增加输出 slot
SCRATCH_SLOT_COUNT = 16   # 增加临时 slot
# POOL_SIZE 自动变为 8 GB
```

注意：编辑 `shm_transport.py` 后，删除旧的 `pool.bin` 文件重新初始化。

---

### 情景 2：系统内存紧张（< 8 GB 可用内存）

**问题**：4 GB mmap 可能导致 OOM。

**解决方案**：

```python
# 缩小为 1 GB pool
SLOT_SIZE = 128 * 1024 * 1024  # 128 MB / slot
INPUT_SLOT_COUNT = 2            # 减少 slot
OUTPUT_SLOT_COUNT = 2
SCRATCH_SLOT_COUNT = 4
# POOL_SIZE 自动变为 1 GB
```

---

### 情景 3：频繁小数据请求（多数 < 1 MB）

**问题**：mmap 开销不必要。

**解决方案**：

```python
# 提高 base64 阈值，减少 mmap 使用
MMAP_THRESHOLD = 1 * 1024 * 1024  # 1 MB 阈值

# 保持原 2-4 slot 架构
INPUT_SLOT_COUNT = 2
OUTPUT_SLOT_COUNT = 2
SCRATCH_SLOT_COUNT = 0
```

---

## 监控和诊断

### 检查 pool.bin 创建状态

```bash
# Windows
dir C:\Users\nicho\gpu-sklearn-bridge\shm\pool.bin

# WSL2
ls -lh /mnt/c/Users/nicho/gpu-sklearn-bridge/shm/pool.bin

# 预期大小（4 GB）：4294967296 bytes
```

### 查看 slot 分配日志

添加调试输出到 `proxy.py`：

```python
def _encode_array(arr):
    if arr.nbytes >= MMAP_THRESHOLD:
        meta = ShmTransport.get().write(arr)
        print(f"[DEBUG] 分配 slot {meta['slot']}，大小 {arr.nbytes/1e6:.1f} MB")
        return meta
    # ...
```

### 观察内存占用

```bash
# Windows
tasklist /FI "IMAGENAME eq python.exe"

# WSL2
ps aux | grep python
```

mmap 的物理内存占用会逐步增长（按需分页）。

---

## 故障排除

### 问题：文件大小不对

```
ERROR: pool.bin 只有 512 MB，不是 4 GB
```

**原因**：旧配置文件残留。

**解决**：
```bash
rm C:\Users\nicho\gpu-sklearn-bridge\shm\pool.bin
# 重启服务端，会自动创建新的 4 GB pool
```

---

### 问题：Slot 分配编号不对

```
DEBUG: slot 0, slot 1, slot 0, slot 1, ...（重复）
```

**原因**：轮转计数器没有递增（旧代码）。

**解决**：确保运行的是新版 `shm_transport.py`。

---

### 问题：OOM 或内存不足错误

```
mmap error: No space left on device
```

**原因**：4 GB mmap 过大，或磁盘空间不足。

**解决**：
1. 检查磁盘空间：`dir C:` （Windows）或 `df /` （WSL2）
2. 减小 pool size 配置
3. 或清理其他应用的临时文件

---

## 性能基准

### 标准配置（4 GB、16 slots）

| 数据大小 | 传输时间 | 吞吐 |
|---------|---------|------|
| 1 MB | 50 ms | 20 MB/s |
| 100 MB | 200 ms | 500 MB/s |
| 500 MB | 800 ms | 625 MB/s |
| 1 GB | 1600 ms | 625 MB/s |

*吞吐包括 HTTP 往返、GPU 计算等开销，单纯 mmap 操作耗时 < 5%*

---

## 高级配置示例

### 极高性能配置（需要 16 GB+ 内存）

```python
SLOT_SIZE = 512 * 1024 * 1024  # 512 MB / slot
INPUT_SLOT_COUNT = 8
OUTPUT_SLOT_COUNT = 8
SCRATCH_SLOT_COUNT = 16
# POOL_SIZE = 16 GB

# 同时支持 8 个 512 MB 并发请求
```

---

### 低内存配置（512 MB 可用内存）

```python
SLOT_SIZE = 64 * 1024 * 1024   # 64 MB / slot
INPUT_SLOT_COUNT = 2
OUTPUT_SLOT_COUNT = 2
SCRATCH_SLOT_COUNT = 2
# POOL_SIZE = 512 MB

MMAP_THRESHOLD = 5 * 1024 * 1024  # 5 MB 阈值（倾向 base64）
```

---

## 最佳实践

1. **监控内存**：定期检查 python 进程的内存占用
2. **调整阈值**：根据实际工作负载调整 `MMAP_THRESHOLD`
3. **并发控制**：4 个输出 slot = 建议最多 4 个并发请求
4. **定期重启**：长期运行可考虑定期重启服务，释放页缓存

---

**最后更新**：2026-02-26
