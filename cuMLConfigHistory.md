# cuML GPU sklearn 环境构建全流程记录

> **记录日期**：2026-02-26  
> **目标**：在 Windows 11 上通过 uv 新建 Python 环境，实现 GPU 版 scikit-learn（RAPIDS cuML），开机自启动，对 Windows 侧 Python 透明可用

---

## 目录

1. [环境基线检测](#1-环境基线检测)
2. [Windows 原生安装尝试（失败）](#2-windows-原生安装尝试失败)
3. [架构决策：WSL2 桥接方案](#3-架构决策wsl2-桥接方案)
4. [WSL2 代理配置](#4-wsl2-代理配置)
5. [WSL2 安装 uv 与 cuML](#5-wsl2-安装-uv-与-cuml)
6. [构建 Flask 桥接服务端](#6-构建-flask-桥接服务端)
7. [构建 Windows 侧代理包 cuml_proxy](#7-构建-windows-侧代理包-cuml_proxy)
8. [创建 Windows uv 环境](#8-创建-windows-uv-环境)
9. [开机自启动配置](#9-开机自启动配置)
10. [修复代理冲突](#10-修复代理冲突)
11. [文件结构总览](#11-文件结构总览)
12. [验证与测试](#12-验证与测试)
13. [已知限制与注意事项](#13-已知限制与注意事项)
14. [传输层演进：共享文件 Transport](#14-传输层演进共享文件-transport)
15. [传输层优化：mmap 共享内存](#15-传输层优化mmap-共享内存)
16. [模型持久化](#16-模型持久化)

---

## 1. 环境基线检测

```powershell
uv --version
# uv 0.10.2 (a788db7e5 2026-02-10)

nvcc --version
# Cuda compilation tools, release 12.8, V12.8.93

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
# NVIDIA GeForce RTX 4060 Laptop GPU, 576.80, 8188 MiB
```

**基线信息汇总：**

| 项目 | 版本/值 |
|---|---|
| OS | Windows 11 |
| uv | 0.10.2 |
| Python（系统） | 3.12.7（Anaconda base） |
| CUDA Toolkit | 12.8 |
| GPU | RTX 4060 Laptop 8 GB |
| 驱动 | 576.80 |
| WSL2 | 已安装，默认版本 2 |
| WSL2 发行版 | Ubuntu 24.04.2 LTS（已有） |

```powershell
wsl --list --verbose
# NAME              STATE    VERSION
# docker-desktop    Stopped  2
# Ubuntu            Stopped  2
```

---

## 2. Windows 原生安装尝试（失败）

### 2.1 尝试 uv 安装 cuml-cu12

```powershell
uv venv C:\Users\nicho\envs\gpu-sklearn --python 3.11
uv pip install cuml-cu12 --extra-index-url https://pypi.nvidia.com
```

**错误：**
```
cuml-cu12>=24.4.0 has no wheels with a matching platform tag (e.g., `win_amd64`)
```

### 2.2 尝试旧版本（Python 3.9）

```powershell
uv venv C:\Users\nicho\envs\gpu-sklearn-py39 --python 3.9
uv pip install "cuml-cu12==24.2.0" --extra-index-url https://pypi.nvidia.com
```

**错误：**
```
Wheels are available for `cuml-cu12` (v24.2.0) on the following platforms:
`manylinux_2_17_x86_64`, `manylinux_2_28_aarch64`
```

### 2.3 根本原因确认

```powershell
pip download cuml-cu12==25.8.0 --extra-index-url https://pypi.nvidia.com --no-deps -d C:\Temp\cuml_test
# RuntimeError: Didn't find wheel for cuml-cu12 25.8.0
# Operating System: Windows 11
# Wheels available: manylinux_2_24_x86_64, manylinux_2_28_aarch64 (Linux only)
```

**结论：NVIDIA RAPIDS cuML 官方从未发布任何 Windows (win_amd64) wheel，所有版本（23.6.0 ~ 26.2.0）均为 Linux 专属。此为 NVIDIA 架构决策，无法绕过。**

---

## 3. 架构决策：WSL2 桥接方案

### 3.1 方案选型

| 方案 | 可行性 | 说明 |
|---|---|---|
| Windows 原生 cuml | ❌ | 无 Windows wheel，永久不可行 |
| WSL2 + 透明桥接 | ✅ | GPU 驱动自动透传，本方案采用 |
| Docker Desktop + GPU | ✅ | 可行但更重 |
| Intel Extension for sklearn | ❌ | 非 NVIDIA GPU |

### 3.2 WSL2 GPU 透传验证

```bash
# 在 WSL2 Ubuntu 中执行
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
# NVIDIA GeForce RTX 4060 Laptop GPU, 576.80
```

Windows 侧驱动（576.80）自动通过 `/usr/lib/wsl/lib/` 透传给 WSL2，**无需在 WSL2 内单独安装 CUDA 驱动**。

### 3.3 桥接架构设计

```
Windows Python (cuml_proxy)
        │
        │  HTTP JSON-RPC  127.0.0.1:19876
        ↓
WSL2 Ubuntu Flask Server (server.py)
        │
        │  cuML Python API
        ↓
RAPIDS cuML 26.02 → RTX 4060 GPU
```

---

## 4. WSL2 代理配置

### 4.1 镜像网络模式尝试（失败）

创建 `C:\Users\nicho\.wslconfig`：
```ini
[wsl2]
networkingMode=mirrored
```

**错误：**
```
wsl: 出现了内部错误。错误代码: CreateInstance/CreateVm/ConfigureNetworking/0x8007054f
wsl: 无法配置网络 (networkingMode Mirrored)，回退到 networkingMode None
```

硬件不支持镜像模式，已回滚：
```ini
[wsl2]
```

### 4.2 netsh portproxy 尝试（失败）

```powershell
netsh interface portproxy add v4tov4 listenaddress=172.28.80.1 listenport=7890 connectaddress=127.0.0.1 connectport=7890
# 请求的操作需要提升（需要管理员权限）
```

### 4.3 最终方案：Windows 下载 → WSL2 执行

WSL2 可直接访问 `pypi.org` 和 `pypi.nvidia.com`（经验证）：
```bash
curl -sv --connect-timeout 10 https://pypi.nvidia.com/   # 200 OK
```

uv 安装脚本通过 Windows 代理下载后传入 WSL2 执行：
```powershell
# Windows 侧用代理下载
$env:https_proxy="http://127.0.0.1:7890"
Invoke-WebRequest -Uri "https://astral.sh/uv/install.sh" -OutFile "C:\Users\nicho\uv_install.sh"

# 在 WSL2 中执行（脚本已在本地，无需网络）
wsl -d Ubuntu -- bash /mnt/c/Users/nicho/uv_install.sh
```

---

## 5. WSL2 安装 uv 与 cuML

### 5.1 在 WSL2 中安装 uv

```bash
# uv 0.10.6 安装到 /home/nicho/.local/bin/
~/.local/bin/uv --version
# uv 0.10.6
```

### 5.2 创建 Python 3.11 虚拟环境

```bash
~/.local/bin/uv venv ~/envs/gpu-sklearn --python 3.11
# Using CPython 3.11.14
# Creating virtual environment at: /home/nicho/envs/gpu-sklearn
```

### 5.3 安装 cuML（直连 NVIDIA PyPI）

```bash
~/.local/bin/uv pip install \
    --python ~/envs/gpu-sklearn/bin/python \
    cuml-cu12 \
    --extra-index-url https://pypi.nvidia.com \
    --index-strategy unsafe-best-match
```

**下载内容（约 3 GB）：**

| 包 | 大小 |
|---|---|
| libcudf-cu12 | ~652 MB |
| nvidia-cublas-cu12 | ~554 MB |
| libcuml-cu12 | ~443 MB |
| nvidia-cusparse-cu12 | ~349 MB |
| nvidia-cusolver-cu12 | ~322 MB |
| nvidia-nccl-cu12 | ~276 MB |
| nvidia-cufft-cu12 | ~191 MB |
| nvidia-cuda-nvrtc-cu12 | ~85 MB |
| nvidia-curand-cu12 | ~65 MB |
| nvidia-cuda-nvcc-cu12 | ~38 MB |
| nvidia-nvjitlink-cu12 | ~37 MB |
| 其余依赖 | ~小包 |

**安装 Flask（供桥接服务使用）：**
```bash
~/.local/bin/uv pip install flask \
    --python ~/envs/gpu-sklearn/bin/python
```

**验证：**
```bash
~/envs/gpu-sklearn/bin/python -c "import cuml; print(cuml.__version__)"
# 26.02.000
```

---

## 6. 构建 Flask 桥接服务端

**文件：** `C:\Users\nicho\gpu-sklearn-bridge\server.py`（同步至 WSL2 `~/gpu-sklearn-bridge/server.py`）

### 6.1 支持的 cuML 类

| 模块 | 类 |
|---|---|
| `linear_model` | LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet |
| `cluster` | KMeans, DBSCAN |
| `decomposition` | PCA, TruncatedSVD |
| `neighbors` | KNeighborsClassifier, KNeighborsRegressor, NearestNeighbors |
| `ensemble` | RandomForestClassifier, RandomForestRegressor |
| `svm` | SVC, SVR |
| `preprocessing` | StandardScaler, MinMaxScaler, LabelEncoder |
| `manifold` | TSNE, UMAP |

### 6.2 API 端点

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/health` | 健康检查，返回 cuML 版本 |
| POST | `/create` | 创建模型实例，返回 `model_id` |
| POST | `/call/<model_id>/<method>` | 调用模型方法（fit/predict/transform 等） |
| GET | `/get_params/<model_id>` | 获取模型参数 |
| DELETE | `/delete/<model_id>` | 销毁模型实例 |
| POST | `/save/<model_id>` | 将模型 pickle 到 `models/{name}.pkl` |
| POST | `/load` | 从 `models/{name}.pkl` 反序列化并注册 |
| GET | `/list_models` | 列出 `models/` 目录中的全部保存文件 |

### 6.3 数组序列化协议（三级传输）

| 数组大小 | 传输方式 | 标识键 |
|---|---|---|
| < 10 KB | HTTP inline Base64 | `__ndarray__` |
| ≥ 10 KB（旧备用） | 共享 `.npy` 文件（`shm/` 目录） | `__file__` |
| ≥ 10 KB（当前默认） | **mmap 预分配共享内存** `shm/pool.bin` | `__mmap__` |

mmap 元数据（通过 HTTP JSON 传递）：
```json
{
  "__mmap__": true,
  "slot": 0,
  "dtype": "float32",
  "shape": [1000, 20],
  "nbytes": 80000
}
```

`slot 0`（INPUT）由 Windows 写入，`slot 1`（OUTPUT）由 WSL2 写入，数组内容不经过 HTTP body。

### 6.4 启动脚本

**`~/gpu-sklearn-bridge/start_server.sh`**：
```bash
PYTHON="$HOME/envs/gpu-sklearn/bin/python"
SERVER="$HOME/gpu-sklearn-bridge/server.py"
nohup "$PYTHON" "$SERVER" >> "$HOME/gpu-sklearn-bridge/server.log" 2>&1 &
```

**`C:\Users\nicho\gpu-sklearn-bridge\start_bridge.bat`**（Windows 入口）：
```bat
wsl -d Ubuntu -- bash /mnt/c/Users/nicho/gpu-sklearn-bridge/start_server.sh
```

---

## 7. 构建 Windows 侧代理包 cuml_proxy

**位置：** `C:\Users\nicho\gpu-sklearn-bridge\cuml_proxy\`

### 7.1 包结构

```
cuml_proxy/
├── __init__.py
├── proxy.py          # 核心代理类 ProxyEstimator
├── linear_model.py
├── cluster.py
├── decomposition.py
├── neighbors.py
├── ensemble.py
├── svm.py
├── preprocessing.py
└── manifold.py
```

### 7.2 核心设计：ProxyEstimator

- 实现 sklearn 标准接口：`fit / predict / transform / fit_transform / fit_predict / score / get_params / set_params`
- 通过 `_session`（`trust_env=False`）发送 HTTP 请求到桥接服务
- 模型实例生命周期由 Python 对象生命周期管理（`__del__` 时自动 DELETE）

---

## 8. 创建 Windows uv 环境

```powershell
# 创建 Python 3.11 虚拟环境
uv venv C:\Users\nicho\envs\cuml-proxy --python 3.11
# Using CPython 3.11.13
# Creating virtual environment at: C:\Users\nicho\envs\cuml-proxy

# 安装依赖
uv pip install --python C:\Users\nicho\envs\cuml-proxy\Scripts\python.exe requests numpy
# Installed: certifi, charset-normalizer, idna, numpy, requests, urllib3
```

### 8.1 注册 cuml_proxy 包路径

无需 `pip install`，直接写 `.pth` 文件实现可导入：

```powershell
Set-Content `
  "C:\Users\nicho\envs\cuml-proxy\Lib\site-packages\cuml_proxy_bridge.pth" `
  "C:\Users\nicho\gpu-sklearn-bridge"
```

**验证：**
```powershell
C:\Users\nicho\envs\cuml-proxy\Scripts\python.exe -c "
from cuml_proxy.linear_model import LinearRegression
print(repr(LinearRegression()))
# LinearRegression() [GPU via WSL2]
"
```

---

## 9. 开机自启动配置

### 9.1 首选方案：Task Scheduler（失败，需管理员）

```powershell
Register-ScheduledTask ...
# 拒绝访问。HRESULT 0x80070005
```

### 9.2 最终方案：注册表 HKCU Run（无需管理员）

```powershell
Set-ItemProperty `
  -Path "HKCU:\Software\Microsoft\Windows\CurrentVersion\Run" `
  -Name "GPU-sklearn-bridge" `
  -Value "C:\Users\nicho\gpu-sklearn-bridge\start_bridge.bat"
```

**验证：**
```powershell
Get-ItemProperty "HKCU:\Software\Microsoft\Windows\CurrentVersion\Run" |
  Select-Object "GPU-sklearn-bridge"
# GPU-sklearn-bridge : C:\Users\nicho\gpu-sklearn-bridge\start_bridge.bat
```

**启动时序：**
```
用户登录
  └─ HKCU\Run → start_bridge.bat
       └─ wsl -d Ubuntu
            └─ start_server.sh
                 └─ nohup python server.py &  (后台运行，端口 19876)
```

---

## 10. 修复代理冲突

### 10.1 问题

`requests` 默认读取 `HTTP_PROXY` / `HTTPS_PROXY` 环境变量。系统代理（Clash/V2Ray，端口 7890）设置这些变量后，发往 `127.0.0.1:19876` 的桥接请求会被代理软件拦截，导致连接失败。

### 10.2 修复

在 `cuml_proxy/proxy.py` 中创建独立 Session：

```python
_session = requests.Session()
_session.trust_env = False                        # 不读取环境变量代理
_session.proxies = {"http": None, "https": None}  # 显式清空
```

所有桥接请求均通过 `_session` 发送，完全绕过系统代理。

### 10.3 验证（模拟代理已设置）

```powershell
$env:HTTP_PROXY="http://127.0.0.1:7890"
$env:HTTPS_PROXY="http://127.0.0.1:7890"
C:\Users\nicho\envs\cuml-proxy\Scripts\python.exe -c "
from cuml_proxy.linear_model import LinearRegression
import numpy as np
lr = LinearRegression()
lr.fit(np.random.rand(100,5).astype('float32'), np.random.rand(100).astype('float32'))
print('OK - 系统代理不干扰桥接请求')
"
# OK - 系统代理不干扰桥接请求
```

---

## 11. 文件结构总览

```
C:\Users\nicho\
├── .wslconfig                          # WSL2 配置（保留，networkingMode 默认）
├── gpu-sklearn-bridge\
│   ├── server.py                       # WSL2 Flask 桥接服务（同步到 WSL2 home）
│   ├── start_bridge.bat                # Windows 启动入口（HKCU Run 指向此文件）
│   ├── start_server.sh                 # WSL2 内部启动脚本
│   └── cuml_proxy\                     # Windows 侧代理包
│       ├── __init__.py
│       ├── proxy.py                    # ProxyEstimator 核心实现
│       ├── linear_model.py
│       ├── cluster.py
│       ├── decomposition.py
│       ├── neighbors.py
│       ├── ensemble.py
│       ├── svm.py
│       ├── preprocessing.py
│       └── manifold.py
└── envs\
    ├── gpu-sklearn\                    # 已删除（Windows 创建失败）
    ├── gpu-sklearn-py39\               # 已删除（Windows 创建失败）
    └── cuml-proxy\                     # ✅ Windows 正式环境
        ├── Scripts\python.exe          # Python 3.11.13
        └── Lib\site-packages\
            └── cuml_proxy_bridge.pth   # 指向 gpu-sklearn-bridge\

WSL2 Ubuntu (~/)
├── envs\
│   └── gpu-sklearn\                   # ✅ Linux GPU 环境
│       └── bin\python                 # Python 3.11.14 + cuML 26.02
└── gpu-sklearn-bridge\
    ├── server.py                      # 桥接服务（从 Windows 同步）
    ├── start_server.sh
    ├── server.log                     # 运行日志
    └── server.pid                     # 进程 ID

注册表
└── HKCU\Software\Microsoft\Windows\CurrentVersion\Run
    └── GPU-sklearn-bridge             # → start_bridge.bat（开机自启）
```

---

## 12. 验证与测试

### 12.1 服务健康检查

```powershell
Invoke-RestMethod "http://127.0.0.1:19876/health"
# cuml_version  status
# 26.02.000     ok
```

### 12.2 端到端功能测试

```powershell
C:\Users\nicho\envs\cuml-proxy\Scripts\python.exe -c "
import numpy as np
from cuml_proxy.cluster import KMeans
from cuml_proxy.decomposition import PCA
from cuml_proxy.preprocessing import StandardScaler

X = np.random.rand(500, 20).astype(np.float32)
X_pca = PCA(n_components=5).fit_transform(X)      # (500,20) -> (500,5)
labels = KMeans(n_clusters=3).fit_predict(X)       # {0,1,2}
X_s = StandardScaler().fit_transform(X)            # mean~0
print('All GPU ops OK!')
"
```

### 12.3 Iris 数据集 ML 实验结果

5-Fold Cross Validation（全量 150 样本）：

| 模型 | 均值 | ±std |
|---|---|---|
| SVC(RBF) | **0.9667** | **±0.0211** |
| RandomForest(100) | 0.9600 | ±0.0249 |
| KNeighborsClassifier(5) | 0.9600 | ±0.0327 |
| LogisticRegression | 0.9533 | ±0.0340 |

---

## 13. 已知限制与注意事项

| 限制 | 说明 |
|---|---|
| WSL2 镜像网络不可用 | 硬件不支持，使用 NAT 模式，Windows 通过自动端口转发访问 WSL2 |
| 首次启动延迟 | WSL2 冷启动约 3-5 秒，服务就绪约 8-10 秒，代理包内置 `wait_for_server()` |
| mmap 单 slot 上限 256 MB | 单次传输超过 256 MB 的数组会报错，建议分批或在 WSL2 内直接处理超大数据 |
| mmap 非真正共享内存 | pool.bin 经由 virtio-fs 文件系统访问，仍有一次内核态 I/O，非 `AF_VSOCK` 级零拷贝 |
| 开机自启动需用户登录 | HKCU Run 在用户登录时触发，非服务级自启动。若需登录前可用，需管理员权限注册 Task Scheduler |
| WSL2 关机后需重启桥接 | 执行 `wsl --shutdown` 后需重新运行 `start_bridge.bat` 或重新登录 |

### 手动重启桥接服务

```powershell
# 方法1：直接运行批处理
C:\Users\nicho\gpu-sklearn-bridge\start_bridge.bat

# 方法2：PowerShell
wsl -d Ubuntu -- bash /mnt/c/Users/nicho/gpu-sklearn-bridge/start_server.sh
```

### 查看服务日志

```bash
# 在 WSL2 中
tail -f ~/gpu-sklearn-bridge/server.log
```

---

## 14. 传输层演进：共享文件 Transport

### 14.1 背景

最初所有 numpy 数组均以 Base64 编码嵌入 HTTP JSON body。当数据集超过几百 KB 时，编码/解码耗时及网络 buffer 均成为瓶颈。

### 14.2 共享文件方案（已被 mmap 取代，作为 fallback 保留）

| 端 | 操作 |
|---|---|
| Windows（写） | `np.save(shm/{uuid}.npy, arr)` |
| HTTP JSON | 传递文件名 `{"__file__": true, "name": "xxx.npy"}` |
| WSL2（读） | `np.load(shm/{uuid}.npy)` → 读后 `os.unlink` |

**阈值**：10 KB，超过则走共享文件，否则 inline base64。  
共享目录 `C:\Users\nicho\gpu-sklearn-bridge\shm\` 在 WSL2 中映射为 `/mnt/c/Users/nicho/gpu-sklearn-bridge/shm/`，两端访问同一物理路径。

### 14.3 缺点（促成 mmap 升级）

- 每次调用产生一次文件创建 + 一次 `unlink`，文件系统 syscall 开销大
- `np.save` 写入含格式头，`np.load` 需解析格式头
- 高频调用时 `shm/` 目录下可能堆积临时文件

---

## 15. 传输层优化：mmap 共享内存

### 15.1 方案设计

**文件：** `C:\Users\nicho\gpu-sklearn-bridge\shm_transport.py`（同步至 WSL2）

预先分配一个 512 MB 的二进制文件 `shm/pool.bin`，分为两个固定 slot：

```
pool.bin（512 MB）
┌──────────────────────────────┬──────────────────────────────┐
│  slot 0 INPUT   256 MB       │  slot 1 OUTPUT  256 MB       │
│  Windows 写 → WSL2 读        │  WSL2 写 → Windows 读        │
└──────────────────────────────┴──────────────────────────────┘
```

### 15.2 核心实现

```python
class ShmTransport:
    _instance = None  # 进程级单例

    def write(self, arr: np.ndarray, slot: int) -> dict:
        """原始字节写入 slot，返回元数据字典（经 HTTP 传输）"""
        arr = np.ascontiguousarray(arr)
        offset = slot * SLOT_SIZE
        self._mm[offset: offset + arr.nbytes] = arr.tobytes()  # 单次 memcpy
        return {"__mmap__": True, "slot": slot,
                "dtype": str(arr.dtype), "shape": list(arr.shape), "nbytes": arr.nbytes}

    def read(self, meta: dict) -> np.ndarray:
        """np.frombuffer 直接引用 mmap 页，零额外分配"""
        offset = meta["slot"] * SLOT_SIZE
        arr = np.frombuffer(self._mm, dtype=meta["dtype"],
                            count=meta["nbytes"] // np.dtype(meta["dtype"]).itemsize,
                            offset=offset).reshape(meta["shape"])
        return arr.copy()  # 返回可写副本
```

### 15.3 与旧方案对比

| 操作 | 共享 .npy 文件 | mmap pool.bin |
|---|---|---|
| 文件创建 | 每次调用 1 次 | 仅首次（预分配） |
| 格式头解析 | `np.save/load` 每次解析 | 无，原始字节 |
| 文件删除 | 每次 `unlink` | 无 |
| 内存拷贝 | ≥ 2 次 | 1 次（`tobytes` → mmap） |
| OS page cache | 冷 | pool.bin 常驻内存，热路径 |

### 15.4 测试结果（2026-02-26）

```
[2] 中等数组（1000×20, 80 KB）—— mmap
    耗时: 29.8 ms

[3] 大数组（5000×480, ~9.6 MB）—— mmap
    耗时: 182.2 ms
    等效吞吐: 52.7 MB/s（含 HTTP 握手 + GPU 计算）

[5] 大数组输入 + 大数组输出往返（10000×20, ~781 KB）
    耗时: 58.3 ms
    全部测试通过 ✅
```

---

## 16. 模型持久化

### 16.1 API

```python
# 保存（将模型 pickle 到 models/{name}.pkl）
model.save("my_model")
# [cuml_proxy] 模型已保存 → C:\Users\nicho\gpu-sklearn-bridge\models\my_model.pkl

# 加载（无需重新训练，直接推理）
from cuml_proxy.proxy import ProxyEstimator
model = ProxyEstimator.load("my_model")

# 列出所有已保存模型
ProxyEstimator.list_saved()
# ['iris_svm', 'test_scaler', 'test_svm', ...]
```

### 16.2 实现原理

```
model.save("name")
  → POST /save/{model_id}  {"name": "name"}
  → WSL2: pickle.dump(model, open("models/name.pkl", "wb"), protocol=5)

ProxyEstimator.load("name")
  → POST /load  {"name": "name"}
  → WSL2: model = pickle.load(open("models/name.pkl", "rb"))
  → 注册新 model_id，返回 {model_id, class, params}
  → Windows: 返回带该 model_id 的 ProxyEstimator 实例
```

模型文件位于 `C:\Users\nicho\gpu-sklearn-bridge\models\`，WSL2 通过 `/mnt/c/...` 写入，Windows 可直接查看 `.pkl` 文件。

### 16.3 验证结果（2026-02-26）

```
训练准确率: 0.98
[cuml_proxy] 模型已保存 → ...\models\test_scaler.pkl
[cuml_proxy] 模型已保存 → ...\models\test_svm.pkl
保存列表: ['iris_svm', 'test_scaler', 'test_svm']
[cuml_proxy] 模型已加载: StandardScaler (id=StandardScaler_8)
[cuml_proxy] 模型已加载: SVC (id=SVC_9)
加载后准确率: 0.98
预测完全一致: True
权重保存/加载 OK ✅
```

---

*环境构建完成，传输层与持久化均已验证正常运行（最后更新：2026-02-26）。*
