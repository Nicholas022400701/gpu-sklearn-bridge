# gpu-sklearn-bridge

<p align="center">
  <img src="https://img.shields.io/badge/platform-Windows%2011%20%2B%20WSL2-blue?logo=windows" alt="platform">
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white" alt="python">
  <img src="https://img.shields.io/badge/CUDA-12.x-76B900?logo=nvidia&logoColor=white" alt="cuda">
  <img src="https://img.shields.io/badge/cuML-26.02-76B900?logo=nvidia&logoColor=white" alt="cuml">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="license">
</p>

> **在 Windows 上透明使用 RAPIDS cuML GPU 加速机器学习**
>
> NVIDIA 从未发布任何 Windows 版 cuML wheel。本项目通过 WSL2 桥接，让你在 Windows Python 里直接 `import cuml`，所有计算在 GPU 上完成，行为与官方 cuML 完全一致。

---

## 目录

- [环境信息](#环境信息)
- [快速开始](#快速开始)
- [架构概览](#架构概览)
- [支持的算法](#支持的算法)
- [安装与部署](#安装与部署)
- [模型保存与加载](#模型保存与加载)
- [开机自启动](#开机自启动)
- [手动管理服务](#手动管理服务)
- [文件结构](#文件结构)
- [性能参考](#性能参考)
- [已知限制](#已知限制)
- [导入方式对比](#导入方式对比)
- [依赖](#依赖)
- [Contributing](#contributing)
- [License](#license)

---

## 环境信息

| 项目 | 版本 |
|---|---|
| OS | Windows 11 |
| GPU | NVIDIA RTX 4060 Laptop 8 GB |
| CUDA Toolkit | 12.8 |
| 驱动 | 576.80 |
| WSL2 发行版 | Ubuntu 24.04.2 LTS |
| cuML | 26.02.000 |
| Python（Windows） | 3.11.13（uv venv） |
| Python（WSL2） | 3.11.14（uv venv） |

---

## 快速开始

### 1. 确认服务已运行

```powershell
Invoke-RestMethod "http://127.0.0.1:19876/health"
# cuml_version  status
# 26.02.000     ok
```

若连接失败，手动启动：

```powershell
C:\Users\nicho\gpu-sklearn-bridge\start_bridge.bat
# 等待约 8 秒后重试
```

### 2. 激活 Windows 环境

```powershell
C:\Users\nicho\envs\cuml-proxy\Scripts\Activate.ps1
```

### 3. 直接 `import cuml` 使用（推荐）

```python
import cuml                              # ← 与官方 cuML 写法完全相同
from cuml.svm import SVC
from cuml.preprocessing import StandardScaler
from cuml.decomposition import PCA
from cuml.linear_model import LogisticRegression
from cuml.cluster import KMeans
import numpy as np

X = np.random.rand(1000, 20).astype("float32")
y = (X[:, 0] > 0.5).astype("float32")

sc = StandardScaler()
X_s = sc.fit_transform(X)

svm = SVC(kernel="rbf", C=1.0)
svm.fit(X_s, y)
print(svm.predict(X_s[:5]))             # GPU 推理

print(cuml.__version__)                  # 26.02.000
```

---

## 架构概览

```
Windows Python
  import cuml          ← cuml/ 是本地 shim，自动转发到 cuml_proxy
  import cuml_proxy    ← 效果相同，显式写法
       │
       │  ① HTTP JSON-RPC  127.0.0.1:19876
       │  ② 数组 ≥ 10 KB → 扩展 mmap pool.bin（4 GB 预分配，16 slots）
       │     Windows 通过 \\wsl.localhost\Ubuntu\... UNC 路径访问
       ↓
WSL2 Ubuntu  server.py  (Flask)
  ~/gpu-sklearn-bridge/shm/pool.bin  ← pool.bin 存储于 WSL2 Linux FS (ext4)
       │
       │  import cuml（真正的 RAPIDS cuML）
       ↓
RAPIDS cuML 26.02 → RTX 4060 GPU
```

### 传输层三级策略

| 数组大小 | 传输方式 | 说明 |
|---|---|---|
| < 10 KB | HTTP inline Base64 | 直接嵌入 JSON body |
| ≥ 10 KB | **扩展 mmap** `pool.bin` | 4 GB 预分配池，16 slots 轮转分配；pool.bin 存于 WSL2 Linux FS，Windows 通过 UNC `\\wsl.localhost\...` 访问 |
| 模型文件 | pickle `.pkl` | 主动调用 `save()` / `load()` 持久化 |

### 扩展 mmap 布局（4 GB，16 slots）

```
pool.bin
┌─────────────────────────┬─────────────────────────┬──────────────────────────────────────────┐
│   Input  slots 0-3      │  Output  slots 4-7       │         Scratch slots 8-15               │
│      1 GB (4×256 MB)    │     1 GB (4×256 MB)      │             2 GB (8×256 MB)              │
│   Windows 写 → WSL2 读  │  WSL2 写 → Windows 读   │         服务端内部临时缓冲区              │
└─────────────────────────┴─────────────────────────┴──────────────────────────────────────────┘
```

客户端和服务端各自维护**轮转计数器**，每次请求自动分配下一个 slot（0→1→2→3→0…），实现最多 4 个并发不阻塞。

---

## 支持的算法

| 模块 | 类 |
|---|---|
| `cuml.linear_model` | `LinearRegression` `LogisticRegression` `Ridge` `Lasso` `ElasticNet` |
| `cuml.svm` | `SVC` `SVR` |
| `cuml.cluster` | `KMeans` `DBSCAN` |
| `cuml.decomposition` | `PCA` `TruncatedSVD` |
| `cuml.neighbors` | `KNeighborsClassifier` `KNeighborsRegressor` `NearestNeighbors` |
| `cuml.ensemble` | `RandomForestClassifier` `RandomForestRegressor` |
| `cuml.preprocessing` | `StandardScaler` `MinMaxScaler` `LabelEncoder` |
| `cuml.manifold` | `TSNE` `UMAP` |

所有类均实现 scikit-learn 标准接口：`fit` / `predict` / `transform` / `fit_transform` / `fit_predict` / `score` / `get_params` / `set_params`。

---

## 安装与部署

> 以下路径中的 `<USER>` 请替换为你自己的 Windows 用户名，`<DISTRO>` 替换为你的 WSL2 发行版名（默认 `Ubuntu`）。

### 前提条件

| 要求 | 说明 |
|---|---|
| Windows 10/11 (x64) | 需支持 WSL2 |
| NVIDIA GPU | 驱动 ≥ 525，CUDA Toolkit 12.x |
| WSL2 + Ubuntu | `wsl --install -d Ubuntu` |
| Python 3.11 | Windows 端和 WSL2 端均需安装 |

### 1. 克隆仓库

```powershell
# Windows 端（PowerShell）
git clone https://github.com/Nicholas022400701/gpu-sklearn-bridge.git C:\Users\<USER>\gpu-sklearn-bridge

# 同步到 WSL2
wsl -d <DISTRO> -- git clone https://github.com/Nicholas022400701/gpu-sklearn-bridge.git ~/gpu-sklearn-bridge
```

### 2. 配置 WSL2 服务端

```bash
cd ~/gpu-sklearn-bridge
pip install flask numpy
# 确认 cuML 已安装（参考 https://docs.rapids.ai/install）
python -c "import cuml; print(cuml.__version__)"
```

### 3. 配置 Windows 客户端

```powershell
uv venv C:\Users\<USER>\envs\cuml-proxy
C:\Users\<USER>\envs\cuml-proxy\Scripts\Activate.ps1
pip install numpy requests

# 将项目目录加入 sys.path（以 .pth 文件方式）
$site = python -c "import site; print(site.getsitepackages()[0])"
"C:\Users\<USER>\gpu-sklearn-bridge" | Out-File "$site\cuml_proxy_bridge.pth" -Encoding ascii
```

### 4. 设置环境变量（可选，自定义路径）

```powershell
$Env:SKLEARN_BRIDGE_PORT   = "19876"
$Env:SKLEARN_BRIDGE_SHARED = "C:\Users\<USER>\gpu-sklearn-bridge\shm"
$Env:SKLEARN_BRIDGE_MODELS = "C:\Users\<USER>\gpu-sklearn-bridge\models"
```

### 5. 配置开机自启（可选）

```powershell
.\scripts\install_windows.ps1
```

### 6. 验证

```powershell
.\start_bridge.bat
Invoke-RestMethod "http://127.0.0.1:19876/health"
# cuml_version  status
# 26.02.000     ok
```

---

## 模型保存与加载

```python
from cuml.svm import SVC
from cuml.preprocessing import StandardScaler
from cuml_proxy.proxy import ProxyEstimator
import numpy as np

X = np.random.rand(200, 10).astype("float32")
y = (X[:, 0] > 0.5).astype("float32")

# 训练
sc = StandardScaler()
X_s = sc.fit_transform(X)
svm = SVC(kernel="rbf")
svm.fit(X_s, y)

# 保存权重（pickle 到 models/ 目录）
sc.save("my_scaler")    # → models/my_scaler.pkl
svm.save("my_svm")      # → models/my_svm.pkl

# 列出所有已保存模型
print(ProxyEstimator.list_saved())   # ['my_scaler', 'my_svm', ...]

# 加载（无需重新训练）
sc2  = ProxyEstimator.load("my_scaler")
svm2 = ProxyEstimator.load("my_svm")

preds = svm2.predict(sc2.transform(X))  # 与保存前预测结果完全一致
```

模型文件存储在 `models/`，WSL2 通过 `/mnt/c/...` 写入，Windows 可直接访问 `.pkl` 文件。

---

## 开机自启动

桥接服务通过注册表 `HKCU\Run` 在用户**登录时自动启动**，无需任何手动操作。

```
用户登录
  └─ HKCU\Run → start_bridge.bat
       └─ wsl -d Ubuntu → start_server.sh
            └─ nohup python server.py &   （后台，端口 19876）
                 └─ 约 1 秒后，19876 端口就绪 ✅
```

验证注册表项：

```powershell
Get-ItemProperty "HKCU:\Software\Microsoft\Windows\CurrentVersion\Run" |
  Select-Object "GPU-sklearn-bridge"
# GPU-sklearn-bridge : C:\Users\nicho\gpu-sklearn-bridge\start_bridge.bat
```

> **注意**：HKCU Run 在用户登录桌面时触发。执行 `wsl --shutdown` 后需手动重启，或重新登录。

---

## 手动管理服务

```powershell
# 启动
C:\Users\nicho\gpu-sklearn-bridge\start_bridge.bat

# 健康检查
Invoke-RestMethod "http://127.0.0.1:19876/health"

# 查看日志（在 WSL2 中）
wsl -d Ubuntu -- tail -f ~/gpu-sklearn-bridge/server.log

# 停止（在 WSL2 中）
wsl -d Ubuntu -- pkill -f server.py
```

---

## 文件结构

```
gpu-sklearn-bridge/
├── server.py                # Flask 桥接服务（运行于 WSL2）
├── shm_transport.py         # 扩展 mmap 共享内存传输层（4 GB，16 slots）
├── start_bridge.bat         # Windows 启动入口
├── start_server.sh          # WSL2 启动脚本
├── quickstart_check.py      # 快速环境验证脚本
├── test_mmap.py             # 基础 mmap 传输层测试
├── test_extended_mmap.py    # 扩展 mmap 集成测试（需要 WSL2+GPU）
├── _local_test.py           # 本地单元测试（无需 WSL2/GPU）
├── _train_test.py           # 端到端训练测试（需要 WSL2+GPU）
├── _e2e_test.py             # 端到端集成测试
├── scripts/
│   ├── install_windows.ps1  # 一键安装脚本（注册自启动）
│   └── start_bridge.ps1
├── shm/                     # 共享内存文件目录（*.npy 旧格式残留）
├── models/                  # 已保存的模型权重（*.pkl）
├── cuml/                    # ← import cuml 别名层（最优体验）
│   └── __init__.py
├── cuml_proxy/              # Windows 代理包（核心）
│   ├── proxy.py             # ProxyEstimator 核心
│   ├── linear_model.py
│   ├── cluster.py
│   ├── decomposition.py
│   ├── neighbors.py
│   ├── ensemble.py
│   ├── svm.py
│   ├── preprocessing.py
│   └── manifold.py
└── windows_bridge/          # Windows hook 层

# WSL2 端路径
~/gpu-sklearn-bridge/shm/pool.bin   # 4 GB 预分配 mmap 池（首次使用自动创建）
# Windows 访问：\\wsl.localhost\Ubuntu\home\<USER>\gpu-sklearn-bridge\shm\pool.bin
```

---

## 性能参考

> 测试日期：2026-02-26 / RTX 4060 Laptop 8 GB / 扩展 mmap（4 GB pool，16 slots 轮转）
> pool.bin 存于 WSL2 Linux FS，Windows 经 UNC `\\wsl.localhost\...` + `fd.seek+read` 读取

**端到端训练测试（_train_test.py，全部 17/17 通过）：**

| 场景 | 数组大小 | 耗时 |
|---|---|---|
| fit_transform（StandardScaler） | 5000×20 | 563 ms |
| fit_transform（PCA, n=5） | 5000×20 | 231 ms |
| fit + predict（LinearRegression，R²=1.0） | 5000×20 | 36 + 32 ms |
| fit + predict（LogisticRegression，acc=0.998） | 5000×20 | 135 + 33 ms |
| fit_predict（KMeans k=3） | 5000×20 | 107 ms |
| fit + predict（RandomForestClassifier，acc=1.0） | 500×20 | 179 + 35 ms |
| fit + predict（SVC rbf，acc=1.0） | 500×20 | 79 + 7 ms |
| fit_transform 压力测试（~51 MB） | 10000×1280 | 4185 ms（12 MB/s 等效吞吐） |

**Iris 数据集 5-Fold Cross Validation：**

| 模型 | 均值准确率 | ±std |
|---|---|---|
| SVC (RBF) | **0.9667** | ±0.0211 |
| RandomForestClassifier (100) | 0.9600 | ±0.0249 |
| KNeighborsClassifier (k=5) | 0.9600 | ±0.0327 |
| LogisticRegression | 0.9533 | ±0.0340 |

---

## 已知限制

| 限制 | 说明 |
|---|---|
| mmap 单次上限 256 MB | 单个数组超过 256 MB 会报错（单 slot 容量），可通过增加 `SLOT_SIZE` 或分批处理解决 |
| 总 pool 上限 4 GB | 若工作集持续超过 4 GB，需扩大 `POOL_SIZE` 并重建 pool.bin |
| 非真正零拷贝 | Windows 通过 P9 协议（`\\wsl.localhost\...` UNC）访问 WSL2 Linux FS，每次 `fd.read` 有一次跨系统 I/O；非 AF_VSOCK 级零拷贝 |
| 需要用户登录才自启 | HKCU Run 在登录桌面时触发，`wsl --shutdown` 后需手动重启 |
| 代理软件冲突 | 已处理（`trust_env=False`），Clash/V2Ray 不影响桥接请求 |
| Windows Server 不支持 WSL2 | 此方案仅适用于 Windows 10/11 桌面系统 |

---

## 导入方式对比

```python
# 方式 1：最优体验，与官方 cuML 写法完全一致 ✅
import cuml
from cuml.svm import SVC

# 方式 2：显式代理包写法，效果相同
from cuml_proxy.svm import SVC

# 方式 3：CPU 版 scikit-learn，不走 GPU
from sklearn.svm import SVC
```

`cuml` 和 `cuml_proxy` 指向完全相同的对象，`cuml` 是 `cuml_proxy` 的别名层，选任意一种写法即可。

---

## 依赖

**Windows 环境**

```
Python  3.11+
numpy
requests
```

**WSL2 环境**

```
Python  3.11+
cuml-cu12   26.02+
flask
```

---

## Contributing

欢迎提交 Issue 或 Pull Request！

- **Bug 报告**：请提供 OS、驱动、cuML 版本，以及完整错误日志（`server.log`）。
- **新算法支持**：在 `cuml_proxy/` 下添加对应模块，并在 `server.py` 的 `_CLASS_MAP` 中注册即可。
- **性能优化**：传输层代码位于 `shm_transport.py`，欢迎探索 AF_VSOCK 或 virtio-fs 等零拷贝方案。

请确保：
1. 新增代码通过 `_local_test.py` 本地测试。
2. 涉及 GPU 计算的功能通过 `_train_test.py` 端到端测试。

---

## License

[MIT](LICENSE) © 2026 区梓灏 (Nicholas Ou)
