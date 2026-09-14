# gpu-sklearn-bridge

中文版: [README.zh-CN.md](README.zh-CN.md)

<p align="center">
  <img src="https://img.shields.io/badge/platform-Windows%2011%20%2B%20WSL2-blue?logo=windows" alt="platform">
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white" alt="python">
  <img src="https://img.shields.io/badge/CUDA-12.x-76B900?logo=nvidia&logoColor=white" alt="cuda">
  <img src="https://img.shields.io/badge/cuML-26.02-76B900?logo=nvidia&logoColor=white" alt="cuml">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="license">
</p>

> **Use RAPIDS cuML GPU-accelerated machine learning transparently on Windows**
>
> NVIDIA has never released a Windows cuML wheel. This project bridges Windows Python to WSL2 so that you can `import cuml` directly on Windows; all computation runs on the GPU inside WSL2, with the same API as the official cuML.

> **Provenance note.** Every version number, test count and benchmark figure in this document is copied verbatim from the original Chinese README (test date 2026-02-26). Nothing was re-run for this English revision.

---

## Contents

- [What it is](#what-it-is)
- [Why it exists](#why-it-exists)
- [Tested environment](#tested-environment)
- [Architecture](#architecture)
- [Quickstart](#quickstart)
- [Supported estimators](#supported-estimators)
- [Saving and loading models](#saving-and-loading-models)
- [Auto-start at logon](#auto-start-at-logon)
- [Managing the service manually](#managing-the-service-manually)
- [Repository layout](#repository-layout)
- [Benchmarks](#benchmarks)
- [Known limits](#known-limits)
- [Environment variables](#environment-variables)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)

---

## What it is

`gpu-sklearn-bridge` is a small client/server bridge:

- **Windows side** – `cuml/` (a shim so that `import cuml` works) and `cuml_proxy/` (a scikit-learn-style proxy package). Every estimator is a `ProxyEstimator` that forwards `fit` / `predict` / `transform` / … to the server over HTTP.
- **WSL2 side** – `server.py`, a Flask HTTP JSON-RPC server that owns the real cuML models on the GPU.
- **Transport** – arrays smaller than 10 KB travel inline as Base64 in the JSON body; arrays of 10 KB or more go through a 4 GB pre-allocated mmap pool (`shm/pool.bin`, 16 slots) that lives on the WSL2 Linux filesystem and is reached from Windows through the `\\wsl.localhost\...` UNC path.

## Why it exists

NVIDIA has never released a Windows cuML wheel. Running cuML inside WSL2 and bridging it to Windows Python lets Windows code use `import cuml` and the standard scikit-learn interface while the actual work happens on the GPU.

---

## Tested environment

(copied from the original README)

| Item | Version |
|---|---|
| OS | Windows 11 |
| GPU | NVIDIA RTX 4060 Laptop 8 GB |
| CUDA Toolkit | 12.8 |
| Driver | 576.80 |
| WSL2 distro | Ubuntu 24.04.2 LTS |
| cuML | 26.02.000 |
| Python (Windows) | 3.11.13 (uv venv) |
| Python (WSL2) | 3.11.14 (uv venv) |

---

## Architecture

```
Windows Python
  import cuml          <- cuml/ is a local shim that forwards to cuml_proxy
  import cuml_proxy    <- same thing, explicit form
       |
       |  (1) HTTP JSON-RPC  127.0.0.1:19876
       |  (2) arrays >= 10 KB -> extended mmap pool.bin (4 GB pre-allocated, 16 slots)
       |      Windows reaches it through the \\wsl.localhost\<DISTRO>\... UNC path
       v
WSL2 Ubuntu  server.py  (Flask)
  ~/gpu-sklearn-bridge/shm/pool.bin  <- pool.bin lives on the WSL2 Linux FS (ext4)
       |
       |  import cuml  (the real RAPIDS cuML)
       v
RAPIDS cuML 26.02 -> RTX 4060 GPU
```

### Three-tier transport

| Array size | Transport | Notes |
|---|---|---|
| < 10 KB | HTTP inline Base64 | embedded in the JSON body |
| >= 10 KB | **extended mmap** `pool.bin` | 4 GB pre-allocated pool, 16 slots allocated round-robin; pool.bin lives on the WSL2 Linux FS, Windows reads it through the UNC path `\\wsl.localhost\...` |
| Model files | pickle `.pkl` | explicit `save()` / `load()` |

### Extended mmap layout (4 GB, 16 slots)

```
pool.bin
+-------------------------+-------------------------+------------------------------------------+
|   Input  slots 0-3      |  Output  slots 4-7      |         Scratch slots 8-15               |
|      1 GB (4x256 MB)    |     1 GB (4x256 MB)     |             2 GB (8x256 MB)              |
|  Windows writes, WSL2   |  WSL2 writes, Windows   |      server-internal scratch buffers     |
|  reads                  |  reads                  |                                          |
+-------------------------+-------------------------+------------------------------------------+
```

Client and server each keep a **round-robin counter** and take the next slot on every request (0 -> 1 -> 2 -> 3 -> 0 ...), so up to 4 requests can be in flight without blocking each other.

---

## Quickstart

> Replace `<DISTRO>` with your WSL2 distro name (default `Ubuntu`). All paths below are relative to your own home directory; nothing in the repository depends on a specific user name any more (see [Environment variables](#environment-variables)).

### Prerequisites

| Requirement | Notes |
|---|---|
| Windows 10/11 (x64) | must support WSL2 |
| NVIDIA GPU | driver >= 525, CUDA Toolkit 12.x |
| WSL2 + Ubuntu | `wsl --install -d Ubuntu` |
| Python 3.11 | on both the Windows side and the WSL2 side |

### WSL2 side (server)

```bash
# inside WSL2
git clone https://github.com/Nicholas022400701/gpu-sklearn-bridge.git ~/gpu-sklearn-bridge
cd ~/gpu-sklearn-bridge
pip install flask numpy
# cuML must already be installed in this environment (see https://docs.rapids.ai/install)
python -c "import cuml; print(cuml.__version__)"
```

`start_server.sh` expects the WSL2 clone at `$HOME/gpu-sklearn-bridge` and a Python interpreter at `$HOME/envs/gpu-sklearn/bin/python`. If yours live elsewhere, set `SKLEARN_BRIDGE_HOME` and `SKLEARN_BRIDGE_PYTHON` before running it, or simply run `python server.py` in that environment.

### Windows side (client)

```powershell
# PowerShell
git clone https://github.com/Nicholas022400701/gpu-sklearn-bridge.git "$env:USERPROFILE\gpu-sklearn-bridge"
uv venv "$env:USERPROFILE\envs\cuml-proxy"
& "$env:USERPROFILE\envs\cuml-proxy\Scripts\Activate.ps1"

# option A: editable install (pyproject.toml)
pip install -e "$env:USERPROFILE\gpu-sklearn-bridge"

# option B: add the repository to sys.path with a .pth file
pip install numpy requests
$site = python -c "import site; print(site.getsitepackages()[0])"
"$env:USERPROFILE\gpu-sklearn-bridge" | Out-File "$site\cuml_proxy_bridge.pth" -Encoding ascii
```

### Start the bridge and check it

```powershell
& "$env:USERPROFILE\gpu-sklearn-bridge\start_bridge.bat"
# wait about 8 seconds, then:
Invoke-RestMethod "http://127.0.0.1:19876/health"
# cuml_version  status
# 26.02.000     ok
```

### Use it

```python
import cuml                              # <- identical to the official cuML spelling
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
print(svm.predict(X_s[:5]))             # GPU inference

print(cuml.__version__)                  # 26.02.000
```

`cuml` and `cuml_proxy` point to exactly the same objects; `cuml` is an alias layer over `cuml_proxy`. `from sklearn.svm import SVC` still gives you the CPU scikit-learn implementation.

---

## Supported estimators

| Module | Classes |
|---|---|
| `cuml.linear_model` | `LinearRegression` `LogisticRegression` `Ridge` `Lasso` `ElasticNet` |
| `cuml.svm` | `SVC` `SVR` |
| `cuml.cluster` | `KMeans` `DBSCAN` |
| `cuml.decomposition` | `PCA` `TruncatedSVD` |
| `cuml.neighbors` | `KNeighborsClassifier` `KNeighborsRegressor` `NearestNeighbors` |
| `cuml.ensemble` | `RandomForestClassifier` `RandomForestRegressor` |
| `cuml.preprocessing` | `StandardScaler` `MinMaxScaler` `LabelEncoder` |
| `cuml.manifold` | `TSNE` `UMAP` |

All classes implement the standard scikit-learn interface: `fit` / `predict` / `transform` / `fit_transform` / `fit_predict` / `score` / `get_params` / `set_params`.

---

## Saving and loading models

```python
from cuml.svm import SVC
from cuml.preprocessing import StandardScaler
from cuml_proxy.proxy import ProxyEstimator
import numpy as np

X = np.random.rand(200, 10).astype("float32")
y = (X[:, 0] > 0.5).astype("float32")

sc = StandardScaler()
X_s = sc.fit_transform(X)
svm = SVC(kernel="rbf")
svm.fit(X_s, y)

sc.save("my_scaler")    # -> models/my_scaler.pkl
svm.save("my_svm")      # -> models/my_svm.pkl

print(ProxyEstimator.list_saved())   # ['my_scaler', 'my_svm', ...]

sc2  = ProxyEstimator.load("my_scaler")
svm2 = ProxyEstimator.load("my_svm")

preds = svm2.predict(sc2.transform(X))  # same predictions as before saving
```

Model files are stored in `models/`; WSL2 writes them through `/mnt/c/...`, so Windows can open the `.pkl` files directly.

---

## Auto-start at logon

The bridge is started from the `HKCU\Run` registry key when the user logs on:

```
user logon
  +- HKCU\Run -> start_bridge.bat
       +- wsl -d <DISTRO> -> start_server.sh
            +- nohup python server.py &   (background, port 19876)
                 +- port 19876 ready about 1 second later
```

```powershell
Get-ItemProperty "HKCU:\Software\Microsoft\Windows\CurrentVersion\Run" |
  Select-Object "GPU-sklearn-bridge"
# GPU-sklearn-bridge : %USERPROFILE%\gpu-sklearn-bridge\start_bridge.bat
```

> **Note:** `HKCU\Run` fires when the user logs on to the desktop. After `wsl --shutdown` you must restart the bridge manually or log on again.

`scripts/install_windows.ps1` registers a Task Scheduler task instead (it fills your user name and repository path into `scripts/GPU_sklearn_bridge.xml`).

---

## Managing the service manually

```powershell
# start
& "$env:USERPROFILE\gpu-sklearn-bridge\start_bridge.bat"

# health check
Invoke-RestMethod "http://127.0.0.1:19876/health"

# logs (inside WSL2)
wsl -d Ubuntu -- tail -f ~/gpu-sklearn-bridge/server.log

# stop (inside WSL2)
wsl -d Ubuntu -- pkill -f server.py
```

---

## Repository layout

```
gpu-sklearn-bridge/
|-- server.py                # Flask bridge server (runs in WSL2)
|-- shm_transport.py         # extended mmap shared-memory transport (4 GB, 16 slots)
|-- start_bridge.bat         # Windows entry point
|-- start_server.sh          # WSL2 start script
|-- quickstart_check.py      # quick environment check
|-- test_mmap.py             # basic mmap transport test
|-- test_extended_mmap.py    # extended mmap integration test (needs WSL2 + GPU)
|-- _local_test.py           # local unit tests (no WSL2/GPU needed)
|-- _train_test.py           # end-to-end training test (needs WSL2 + GPU)
|-- _e2e_test.py             # end-to-end integration test
|-- pyproject.toml           # pip install -e . for the Windows-side packages
|-- scripts/
|   |-- install_windows.ps1  # one-shot installer (registers auto-start)
|   |-- start_bridge.ps1
|   +-- GPU_sklearn_bridge.xml
|-- docs/                    # design notes (Chinese)
|-- shm/                     # shared-memory files
|-- models/                  # saved model weights (*.pkl)
|-- cuml/                    # <- import cuml alias layer
|   +-- __init__.py
|-- cuml_proxy/              # Windows proxy package (core)
|   |-- proxy.py             # ProxyEstimator
|   |-- linear_model.py
|   |-- cluster.py
|   |-- decomposition.py
|   |-- neighbors.py
|   |-- ensemble.py
|   |-- svm.py
|   |-- preprocessing.py
|   +-- manifold.py
|-- windows_bridge/          # legacy rpyc import hook
+-- wsl_server/              # legacy rpyc server + setup script

# WSL2 side
~/gpu-sklearn-bridge/shm/pool.bin   # 4 GB pre-allocated mmap pool (created on first use)
# reached from Windows as: \\wsl.localhost\<DISTRO>\home\<USER>\gpu-sklearn-bridge\shm\pool.bin
```

---

## Benchmarks

> Copied verbatim from the original README; **not re-run** for this revision.
> Test date: 2026-02-26 / RTX 4060 Laptop 8 GB / extended mmap (4 GB pool, 16 rotating slots)
> pool.bin on the WSL2 Linux FS, read from Windows through UNC `\\wsl.localhost\...` + `fd.seek+read`

**End-to-end training test (`_train_test.py`, 17/17 passed):**

| Scenario | Array size | Time |
|---|---|---|
| fit_transform (StandardScaler) | 5000x20 | 563 ms |
| fit_transform (PCA, n=5) | 5000x20 | 231 ms |
| fit + predict (LinearRegression, R²=1.0) | 5000x20 | 36 + 32 ms |
| fit + predict (LogisticRegression, acc=0.998) | 5000x20 | 135 + 33 ms |
| fit_predict (KMeans k=3) | 5000x20 | 107 ms |
| fit + predict (RandomForestClassifier, acc=1.0) | 500x20 | 179 + 35 ms |
| fit + predict (SVC rbf, acc=1.0) | 500x20 | 79 + 7 ms |
| fit_transform stress test (~51 MB) | 10000x1280 | 4185 ms (12 MB/s effective throughput) |

**Iris dataset, 5-fold cross validation:**

| Model | Mean accuracy | ±std |
|---|---|---|
| SVC (RBF) | **0.9667** | ±0.0211 |
| RandomForestClassifier (100) | 0.9600 | ±0.0249 |
| KNeighborsClassifier (k=5) | 0.9600 | ±0.0327 |
| LogisticRegression | 0.9533 | ±0.0340 |

---

## Known limits

| Limit | Notes |
|---|---|
| 256 MB per mmap transfer | a single array larger than 256 MB (one slot) raises an error; increase `SLOT_SIZE` or split the data |
| 4 GB total pool | if the working set keeps exceeding 4 GB, increase `POOL_SIZE` and recreate pool.bin |
| Not true zero-copy | Windows reaches the WSL2 Linux FS over the P9 protocol (`\\wsl.localhost\...` UNC); every `fd.read` is one cross-system I/O, not AF_VSOCK-level zero-copy |
| Auto-start needs a logon | `HKCU\Run` fires at desktop logon; after `wsl --shutdown` the bridge must be restarted by hand |
| Proxy software | handled (`trust_env=False`); Clash/V2Ray do not intercept bridge requests |
| Windows Server | WSL2 is not available there; this only targets Windows 10/11 desktop |

---

## Environment variables

All variables are optional. **When none of them is set, every script and module behaves exactly as before** (paths are derived from the location of the script, from `%USERPROFILE%` / `$HOME`, or from the current user name).

| Variable | Used by | Default |
|---|---|---|
| `SKLEARN_BRIDGE_PORT` | `server.py`, `cuml_proxy` | `19876` |
| `SKLEARN_BRIDGE_SHARED` | `server.py`, `cuml_proxy` | `<repo>/shm` (Windows) / `/mnt/c/Users/<win user>/gpu-sklearn-bridge/shm` (WSL2) |
| `SKLEARN_BRIDGE_MODELS` | `server.py`, `cuml_proxy` | `<repo>/models` (Windows) / `/mnt/c/Users/<win user>/gpu-sklearn-bridge/models` (WSL2) |
| `SKLEARN_BRIDGE_HOME` | `start_bridge.bat`, `scripts/*.ps1`, `wsl_server/setup.sh`, `cuml_proxy`, tests | directory of the script / repository root; in `start_server.sh` it is the WSL2 clone, default `$HOME/gpu-sklearn-bridge` |
| `SKLEARN_BRIDGE_PYTHON` | `start_server.sh` | `$HOME/envs/gpu-sklearn/bin/python` |
| `SKLEARN_BRIDGE_POOL` | `shm_transport.py` | Windows: `\\wsl.localhost\<DISTRO>\home\<wsl user>\gpu-sklearn-bridge\shm\pool.bin`; WSL2: `$HOME/gpu-sklearn-bridge/shm/pool.bin` |
| `SKLEARN_BRIDGE_WSL_DISTRO` | `start_bridge.bat`, `scripts/*.ps1`, `shm_transport.py` | `Ubuntu` |
| `SKLEARN_BRIDGE_WSL_USER` | `scripts/*.ps1`, `shm_transport.py` | the Windows user name (`%USERNAME%`) |
| `SKLEARN_BRIDGE_WIN_USER` | `server.py` (inside WSL2) | the current Linux user name |
| `SKLEARN_BRIDGE_VENV` | `scripts/install_windows.ps1` | `%USERPROFILE%\envs\gpu-sklearn` |

Example (PowerShell):

```powershell
$Env:SKLEARN_BRIDGE_PORT   = "19876"
$Env:SKLEARN_BRIDGE_SHARED = "$env:USERPROFILE\gpu-sklearn-bridge\shm"
$Env:SKLEARN_BRIDGE_MODELS = "$env:USERPROFILE\gpu-sklearn-bridge\models"
```

---

## Dependencies

**Windows**

```
Python  3.11+
numpy
requests
```

**WSL2**

```
Python  3.11+
cuml-cu12   26.02+
flask
```

---

## Contributing

Issues and pull requests are welcome.

- **Bug reports**: include OS, driver and cuML versions plus the full error log (`server.log`).
- **New estimators**: add the class to the matching module in `cuml_proxy/` and register it in `_CLASS_MAP` in `server.py`.
- **Performance**: the transport layer is `shm_transport.py`; zero-copy approaches such as AF_VSOCK or virtio-fs are open for exploration.

Please make sure that:
1. new code passes `_local_test.py` locally;
2. anything touching GPU computation passes the `_train_test.py` end-to-end test.

---

## License

[MIT](LICENSE) © 2026 区梓灏 (Nicholas Ou)
