# gpu-sklearn-bridge

<p align="center">
  <img src="https://img.shields.io/badge/platform-Windows%2011%20%2B%20WSL2-blue?logo=windows" alt="platform">
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white" alt="python">
  <img src="https://img.shields.io/badge/CUDA-12.x-76B900?logo=nvidia&logoColor=white" alt="cuda">
  <img src="https://img.shields.io/badge/cuML-26.02-76B900?logo=nvidia&logoColor=white" alt="cuml">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="license">
</p>

> **鍦?Windows 涓婇€忔槑浣跨敤 RAPIDS cuML GPU 鍔犻€熸満鍣ㄥ涔?*
>
> NVIDIA 浠庢湭鍙戝竷浠讳綍 Windows 鐗?cuML wheel銆傛湰椤圭洰閫氳繃 WSL2 妗ユ帴锛岃浣犲湪 Windows Python 閲岀洿鎺?`import cuml`锛屾墍鏈夎绠楀湪 GPU 涓婂畬鎴愶紝琛屼负涓庡畼鏂?cuML 瀹屽叏涓€鑷淬€?

---

## 鐩綍

- [鐜淇℃伅](#鐜淇℃伅)
- [蹇€熷紑濮媇(#蹇€熷紑濮?
- [鏋舵瀯姒傝](#鏋舵瀯姒傝)
- [鏀寔鐨勭畻娉昡(#鏀寔鐨勭畻娉?
- [瀹夎涓庨儴缃瞉(#瀹夎涓庨儴缃?
- [妯″瀷淇濆瓨涓庡姞杞絔(#妯″瀷淇濆瓨涓庡姞杞?
- [寮€鏈鸿嚜鍚姩](#寮€鏈鸿嚜鍚姩)
- [鎵嬪姩绠＄悊鏈嶅姟](#鎵嬪姩绠＄悊鏈嶅姟)
- [鏂囦欢缁撴瀯](#鏂囦欢缁撴瀯)
- [鎬ц兘鍙傝€僝(#鎬ц兘鍙傝€?
- [宸茬煡闄愬埗](#宸茬煡闄愬埗)
- [瀵煎叆鏂瑰紡瀵规瘮](#瀵煎叆鏂瑰紡瀵规瘮)
- [渚濊禆](#渚濊禆)
- [Contributing](#contributing)
- [License](#license)

---

## 鐜淇℃伅

| 椤圭洰 | 鐗堟湰 |
|---|---|
| OS | Windows 11 |
| GPU | NVIDIA RTX 4060 Laptop 8 GB |
| CUDA Toolkit | 12.8 |
| 椹卞姩 | 576.80 |
| WSL2 鍙戣鐗?| Ubuntu 24.04.2 LTS |
| cuML | 26.02.000 |
| Python锛圵indows锛?| 3.11.13锛坲v venv锛?|
| Python锛圵SL2锛?| 3.11.14锛坲v venv锛?|

---

## 蹇€熷紑濮?

### 1. 纭鏈嶅姟宸茶繍琛?

```powershell
Invoke-RestMethod "http://127.0.0.1:19876/health"
# cuml_version  status
# 26.02.000     ok
```

鑻ヨ繛鎺ュけ璐ワ紝鎵嬪姩鍚姩锛?

```powershell
C:\Users\nicho\gpu-sklearn-bridge\start_bridge.bat
# 绛夊緟绾?8 绉掑悗閲嶈瘯
```

### 2. 婵€娲?Windows 鐜

```powershell
C:\Users\nicho\envs\cuml-proxy\Scripts\Activate.ps1
```

### 3. 鐩存帴 `import cuml` 浣跨敤锛堟帹鑽愶級

```python
import cuml                              # 鈫?涓庡畼鏂?cuML 鍐欐硶瀹屽叏鐩稿悓
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
print(svm.predict(X_s[:5]))             # GPU 鎺ㄧ悊

print(cuml.__version__)                  # 26.02.000
```

---

## 鏋舵瀯姒傝

```
Windows Python
  import cuml          鈫?cuml/ 鏄湰鍦?shim锛岃嚜鍔ㄨ浆鍙戝埌 cuml_proxy
  import cuml_proxy    鈫?鏁堟灉鐩稿悓锛屾樉寮忓啓娉?
       鈹?
       鈹? 鈶?HTTP JSON-RPC  127.0.0.1:19876
       鈹? 鈶?鏁扮粍 鈮?10 KB 鈫?鎵╁睍 mmap pool.bin锛? GB 棰勫垎閰嶏紝16 slots锛?
       鈹?    Windows 閫氳繃 \\wsl.localhost\Ubuntu\... UNC 璺緞璁块棶
       鈫?
WSL2 Ubuntu  server.py  (Flask)
  ~/gpu-sklearn-bridge/shm/pool.bin  鈫?pool.bin 瀛樺偍浜?WSL2 Linux FS (ext4)
       鈹?
       鈹? import cuml锛堢湡姝ｇ殑 RAPIDS cuML锛?
       鈫?
RAPIDS cuML 26.02 鈫?RTX 4060 GPU
```

### 浼犺緭灞備笁绾х瓥鐣?

| 鏁扮粍澶у皬 | 浼犺緭鏂瑰紡 | 璇存槑 |
|---|---|---|
| < 10 KB | HTTP inline Base64 | 鐩存帴宓屽叆 JSON body |
| 鈮?10 KB | **鎵╁睍 mmap** `pool.bin` | 4 GB 棰勫垎閰嶆睜锛?6 slots 杞浆鍒嗛厤锛沺ool.bin 瀛樹簬 WSL2 Linux FS锛學indows 閫氳繃 UNC `\wsl.localhost\...` 璁块棶 |
| 妯″瀷鏂囦欢 | pickle `.pkl` | 涓诲姩璋冪敤 `save()` / `load()` 鎸佷箙鍖?|

### 鎵╁睍 mmap 甯冨眬锛? GB锛?6 slots锛?

```
pool.bin
鈹屸攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?
鈹?  Input  slots 0-3      鈹? Output  slots 4-7       鈹?        Scratch slots 8-15               鈹?
鈹?     1 GB (4脳256 MB)    鈹?    1 GB (4脳256 MB)      鈹?            2 GB (8脳256 MB)              鈹?
鈹?  Windows 鍐?鈫?WSL2 璇? 鈹? WSL2 鍐?鈫?Windows 璇?  鈹?        鏈嶅姟绔唴閮ㄤ复鏃剁紦鍐插尯              鈹?
鈹斺攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹粹攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹粹攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?
```

瀹㈡埛绔拰鏈嶅姟绔悇鑷淮鎶?*杞浆璁℃暟鍣?*锛屾瘡娆¤姹傝嚜鍔ㄥ垎閰嶄笅涓€涓?slot锛?鈫?鈫?鈫?鈫?鈥︼級锛屽疄鐜版渶澶?4 涓苟鍙戜笉闃诲銆?

---

## 鏀寔鐨勭畻娉?

| 妯″潡 | 绫?|
|---|---|
| `cuml.linear_model` | `LinearRegression` `LogisticRegression` `Ridge` `Lasso` `ElasticNet` |
| `cuml.svm` | `SVC` `SVR` |
| `cuml.cluster` | `KMeans` `DBSCAN` |
| `cuml.decomposition` | `PCA` `TruncatedSVD` |
| `cuml.neighbors` | `KNeighborsClassifier` `KNeighborsRegressor` `NearestNeighbors` |
| `cuml.ensemble` | `RandomForestClassifier` `RandomForestRegressor` |
| `cuml.preprocessing` | `StandardScaler` `MinMaxScaler` `LabelEncoder` |
| `cuml.manifold` | `TSNE` `UMAP` |

鎵€鏈夌被鍧囧疄鐜?scikit-learn 鏍囧噯鎺ュ彛锛歚fit` / `predict` / `transform` / `fit_transform` / `fit_predict` / `score` / `get_params` / `set_params`銆?

---

## 妯″瀷淇濆瓨涓庡姞杞?

```python
from cuml.svm import SVC
from cuml.preprocessing import StandardScaler
from cuml_proxy.proxy import ProxyEstimator
import numpy as np

X = np.random.rand(200, 10).astype("float32")
y = (X[:, 0] > 0.5).astype("float32")

# 璁粌
sc = StandardScaler()
X_s = sc.fit_transform(X)
svm = SVC(kernel="rbf")
svm.fit(X_s, y)

# 淇濆瓨鏉冮噸锛坧ickle 鍒?models/ 鐩綍锛?
sc.save("my_scaler")    # 鈫?models/my_scaler.pkl
svm.save("my_svm")      # 鈫?models/my_svm.pkl

# 鍒楀嚭鎵€鏈夊凡淇濆瓨妯″瀷
print(ProxyEstimator.list_saved())   # ['my_scaler', 'my_svm', ...]

# 鍔犺浇锛堟棤闇€閲嶆柊璁粌锛?
sc2  = ProxyEstimator.load("my_scaler")
svm2 = ProxyEstimator.load("my_svm")

X_s2 = sc2.transform(X)
preds = svm2.predict(X_s2)          # 涓庝繚瀛樺墠棰勬祴缁撴灉瀹屽叏涓€鑷?
```

妯″瀷鏂囦欢瀛樺偍鍦?`C:\Users\nicho\gpu-sklearn-bridge\models\`锛學SL2 閫氳繃 `/mnt/c/...` 鍐欏叆锛學indows 鍙洿鎺ヨ闂?`.pkl` 鏂囦欢銆?

---

## 寮€鏈鸿嚜鍚姩

妗ユ帴鏈嶅姟閫氳繃娉ㄥ唽琛?`HKCU\Run` 鍦ㄧ敤鎴?*鐧诲綍鏃惰嚜鍔ㄥ惎鍔?*锛屾棤闇€浠讳綍鎵嬪姩鎿嶄綔銆?

```
鐢ㄦ埛鐧诲綍
  鈹斺攢 HKCU\Run 鈫?start_bridge.bat
       鈹斺攢 wsl -d Ubuntu 鈫?start_server.sh
            鈹斺攢 nohup python server.py &   锛堝悗鍙帮紝绔彛 19876锛?
                 鈹斺攢 绾?1 绉掑悗锛?9876 绔彛灏辩华 鉁?
```

楠岃瘉娉ㄥ唽琛ㄩ」锛?

```powershell
Get-ItemProperty "HKCU:\Software\Microsoft\Windows\CurrentVersion\Run" |
  Select-Object "GPU-sklearn-bridge"
# GPU-sklearn-bridge : C:\Users\nicho\gpu-sklearn-bridge\start_bridge.bat
```

> **娉ㄦ剰**锛欻KCU Run 鍦ㄧ敤鎴风櫥褰曟闈㈡椂瑙﹀彂銆傛墽琛?`wsl --shutdown` 鍚庨渶鎵嬪姩閲嶅惎锛屾垨閲嶆柊鐧诲綍銆?

---

## 鎵嬪姩绠＄悊鏈嶅姟

```powershell
# 鍚姩
C:\Users\nicho\gpu-sklearn-bridge\start_bridge.bat

# 鍋ュ悍妫€鏌?
Invoke-RestMethod "http://127.0.0.1:19876/health"

# 鏌ョ湅鏃ュ織锛堝湪 WSL2 涓級
wsl -d Ubuntu -- tail -f ~/gpu-sklearn-bridge/server.log

# 鍋滄锛堝湪 WSL2 涓級
wsl -d Ubuntu -- bash -c "kill $(pgrep -f server.py)"
```

---

## 鏂囦欢缁撴瀯

```
C:\Users\nicho\gpu-sklearn-bridge\
鈹溾攢鈹€ server.py                # Flask 妗ユ帴鏈嶅姟锛堝悓姝ヨ嚦 WSL2锛?
鈹溾攢鈹€ shm_transport.py         # 鎵╁睍 mmap 鍏变韩鍐呭瓨浼犺緭灞傦紙4 GB锛?6 slots锛?
鈹溾攢鈹€ start_bridge.bat         # Windows 鍚姩鍏ュ彛
鈹溾攢鈹€ start_server.sh          # WSL2 鍚姩鑴氭湰
鈹溾攢鈹€ test_mmap.py             # 鍩虹 mmap 浼犺緭灞傛祴璇?
鈹溾攢鈹€ test_extended_mmap.py    # 鎵╁睍 mmap 闆嗘垚娴嬭瘯锛堥渶瑕?WSL2+GPU锛?
鈹溾攢鈹€ _local_test.py           # 鏈湴鍗曞厓娴嬭瘯锛堟棤闇€ WSL2/GPU锛?
鈹溾攢鈹€ shm\
鈹?  鈹斺攢鈹€ *.npy / *.bin        # 锛堟棫鏍煎紡娈嬬暀锛屽彲蹇界暐锛?
鈹溾攢鈹€ models\
鈹?  鈹斺攢鈹€ *.pkl                # 宸蹭繚瀛樼殑妯″瀷鏉冮噸
鈹溾攢鈹€ cuml\                    # 鈫?import cuml 鍒悕灞傦紙鏈€浼樹綋楠岋級
鈹?  鈹斺攢鈹€ __init__.py
鈹斺攢鈹€ cuml_proxy\              # Windows 浠ｇ悊鍖?
    鈹溾攢鈹€ __init__.py
    鈹溾攢鈹€ proxy.py             # ProxyEstimator 鏍稿績
    鈹溾攢鈹€ linear_model.py
    鈹溾攢鈹€ cluster.py
    鈹溾攢鈹€ decomposition.py
    鈹溾攢鈹€ neighbors.py
    鈹溾攢鈹€ ensemble.py
    鈹溾攢鈹€ svm.py
    鈹溾攢鈹€ preprocessing.py
    鈹斺攢鈹€ manifold.py

C:\Users\nicho\envs\cuml-proxy\      # Windows uv 铏氭嫙鐜
鈹溾攢鈹€ Scripts\python.exe               # Python 3.11.13
鈹斺攢鈹€ Lib\site-packages\
    鈹斺攢鈹€ cuml_proxy_bridge.pth        # 灏?gpu-sklearn-bridge\ 鍔犲叆 sys.path

WSL2 ~/gpu-sklearn-bridge\
鈹溾攢鈹€ server.py
鈹溾攢鈹€ shm_transport.py
鈹溾攢鈹€ start_server.sh
鈹溾攢鈹€ server.log
鈹斺攢鈹€ shm/
    鈹斺攢鈹€ pool.bin             # 4 GB 棰勫垎閰嶆墿灞?mmap 姹狅紙棣栨浣跨敤鑷姩鍒涘缓浜?Linux FS锛?
                             # Windows 璁块棶璺緞锛歕\wsl.localhost\Ubuntu\home\nicho\gpu-sklearn-bridge\shm\pool.bin

WSL2 ~/envs/gpu-sklearn\
鈹斺攢鈹€ bin/python                       # Python 3.11.14 + cuML 26.02
```

---

## 鎬ц兘鍙傝€?

> 娴嬭瘯鏃ユ湡锛?026-02-26 / RTX 4060 Laptop 8 GB / 鎵╁睍 mmap锛? GB pool锛?6 slots 杞浆锛?
> pool.bin 瀛樹簬 WSL2 Linux FS锛學indows 缁?UNC `\\wsl.localhost\...` + `fd.seek+read` 璇诲彇

**绔埌绔缁冩祴璇曪紙_train_test.py锛屽叏閮?17/17 閫氳繃锛夛細**

| 鍦烘櫙 | 鏁扮粍澶у皬 | 鑰楁椂 |
|---|---|---|
| fit_transform锛圫tandardScaler锛墊 5000脳20 | 563 ms |
| fit_transform锛圥CA, n=5锛墊 5000脳20 | 231 ms |
| fit + predict锛圠inearRegression锛孯虏=1.0锛墊 5000脳20 | 36 + 32 ms |
| fit + predict锛圠ogisticRegression锛宎cc=0.998锛墊 5000脳20 | 135 + 33 ms |
| fit_predict锛圞Means k=3锛墊 5000脳20 | 107 ms |
| fit + predict锛圧andomForestClassifier锛宎cc=1.0锛墊 500脳20 | 179 + 35 ms |
| fit + predict锛圫VC rbf锛宎cc=1.0锛墊 500脳20 | 79 + 7 ms |
| fit_transform 鍘嬪姏娴嬭瘯锛垀51 MB锛墊 10000脳1280 | 4185 ms锛?2 MB/s 绛夋晥鍚炲悙锛?|

Iris 鏁版嵁闆?5-Fold Cross Validation锛?

| 妯″瀷 | 鍧囧€煎噯纭巼 | 卤std |
|---|---|---|
| SVC (RBF) | **0.9667** | 卤0.0211 |
| RandomForestClassifier (100) | 0.9600 | 卤0.0249 |
| KNeighborsClassifier (k=5) | 0.9600 | 卤0.0327 |
| LogisticRegression | 0.9533 | 卤0.0340 |

---

## 宸茬煡闄愬埗

| 闄愬埗 | 璇存槑 |
|---|---|
| mmap 鍗曟涓婇檺 256 MB | 鍗曚釜鏁扮粍瓒呰繃 256 MB 浼氭姤閿欙紙鍗?slot 瀹归噺锛夛紝鍙€氳繃澧炲姞 `SLOT_SIZE` 鎴栧垎鎵瑰鐞嗚В鍐?|
| 鎬?pool 涓婇檺 4 GB | 鑻ュ伐浣滈泦鎸佺画瓒呰繃 4 GB锛岄渶鎵╁ぇ `POOL_SIZE` 骞堕噸寤?pool.bin |
| 闈炵湡姝ｉ浂鎷疯礉 | Windows 閫氳繃 P9 鍗忚锛坄\\wsl.localhost\...` UNC锛夎闂?WSL2 Linux FS锛屾瘡娆?`fd.read` 鏈変竴娆¤法绯荤粺 I/O锛涢潪 AF_VSOCK 绾ч浂鎷疯礉 |
| 闇€瑕佺敤鎴风櫥褰曟墠鑷惎 | HKCU Run 鍦ㄧ櫥褰曟闈㈡椂瑙﹀彂锛宍wsl --shutdown` 鍚庨渶鎵嬪姩閲嶅惎 |
| 浠ｇ悊杞欢鍐茬獊 | 宸插鐞嗭紙`trust_env=False`锛夛紝Clash/V2Ray 涓嶅奖鍝嶆ˉ鎺ヨ姹?|
| Windows Server 涓嶆敮鎸?WSL2 | 姝ゆ柟妗堜粎閫傜敤浜?Windows 10/11 妗岄潰绯荤粺 |

---

## 瀵煎叆鏂瑰紡瀵规瘮

```python
# 鏂瑰紡 1锛氭渶浼樹綋楠岋紝涓庡畼鏂?cuML 鍐欐硶瀹屽叏涓€鑷?鉁?
import cuml
from cuml.svm import SVC

# 鏂瑰紡 2锛氭樉寮忎唬鐞嗗寘鍐欐硶锛屾晥鏋滅浉鍚?
from cuml_proxy.svm import SVC

# 鏂瑰紡 3锛欳PU 鐗?scikit-learn锛屼笉璧?GPU
from sklearn.svm import SVC
```

`cuml` 鍜?`cuml_proxy` 鎸囧悜瀹屽叏鐩稿悓鐨勫璞★紝`cuml` 鏄?`cuml_proxy` 鐨勫埆鍚嶅眰锛岄€変换鎰忎竴绉嶅啓娉曞嵆鍙€?

---

## 渚濊禆

**Windows 鐜**锛坄C:\Users\nicho\envs\cuml-proxy\`锛?

```
Python  3.11.13
numpy
requests
```

**WSL2 鐜**锛坄~/envs/gpu-sklearn/`锛?

```
Python  3.11.14
cuml-cu12   26.02.000
flask
```


---

## 安装与部署

> 以下路径中的 `<USER>` 请替换为你自己的 Windows 用户名，`<DISTRO>` 替换为你的 WSL2 发行版名（默认 `Ubuntu`）。

### 前提条件

| 要求 | 说明 |
|---|---|
| Windows 10/11 (x64) | 需支持 WSL2 |
| NVIDIA GPU | 驱动  525，CUDA Toolkit 12.x |
| WSL2 + Ubuntu | `wsl --install -d Ubuntu` |
| Python 3.11 | Windows 端和 WSL2 端均需安装 |

### 1. 克隆仓库

```powershell
git clone https://github.com/<your-username>/gpu-sklearn-bridge.git C:\Users\<USER>\gpu-sklearn-bridge
wsl -d <DISTRO> -- git clone https://github.com/<your-username>/gpu-sklearn-bridge.git ~/gpu-sklearn-bridge
```

### 2. 配置 WSL2 服务端

```bash
cd ~/gpu-sklearn-bridge
pip install flask numpy
python -c "import cuml; print(cuml.__version__)"
```

### 3. 配置 Windows 客户端

```powershell
uv venv C:\Users\<USER>\envs\cuml-proxy
C:\Users\<USER>\envs\cuml-proxy\Scripts\Activate.ps1
pip install numpy requests
# 将项目目录加入 sys.path
$site = python -c "import site; print(site.getsitepackages()[0])"
"C:\Users\<USER>\gpu-sklearn-bridge" | Out-File "$site\cuml_proxy_bridge.pth" -Encoding ascii
```

### 4. 验证

```powershell
.\start_bridge.bat
Invoke-RestMethod "http://127.0.0.1:19876/health"
```

---

## Contributing

欢迎提交 Issue 或 Pull Request！

- **Bug 报告**：请提供 OS、驱动、cuML 版本及完整 `server.log`。
- **新算法支持**：在 `cuml_proxy/` 下添加对应模块，并在 `server.py` 的 `_CLASS_MAP` 中注册。
- **性能优化**：传输层代码位于 `shm_transport.py`，欢迎探索 AF_VSOCK 等零拷贝方案。

---

## License

[MIT](LICENSE)  2026 区梓灏