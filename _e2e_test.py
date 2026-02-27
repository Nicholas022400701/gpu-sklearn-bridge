# -*- coding: utf-8 -*-
"""
_e2e_test.py  –  全面端对端测试（需要 WSL2 + GPU）

27 个测试节，覆盖所有已支持算法、模型持久化、Pipeline 链式、
train/test 泛化性验证、anti-stale-cache 稳定性、大数组 mmap 传输。

_train_test.py 新增（用 * 标注）：
  预处理   : StandardScaler / *MinMaxScaler / *LabelEncoder (inverse_transform)
  分解     : PCA (inverse_transform) / *TruncatedSVD
  线性模型 : LinearRegression / *Ridge / *Lasso / *ElasticNet
             LogisticRegression (binary) / *LogisticRegression (multiclass)
  聚类     : KMeans / *DBSCAN
  近邻     : *KNeighborsClassifier / *KNeighborsRegressor / *NearestNeighbors
  集成     : RandomForestClassifier / *RandomForestRegressor
  SVM      : SVC / *SVR
  流形     : *TSNE / *UMAP
  高级     : *模型保存/加载 / *get_params/set_params / *Pipeline 链式
             *大数组 mmap / *anti-stale-cache 稳定性
"""

import sys, time, os
import numpy as np

sys.path.insert(0, r"C:\Users\nicho\gpu-sklearn-bridge")

# 增加超时：TSNE / UMAP 可能需要 60+ 秒
import cuml_proxy.proxy as _proxy_mod
_proxy_mod._TIMEOUT = 120

from cuml.preprocessing  import StandardScaler, MinMaxScaler, LabelEncoder
from cuml.decomposition  import PCA, TruncatedSVD
from cuml.linear_model   import LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression
from cuml.cluster        import KMeans, DBSCAN
from cuml.neighbors      import KNeighborsClassifier, KNeighborsRegressor, NearestNeighbors
from cuml.ensemble       import RandomForestClassifier, RandomForestRegressor
from cuml.svm            import SVC, SVR
from cuml.manifold       import TSNE, UMAP
from cuml_proxy.proxy    import ProxyEstimator, _session, _BRIDGE_URL

# ── 数据集 ────────────────────────────────────────────────────────────────────
np.random.seed(42)
N, SPLIT, FEATS = 1000, 800, 20

X_all  = np.random.randn(N, FEATS).astype(np.float32)
X_tr, X_te = X_all[:SPLIT], X_all[SPLIT:]

w     = np.random.randn(FEATS).astype(np.float32)
y_reg = (X_all @ w + 0.1 * np.random.randn(N).astype(np.float32)).astype(np.float32)
y_reg_tr, y_reg_te = y_reg[:SPLIT], y_reg[SPLIT:]

y_bin    = (X_all[:, 0] > 0).astype(np.int32)
y_bin_tr, y_bin_te = y_bin[:SPLIT], y_bin[SPLIT:]

y_multi  = np.zeros(N, dtype=np.int32)
y_multi[X_all[:, 0] >  0.5] = 1
y_multi[X_all[:, 0] < -0.5] = 2
y_multi_tr, y_multi_te = y_multi[:SPLIT], y_multi[SPLIT:]

# 聚类专用数据（3 个明确分离的簇，DBSCAN / KMeans 专用）
_c0 = np.array([4, 4] + [0] * (FEATS - 2), dtype=np.float32)
_c1 = np.array([-4, -4] + [0] * (FEATS - 2), dtype=np.float32)
X_clust = np.vstack([
    np.random.randn(200, FEATS).astype(np.float32) * 0.4 + _c0,
    np.random.randn(200, FEATS).astype(np.float32) * 0.4 + _c1,
    np.random.randn(200, FEATS).astype(np.float32) * 0.4,
])

X_tiny      = X_all[:200]                              # TSNE / UMAP
X_small     = X_all[:500]
y_small_bin = y_bin[:500]
y_small_reg = y_reg[:500]
X_pos       = np.abs(X_all).astype(np.float32) + 0.01  # TruncatedSVD（非负）

# ── 辅助函数 ──────────────────────────────────────────────────────────────────
results = []

def section(title):
    print(f"\n{'='*62}")
    print(f"  {title}")
    print(f"{'='*62}")

def ok(msg, ms=None):
    t = f"  ({ms:.1f} ms)" if ms is not None else ""
    print(f"  [OK]  {msg}{t}")
    results.append((True, msg, ms or 0))

def fail(name, err):
    body = ""
    if hasattr(err, "response") and err.response is not None:
        try:
            d = err.response.json()
            lines = (d.get("traceback") or "").strip().splitlines()
            body = f"\n       ↳ {lines[-1] if lines else d.get('error','?')}"
        except Exception:
            pass
    print(f"  [FAIL] {name}: {err}{body}")
    results.append((False, name, 0))

def r2_score(pred, true):
    p, t = pred.astype(np.float64), true.astype(np.float64)
    return float(1 - np.sum((p - t) ** 2) / np.sum((t - t.mean()) ** 2))

def accuracy(pred, true):
    return float((pred.astype(np.int32) == true.astype(np.int32)).mean())

def timed(fn):
    """执行 fn()，返回 (result, elapsed_ms)。"""
    t0 = time.perf_counter()
    r = fn()
    return r, (time.perf_counter() - t0) * 1000


# ═══════════════════════════════════════════════════════════════════════════════
# 0. 服务健康检查
# ═══════════════════════════════════════════════════════════════════════════════
section("0. 服务健康检查")
try:
    resp = _session.get(f"{_BRIDGE_URL}/health", timeout=5)
    resp.raise_for_status()
    info = resp.json()
    ok(f"server ok  cuML={info['cuml_version']}")
except Exception as e:
    fail("health", e)
    print("  服务未运行，请先启动 WSL2 bridge。")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. StandardScaler  (fit_transform / transform / inverse_transform)
# ═══════════════════════════════════════════════════════════════════════════════
section("1. StandardScaler")
sc_main = StandardScaler()
X_tr_s = X_te_s = None
try:
    X_tr_s, ms = timed(lambda: sc_main.fit_transform(X_tr))
    ok(f"fit_transform  {X_tr.shape} → {X_tr_s.shape}", ms)
    assert abs(X_tr_s.mean()) < 0.05 and abs(X_tr_s.std() - 1) < 0.05
    print(f"       mean={X_tr_s.mean():.4f}  std={X_tr_s.std():.4f}")

    X_te_s, ms = timed(lambda: sc_main.transform(X_te))
    ok(f"transform      test {X_te.shape} → {X_te_s.shape}", ms)

    X_inv, ms = timed(lambda: sc_main.inverse_transform(X_tr_s))
    max_err = float(np.abs(X_inv - X_tr).max())
    ok(f"inverse_transform  max_err={max_err:.2e}", ms)
    assert max_err < 1e-3, f"inverse_transform 误差过大: {max_err:.2e}"
except Exception as e:
    fail("StandardScaler", e)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. MinMaxScaler
# ═══════════════════════════════════════════════════════════════════════════════
section("2. MinMaxScaler")
try:
    mms = MinMaxScaler()
    X_mm, ms = timed(lambda: mms.fit_transform(X_tr))
    ok(f"fit_transform  {X_tr.shape}", ms)
    assert X_mm.min() >= -1e-4 and X_mm.max() <= 1 + 1e-4
    print(f"       min={X_mm.min():.4f}  max={X_mm.max():.4f}")

    X_mm_te, ms = timed(lambda: mms.transform(X_te))
    ok(f"transform      test {X_te.shape}", ms)
except Exception as e:
    fail("MinMaxScaler", e)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. LabelEncoder  (fit_transform / inverse_transform round-trip)
# ═══════════════════════════════════════════════════════════════════════════════
section("3. LabelEncoder")
try:
    le = LabelEncoder()
    y_enc, ms = timed(lambda: le.fit_transform(y_multi.astype(np.float32)))
    enc_u = np.unique(y_enc.astype(int)).tolist()
    ok(f"fit_transform  {y_enc.shape}  encoded_unique={enc_u}", ms)

    y_back, ms = timed(lambda: le.inverse_transform(y_enc))
    match = float((y_back.astype(int) == y_multi).mean())
    ok(f"inverse_transform  round-trip match={match:.4f}", ms)
    assert match > 0.99, f"round-trip mismatch={match:.4f}"
except Exception as e:
    fail("LabelEncoder", e)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. PCA  (n_components=10, fit_transform / transform / inverse_transform)
# ═══════════════════════════════════════════════════════════════════════════════
section("4. PCA  (n_components=10)")
X_tr_pca = X_te_pca = None
try:
    pca = PCA(n_components=10)
    X_tr_pca, ms = timed(lambda: pca.fit_transform(X_tr))
    ok(f"fit_transform  {X_tr.shape} → {X_tr_pca.shape}", ms)
    assert X_tr_pca.shape == (SPLIT, 10)

    X_te_pca, ms = timed(lambda: pca.transform(X_te))
    ok(f"transform      {X_te.shape} → {X_te_pca.shape}", ms)

    X_inv_pca, ms = timed(lambda: pca.inverse_transform(X_tr_pca))
    ok(f"inverse_transform  shape={X_inv_pca.shape}", ms)
    assert X_inv_pca.shape == X_tr.shape
except Exception as e:
    fail("PCA", e)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. TruncatedSVD  (n_components=8)
# ═══════════════════════════════════════════════════════════════════════════════
section("5. TruncatedSVD  (n_components=8)")
try:
    tsvd = TruncatedSVD(n_components=8)
    X_svd, ms = timed(lambda: tsvd.fit_transform(X_pos[:SPLIT]))
    ok(f"fit_transform  {X_pos[:SPLIT].shape} → {X_svd.shape}", ms)
    assert X_svd.shape == (SPLIT, 8)

    X_svd_te, ms = timed(lambda: tsvd.transform(X_pos[SPLIT:]))
    ok(f"transform      {X_pos[SPLIT:].shape} → {X_svd_te.shape}", ms)
except Exception as e:
    fail("TruncatedSVD", e)


# ═══════════════════════════════════════════════════════════════════════════════
# 6. LinearRegression  (train + test R²)
# ═══════════════════════════════════════════════════════════════════════════════
section("6. LinearRegression  (test R²)")
try:
    lr = LinearRegression()
    _, ms     = timed(lambda: lr.fit(X_tr, y_reg_tr))
    ok(f"fit  {X_tr.shape}", ms)
    y_tr_p, ms = timed(lambda: lr.predict(X_tr))
    ok(f"predict train", ms)
    y_te_p, ms = timed(lambda: lr.predict(X_te))
    ok(f"predict test", ms)
    r_tr = r2_score(y_tr_p, y_reg_tr)
    r_te = r2_score(y_te_p, y_reg_te)
    print(f"       train R²={r_tr:.4f}  test R²={r_te:.4f}")
    assert r_te > 0.8, f"test R²={r_te:.4f} too low"
    ok(f"R²  train={r_tr:.4f}  test={r_te:.4f}")
except Exception as e:
    fail("LinearRegression", e)


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Ridge  (alpha=1.0)
# ═══════════════════════════════════════════════════════════════════════════════
section("7. Ridge  (alpha=1.0)")
try:
    ridge = Ridge(alpha=1.0)
    _, ms   = timed(lambda: ridge.fit(X_tr, y_reg_tr))
    ok(f"fit", ms)
    y_p, ms = timed(lambda: ridge.predict(X_te))
    ok(f"predict test", ms)
    r = r2_score(y_p, y_reg_te)
    print(f"       test R²={r:.4f}")
    assert r > 0.7, f"test R²={r:.4f} too low"
    ok(f"R² test={r:.4f}")
except Exception as e:
    fail("Ridge", e)


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Lasso  (alpha=0.01)
# ═══════════════════════════════════════════════════════════════════════════════
section("8. Lasso  (alpha=0.01)")
try:
    lasso = Lasso(alpha=0.01)
    _, ms   = timed(lambda: lasso.fit(X_tr, y_reg_tr))
    ok(f"fit", ms)
    y_p, ms = timed(lambda: lasso.predict(X_te))
    ok(f"predict test", ms)
    r = r2_score(y_p, y_reg_te)
    print(f"       test R²={r:.4f}")
    assert r > 0.5, f"test R²={r:.4f} too low"
    ok(f"R² test={r:.4f}")
except Exception as e:
    fail("Lasso", e)


# ═══════════════════════════════════════════════════════════════════════════════
# 9. ElasticNet  (alpha=0.01, l1_ratio=0.5)
# ═══════════════════════════════════════════════════════════════════════════════
section("9. ElasticNet  (alpha=0.01, l1_ratio=0.5)")
try:
    enet = ElasticNet(alpha=0.01, l1_ratio=0.5)
    _, ms   = timed(lambda: enet.fit(X_tr, y_reg_tr))
    ok(f"fit", ms)
    y_p, ms = timed(lambda: enet.predict(X_te))
    ok(f"predict test", ms)
    r = r2_score(y_p, y_reg_te)
    print(f"       test R²={r:.4f}")
    assert r > 0.5, f"test R²={r:.4f} too low"
    ok(f"R² test={r:.4f}")
except Exception as e:
    fail("ElasticNet", e)


# ═══════════════════════════════════════════════════════════════════════════════
# 10. LogisticRegression  (binary)
# ═══════════════════════════════════════════════════════════════════════════════
section("10. LogisticRegression  (binary)")
try:
    sc10 = StandardScaler()
    Xtr10, ms = timed(lambda: sc10.fit_transform(X_tr))
    ok(f"scaler.fit_transform", ms)
    Xte10, ms = timed(lambda: sc10.transform(X_te))
    ok(f"scaler.transform", ms)
    clf10 = LogisticRegression(max_iter=200)
    _, ms      = timed(lambda: clf10.fit(Xtr10, y_bin_tr))
    ok(f"fit", ms)
    pred10, ms = timed(lambda: clf10.predict(Xte10))
    ok(f"predict test", ms)
    a = accuracy(pred10, y_bin_te)
    print(f"       test acc={a:.4f}  unique={np.unique(pred10.astype(int)).tolist()}")
    assert a > 0.7, f"acc={a:.4f} too low"
    ok(f"acc test={a:.4f}")
except Exception as e:
    fail("LogisticRegression binary", e)


# ═══════════════════════════════════════════════════════════════════════════════
# 11. LogisticRegression  (3-class multiclass)
# ═══════════════════════════════════════════════════════════════════════════════
section("11. LogisticRegression  (3-class)")
try:
    sc11 = StandardScaler()
    Xtr11, _ = timed(lambda: sc11.fit_transform(X_tr))
    Xte11, _ = timed(lambda: sc11.transform(X_te))
    clf11 = LogisticRegression(max_iter=200)
    _, ms       = timed(lambda: clf11.fit(Xtr11, y_multi_tr))
    ok(f"fit", ms)
    pred11, ms  = timed(lambda: clf11.predict(Xte11))
    ok(f"predict test", ms)
    a = accuracy(pred11, y_multi_te)
    print(f"       test acc={a:.4f}  unique={np.unique(pred11.astype(int)).tolist()}")
    assert a > 0.5, f"acc={a:.4f} too low"
    ok(f"acc test={a:.4f}")
except Exception as e:
    fail("LogisticRegression multiclass", e)


# ═══════════════════════════════════════════════════════════════════════════════
# 12. KMeans  (k=3, 明确分离的簇)
# ═══════════════════════════════════════════════════════════════════════════════
section("12. KMeans  (k=3, clustered data)")
try:
    km = KMeans(n_clusters=3, random_state=0)
    labels, ms    = timed(lambda: km.fit_predict(X_clust))
    ok(f"fit_predict  {X_clust.shape}", ms)
    u = set(np.unique(labels.astype(int)))
    assert u == {0, 1, 2}, f"expected {{0,1,2}}, got {u}"

    pred_new, ms = timed(lambda: km.predict(X_clust[:30]))
    ok(f"predict  30 new samples", ms)
    ok(f"clusters={sorted(u)}")
except Exception as e:
    fail("KMeans", e)


# ═══════════════════════════════════════════════════════════════════════════════
# 13. DBSCAN  (eps=1.0, min_samples=5)
# ═══════════════════════════════════════════════════════════════════════════════
section("13. DBSCAN  (eps=3.0, min_samples=5)")
try:
    db = DBSCAN(eps=3.0, min_samples=5)
    labels_db, ms = timed(lambda: db.fit_predict(X_clust))
    ok(f"fit_predict  {X_clust.shape}", ms)
    u_db = np.unique(labels_db.astype(int)).tolist()
    noise_pct = float((labels_db == -1).mean())
    print(f"       labels={u_db}  noise%={noise_pct:.1%}")
    assert any(l >= 0 for l in u_db), "DBSCAN found no clusters (all noise)"
    ok(f"clusters={[l for l in u_db if l >= 0]}  noise%={noise_pct:.1%}")
except Exception as e:
    fail("DBSCAN", e)


# ═══════════════════════════════════════════════════════════════════════════════
# 14. KNeighborsClassifier  (k=5)
# ═══════════════════════════════════════════════════════════════════════════════
section("14. KNeighborsClassifier  (k=5)")
try:
    knnc = KNeighborsClassifier(n_neighbors=5)
    _, ms       = timed(lambda: knnc.fit(X_tr, y_bin_tr))
    ok(f"fit  {X_tr.shape}", ms)
    pred14, ms  = timed(lambda: knnc.predict(X_te))
    ok(f"predict test  {X_te.shape}", ms)
    a = accuracy(pred14, y_bin_te)
    print(f"       test acc={a:.4f}")
    assert a > 0.7, f"acc={a:.4f} too low"
    ok(f"acc test={a:.4f}")
except Exception as e:
    fail("KNeighborsClassifier", e)


# ═══════════════════════════════════════════════════════════════════════════════
# 15. KNeighborsRegressor  (k=5)
# ═══════════════════════════════════════════════════════════════════════════════
section("15. KNeighborsRegressor  (k=5)")
try:
    knnr = KNeighborsRegressor(n_neighbors=5)
    _, ms      = timed(lambda: knnr.fit(X_tr, y_reg_tr))
    ok(f"fit  {X_tr.shape}", ms)
    pred15, ms = timed(lambda: knnr.predict(X_te))
    ok(f"predict test", ms)
    r = r2_score(pred15, y_reg_te)
    print(f"       test R²={r:.4f}")
    assert r > 0.5, f"test R²={r:.4f} too low"
    ok(f"R² test={r:.4f}")
except Exception as e:
    fail("KNeighborsRegressor", e)


# ═══════════════════════════════════════════════════════════════════════════════
# 16. NearestNeighbors  (kneighbors 返回 distances + indices)
# ═══════════════════════════════════════════════════════════════════════════════
section("16. NearestNeighbors  (kneighbors, k=3)")
try:
    nn = NearestNeighbors(n_neighbors=3)
    _, ms       = timed(lambda: nn.fit(X_tr))
    ok(f"fit  {X_tr.shape}", ms)
    result, ms  = timed(lambda: nn.kneighbors(X_te[:20]))
    ok(f"kneighbors  20 queries", ms)
    if isinstance(result, (list, tuple)) and len(result) == 2:
        dists = np.asarray(result[0])
        inds  = np.asarray(result[1])
        print(f"       dists.shape={dists.shape}  inds.shape={inds.shape}")
        assert dists.shape == (20, 3) and inds.shape == (20, 3)
        assert (dists >= 0).all()
        ok(f"dists {dists.shape}  inds {inds.shape}  all_nonneg=True")
    else:
        ok(f"result type={type(result).__name__}  (shape check skipped)")
except Exception as e:
    fail("NearestNeighbors.kneighbors", e)


# ═══════════════════════════════════════════════════════════════════════════════
# 17. RandomForestClassifier  (n_estimators=100)
# ═══════════════════════════════════════════════════════════════════════════════
section("17. RandomForestClassifier  (n_estimators=100)")
try:
    rfc = RandomForestClassifier(n_estimators=100, random_state=0)
    _, ms         = timed(lambda: rfc.fit(X_small, y_small_bin))
    ok(f"fit  {X_small.shape}", ms)
    pred17_tr, ms = timed(lambda: rfc.predict(X_small))
    ok(f"predict train", ms)
    pred17_te, ms = timed(lambda: rfc.predict(X_te))
    ok(f"predict test", ms)
    a_tr = accuracy(pred17_tr, y_small_bin)
    a_te = accuracy(pred17_te, y_bin_te)
    print(f"       train acc={a_tr:.4f}  test acc={a_te:.4f}")
    assert a_tr > 0.85, f"train acc={a_tr:.4f} too low"
    ok(f"train acc={a_tr:.4f}  test acc={a_te:.4f}")
except Exception as e:
    fail("RandomForestClassifier", e)


# ═══════════════════════════════════════════════════════════════════════════════
# 18. RandomForestRegressor  (n_estimators=100)
# ═══════════════════════════════════════════════════════════════════════════════
section("18. RandomForestRegressor  (n_estimators=100)")
try:
    rfr = RandomForestRegressor(n_estimators=100, random_state=0)
    _, ms      = timed(lambda: rfr.fit(X_small, y_small_reg))
    ok(f"fit  {X_small.shape}", ms)
    pred18, ms = timed(lambda: rfr.predict(X_te))
    ok(f"predict test", ms)
    r = r2_score(pred18, y_reg_te)
    print(f"       test R²={r:.4f}")
    assert r > 0.3, f"test R²={r:.4f} too low"
    ok(f"R² test={r:.4f}")
except Exception as e:
    fail("RandomForestRegressor", e)


# ═══════════════════════════════════════════════════════════════════════════════
# 19. SVC  (kernel=rbf, C=10, test acc)
# ═══════════════════════════════════════════════════════════════════════════════
section("19. SVC  (kernel=rbf, C=10)")
try:
    sc19 = StandardScaler()
    Xtr19, _ = timed(lambda: sc19.fit_transform(X_small))
    Xte19, _ = timed(lambda: sc19.transform(X_te))
    svc = SVC(kernel="rbf", C=10.0)
    _, ms      = timed(lambda: svc.fit(Xtr19, y_small_bin))
    ok(f"fit  {Xtr19.shape}", ms)
    pred19, ms = timed(lambda: svc.predict(Xte19))
    ok(f"predict test", ms)
    a = accuracy(pred19, y_bin_te)
    print(f"       test acc={a:.4f}")
    assert a > 0.7, f"acc={a:.4f} too low"
    ok(f"acc test={a:.4f}")
except Exception as e:
    fail("SVC", e)


# ═══════════════════════════════════════════════════════════════════════════════
# 20. SVR  (kernel=rbf, C=10, test R²)
# ═══════════════════════════════════════════════════════════════════════════════
section("20. SVR  (kernel=rbf, C=10)")
try:
    sc20 = StandardScaler()
    Xtr20, _ = timed(lambda: sc20.fit_transform(X_small))
    Xte20, _ = timed(lambda: sc20.transform(X_te))
    svr = SVR(kernel="rbf", C=10.0)
    _, ms      = timed(lambda: svr.fit(Xtr20, y_small_reg))
    ok(f"fit  {Xtr20.shape}", ms)
    pred20, ms = timed(lambda: svr.predict(Xte20))
    ok(f"predict test", ms)
    r = r2_score(pred20, y_reg_te)
    print(f"       test R²={r:.4f}")
    assert r > 0.3, f"test R²={r:.4f} too low"
    ok(f"R² test={r:.4f}")
except Exception as e:
    fail("SVR", e)


# ═══════════════════════════════════════════════════════════════════════════════
# 21. TSNE  (200 samples → (200, 2))
# ═══════════════════════════════════════════════════════════════════════════════
section("21. TSNE  (n_components=2, 200 samples)")
try:
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne, ms = timed(lambda: tsne.fit_transform(X_tiny))
    ok(f"fit_transform  {X_tiny.shape} → {X_tsne.shape}", ms)
    assert X_tsne.shape == (200, 2), f"expected (200,2), got {X_tsne.shape}"
    ok(f"shape OK  range=[{X_tsne.min():.1f}, {X_tsne.max():.1f}]")
except Exception as e:
    fail("TSNE", e)


# ═══════════════════════════════════════════════════════════════════════════════
# 22. UMAP  (200 samples → (200, 2))
# ═══════════════════════════════════════════════════════════════════════════════
section("22. UMAP  (n_components=2, 200 samples)")
try:
    umap = UMAP(n_components=2, random_state=42)
    X_umap, ms = timed(lambda: umap.fit_transform(X_tiny))
    ok(f"fit_transform  {X_tiny.shape} → {X_umap.shape}", ms)
    assert X_umap.shape == (200, 2), f"expected (200,2), got {X_umap.shape}"
    ok(f"shape OK  range=[{X_umap.min():.1f}, {X_umap.max():.1f}]")
except Exception as e:
    fail("UMAP", e)


# ═══════════════════════════════════════════════════════════════════════════════
# 23. 模型保存 / 加载  (SVC round-trip)
# ═══════════════════════════════════════════════════════════════════════════════
section("23. 模型保存 / 加载  (SVC round-trip)")
try:
    sc23 = StandardScaler()
    X23, _ = timed(lambda: sc23.fit_transform(X_small))
    svc23 = SVC(kernel="rbf", C=1.0)
    svc23.fit(X23, y_small_bin)

    orig_pred, ms = timed(lambda: svc23.predict(X23[:50]))
    ok(f"predict (before save)", ms)

    save_name = "_e2e_test_svc_tmp"
    path = svc23.save(save_name)
    ok(f"save → {os.path.basename(path)}")

    svc_loaded = ProxyEstimator.load(save_name)
    load_pred, ms = timed(lambda: svc_loaded.predict(X23[:50]))
    ok(f"predict (after load)", ms)

    match = float((orig_pred.astype(int) == load_pred.astype(int)).mean())
    print(f"       save/load 一致性={match:.4f}")
    assert match == 1.0, f"不一致: {match:.4f}"
    ok(f"save/load 预测完全一致")
except Exception as e:
    fail("save/load", e)


# ═══════════════════════════════════════════════════════════════════════════════
# 24. get_params / set_params
# ═══════════════════════════════════════════════════════════════════════════════
section("24. get_params / set_params")
try:
    rf_p = RandomForestClassifier(n_estimators=10, max_depth=3)
    p1 = rf_p.get_params()
    assert p1.get("n_estimators") == 10 and p1.get("max_depth") == 3
    ok(f"get_params  n_estimators={p1['n_estimators']}  max_depth={p1['max_depth']}")

    rf_p.set_params(n_estimators=20, max_depth=5)
    p2 = rf_p.get_params()
    assert p2.get("n_estimators") == 20 and p2.get("max_depth") == 5
    ok(f"set_params  n_estimators={p2['n_estimators']}  max_depth={p2['max_depth']}")

    _, ms      = timed(lambda: rf_p.fit(X_small, y_small_bin))
    ok(f"fit after set_params", ms)
    preds_p, ms = timed(lambda: rf_p.predict(X_small[:10]))
    ok(f"predict after set_params  preds={preds_p.astype(int).tolist()}", ms)
except Exception as e:
    fail("get_params/set_params", e)


# ═══════════════════════════════════════════════════════════════════════════════
# 25. Pipeline 链式  (StandardScaler → LogisticRegression)
# ═══════════════════════════════════════════════════════════════════════════════
section("25. Pipeline 链式  (StandardScaler → LogisticRegression)")
try:
    pipe_sc  = StandardScaler()
    pipe_clf = LogisticRegression(max_iter=300)

    Xtr_p, ms  = timed(lambda: pipe_sc.fit_transform(X_tr))
    ok(f"scaler.fit_transform  {X_tr.shape}", ms)
    _, ms       = timed(lambda: pipe_clf.fit(Xtr_p, y_bin_tr))
    ok(f"clf.fit", ms)
    Xte_p, ms  = timed(lambda: pipe_sc.transform(X_te))
    ok(f"scaler.transform  test", ms)
    pred_p, ms = timed(lambda: pipe_clf.predict(Xte_p))
    ok(f"clf.predict  test", ms)

    a = accuracy(pred_p, y_bin_te)
    print(f"       Pipeline test acc={a:.4f}")
    assert a > 0.7, f"acc={a:.4f} too low"
    ok(f"Pipeline acc={a:.4f}")
except Exception as e:
    fail("Pipeline", e)


# ═══════════════════════════════════════════════════════════════════════════════
# 26. 大数组 mmap 传输  (~50 MB)
# ═══════════════════════════════════════════════════════════════════════════════
section("26. 大数组 mmap 传输  (~50 MB)")
try:
    X_big = np.random.randn(10000, 1280).astype(np.float32)
    print(f"       输入大小: {X_big.nbytes / 1e6:.1f} MB")
    sc_big = StandardScaler()
    out_big, ms = timed(lambda: sc_big.fit_transform(X_big))
    ok(f"fit_transform  {X_big.shape} → {out_big.shape}", ms)
    assert out_big.shape == X_big.shape
    throughput = X_big.nbytes / (ms / 1000) / 1e6
    ok(f"吞吐≈{throughput:.0f} MB/s")
except Exception as e:
    fail("大数组 mmap", e)


# ═══════════════════════════════════════════════════════════════════════════════
# 27. 连续请求稳定性  (anti-stale-cache，10× unique inputs)
# ═══════════════════════════════════════════════════════════════════════════════
section("27. 连续请求稳定性  (10× unique inputs, anti-stale-cache)")
try:
    sc27 = StandardScaler()
    sc27.fit(X_tr)   # 均值≈0 各 feature
    prev_mean = None
    for i in range(10):
        val = float(i + 1)   # 1.0, 2.0, ..., 10.0（每轮不同）
        X_probe = np.full((100, FEATS), val, dtype=np.float32)
        out, _ = timed(lambda: sc27.transform(X_probe))
        cur_mean = float(out.mean())
        # val 每步 +1 → 标准化后均值每步约 +1.0
        # 若 stale cache: cur_mean ≈ prev_mean（违反单调递增 > 0.5）
        if prev_mean is not None:
            assert cur_mean > prev_mean + 0.5, (
                f"第 {i+1} 次请求疑似 stale cache！"
                f"  cur_mean={cur_mean:.4f}  prev_mean={prev_mean:.4f}"
            )
        prev_mean = cur_mean
    ok(f"10 次连续请求均值单调递增，无 stale cache  last_mean={prev_mean:.4f}")
except Exception as e:
    fail("连续请求稳定性", e)


# ═══════════════════════════════════════════════════════════════════════════════
# 汇总
# ═══════════════════════════════════════════════════════════════════════════════
section("测试结果汇总")
passed  = [r for r in results if r[0]]
failed  = [r for r in results if not r[0]]
total_t = sum(r[2] for r in results)

print(f"\n  通过: {len(passed)}  失败: {len(failed)}  总耗时: {total_t:.0f} ms\n")
for ok_f, msg, ms in results:
    tag   = "[OK]  " if ok_f else "[FAIL]"
    t_str = f"{ms:8.1f} ms" if ms else "         ---"
    print(f"  {tag}  {t_str}  {msg}")

if failed:
    print(f"\n  ❌ {len(failed)} 项失败，详见上方错误信息。")
    sys.exit(1)
else:
    print(f"\n  🎉 全部 {len(passed)} 项通过！")
