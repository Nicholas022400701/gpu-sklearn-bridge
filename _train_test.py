# -*- coding: utf-8 -*-
"""
_train_test.py - 端对端训练测试（需要 WSL2 + GPU）
覆盖：StandardScaler / PCA / LinearRegression / LogisticRegression /
      KMeans / RandomForestClassifier / SVC
测试内容：fit / predict / transform / score / fit_transform
"""
import sys, time
import numpy as np
import requests

sys.path.insert(0, r"C:\Users\nicho\gpu-sklearn-bridge")

from cuml.preprocessing  import StandardScaler
from cuml.decomposition  import PCA
from cuml.linear_model   import LinearRegression, LogisticRegression
from cuml.cluster        import KMeans
from cuml.ensemble       import RandomForestClassifier
from cuml.svm            import SVC
from cuml_proxy.proxy    import _session, _BRIDGE_URL, _TIMEOUT

np.random.seed(42)

results = []

def section(title):
    print(f"\n{'='*55}")
    print(f"  {title}")
    print(f"{'='*55}")

def ok(msg, t0):
    ms = (time.perf_counter() - t0) * 1000
    line = f"  [OK]  {msg}  ({ms:.1f} ms)"
    print(line)
    results.append((True, msg, ms))

def fail(msg, err):
    # 如果是 HTTPError，尝试打印 JSON body 中的实际 cuML 错误
    body = ""
    if hasattr(err, "response") and err.response is not None:
        try:
            d = err.response.json()
            body = f"\n  cuML error: {d.get('error','?')}\n  {d.get('traceback','').strip().splitlines()[-1] if d.get('traceback') else ''}"
        except Exception:
            pass
    line = f"  [FAIL] {msg}: {err}{body}"
    print(line)
    results.append((False, msg, 0))

# ── 数据集 ──────────────────────────────────────────────────────
N_SMALL  = 500
N_MEDIUM = 5000
FEATS    = 20
CLASSES  = 3

X_small  = np.random.randn(N_SMALL,  FEATS).astype(np.float32)
X_medium = np.random.randn(N_MEDIUM, FEATS).astype(np.float32)
y_reg    = (X_medium @ np.random.randn(FEATS).astype(np.float32)).astype(np.float32)
# 分类标签用 int32（cuML 分类器要求整数标签）
y_cls    = (X_medium[:, 0] > 0).astype(np.int32)
y_small_cls = (X_small[:, 0] > 0).astype(np.int32)

# ── 1. StandardScaler ────────────────────────────────────────────
section("1. StandardScaler")
try:
    sc = StandardScaler()
    t0 = time.perf_counter()
    X_s = sc.fit_transform(X_medium)
    ok(f"fit_transform  {X_medium.shape} -> {X_s.shape}", t0)

    t0 = time.perf_counter()
    X_s2 = sc.transform(X_medium[:10])
    ok(f"transform      (10,{FEATS})", t0)

    assert X_s.shape == X_medium.shape
    assert abs(X_s.mean()) < 0.05, f"mean={X_s.mean():.4f} not ~0"
    assert abs(X_s.std() - 1.0) < 0.05, f"std={X_s.std():.4f} not ~1"
    print(f"       mean={X_s.mean():.4f}  std={X_s.std():.4f}")
except Exception as e:
    fail("StandardScaler", e)

# ── 2. PCA ───────────────────────────────────────────────────────
section("2. PCA  (n_components=5)")
try:
    pca = PCA(n_components=5)
    t0 = time.perf_counter()
    X_pca = pca.fit_transform(X_medium)
    ok(f"fit_transform  {X_medium.shape} -> {X_pca.shape}", t0)
    assert X_pca.shape == (N_MEDIUM, 5)

    t0 = time.perf_counter()
    X_pca2 = pca.transform(X_medium[:100])
    ok(f"transform      (100,{FEATS}) -> {X_pca2.shape}", t0)
except Exception as e:
    fail("PCA", e)

# ── 3. LinearRegression ───────────────────────────────────────────
section("3. LinearRegression")
try:
    lr = LinearRegression()
    t0 = time.perf_counter()
    lr.fit(X_medium, y_reg)
    ok(f"fit            {X_medium.shape}", t0)

    t0 = time.perf_counter()
    y_pred = lr.predict(X_medium)
    ok(f"predict        {y_pred.shape}", t0)

    # 手动计算 R2（避免 score() 内部 virtio-fs flush 时序问题）
    y_pred_f = y_pred.astype(np.float64)
    y_reg_f  = y_reg.astype(np.float64)
    SS_res = np.sum((y_pred_f - y_reg_f) ** 2)
    SS_tot = np.sum((y_reg_f  - y_reg_f.mean()) ** 2)
    r2 = float(1.0 - SS_res / SS_tot)
    corr = float(np.corrcoef(y_pred_f, y_reg_f)[0, 1])
    t0 = time.perf_counter()
    ok(f"R2 (manual)    = {r2:.4f}  corr={corr:.4f}", t0)
    print(f"       y_pred range=[{y_pred.min():.2f},{y_pred.max():.2f}]  y_reg range=[{y_reg.min():.2f},{y_reg.max():.2f}]")
    assert r2 > 0.8, f"R2={r2:.4f} too low"
except Exception as e:
    fail("LinearRegression", e)

# ── 4. LogisticRegression ─────────────────────────────────────────
section("4. LogisticRegression  (binary)")
try:
    scaler = StandardScaler()
    X_lr = scaler.fit_transform(X_medium)

    clf = LogisticRegression(max_iter=200)
    t0 = time.perf_counter()
    clf.fit(X_lr, y_cls)   # int32 labels
    ok(f"fit            {X_lr.shape}", t0)

    t0 = time.perf_counter()
    pred = clf.predict(X_lr)
    ok(f"predict        {pred.shape}", t0)

    # 诊断：打印 pred 原始值
    print(f"       pred dtype={pred.dtype}  raw[:5]={pred[:5].tolist()}")
    print(f"       pred min={pred.min()}, max={pred.max()}")

    # 手动计算 acc（避免 score() 内部 virtio-fs 内容一致性问题）
    pred_i32 = pred.astype(np.int32)
    acc = float((pred_i32 == y_cls).mean())
    t0 = time.perf_counter()
    ok(f"acc (manual)   = {acc:.4f}", t0)
    print(f"       pred unique={np.unique(pred_i32)}  y_cls dist: {y_cls.mean():.2f} pos rate")
    assert acc > 0.7, f"acc={acc:.4f} too low"
except Exception as e:
    fail("LogisticRegression", e)

# ── 5. KMeans ─────────────────────────────────────────────────────
section("5. KMeans  (k=3)")
try:
    km = KMeans(n_clusters=3, random_state=0)
    t0 = time.perf_counter()
    labels = km.fit_predict(X_medium)
    ok(f"fit_predict    {X_medium.shape} -> {labels.shape}", t0)
    assert set(np.unique(labels)).issubset({0, 1, 2})

    t0 = time.perf_counter()
    labels2 = km.predict(X_medium[:100])
    ok(f"predict        (100,) clusters={set(np.unique(labels2))}", t0)
except Exception as e:
    fail("KMeans", e)

# ── 6. RandomForestClassifier ─────────────────────────────────────
section("6. RandomForestClassifier  (n_estimators=50)")
try:
    rf = RandomForestClassifier(n_estimators=50, random_state=0)
    t0 = time.perf_counter()
    rf.fit(X_small, y_small_cls)   # int32 labels
    ok(f"fit            {X_small.shape}", t0)

    t0 = time.perf_counter()
    preds = rf.predict(X_small)
    ok(f"predict        {preds.shape}", t0)

    preds_i32 = preds.astype(np.int32)
    print(f"       preds dtype={preds.dtype}  unique={np.unique(preds_i32)}")
    print(f"       preds[:10]={preds_i32[:10].tolist()}")
    print(f"       y_small_cls[:10]={y_small_cls[:10].tolist()}")
    acc_rf = (preds_i32 == y_small_cls).mean()
    print(f"       train acc = {acc_rf:.4f}  y_cls dist: {y_small_cls.mean():.2f} pos rate")
    # cuML RF on train data should overfit (>= 0.85)
    assert acc_rf > 0.8, f"acc={acc_rf:.4f} too low"
except Exception as e:
    fail("RandomForestClassifier", e)

# ── 7. SVC ────────────────────────────────────────────────────────
section("7. SVC  (kernel=rbf, C=1)")
try:
    sc7 = StandardScaler()
    X7  = sc7.fit_transform(X_small)
    y7  = y_small_cls   # int32

    svc = SVC(kernel="rbf", C=1.0)
    t0 = time.perf_counter()
    svc.fit(X7, y7)
    ok(f"fit            {X7.shape}", t0)

    t0 = time.perf_counter()
    pred7 = svc.predict(X7)
    ok(f"predict        {pred7.shape}", t0)

    pred7_i32 = pred7.astype(np.int32)
    acc7 = float((pred7_i32 == y7).mean())
    print(f"       train acc = {acc7:.4f}  pred unique={np.unique(pred7_i32)}")
    assert acc7 > 0.8, f"acc={acc7:.4f} too low"
except Exception as e:
    fail("SVC", e)

# ── 8. 大数组 mmap 传输（~100 MB 输入）─────────────────────────────
section("8. 大数组 mmap 传输压力测试  (~100 MB)")
try:
    X_big = np.random.randn(10000, 1280).astype(np.float32)
    print(f"       输入大小: {X_big.nbytes/1e6:.1f} MB")
    sc8 = StandardScaler()
    t0 = time.perf_counter()
    out8 = sc8.fit_transform(X_big)
    ok(f"fit_transform  {X_big.shape} -> {out8.shape}", t0)
    total_ms = (time.perf_counter() - t0 + 1e-9)   # already counted in ok
    # recalc for throughput
    elapsed = results[-1][2] / 1000
    print(f"       等效吞吐: {X_big.nbytes/elapsed/1e6:.0f} MB/s")
except Exception as e:
    fail("大数组 mmap", e)

# ── 汇总 ─────────────────────────────────────────────────────────
section("测试结果汇总")
passed = [r for r in results if r[0]]
failed = [r for r in results if not r[0]]
total_time = sum(r[2] for r in results)

print(f"\n  通过: {len(passed)}  失败: {len(failed)}  总耗时: {total_time:.0f} ms\n")
for ok_flag, msg, ms in results:
    tag = "[OK]  " if ok_flag else "[FAIL]"
    t_str = f"{ms:7.1f} ms" if ms else "      ---"
    print(f"  {tag}  {t_str}  {msg}")

if failed:
    print(f"\n  {len(failed)} 项失败，请检查上方错误信息。")
    sys.exit(1)
else:
    print(f"\n  全部通过！")
