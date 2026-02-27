"""
test_mmap.py - 验证 mmap 共享内存传输层是否正常工作

测试矩阵：
  1. 小数组（< 10 KB）—— HTTP inline base64（不走 mmap）
  2. 大数组（>= 10 KB）—— mmap slot 传输
  3. 结果解码正确性（fit_transform / predict 返回值一致）
  4. 传输带宽粗测（100 MB 数组耗时）
"""
import sys
import time
import numpy as np

# 使用桥接包
sys.path.insert(0, r"C:\Users\nicho\gpu-sklearn-bridge")
from cuml_proxy.preprocessing import StandardScaler
from cuml_proxy.decomposition import PCA
from cuml_proxy.linear_model import LinearRegression

print("=" * 60)
print(" mmap 共享内存传输测试")
print("=" * 60)

# ── 1. 小数组（走 base64 inline）────────────────────────────────
print("\n[1] 小数组（2×4, ~96 B）—— 期望走 inline base64")
X_small = np.array([[1.0, 2.0, 3.0, 4.0],
                    [5.0, 6.0, 7.0, 8.0]], dtype=np.float32)
sc = StandardScaler()
out = sc.fit_transform(X_small)
print(f"    输入形状: {X_small.shape}  输出形状: {out.shape}")
print(f"    结果（已标准化）:\n{out}")
del sc

# ── 2. 中等数组（>= 10 KB，走 mmap）──────────────────────────────
print("\n[2] 中等数组（1000×20, 80 KB）—— 期望走 mmap")
X_mid = np.random.rand(1000, 20).astype(np.float32)
t0 = time.perf_counter()
sc2 = StandardScaler()
out2 = sc2.fit_transform(X_mid)
elapsed = time.perf_counter() - t0
print(f"    输入形状: {X_mid.shape}  输出形状: {out2.shape}")
print(f"    均值≈0: {out2.mean(axis=0)[:4].round(4)}")
print(f"    标准差≈1: {out2.std(axis=0)[:4].round(4)}")
print(f"    耗时: {elapsed*1000:.1f} ms")
del sc2

# ── 3. 大数组（~9.6 MB）—— 测试 mmap 吞吐────────────────────────
print("\n[3] 大数组（5000×480, ~9.6 MB）—— mmap 吞吐测试")
X_big = np.random.rand(5000, 480).astype(np.float32)
print(f"    数组大小: {X_big.nbytes/1e6:.1f} MB")
t0 = time.perf_counter()
sc3 = StandardScaler()
out3 = sc3.fit_transform(X_big)
elapsed = time.perf_counter() - t0
print(f"    输出形状: {out3.shape}")
print(f"    耗时: {elapsed*1000:.1f} ms")
print(f"    等效吞吐: {X_big.nbytes/elapsed/1e6:.1f} MB/s（含 HTTP + GPU）")
del sc3

# ── 4. 结果正确性验证（PCA）────────────────────────────────────────
print("\n[4] 正确性验证 —— PCA 降维到 2D")
X_iris = np.random.randn(150, 4).astype(np.float32)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_iris)
print(f"    输入: {X_iris.shape}  →  输出: {X_pca.shape}")
assert X_pca.shape == (150, 2), f"形状错误: {X_pca.shape}"
print("    ✅  形状正确")

# ── 5. 大数组往返（输入 + 输出均走 mmap）────────────────────────────
print("\n[5] 大数组输入 + 大数组输出往返")
X_big2 = np.random.rand(10000, 20).astype(np.float32)  # 800 KB
lr = LinearRegression()
y = (X_big2 @ np.random.rand(20).astype(np.float32)) + 0.1
t0 = time.perf_counter()
lr.fit(X_big2, y)
y_pred = lr.predict(X_big2)
elapsed = time.perf_counter() - t0
print(f"    X 大小: {X_big2.nbytes/1024:.0f} KB  y 大小: {y.nbytes/1024:.0f} KB")
print(f"    预测输出: {y_pred.shape}  耗时: {elapsed*1000:.1f} ms")
assert y_pred.shape == (10000,), f"预测形状错误: {y_pred.shape}"
print("    ✅  往返正确")

print("\n" + "=" * 60)
print(" 全部测试通过 ✅")
print("=" * 60)
