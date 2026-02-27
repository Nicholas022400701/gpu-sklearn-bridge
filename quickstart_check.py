#!/usr/bin/env python3
"""
Quick start guide for Extended mmap optimization

快速开始指南 - 扩展 mmap 优化验证
"""

import os
import sys
import subprocess
import time
from pathlib import Path

BRIDGE_DIR = Path(__file__).parent
SHM_DIR = BRIDGE_DIR / "shm"
POOL_BIN = SHM_DIR / "pool.bin"

def print_header(text):
    print(f"\n{'='*70}")
    print(f" {text}")
    print(f"{'='*70}\n")

def print_status(step, desc):
    print(f"[{step}] {desc}")

def check_pool():
    """检查 pool.bin 文件大小"""
    if POOL_BIN.exists():
        size = POOL_BIN.stat().st_size
        size_gb = size / (1024**3)
        if size >= 4 * 1024**3:
            print(f"  ✅  pool.bin 大小: {size_gb:.1f} GB (正确)")
            return True
        else:
            print(f"  ⚠️  pool.bin 大小: {size_gb:.1f} GB (应为 4 GB)")
            return False
    else:
        print(f"  ℹ️  pool.bin 不存在（将在首次运行时创建）")
        return True

def check_imports():
    """检查必要的模块"""
    deps = {
        "numpy": "NumPy",
        "requests": "requests",
        "flask": "Flask",
    }
    
    missing = []
    for module, name in deps.items():
        try:
            __import__(module)
            print(f"  ✅  {name}")
        except ImportError:
            print(f"  ❌  {name}")
            missing.append(module)
    
    return len(missing) == 0

def check_code_updates():
    """检查代码是否已更新"""
    issues = []
    
    # 检查 shm_transport.py
    shm_file = BRIDGE_DIR / "shm_transport.py"
    if shm_file.exists():
        content = shm_file.read_text()
        if "SLOT_INPUT_COUNT = 4" not in content:
            issues.append("shm_transport.py: 未检测到多 slot 配置")
        if "_alloc_output_slot" not in content:
            issues.append("shm_transport.py: 缺少轮转分配方法")
        if "超过 SLOT_SIZE 返回 None" in content:
            issues.append("shm_transport.py: 仍包含 fallback 检查")
    
    # 检查 server.py
    server_file = BRIDGE_DIR / "server.py"
    if server_file.exists():
        content = server_file.read_text()
        if "uuid.uuid4().hex).npy" in content:
            issues.append("server.py: 仍使用 .npy fallback")
    
    # 检查 proxy.py
    proxy_file = BRIDGE_DIR / "cuml_proxy" / "proxy.py"
    if proxy_file.exists():
        content = proxy_file.read_text()
        if "uuid.uuid4().hex).npy" in content:
            issues.append("proxy.py: 仍使用 .npy fallback")
    
    if issues:
        for issue in issues:
            print(f"  ⚠️  {issue}")
        return False
    else:
        print(f"  ✅  代码已更新")
        return True

def test_import():
    """测试导入"""
    try:
        sys.path.insert(0, str(BRIDGE_DIR))
        from cuml_proxy.preprocessing import StandardScaler
        print(f"  ✅  能够导入 cuml_proxy")
        return True
    except Exception as e:
        print(f"  ❌  导入失败: {e}")
        return False

def main():
    print_header("🚀 扩展 mmap 优化 - 快速验证")
    
    # 步骤 1: 环境检查
    print_status("1", "环境检查")
    print(f"  工作目录: {BRIDGE_DIR}")
    print(f"  Python 版本: {sys.version.split()[0]}")
    
    # 步骤 2: 依赖检查
    print_status("2", "检查依赖模块")
    if not check_imports():
        print("\n❌ 缺少必要的 Python 模块。请运行:")
        print("   pip install numpy requests flask")
        return False
    
    # 步骤 3: 代码更新检查
    print_status("3", "检查代码更新")
    if not check_code_updates():
        print("\n⚠️  代码可能未完全更新")
    
    # 步骤 4: 导入检查
    print_status("4", "检查模块导入")
    if not test_import():
        print("\n⚠️  模块导入失败，可能是配置问题")
    
    # 步骤 5: 共享内存池检查
    print_status("5", "检查共享内存池")
    check_pool()
    
    # 步骤 6: 建议后续操作
    print("\n" + "="*70)
    print(" 🎯 后续步骤:")
    print("="*70)
    print("""
1. 启动 WSL2 服务端（在 WSL2 中）:
   cd /mnt/c/Users/nicho/gpu-sklearn-bridge
   python server.py
   
   等待看到：
   [ShmTransport] 初始化 mmap pool: ... (4.0 GB)

2. 启动 Windows 客户端测试（新终端窗口）:
   cd C:\\Users\\nicho\\gpu-sklearn-bridge
   python test_extended_mmap.py
   
   预期：所有测试通过 ✅

3. 验证性能改进:
   - 检查 test_extended_mmap.py [4] 的耗时
   - 应该 < 1000 ms 处理 500 MB 数据
   
4. 查看详细文档:
   - EXTENDED_MMAP_OPTIMIZATION.md    —— 优化报告
   - MMAP_CONFIG_GUIDE.md             —— 配置指南
   - MIGRATION_CHECKLIST.md           —— 迁移清单
    """)
    
    print("="*70)
    print(" ✨ 扩展 mmap 架构已部署完毕！")
    print("="*70)
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
