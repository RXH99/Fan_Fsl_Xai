"""
Step 1 (CWRU 版): 数据预处理

从 agent_model_projects 的 CWRU .mat 文件读取数据，
生成与风机数据格式完全一致的 preprocessed.npz。

输出:
  data_cwru/processed/preprocessed.npz
  → 包含 X_train, y_train, X_test, y_test, X_val, y_val

用法:
  python step1_preprocess_cwru.py
"""

import os, sys, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
from scipy.io import loadmat
from collections import Counter

# CWRU 原始 .mat 路径（指向 agent_model_projects 下的数据）
AGENT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "..", "agent_model_projects", "data")
CWRU_BASE = os.path.join(AGENT_DIR, "CWRU data", "12k Drive End Bearing Fault Data")
NORMAL_DIR = os.path.join(AGENT_DIR, "CWRU data", "Normal Baseline")

WINDOW_SIZE = 1024
STEP = 512  # 50% 重叠


def extract_signal(path):
    """从 .mat 文件中提取振动信号（DE_time）"""
    mat = loadmat(path)
    keys = [k for k in mat.keys() if not k.startswith("__")]
    # 优先选 DE_time
    de_key = next((k for k in keys if "DE" in k.upper()), keys[0])
    return mat[de_key].flatten().astype(np.float32)


def windows_from_signal(signal, window_size, step):
    """将长信号切片为固定长度窗口"""
    windows = []
    for i in range(0, len(signal) - window_size + 1, step):
        windows.append(signal[i : i + window_size])
    return np.array(windows)


def normalize(X):
    """逐样本独立归一化"""
    mean = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True) + 1e-8
    return (X - mean) / std


def run():
    # ===== 定义 CWRU 类（4 故障类型 × 4 负载 = 16 类） =====
    # 使用 12k Drive End 数据，构建 16 类
    CLASS_CONFIG = []

    # 正常（4 种负载）
    loads = [0, 1, 2, 3]
    for hp in loads:
        path = f"{NORMAL_DIR}/normal_{hp}.mat"
        if os.path.exists(path):
            CLASS_CONFIG.append({
                "path": path,
                "class_name": f"Normal_{hp}hp",
            })

    # 内圈故障
    for hp in loads:
        for diam in ["0007", "0014", "0021"]:
            fname = f"IR{diam[1:]}_{hp}.mat"  # 目录 0007 → 文件名 IR007_0.mat
            path = f"{CWRU_BASE}/Inner Race/{diam}/{fname}"
            if os.path.exists(path):
                CLASS_CONFIG.append({
                    "path": path,
                    "class_name": f"IR_{diam}_{hp}hp",
                })

    # 滚动体故障
    for hp in loads:
        for diam in ["0007", "0014", "0021"]:
            fname = f"B{diam[1:]}_{hp}.mat"
            path = f"{CWRU_BASE}/Ball/{diam}/{fname}"
            if os.path.exists(path):
                CLASS_CONFIG.append({
                    "path": path,
                    "class_name": f"Ball_{diam}_{hp}hp",
                })

    # 外圈故障（Centered @6）
    for hp in loads:
        for diam in ["0007", "0014", "0021"]:
            fname = f"OR{diam[1:]}@6_{hp}.mat"
            path = f"{CWRU_BASE}/Outer Race/Centered/{diam}/{fname}"
            if os.path.exists(path):
                CLASS_CONFIG.append({
                    "path": path,
                    "class_name": f"OR_{diam}_{hp}hp",
                })

    print(f"找到 {len(CLASS_CONFIG)} 个类")
    
    # ===== 分配 train/val/test =====
    # 0hp, 1hp → 训练（8 类）
    # 2hp → 验证（4 类）
    # 3hp → 测试（4 类）
    all_classes = list(set(c["class_name"] for c in CLASS_CONFIG))
    train_classes = [c for c in all_classes if "0hp" in c or "1hp" in c]
    val_classes = [c for c in all_classes if "2hp" in c]
    test_classes = [c for c in all_classes if "3hp" in c]

    print(f"训练类: {len(train_classes)} ({[c.split('_')[0] for c in train_classes[:5]]}...)")
    print(f"验证类: {len(val_classes)}")
    print(f"测试类: {len(test_classes)}")

    # 建立类名 → 标签映射
    sorted_classes = sorted(all_classes)
    name_to_label = {name: i for i, name in enumerate(sorted_classes)}

    X_train, y_train = [], []
    X_val, y_val = [], []
    X_test, y_test = [], []

    for cfg in CLASS_CONFIG:
        path = cfg["path"]
        if not os.path.exists(path):
            print(f"  跳过 {cfg['class_name']}: 找不到 {os.path.basename(path)}")
            continue
        
        signal = extract_signal(path)
        windows = windows_from_signal(signal, WINDOW_SIZE, STEP)
        label = name_to_label[cfg["class_name"]]

        # 按负载分配
        if "0hp" in cfg["class_name"] or "1hp" in cfg["class_name"]:
            X_train.append(windows)
            y_train.extend([label] * len(windows))
        elif "2hp" in cfg["class_name"]:
            X_val.append(windows)
            y_val.extend([label] * len(windows))
        elif "3hp" in cfg["class_name"]:
            X_test.append(windows)
            y_test.extend([label] * len(windows))

        print(f"  {cfg['class_name']:<25} {len(windows):4d} 样本")

    # 拼接
    X_train = normalize(np.concatenate(X_train)) if X_train else np.array([])
    X_val = normalize(np.concatenate(X_val)) if X_val else np.array([])
    X_test = normalize(np.concatenate(X_test)) if X_test else np.array([])
    y_train = np.array(y_train, dtype=np.int64)
    y_val = np.array(y_val, dtype=np.int64)
    y_test = np.array(y_test, dtype=np.int64)

    print(f"\n{'='*50}")
    print("CWRU 预处理完成")
    print(f"{'='*50}")
    print(f"  训练: {len(X_train)} 样本, {len(set(y_train.tolist()))} 类")
    print(f"  验证: {len(X_val)} 样本, {len(set(y_val.tolist()))} 类")
    print(f"  测试: {len(X_test)} 样本, {len(set(y_test.tolist()))} 类")

    # 保存
    output_dir = "data_cwru/processed"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "preprocessed.npz")
    np.savez(output_path,
             X_train=X_train, y_train=y_train,
             X_val=X_val, y_val=y_val,
             X_test=X_test, y_test=y_test)
    print(f"\n✅ 已保存到 {output_path}")
    print(f"   验证类与训练类不交叉：" , end="")
    if set(y_train.tolist()) & set(y_val.tolist()):
        print("❌ 有重叠（bug）")
    else:
        print("✅ 无重叠")
    print(f"   测试类与训练类不交叉：" , end="")
    if set(y_train.tolist()) & set(y_test.tolist()):
        print("❌ 有重叠（bug）")
    else:
        print("✅ 无重叠")


if __name__ == "__main__":
    run()
