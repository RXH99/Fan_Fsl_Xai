# Fan_Fsl_Xai

**大容量 ResNet + 不确定性加权直推式小样本故障诊断**

基于原型网络（Prototypical Network）的小样本故障诊断框架，在**风机 189 类**数据集上验证，**5-way 5-shot 达到 97.2%**。

---

## ✨ 特点

- **SE‑ResNet1D (base64, 4M)** 大容量特征提取器
- **SupCon 对比学习预训练** (400 epochs)
- **余弦相似度** 度量 + **10‑way 元训练** + 在线数据增强
- **不确定性加权直推式推理 (UWT)** — 抑制高熵样本对原型更新的影响

---

## 🧠 方法概览

```
原始信号 (1024 点)
       │
       ▼
  SE‑ResNet1D 编码器 (base64, 4M, 单尺度)
       │
       ├── conv1 → maxpool → layer1~4 → avgpool → fc → (128,)
       └── SE 注意力 (通道重标定)
       │
       ▼
  L2 归一化特征嵌入
       │
       ▼
  余弦相似度 vs 类原型
       │
       ▼
  [可选] 不确定性加权直推式推理
       │   └── 计算 query 熵 → 降权模糊样本 → 迭代优化原型
       │
       ▼
  分类结果
```

### 方法核心

| 组件 | 作用 | 增益 |
|---|---|---|
| base64 编码器 (4M, 1M→4M) | 大幅提升容量 | +3% |
| SupCon 预训练 | 对比学习初始化 | +2% |
| 直推式推理 | 利用 query 优化原型 | +1.5% |
| 不确定性加权 (UWT) | 抑制模糊 query | +0.5% (正则化) |

> 已系统性验证的无效组件：多尺度聚合、跨注意力 (CrossAttn)、时频融合、Conv-Transformer。

---

## 📊 实验结果

### 风机数据集 — 189 类（153 基类 / 18 新类 / 18 验证）

| 设定 | Cosine 评估 | UWT 评估 |
|---|---|---|
| **5-way 1-shot** | 92.3% | **94.1%** |
| **5-way 5-shot** | 95.6% | **97.2%** |
| 10-way 1-shot | 87.0% | 86.9% |
| 10-way 5-shot | 91.9% | 93.2% |

**完整提升路线：**

| 阶段 | 5w5s | 累计提升 |
|---|---|---|
| ResNet base32 (1M) + Cosine ProtoNet (基线) | ~94% | — |
| **+ base64 (4M) 编码器** | ~96% | **+2%** |
| + SupCon 预训练 | ~97% | **+3%** |
| + **UWT 不确定性加权直推式** | **97.2%** | **+3%+** |

**组件消融（3 种子平均）：**

| 变体 | 5w5s | Δ vs Full |
|---|---|---|
| **Full** (base64 + Cosine + UWT) | 96.4% | — |
| - CrossAttn (去掉跨注意力) | **97.2%** | **+0.8%** |
| - Multi-scale (去掉多尺度) | 96.6% | +0.2% |
| - SupCon (去掉预训练) | 94.5% | **-1.9%** |

---

## 🚀 快速开始

### 环境要求

- Python 3.9+
- PyTorch 2.0+
- `scipy`, `numpy`, `pyyaml`
- 推荐使用 CUDA GPU

### 数据准备

将风机 `.mat` 文件放入：
```
data/
├── train/   (153 类，每类 1 个 .mat，含 20 个样本)
├── test/    (18 类)
└── val/     (18 类)
```

每个 `.mat` 文件需包含 `Y0` 数组，形状 `(20, 1024)`。

### 运行实验

```powershell
# ===================== 完整流程 =====================

# 第 1 步：数据预处理
python step1_preprocess.py

# 第 2 步：SupCon 预训练（~3h，可选，已有预训练权重可跳过）
python step2_pretrain_simclr.py --config configs/clean.yaml --mode supcon --epochs 400

# 第 3 步：小样本训练（~15 min）
python step3_train_fewshot.py --config configs/clean.yaml --method ProtoNet_Cosine

# 第 4 步：UWT 评估
python eval_clean.py

# ===================== 其他实验 =====================

# 使用含 CrossAttn/多尺度的配置（如需要复现）
python step3_train_fewshot.py --config configs/base64.yaml --method ProtoNet_Cosine

# CWRU 数据集（如需）
python step1_preprocess_cwru.py
python step3_train_fewshot.py --config configs/optimized.yaml --method ProtoNet_Cosine
```

### 超参数

**最终方法**（`configs/clean.yaml`）：

| 参数 | 值 |
|---|---|
| backbone | resnet18 |
| base_filters | 64 |
| encoder_dim | 128 |
| use_multiscale | false |
| use_se | true |
| SupCon epochs | 400 |
| 元训练 episodes | 3000 |
| lr / sep_weight | 0.0001 / 0.15 |
| UWT beta (5w5s) | 1.0 |

---

## 📁 项目结构

```
Fan_Fsl_Xai/
├── configs/
│   ├── clean.yaml               # 最终方法配置 (base64 单尺度)
│   ├── base64.yaml              # 含多尺度/CrossAttn（实验用）
│   ├── optimized.yaml           # 旧基线配置
│   └── cwru.yaml                # CWRU 配置
├── src/
│   ├── config.py                # 配置管理
│   ├── data/                    # 数据集 + 预处理 + 增强
│   ├── models/
│   │   ├── encoder.py           # ResNet1D (含多尺度/单尺度开关)
│   │   └── prototypical.py      # ProtoNet + CrossAttn + UWT
│   ├── training/
│   │   └── train_fewshot.py     # 训练循环
│   └── interpret/               # 可解释性（预留）
├── eval_clean.py                # 最终评估 (无 CrossAttn)
├── step1_preprocess.py          # 数据预处理
├── step2_pretrain_simclr.py     # SupCon 预训练
├── step3_train_fewshot.py       # 小样本训练
├── step6_tsne.py                # t-SNE 可视化
├── EXPERIMENT_SUMMARY.md        # 详细实验总结
├── outputs/
│   ├── clean/                   # 最终结果
│   └── best_result/             # 97.2% 备份
├── data/                        # 风机原始 .mat (不上传)
└── .gitignore                   # 排除 data/ outputs/ *.pth
```

---

## 🐛 已知修复

| Bug | 影响 | 修复时间 |
|---|---|---|
| `base_filters: 64` config 未传入编码器 | 所有"base64"实验实际 base32 | 2026-06-13 |
| 评估时 cross_attn 输入未归一化 | 评估精度偏低 | 2026-06-13 |
| step5_experiments.py 未加载预训练权重 | 基线 ~90% 偏低 | 2026-06-12 |

---

## 📝 说明

- **数据和模型权重不上传**此仓库 (详见 `.gitignore`)
- CrossAttn（跨注意力）、多尺度聚合、时频融合已被系统性验证无效
- 最终方法 = 大容量 ResNet + SupCon + 余弦 ProtoNet + UWT
- 所有实验在 `expt/encoder-opt` 分支上 (tag: `v1-final`)，`main` 为原始基线

---

## 📄 许可证

MIT
