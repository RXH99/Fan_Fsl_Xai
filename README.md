# Fan_Fsl_Xai

**多尺度聚合 + 跨注意力自适应 + 不确定性加权直推式小样本故障诊断**

基于原型网络（Prototypical Network）的小样本故障诊断框架，在**风机 189 类**数据集上验证，**5-way 5-shot 达到 97.0% ± 0.1%**。

---

## ✨ 特点

- **SE‑ResNet1D (base64, 4M)** 特征提取器
- **多尺度特征聚合** — 融合 layer1~4 的 4 层特征图
- **跨注意力任务自适应 (CrossAttn)** — query 参照 support 集动态调整特征
- **SupCon 对比学习预训练** (400 epochs)
- **余弦相似度** 度量 + **10‑way 元训练** + 在线数据增强
- **不确定性加权直推式推理 (UWT)** — 抑制高熵样本对原型更新的影响
- **3 轮实验标准差 < 1%，可重复性高**

---

## 🧠 方法概览

```
原始信号 (1024 点)
       │
       ▼
  SE‑ResNet1D 编码器 (base64, 4M)
       │
       ├── conv1 → maxpool
       ├── layer1 → AdaptivePool ─┐
       ├── layer2 → AdaptivePool ─┤
       ├── layer3 → AdaptivePool ─┤── concat → FC → (128,)
       └── layer4 → AdaptivePool ─┘
       │
       ▼
  跨注意力任务自适应
       │
       ├── query → Q_proj
       ├── support → K_proj, V_proj
       ├── MultiheadCrossAttn(query, support)
       └── LayerNorm + FFN
       │
       ▼
  余弦相似度 vs 类原型
       │
       ▼
  [可选] 不确定性加权直推式推理
       │   计算 query 熵 → 降权模糊样本 → 迭代优化原型
       │
       ▼
  分类结果
```

### 核心创新

| 组件 | 作用 | 增益 |
|---|---|---|
| 多尺度聚合 | 保留低层高分辨率特征 | +0.1~1.1% |
| base64 编码器 (4M) | 增大模型容量 | +1~2% |
| 跨注意力 (CrossAttn) | query 按 task 自适应 | +1~3% (主要) |
| 不确定性加权 (UWT) | 抑制模糊query干扰 | +0.2~0.5% (正则化) |

---

## 📊 最终实验结果

### 风机数据集 — 189 类（153 基类 / 18 新类 / 18 验证）

**最终方法：** base64 + 多尺度聚合 + CrossAttn V1 + UWT
**3 轮均值 ± 标准差（论文可用数据）：**

| 设定 | 本文方法 (3 runs) |
|---|---|
| **5-way 1-shot** | **93.7% ± 0.4%** |
| **5-way 5-shot** | **97.0% ± 0.1%** |
| 10-way 1-shot | 87.6% ± 0.7% |
| 10-way 5-shot | 93.8% ± 0.3% |

**完整进化路线（5-way 5-shot）：**

| 阶段 | 精度 | 累计提升 |
|---|---|---|
| 原始 ResNet (base32, 单尺度) | ~94% | — |
| + 余弦度量 + 10-way + 增强 | ~95% | +1% |
| + 不确定性加权直推式 (UWT) | ~96% | +2% |
| **+ 多尺度聚合 + base64 + CrossAttn V1** | **97.0%** | **+3%** |

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
python step2_pretrain_simclr.py --config configs/base64.yaml --mode supcon --epochs 400

# 第 3 步：小样本训练 + 跨注意力（~15 min）
python step3_train_fewshot.py --config configs/base64.yaml --method ProtoNet_CrossAttn

# 第 4 步：UWT 评估
python eval_final.py --v 1

# 第 5 步（可选）：3 轮取平均（~45 min）
python run_final.py

# ===================== 基线实验 =====================

# 标准 ProtoNet（无跨注意力）
python step3_train_fewshot.py --config configs/base64.yaml --method ProtoNet_Cosine

# CWRU 数据集（如需）
python step1_preprocess_cwru.py
python step3_train_fewshot.py --config configs/optimized.yaml --method ProtoNet_Cosine
```

### 超参数

**最终方法**（`configs/base64.yaml`）：

| 参数 | 值 |
|---|---|
| backbone | resnet18 |
| base_filters | 64 |
| encoder_dim | 128 |
| 多尺度聚合 | layer1~4 concat |
| use_se | true |
| SupCon epochs | 400 |
| 元训练 ways / shot / query | 10 / 5 / 5 |
| 元训练 episodes | 2000 |
| lr / sep_weight | 0.0001 / 0.15 |
| CrossAttn V1 | d_model=128, dropout=0.1 |
| UWT steps / tau / mix_ratio | 3 / 0.3 / 0.8 |
| UWT beta (5w5s) | 1.0 |

---

## 📁 项目结构

```
Fan_Fsl_Xai/
├── configs/
│   ├── base64.yaml             # 最终方法配置 (base64 多尺度 + CrossAttn)
│   ├── optimized.yaml          # 旧基线配置 (base32 单尺度)
│   ├── baseline.yaml           # 原始基线配置
│   └── cwru.yaml               # CWRU 实验配置
├── src/
│   ├── config.py               # 配置管理 (YAML 加载)
│   ├── data/
│   │   ├── dataset.py          # FaultDataset + EpisodicSampler
│   │   ├── preprocess.py       # 风机 189 类数据预处理
│   │   └── augmentation.py     # 数据增强 (噪声/掩码/缩放)
│   ├── models/
│   │   ├── encoder.py          # ResNet1D + 多尺度聚合 + STFT 频域分支
│   │   └── prototypical.py     # ProtoNet 损失 + CrossAttn V1/V2 + UWT
│   ├── training/
│   │   └── train_fewshot.py    # 训练循环 (增广/早停/支持跨注意力)
│   └── interpret/              # 可解释性模块 (预留)
├── eval_final.py               # 最终评估 (CrossAttn + UWT, 支持 V1/V2)
├── run_final.py                # 3 轮取平均实验脚本
├── run_ablation.py             # 消融实验脚本
├── step1_preprocess.py         # 风机数据预处理入口
├── step1_preprocess_cwru.py    # CWRU 数据预处理入口
├── step2_pretrain_simclr.py    # SupCon/SimCLR 对比学习预训练
├── step2_train_cnn.py          # (已弃用) 旧 CNN 预训练
├── step3_train_fewshot.py      # 小样本训练 (支持所有方法)
├── step5_experiments.py        # 批量对比实验
├── step6_tsne.py               # t-SNE 特征可视化
├── EXPERIMENT_SUMMARY.md       # 详细实验总结
├── outputs/
│   ├── base64/                 # 最终结果 (权重 + 3 轮汇总)
│   └── best_result/            # 97.2% 单次最优备份
├── data/                       # 风机原始 .mat (不上传)
├── data_cwru/                  # CWRU 数据 (不上传)
└── .gitignore                  # 排除 data/ outputs/ *.pth
```

### 每个文件的作用

| 文件 | 作用 |
|---|---|
| `src/models/encoder.py` | 编码器：CNN / ResNet1D (含多尺度聚合) / STFT频域分支 |
| `src/models/prototypical.py` | 原型网络 + CrossAttnV1/V2 + 不确定性加权直推式 |
| `src/training/train_fewshot.py` | 训练循环，支持跨注意力模块联合训练 |
| `src/data/dataset.py` | FaultDataset + 标准/半监督 EpisodicSampler |
| `src/data/augmentation.py` | 振动信号数据增强 |
| `src/data/preprocess.py` | 189 类风机数据预处理 (mat → npz) |
| `configs/base64.yaml` | **最终方法配置** (base64 + 多尺度 + CrossAttn) |
| `eval_final.py` | CrossAttn + UWT 联合评估，支持 V1/V2 切换 |
| `run_final.py` | 自动 3 轮训练 + 评估，输出均值±标准差 |
| `run_ablation.py` | 消融实验脚本 (7 个变体，需更新到新架构) |
| `step3_train_fewshot.py` | 小样本训练入口，支持 6 种方法 |
| `step2_pretrain_simclr.py` | SupCon/SimCLR 对比学习预训练 |
| `step5_experiments.py` | 批量对比实验入口 |
| `step6_tsne.py` | t-SNE 特征可视化 |

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
- 跨注意力与 UWT 存在**协同效应**：跨注意力拉大置信度差距，使 UWT 的熵加权更有效
- 所有实验在 `expt/encoder-opt` 分支上 (tag: `v1-final`)，`main` 为原始基线

---

## 📄 许可证

MIT
