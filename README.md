# Fan_Fsl_Xai

**大容量 ResNet + 不确定性加权直推式小样本故障诊断**

基于原型网络的小样本故障诊断框架，在**风机 189 类**数据集上达到 **97.1% ± 0.4%**（5-way 5-shot, 3 seeds × 1000 episodes）。

---

## Method

```
原始信号 (1024 点)
 │
 ▼
 SE-ResNet1D Encoder (base64, 4M, 单尺度)
 │
 ├── conv1 → maxpool → layer1~4 → avgpool → fc → (128,)
 ├── SE attention (通道重标定)
 └── SupCon 对比学习预训练 (400 epochs)
 │
 ▼
 L2 归一化特征嵌入
 │
 ▼
 余弦相似度 vs 类原型
 │
 ▼
 UWT (不确定性加权直推式推理)
 │ └── 计算 query 熵 → 降权模糊样本 → 迭代优化原型
 │
 ▼
 分类结果
```

### 关键发现

| 组件 | 增益 | 结论 |
|---|---|---|
| base64 编码器 (4M, base32→base64) | +3% | 容量提升是最大增益 |
| SupCon 对比学习预训练 | +2% | 不可替代 |
| UWT 不确定性加权直推式 | +1.6~2.2% | 核心创新点，无需额外参数 |
| CrossAttn / 多尺度 / 时频融合 | ≤±0.5% | 已系统性排除 |

---

## Results

### 风机 189 类 — Clean 模型（3 seeds 均值）

| Setting | Cosine | UWT | Δ |
|---|---|---|---|
| **5-way 1-shot** | 92.3% | 93.3% | +1.0% |
| **5-way 5-shot** | 94.9% | **97.1% ± 0.4%** | **+2.2%** |
| 10-way 1-shot | 85.4% | 87.3% | +1.9% |
| 10-way 5-shot | 90.5% | 94.2% | +3.7% |

### 方法对比（5-way 5-shot）

| 方法 | 预训练 | 5w5s |
|---|---|---|
| ProtoNet (Cosine) | ✓ SupCon | 94.9% |
| + **UWT** | ✓ SupCon | **97.1% ± 0.4%** |
| RelationNet | ✓ SupCon | 96.0% |
| MAML | — | < 20%* |
| MAML | ✓ SupCon | < 27%* |
> *MAML 在 189 类 × 10-way 下无法有效收敛

### 多种子消融（3 seeds, 5w5s）

| 变体 | 精度 | Δ vs Full |
|---|---|---|
| **Full** (CrossAttn+多尺度) | 97.0% | — |
| - CrossAttn | 96.9% | -0.1% (噪音) |
| - Multi-scale | 96.7% | -0.3% (噪音) |
| - SupCon | 94.7% | **-2.3%** |

### SupCon × UWT 交互效应（5w5s）

```
                   Cosine      UWT      Δ UWT
    ┌────────────┬────────┬────────┬─────────┐
    │ 无 SupCon  │ 93.6%  │ 94.0%  │  +0.4%  │
    ├────────────┼────────┼────────┼─────────┤
    │ 有 SupCon  │ 94.9%  │ 97.1%  │  +2.2%  │
    ├────────────┼────────┼────────┼─────────┤
    │ Δ SupCon   │ +1.3%  │ +3.1%  │         │
    └────────────┴────────┴────────┴─────────┘

交互效应 = +1.8%（强协同：SupCon 和 UWT 各自单独增益 < 2%，结合增益 > 3.5%）
```

- SupCon 单独（无 UWT）：+1.3%（93.6% → 94.9%）
- UWT 单独（无 SupCon）：+0.4%（93.6% → 94.0%）
- 两者结合：+3.5%（93.6% → 97.1%）
- 交互效应：+1.8%（协同放大）

**结论：SupCon 为 UWT 提供了结构良好的特征空间，是 UWT 直推修正发挥效果的前提。**

### 多尺度 vs 单尺度专项验证

| Setting | 多尺度 | 单尺度 | 差值 |
|---|---|---|---|
| 5w5s | 96.7% | **96.8%** | -0.1% |
| **平均** | | | **-0.3%** |

---

## Quick Start

### 环境
```
Python 3.9+
PyTorch 2.0+
scipy, numpy, pyyaml, matplotlib, scikit-learn
```

### 数据准备

风机 `.mat` 文件按以下目录结构放置：
```
data/
├── train/   (153 类, 每类 1 个 .mat, 每文件 20 个 1024 点样本)
├── test/    (18 类)
└── val/     (18 类)
```

### 完整工作流

**第 1 步：数据预处理**
```powershell
python step1_preprocess.py
```
输出: `data/processed/preprocessed.npz`

**第 2 步：SupCon 对比学习预训练（已有预训练可跳过）**
```powershell
python step2_pretrain_simclr.py --config configs/base64.yaml --mode supcon --epochs 400
```
输出: `outputs/base64/pretrained_resnet18_encoder.pth`

**第 3 步：小样本元训练**
```powershell
# 最终方法：Clean 模型（单尺度 + 无 CrossAttn）
python step3_train_fewshot.py --config configs/clean.yaml --method ProtoNet_Cosine
```
输出: `outputs/clean/fewshot_encoder_ProtoNet_Cosine.pth`

**第 4 步：UWT 评估（核心结果）**
```powershell
python eval_clean.py
```
输出: 打印 5w1s / 5w5s / 10w1s / 10w5s 的 Cosine→UWT 精度

**第 5 步：t-SNE 特征可视化**
```powershell
python step6_tsne.py
```
输出: `outputs/clean/tsne/tsne_train.png`, `tsne_test.png`, `tsne_comparison.png`

---

### 对比实验

**RelationNet 对比**
```powershell
python eval_relationnet.py
```
冻结 SupCon 编码器，训练 Relation Module，评估含/不含 UWT

**MAML 对比**
```powershell
# 训练（3-5 小时）
python train_maml.py --config configs/clean.yaml --meta_lr 0.001 --inner_steps 5
# 评估
python eval_maml.py
```

**多种子消融**
```powershell
python run_seeded_ablation.py --seeds 42 123 999
```

**SupCon × UWT 交互效应验证**
```powershell
# 1. 训练无 SupCon 的 Clean 模型
python step3_train_fewshot.py --config configs/clean_nopretrain.yaml --method ProtoNet_Cosine --no_pretrain
# 2. 评估 2×2 对比矩阵
python eval_supcon_2x2.py
```

**多尺度 vs 单尺度验证**
```powershell
python eval_multiscale_check.py
```

---

## 项目结构

```
Fan_Fsl_Xai/
├── configs/
│   ├── clean.yaml               # 🔴 最终方法配置 (单尺度 base64)
│   ├── clean_nopretrain.yaml    #   无 SupCon 对照实验配置
│   ├── base64.yaml              #   旧实验配置 (含多尺度/CrossAttn)
│   ├── optimized.yaml           #   旧基线配置 (base32)
│   ├── baseline.yaml            #   更旧基线 (CNN 编码器)
│   ├── cwru.yaml                #   CWRU 数据集配置
│   ├── compare_relnet.yaml      #   RelationNet 对比
│   └── compare_maml.yaml        #   MAML 对比
├── src/
│   ├── models/
│   │   ├── encoder.py           #   ResNet1D (SE + 多尺度/单尺度)
│   │   ├── prototypical.py      #   ProtoNet + CrossAttn + UWT
│   │   ├── relationnet.py       #   RelationNet 模块
│   │   └── maml.py              #   MAML (纯 PyTorch)
│   ├── data/
│   │   ├── dataset.py           #   FaultDataset + EpisodicSampler
│   │   ├── augmentation.py      #   振动信号增强 (噪声/掩码/缩放)
│   │   └── preprocess.py        #   .mat 读取 + 类别构建逻辑
│   ├── training/
│   │   └── train_fewshot.py     #   小样本训练循环
│   └── interpret/               #   可解释性 (预留)
├── step1_preprocess.py          # 🔴 数据预处理
├── step2_pretrain_simclr.py     #   SupCon 对比学习预训练
├── step3_train_fewshot.py       # 🔴 小样本元训练
├── step6_tsne.py                #   特征可视化
├── step1_preprocess_cwru.py     #   CWRU 预处理 (补充数据)
├── eval_clean.py                # 🔴 Clean + UWT 最终评估
├── eval_supcon_2x2.py           #   SupCon × UWT 交互效应
├── eval_relationnet.py          #   RelationNet 对比
├── eval_maml.py                 #   MAML 评估
├── train_maml.py                #   MAML 训练
├── eval_multiscale_check.py     #   多尺度 vs 单尺度验证
├── run_seeded_ablation.py       # 🔴 推荐消融脚本 (3 seeds)
├── outputs/
│   ├── clean/                   # 🔴 最终模型 + 3 种子结果
│   │   ├── fewshot_encoder_ProtoNet_Cosine.pth
│   │   ├── fewshot_encoder_seed{42,123,999}.pth
│   │   ├── 3seeds_summary.txt
│   │   └── uwt_ablation_summary.txt
│   ├── clean_nopretrain/        #   无 SupCon 对照实验
│   │   ├── fewshot_encoder_ProtoNet_Cosine.pth
│   │   └── 2x2_comparison.txt
│   ├── base64/                  #   SupCon 预训练权重 + 消融
│   │   ├── pretrained_resnet18_encoder.pth  # ← 预训练权重
│   │   └── ablation_v2/                     # 消融结果
│   └── relationnet/             #   RelationNet 权重
├── data/                        #   风机原始 .mat 文件
├── data_cwru/                   #   CWRU 数据 (补充)
├── EXPERIMENT_SUMMARY.md        # 🔴 详细实验日志
└── README.md
```
> 🔴 = 核心文件/流程

---

### 超参数（最终方法 `clean.yaml`）

| 参数 | 值 |
|---|---|
| backbone | resnet18 |
| base_filters | 64 (4M params) |
| encoder_dim | 128 |
| use_se | true |
| use_multiscale | false |
| CrossAttn | 不使用 |
| SupCon epochs | 400 |
| 元训练 episodes | 3000 |
| 元训练 ways / shot / query | 10 / 5 / 5 |
| lr / sep_weight | 0.0001 / 0.15 |
| 数据增强 | noise=0.02, mask=0.15, scale=0.03 |
| UWT steps / tau / mix_ratio | 3 / 0.3 / 0.8 |
| UWT beta (5w5s) | 1.0 |

---

## 论文数据清单

| 数据项 | 值 | 来源 |
|---|---|---|
| 最终 5w5s | **97.1% ± 0.4%** | 3 seeds × 1000 episodes |
| UWT 贡献 | +2.2% (Cosine 94.9% → UWT 97.1%) | 3 seeds 均值 |
| SupCon 贡献 | +2.3% (去掉降 2.3%) | 3 seeds 消融 |
| SupCon × UWT 交互效应 | **+1.8%（协同）** | 2×2 对比矩阵 |
| 编码器容量贡献 | +3% (base32[1M] → base64[4M]) | 对照实验 |
| RelationNet 对比 | 96.0% (vs ProtoNet+UWT 97.1%) | 同编码器冻结 |
| MAML 对比 | <27% (无法收敛) | 10-way 从零/预训练 |
| 多尺度 vs 单尺度 | 差值 < 0.5% (无影响) | 500 episodes 验证 |

---

## 已修复的 Bug

| Bug | 影响 | 修复日期 |
|---|---|---|
| `use_multiscale` 未从 config 读取 | 所有模型实际多尺度训练 (但无影响) | 2026-06-14 |
| `base_filters` 未传入编码器 | 部分实验实际使用 base32 而非 base64 | 2026-06-13 |
| 评估时 cross_attn 未归一化 | CrossAttn 评估精度偏低 | 2026-06-13 |
| v1 消融脚本路径不匹配 | SupCon 预训练权重未加载，对比无效 | 2026-06-14 |

---

## License

MIT
