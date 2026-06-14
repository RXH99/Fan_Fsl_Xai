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
       │   └── 计算 query 熵 → 降权模糊样本 → 迭代优化原型
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

### 多尺度 vs 单尺度专项验证

| Setting | 多尺度 | 单尺度 | 差值 |
|---|---|---|---|
| 5w5s | 96.7% | **96.8%** | -0.1% |
| **平均** | | | **-0.3%** |

---

## Quick Start

### 环境
- Python 3.9+ / PyTorch 2.0+
- `scipy`, `numpy`, `pyyaml`

### 数据
风机 `.mat` 文件放 `data/` 下：
```
data/
├── train/   (153 类, 每类 1 个 .mat, 含 20 个样本)
├── test/    (18 类)
└── val/     (18 类)
```

### 完整流程

```powershell
# 1. 数据预处理
python step1_preprocess.py

# 2. SupCon 预训练（已有预训练可跳过）
python step2_pretrain_simclr.py --config configs/base64.yaml --mode supcon --epochs 400

# 3. 小样本训练
python step3_train_fewshot.py --config configs/clean.yaml --method ProtoNet_Cosine

# 4. UWT 评估
python eval_clean.py

# 5. t-SNE 可视化
python step6_tsne.py
```

### 对比实验

```powershell
# RelationNet（编码器冻结，5min）
python eval_relationnet.py

# MAML 训练 + 评估（3-5h）
python train_maml.py --config configs/clean.yaml --meta_lr 0.0005 --inner_steps 10
python eval_maml.py

# 多种子消融
python run_seeded_ablation.py --seeds 42 123 999
```

---

## 项目结构

```
Fan_Fsl_Xai/
├── configs/
│   ├── clean.yaml               # 最终方法配置 (单尺度 base64)
│   ├── base64.yaml              # 实验用 (含多尺度/CrossAttn)
│   ├── optimized.yaml           # 旧基线配置 (base32)
│   ├── cwru.yaml                # CWRU 配置
│   ├── compare_relnet.yaml      # RelationNet 对比
│   └── compare_maml.yaml        # MAML 对比
├── src/
│   ├── models/
│   │   ├── encoder.py           # ResNet1D (SE + 多尺度/单尺度)
│   │   ├── prototypical.py      # ProtoNet + CrossAttn + UWT
│   │   ├── relationnet.py       # RelationNet 模块 (MLP)
│   │   └── maml.py              # MAML (纯 PyTorch, 无外部依赖)
│   ├── data/
│   │   ├── dataset.py           # FaultDataset + EpisodicSampler
│   │   └── augmentation.py      # 振动信号数据增强
│   ├── training/
│   │   └── train_fewshot.py     # 训练循环
│   └── interpret/               # 可解释性 (预留)
├── eval_clean.py                # ✅ 最终评估 (Clean + UWT)
├── eval_relationnet.py          # RelationNet 对比
├── eval_maml.py                 # MAML 评估
├── train_maml.py                # MAML 训练
├── eval_multiscale_check.py     # 多尺度 vs 单尺度验证
├── run_seeded_ablation.py       # ✅ 推荐消融脚本 (3 seeds)
├── step1_preprocess.py          # 数据预处理
├── step2_pretrain_simclr.py     # SupCon 预训练
├── step3_train_fewshot.py       # 小样本训练
├── step6_tsne.py                # t-SNE 可视化
├── outputs/
│   ├── clean/                   # 最终模型权重 + 3seeds_summary
│   ├── base64/                  # 预训练权重 + 消融结果
│   └── relationnet/            # RelationNet 权重
├── EXPERIMENT_SUMMARY.md        # 详细实验日志
└── README.md
```

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
| ways / shot / query | 10 / 5 / 5 |
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
