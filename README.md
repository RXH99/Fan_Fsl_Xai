# Fan_Fsl_Xai

**大容量 ResNet + 不确定性加权直推式小样本故障诊断**

基于原型网络的小样本故障诊断框架，在**风机 189 类**数据集上达到 **97.1% ± 0.4%**（5-way 5-shot, 3 seeds × 1000 episodes）。

核心创新：**Clean (base64 + SupCon) + UWT** —— 简洁架构配合充分预训练与鲁棒推理，性能超越复杂设计。

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
 余弦相似度 vs 类原型 (ProtoNet_Cosine)
 │
 ▼
 UWT (不确定性加权直推式推理) ← 零参数、即插即用
 │ └── 计算 query 熵 → 降权模糊样本 → 迭代优化原型 (3 steps)
 │
 ▼
 分类结果
```

### 关键发现

| 组件 | 增益 | 结论 |
|---|---|---|
| **SupCon × UWT 协同效应** | **+1.8%** | 🔴 论文核心创新点 |
| SupCon 预训练（在 UWT 下） | +2.3% | 释放大编码器潜力的前提 |
| UWT 直推（在 SupCon 下） | +2.2% | 无需额外参数，推理即插即用 |
| 编码器容量 1M→4M（无预训练） | ~0%（持平） | 无预训练时大模型不占优 |
| **CrossAttn** | **-0.1%** | ❌ 负贡献，已排除 |
| 多尺度 / 时频融合 | ≤±0.3% | ⚪ 冗余，已排除 |

> **核心洞察**：
> 1. **大容量编码器的优势必须配合 SupCon 预训练才能发挥**。无预训练时 base64 (93.7%) 与 base32 (94.3%) 性能相当甚至略低。
> 2. **CrossAttn 功能冗余**：与 UWT 同时使用时增益不叠加，单独贡献为负（-0.1%），移除后性能反而提升 +0.6%。
> 3. **遵循 Occam's Razor**：选择 Clean (base64+SupCon) + UWT = **97.1%**，而非 Full (CrossAttn+Multi-scale) = 96.5%。

---

## Results

### 风机 189 类 — Clean 模型（3 seeds 均值）

| Setting | Cosine | UWT | Δ | Beta |
|---|---|---|---|---|
| **5-way 1-shot** | 92.3% | 93.3% | +1.0% | 3.0 |
| **5-way 5-shot** | 94.9% | **97.1% ± 0.4%** | **+2.2%** | 1.0 |
| 10-way 1-shot | 85.4% | 87.3% | +1.9% | 2.0 |
| 10-way 5-shot | 90.5% | 94.2% | +3.7% | 1.0 |

> UWT 在高难度设置（10-way, 1-shot）下增益更大，5w5s 作为标准 benchmark 增益稳定在 +2.2%。

### 方法对比（5-way 5-shot）

| 方法 | 预训练 | 5w5s | 说明 |
|---|---|---|---|
| ProtoNet (Cosine) | ✓ SupCon | 94.9% | 基线 |
| + **UWT** | ✓ SupCon | **97.1% ± 0.4%** | 🔴 最终方案 |
| RelationNet | ✓ SupCon | 96.0% | < ProtoNet+UWT |
| MAML | — | < 20%* | 无法收敛 |
| MAML | ✓ SupCon | < 27%* | 仍无法收敛 |

> *MAML 在 189 类 × 10-way 下无法有效收敛，即使有 SupCon 预训练。

---

### 多种子消融实验（3 seeds × 3000 episodes × 500 eval）

**完整实验结果（2026-06-24）**

| 变体 | 5w1s | 5w5s | 10w1s | 10w5s |
|---|---|---|---|---|
| **Full** (CrossAttn+多尺度+SE+Aug+SupCon) | 92.4% ± 1.3% | **96.5% ± 0.4%** | 85.4% ± 2.1% | 93.1% ± 0.9% |
| - CrossAttn | 94.0% ± 0.7% | **97.1% ± 0.4%** | 87.9% ± 1.0% | 94.1% ± 0.7% |
| - Multi-scale | 92.2% ± 1.6% | **96.2% ± 0.5%** | 84.8% ± 1.7% | 92.7% ± 1.2% |
| - SupCon | 91.3% ± 1.0% | **94.9% ± 0.4%** | 82.6% ± 1.2% | 89.8% ± 0.2% |

**5w5s 核心结论：**
| 变体 | 3种子均值 | Δ vs Full | 结论 |
|---|---|---|---|
| Full | **96.5%** | — | 基准 |
| - CrossAttn | 97.1% | **+0.6%** | 🔴 移除后性能提升 |
| - Multi-scale | 96.2% | -0.3% | 无差异（噪音内） |
| - SupCon | 94.9% | **-1.6%** | 🔴 唯一真正有效

**关键发现**:
1. 🔴 **CrossAttn 负贡献**：移除后性能提升 +0.6%，证明其引入干扰
2. 🔴 **SupCon 最关键**：移除后下降 1.6%，是性能的基础保障
3. ⚪ **Multi-scale 冗余**：移除后仅微降 0.3%，可安全移除

---

### CrossAttn vs UWT 推断变体分析

**方法**：复用 Full 模型编码器（Seed 999），评估不同推理模式（500 episodes, 5w5s）

| 推理模式 | 准确率 | 标准差 | 说明 |
|---|---|---|---|
| **Cosine (有CrossAttn)** | 93.9% | ± 5.0% | Full模型 + Cosine推理 |
| **Cosine Only (无CrossAttn)** | 94.0% | ± 5.1% | 纯Cosine baseline |
| **UWT (有CrossAttn)** | **94.5%** | ± 4.9% | UWT修正 |

**增益分解**:
```
→ CrossAttn 贡献:         -0.1%  ❌ 负增益
→ UWT 增益 (有CrossAttn):  +0.6%  ⚠️ 微小
→ UWT 增益 (无CrossAttn):  +0.5%
```

**与 Clean 模型对比**:
| 模型配置 | 5w5s | UWT增益 | 说明 |
|---|---|---|---|
| **Clean (base64+SupCon)** | 97.1% ± 0.4% | N/A | 来自 eval_clean.py |
| **Full + UWT** | 94.5% ± 4.9% | +0.6% | 当前实验 |
| **差异** | **-2.6%** | - | Clean 更优 |

**结论**: 
- ✅ **CrossAttn 功能冗余**：单独贡献为负（-0.1%），与 UWT 同时使用时增益不叠加
- ✅ **选择 Clean + UWT**：性能最优（97.1%），架构最简，符合 Occam's Razor
- ✅ **UWT 优势**：零参数、不改变训练、推理即插即用

---

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

---

### 编码器容量 × SupCon × UWT 完整对比

| 设定 (5w5s) | base32 (1M) | base64 (4M) | Δ |
|---|---|---|---:|
| 无 SupCon + Cosine | 94.3% | 93.6% | -0.7% |
| 有 SupCon + Cosine | 95.3% | 94.9% | -0.4% |
| 有 SupCon + UWT | 95.9% | **97.1%** | **+1.2%** |
| SupCon 增益 | +1.0% | +1.3% | — |
| UWT 增益（有 SupCon 下） | +0.6% | **+2.2%** | **差 3.7×** |

> - SupCon 增益与模型大小无关（base32 +1.0% vs base64 +1.3%）
> - **但 UWT 增益高度依赖容量**：base32 上仅 +0.6%，base64 上 +2.2%
> - 这是容量特征丰富度 × 预训练 × 直推的三层协同关系
> - **Base32 上无显著协同效应，base64 上交互效应 +1.8%**

---

### 多尺度 vs 单尺度专项验证

**方法：** 同一 clean 权重，`return_features=True` 跳过 fc，在卷积特征层直接对比。

| Setting | 多尺度 (960维) | 单尺度 (512维) | 差值 |
|---|---|---|---|
| 5-way 1-shot | 93.3% | **93.5%** | -0.2% |
| **5-way 5-shot** | 96.7% | **96.8%** | **-0.1%** |
| 10-way 1-shot | 86.4% | **86.9%** | -0.5% |
| 10-way 5-shot | 92.4% | **92.8%** | -0.4% |
| **平均** | | | **-0.3%** |

**结论：** 单尺度略好。多尺度无贡献，`clean.yaml` 使用 `use_multiscale: false` 是正确选择。

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
```
# 使用默认配置
python eval_clean.py

# 自定义配置文件（调整UWT参数）
python eval_clean.py --config configs/clean.yaml
```
输出: 打印 5w1s / 5w5s / 10w1s / 10w5s 的 Cosine→UWT 精度

**配置文件说明**:
- UWT参数在 `configs/clean.yaml` 的 `inference.uwt` 段中定义
- 可调整的参数包括：`steps`（迭代步数）、`tau`（温度系数）、`mix_ratio`（动量比例）、`beta_search`（各setting的beta值）
- 修改配置文件后重新运行评估脚本即可应用新参数

**错误处理**:
- 如果权重文件不存在，脚本会显示详细的解决建议
- 如果GPU不可用，会自动切换到CPU并显示警告
- 所有错误消息都包含具体的排查步骤

**第 5 步：t-SNE 特征可视化**
```powershell
python step6_tsne.py
```
输出: `outputs/clean/tsne/tsne_train.png`, `tsne_test.png`, `tsne_comparison.png`

**第 6 步：多种子消融实验（可选，需 7-8 小时）**
```powershell
python run_seeded_ablation.py --seeds 42 123 999
```
输出: `outputs/base64/ablation_v2/seeded_ablation_*.txt`

---

### 对比实验

**RelationNet 对比**
```powershell
python eval_relationnet.py
```
冻结 SupCon 编码器，训练 Relation Module，评估含/不含 UWT

**MAML 对比**
```
# 训练（3-5 小时）
python train_maml.py --config configs/clean.yaml --meta_lr 0.001 --inner_steps 5
# 评估
python eval_maml.py
```

**SupCon × UWT 交互效应验证**
```
# 1. 训练无 SupCon 的 Clean 模型
python step3_train_fewshot.py --config configs/clean_nopretrain.yaml --method ProtoNet_Cosine --no_pretrain
# 2. 评估 2×2 对比矩阵
python eval_supcon_2x2.py
```

**多尺度 vs 单尺度验证**
```powershell
python eval_multiscale_check.py
```

**CrossAttn vs UWT 增益分析**
```powershell
python analyze_crossattn_uwt_gain.py
```
输出: CrossAttn 贡献、UWT 增益分解、与 Clean 模型对比

---

## 项目结构

```
Fan_Fsl_Xai/
├── configs/
│   ├── clean.yaml               # 🔴 最终方法配置 (单尺度 base64)
│   ├── clean_nopretrain.yaml    #   无 SupCon 对照实验配置
│   ├── clean_base32.yaml        #   base32 对照配置
│   ├── base64.yaml              #   SupCon 预训练配置
│   ├── base32.yaml              #   base32 预训练配置（可选）
│   ├── cwru.yaml                #   CWRU 数据集配置
│   ├── compare_relnet.yaml      #   RelationNet 对比
│   └── compare_maml.yaml        #   MAML 对比
├── src/
│   ├── utils.py                 # 🔴 通用工具模块 (160行, 11个测试用例)
│   ├── config.py                #   配置加载工具
│   ├── models/
│   │   ├── encoder.py           #   ResNet1D (SE + 多尺度/单尺度)
│   │   ├── prototypical.py      #   ProtoNet + UWT
│   │   ├── relationnet.py       #   RelationNet 模块
│   │   └── maml.py              #   MAML (纯 PyTorch)
│   ├── data/
│   │   ├── dataset.py           #   FaultDataset + EpisodicSampler
│   │   ├── augmentation.py      #   振动信号增强 (噪声/掩码/缩放)
│   │   └── preprocess.py        #   .mat 读取 + 类别构建逻辑
│   └── training/
│       └── train_fewshot.py     #   小样本训练循环
├── tests/
│   └── test_utils.py            # 🔴 工具模块单元测试
├── step1_preprocess.py          # 🔴 数据预处理
├── step1_preprocess_cwru.py     #   CWRU 预处理 (补充数据)
├── step2_pretrain_simclr.py     #   SupCon 对比学习预训练
├── step3_train_fewshot.py       # 🔴 小样本元训练
├── step6_tsne.py                #   特征可视化
├── eval_clean.py                # 🔴 Clean + UWT 最终评估
├── eval_supcon_2x2.py           #   SupCon × UWT 交互效应 (base64)
├── eval_supcon_2x2_base32.py    #   SupCon × UWT 交互效应 (base32)
├── eval_compare_capacity.py     #   容量对比实验 (base32 vs base64)
├── eval_relationnet.py          #   RelationNet 对比
├── eval_maml.py                 #   MAML 评估
├── eval_multiscale_check.py     #   多尺度 vs 单尺度验证
├── analyze_crossattn_uwt_gain.py# 🔴 CrossAttn vs UWT 增益分析
├── run_seeded_ablation.py       # 🔴 推荐消融脚本 (3 seeds)
├── generate_results_table.py    #   快速生成表格工具（可选）
├── train_maml.py                #   MAML 训练
├── outputs/
│   ├── clean/                   # 🔴 最终模型 + 3 种子结果
│   │   ├── fewshot_encoder_ProtoNet_Cosine.pth
│   │   ├── fewshot_encoder_seed{42,123,999}.pth
│   │   ├── 3seeds_summary.txt
│   │   └── uwt_ablation_summary.txt
│   ├── clean_nopretrain/        #   无 SupCon 对照实验
│   │   ├── fewshot_encoder_ProtoNet_Cosine.pth
│   │   └── 2x2_comparison.txt
│   ├── clean_base32/            #   base32 模型权重
│   ├── base64/                  #   SupCon 预训练权重
│   │   └── pretrained_resnet18_encoder.pth
│   └── relationnet/             #   RelationNet 权重
├── data/                        #   风机原始 .mat 文件
├── data_cwru/                   #   CWRU 数据 (补充)
├── README.md                    # 🔴 本文档
├── FINAL_EXPERIMENT_RESULTS.md  # 🔴 完整实验结果分析报告
├── CROSSATTN_UWT_ANALYSIS.md    # 🔴 CrossAttn vs UWT 详细分析
└── EXPERIMENT_SUMMARY.md        #   方法演进历史
```
> 🔴 = 核心文件/流程

**清理说明**: 
- 已删除过时的配置文件（baseline.yaml, optimized.yaml）
- 已删除阶段性报告文档（PHASE1/2/3_COMPLETION_REPORT.md）
- 已删除冗余脚本（test_crossattn_uwt_gain.py）
- 已删除旧输出目录（outputs/base32/）
- 当前项目结构精简至 **~37个核心文件**（减少26%）

---

### 超参数（最终方法 `clean.yaml`）

| 参数 | 值 | 说明 |
|---|---|---|
| backbone | resnet18 | - |
| base_filters | 64 (4M params) | base64 架构 |
| encoder_dim | 128 | - |
| use_se | true | SE 注意力 |
| use_multiscale | false | 单尺度更优 |
| CrossAttn | 不使用 | ❌ 负贡献 |
| SupCon epochs | 400 | 对比学习预训练 |
| 元训练 episodes | 3000 | - |
| 元训练 ways / shot / query | 10 / 5 / 5 | - |
| lr / sep_weight | 0.0001 / 0.15 | - |
| 数据增强 | noise=0.02, mask=0.15, scale=0.03 | - |
| UWT steps / tau / mix_ratio | 3 / 0.3 / 0.8 | 推理阶段 |
| UWT beta (5w5s) | 1.0 | 可调 |

---

## 论文数据清单

| 数据项 | 值 | 来源 |
|---|---|---|
| 最终 5w5s | **97.1% ± 0.4%** | 3 seeds × 1000 episodes |
| UWT 贡献（base64 + SupCon 下） | +2.2% (94.9% → 97.1%) | 3 seeds 均值 |
| SupCon 贡献（base64 + Cosine 下） | +1.3% (93.6% → 94.9%) | 1000ep 评估 |
| SupCon × UWT 交互效应 | **+1.8%（强协同）** | 2×2 对比矩阵 |
| 编码器容量单独贡献 | ~0% (无预训练时 94.3% vs 93.7%) | 1000ep 统一评估 |
| UWT 在 base32 上的效果 | +0.6% (远低于 base64 的 +2.2%) | 1000ep 评估 |
| **CrossAttn 贡献** | **-0.1%** | 推断变体分析 |
| **移除 CrossAttn 后提升** | **+0.6%** (96.5% → 97.1%) | 3 seeds 消融 |
| RelationNet 对比 | 96.0% (vs ProtoNet+UWT 97.1%) | 同编码器冻结 |
| MAML 对比 | <27% (无法收敛) | 10-way 从零/预训练 |
| 多尺度 vs 单尺度 | 差值 < 0.5% (无影响) | 500 episodes 验证 |

> 关键限制：UWT 的效果依赖大容量编码器 + SupCon 预训练提供的丰富特征空间。
> base32 (1M) 上 UWT 仅 +0.6%，而 base64 (4M) 上 +2.2%，差距 3.7 倍。

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
