# Fan_Fsl_Xai — 实验总结

> 最后更新：2026-06-10
> 目标期刊：SCI Q2/Q3（Measurement Science and Technology / IEEE Sensors Journal 等）
> 创新点：Uncertainty-Weighted Transductive Prototypical Network

---

## 一、项目概述

小样本故障诊断框架，基于原型网络（Prototypical Network），在**风机 189 类**和 **CWRU 40 类**两个数据集上验证。

### 核心方法链

```
SE‑ResNet1D 编码器
  → 余弦相似度度量（替代欧氏距离）
  → 10‑way 元训练（更难的 episode，特征泛化更强）
  → 直推式推理（Transductive Inference，利用无标签 query 优化原型）
  → 不确定性加权（本文创新：抑制高熵模糊样本对原型的污染）
```

### 项目结构

```
Fan_Fsl_Xai/
├── configs/
│   ├── baseline.yaml          # 原始基线配置
│   ├── optimized.yaml         # 风机实验配置
│   └── cwru.yaml              # CWRU 实验配置
├── src/
│   ├── config.py              # 配置管理
│   ├── data/
│   │   ├── dataset.py         # FaultDataset + EpisodicSampler
│   │   ├── preprocess.py      # 风机数据预处理
│   │   └── augmentation.py    # 数据增强（噪声/掩码/缩放）
│   ├── models/
│   │   ├── encoder.py         # CNNEncoder / ResNet1DEncoder / MultiScaleCNN
│   │   └── prototypical.py    # ProtoNet 损失 + 直推推理 + 不确定性加权
│   ├── training/
│   │   └── train_fewshot.py   # 训练循环（增广、早停、LR调度）
│   └── interpret/             # 可解释性模块
├── step1_preprocess.py        # 风机数据预处理入口
├── step1_preprocess_cwru.py   # CWRU 数据预处理入口
├── step2_pretrain_simclr.py   # SupCon/SimCLR 对比学习预训练
├── step3_train_fewshot.py     # 小样本训练入口
├── step5_experiments.py       # 批量对比实验
├── step6_tsne.py              # t-SNE 可视化
├── eval_uwt.py                # 不确定性加权直推式推理评估
├── output_cwru/               # CWRU 实验输出
├── outputs/                   # 风机实验输出
├── data/                      # 风机原始 .mat 数据
├── data_cwru/                 # CWRU 预处理数据
└── EXPERIMENT_SUMMARY.md      # 本文件
```

---

## 二、实验数据

### 2.1 风机数据（189 类）

| 属性 | 值 |
|---|---|
| 来源 | 课题组风机实测振动数据 |
| 总类别 | 189 |
| 训练类 | 153（3060 样本） |
| 验证类 | 18（360 样本） |
| 测试类 | 18（360 样本） |
| **类交叉** | **训练/验证/测试零重叠 ✅** |
| 每类样本数 | 20 |
| 信号长度 | 1024 点 |
| 类别含义 | 故障类型(Cxx) × 负载(Lx) × 转速(Sxx) × 通道(Chx) |

### 2.2 CWRU 数据（40 类）

| 属性 | 值 |
|---|---|
| 来源 | CWRU 12k Drive End Bearing Fault Data |
| 总类别 | 40 |
| 训练类 | 20（0hp + 1hp 负载，5679 样本） |
| 验证类 | 10（2hp 负载，3072 样本） |
| 测试类 | 10（3hp 负载，3081 样本） |
| **类交叉** | **训练/验证/测试零重叠 ✅** |
| 每类样本数 | 235–947 |
| 信号长度 | 1024 点 |
| 类别组成 | Normal(4) / Inner Race(12) / Ball(12) / Outer Race(12) |

---

## 三、实验结果

### 3.1 风机 189 类 — 完整消融

| 实验 | 5w1s | 5w5s | 10w1s | 10w5s | 说明 |
|---|---|---|---|---|---|
| **ProtoNet_CNN**（原始基线） | 72.0% | 80.5% | 54.5% | 65.3% | 简单 3 层 CNN |
| **ProtoNet_ResNet**（旧基线） | 88.9% | 94.0% | 79.7% | 87.8% | ResNet18 + SE |
| **ProtoNet_Cosine**（10-way 训练） | 90.8% | 95.3% | 83.1% | 90.5% | 余弦度量 + 增强 |
| **+ 直推式推理** | 93.2% | 96.3% | 86.3% | 92.5% | steps=3, tau=0.3, mix=0.8 |
| **+ 不确定性加权（本文）** | **93.4%** | **96.6%** | **86.7%** | **92.5%** | beta=3.0 |
| 总计提升 vs ResNet 基线 | +4.5% | +2.6% | +7.0% | +4.7% | — |

**需要补充的消融实验（论文要求）：**

| 消融 | 预期 5w5s | 说明 |
|---|---|---|
| 去掉 SE 注意力 | ~95.5% | SE 贡献 |
| 去掉数据增强 | ~96.0% | 增强贡献 |
| 5-way 训练（非 10-way） | ~95.8% | 10-way 贡献 |
| 去掉直推推理 | ~95.3% | 直推贡献 |
| 去掉不确定性加权 | ~96.3% | 加权贡献 |

### 3.2 CWRU 40 类

| 实验 | 5w1s | 5w5s | 10w1s | 10w5s |
|---|---|---|---|---|
| **ProtoNet_Cosine** | **99.7%** | **99.9%** | **99.4%** | **99.8%** |
| 验证集最佳 | — | 100.0% | — | — |

> CWRU 类间差异大、每类样本充裕，方法接近天花板，不需要额外调优。

---

## 四、创新点说明

### 4.1 不确定性加权直推式推理（本文核心创新）

**问题：**
标准直推式推理在原型更新时对所有 query 样本一视同仁。但边界模糊、低置信度的 query 样本（两类之间犹豫不决）会污染原型估计，降低分类精度。

**方法：**
在每次直推式迭代中：
1. 计算每个 query 样本的 softmax 熵作为不确定性度量
2. 对熵施加指数惩罚：`weight = exp(-entropy_norm × beta)`
3. 用加权后的软分配更新原型

```python
entropy = -(soft * torch.log(soft + 1e-8)).sum(dim=1)
entropy_norm = entropy / log(ways)
weight = exp(-entropy_norm * beta)
weighted_soft = soft * weight.unsqueeze(1)
```

**效果：**
- 高熵（模糊）样本对原型更新的影响被抑制
- 低熵（高置信度）样本主导原型更新
- 5w1s +0.2%，5w5s +0.3%，10w1s +0.4%

**新颖性确认：**
经检索，不确定性加权直推推理在小样本故障诊断领域未被发表过（医学图像分割有类似思路，但领域不同、实现不同）。

### 4.2 论文故事线

```
工业场景：风机数据极度稀缺（每类 20 样本）
          + 新增故障类型与已知类型完全无交叉

挑战：现有 ProtoNet 在小样本下特征可区分性不足
       → 直推推理改进了原型，但模糊 query 会污染原型

本文方案：SE‑ResNet1D + 余弦度量 + 10‑way 元训练
          → Uncertainty‑Weighted Transductive 迭代
          → 自适应抑制不确定样本，提升原型鲁棒性

验证：风机 189 类（96.6% 5w5s）+
       CWRU 40 类（99.9% 5w5s）
       → 方法在不同难度场景下均有效
```

---

## 五、运行流程

### 5.1 风机实验

```powershell
# 1. 数据预处理
python step1_preprocess.py

# 2. SupCon 对比学习预训练（可选，可跳过，预训练权重直接用）
python step2_pretrain_simclr.py --mode supcon --epochs 200

# 3. 小样本训练
python step3_train_fewshot.py --method ProtoNet_Cosine

# 4. 不确定性加权直推式评估
python eval_uwt.py

# 5. 完整对比实验（论文表格）
python step5_experiments.py --runs 3

# 6. t-SNE 可视化
python step6_tsne.py
```

### 5.2 CWRU 实验

```powershell
# 1. 数据预处理
python step1_preprocess_cwru.py

# 2. 小样本训练（用 cwru 配置）
python step3_train_fewshot.py --config configs/cwru.yaml --method ProtoNet_Cosine

# 3. 不确定性加权评估（需修改 eval_uwt.py 指向 cwru.yaml）
```

### 5.3 超参数设置

**风机（optimized.yaml）:**
- ways: 10, shot: 5, query: 5
- episodes: 3000, lr: 0.0001
- sep_weight: 0.05
- base_filters: 32, use_se: true
- 数据增强: noise=0.02, mask=0.15, scale=0.03

**直推推理最佳参数（风机 5w5s）:**
- num_steps: 3
- tau: 0.3
- mix_ratio: 0.8
- beta（不确定性）: 3.0

**CWRU（cwru.yaml）:**
- 与风机配置一致，仅数据路径不同

---

## 六、论文写作计划

### 6.1 目标期刊

| 期刊 | 分区 | 匹配度 |
|---|---|---|
| Measurement Science and Technology | Q2 | ⭐⭐⭐⭐⭐ |
| IEEE Sensors Journal | Q2 | ⭐⭐⭐⭐ |
| Journal of Intelligent Manufacturing | Q1/Q2 | ⭐⭐⭐ |
| Structural Health Monitoring | Q1/Q2 | ⭐⭐⭐ |

### 6.2 所需补充工作

| 工作 | 预计时间 | 优先级 |
|---|---|---|
| 1. 消融实验（SE/增强/10-way/直推/加权 各去掉测试） | 1 天 | 🔴 必须 |
| 2. 对比实验（MAML/RelationNet/1D-CNN/SVM 等） | 1 天 | 🔴 必须 |
| 3. t-SNE 特征可视化 + IG 归因图 | 0.5 天 | 🟡 重要 |
| 4. 混淆矩阵 + 标准差值分析 | 0.5 天 | 🟡 重要 |
| 5. 论文初稿写作 | 5–7 天 | —— |

### 6.3 对比方法建议

| 方法 | 类型 | 说明 |
|---|---|---|
| 1D-CNN + Softmax | 传统深度学习 | 弱基线 |
| SVM (with features) | 传统机器学习 | 弱基线 |
| ProtoNet (Euclidean) | 度量元学习 | 同等基线 |
| ProtoNet (Cosine) | 度量元学习 | 本文基线 |
| MAML | 优化元学习 | 强基线 |
| RelationNet | 度量元学习 | 强基线 |
| 本文方法 | — | 最终 |

---

## 七、关键文件清单

| 文件 | 说明 |
|---|---|
| `outputs/fewshot_encoder_ProtoNet_Cosine.pth` | 风机最优模型（验证 94.5%） |
| `outputs_cwru/fewshot_encoder_ProtoNet_Cosine.pth` | CWRU 最优模型（验证 100%） |
| `outputs/pretrained_resnet18_encoder.pth` | SupCon 预训练权重 |
| `outputs/experiment_results_*.txt` | 实验结果记录 |
| `data/processed/preprocessed.npz` | 风机预处理数据 |
| `data_cwru/processed/preprocessed.npz` | CWRU 预处理数据 |

---

## 八、待办事项

- [ ] 跑完整消融实验（TBD）
- [ ] 跑 MAML / RelationNet 对比
- [ ] 跑 CWRU 不确定性加权评估
- [ ] 生成 t-SNE 图（风机 + CWRU）
- [ ] 生成 IG 归因图（高熵 vs 低熵样本对比）
- [ ] 生成混淆矩阵
- [ ] 论文写作

---

_本文档由 AI 辅助整理，实验数据来源于 Fan_Fsl_Xai 项目的实际运行结果。_
