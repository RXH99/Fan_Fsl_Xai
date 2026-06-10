# Fan_Fsl_Xai

**不确定性加权直推式原型网络的小样本故障诊断**

基于原型网络（Prototypical Network）的小样本故障诊断框架，在**风机 189 类**和 **CWRU 轴承 40 类**数据集上验证。

---

## ✨ 特点

- **SE‑ResNet1D** 特征提取器，集成压缩激励注意力机制
- **余弦相似度** 度量，替代标准欧氏距离
- **10‑way 元训练**，搭配在线数据增强（噪声 / 时间掩码 / 随机缩放）
- **直推式推理** — 利用无标签 query 样本迭代优化原型
- **不确定性加权直推式** — 抑制高熵（模糊）query 样本对原型更新的影响
- **双数据集验证** — 工业风机数据（189 类，每类 20 样本）+ CWRU 轴承数据（40 类）

---

## 🧠 方法概览

```
原始信号 (1024 点)
       │
       ▼
  SE‑ResNet1D 编码器
       │
       ├── 时域分支 (1D 卷积)
       └── SE 注意力 (通道重标定)
       │
       ▼
  L2 归一化特征嵌入
       │
       ▼
  余弦相似度 vs 类原型
       │
       ▼
  [可选] 直推式推理
       │   └── 不确定性加权 (本文)
       │
       ▼
  分类结果
```

### 不确定性加权直推式推理

标准直推式推理对所有 query 样本一视同仁。但位于决策边界附近的模糊样本（高预测不确定性）会拉偏原型，降低分类精度。

**改进方法：** 计算每个 query 的 softmax 熵，对高熵样本降权：

```python
entropy = -(soft * torch.log(soft + 1e-8)).sum(dim=1)
entropy_norm = entropy / log(ways)
weight = exp(-entropy_norm * beta)
weighted_soft = soft * weight.unsqueeze(1)
```

这是一个**正则化机制**，而非主要涨分手段——它的作用是提升原型的鲁棒性，而非直接推高准确率。

---

## 📊 实验结果

### 风机数据集 — 189 类（153 基类 / 18 新类 / 18 验证）

| 方法 | 5w1s | 5w5s | 10w1s | 10w5s |
|---|---|---|---|---|
| ProtoNet_CNN（基线） | 72.0% | 80.5% | 54.5% | 65.3% |
| ProtoNet_ResNet18 | 88.9% | 94.0% | 79.7% | 87.8% |
| **ProtoNet_Cosine**（10-way 训练） | 90.8% | 95.3% | 83.1% | 90.5% |
| + 直推式推理 | 93.2% | 96.3% | 86.3% | 92.5% |
| + **不确定性加权（本文）** | **93.4%** | **96.6%** | **86.7%** | **92.5%** |

**关键分析：**
- 余弦度量 + 10‑way 训练 + 数据增强：**较 ResNet 基线提升 +1.3%–1.9%**
- 直推式推理：**+1.0%–2.4%（主要涨分来源）**
- 不确定性加权：**+0.2%–0.4%（鲁棒性机制，非性能驱动）**

### CWRU 数据集 — 40 类（20 基类 / 10 新类 / 10 验证）

| 方法 | 5w1s | 5w5s | 10w1s | 10w5s |
|---|---|---|---|---|
| **ProtoNet_Cosine** | **99.7%** | **99.9%** | **99.4%** | **99.8%** |

> CWRU 各类间区分度大，样本充足（约 235–947 样本/类），性能已达天花板。

---

## 🚀 快速开始

### 环境要求

- Python 3.9+
- PyTorch 2.0+
- `scipy`, `numpy`, `pyyaml`
- 推荐使用 CUDA GPU

### 数据准备

**1. 风机数据** — 将 `.mat` 文件放入：
```
data/
├── train/   (153 类)
├── test/    (18 类)
└── val/     (18 类)
```

**2. CWRU 数据** — 将 CWRU `.mat` 文件放入：
```
data/CWRU data/12k Drive End Bearing Fault Data/
data/CWRU data/Normal Baseline/
```

### 运行实验

```powershell
# === 风机实验 ===

# 第 1 步：数据预处理
python step1_preprocess.py

# 第 2 步：SupCon 预训练（可选）
python step2_pretrain_simclr.py --mode supcon --epochs 200

# 第 3 步：小样本训练
python step3_train_fewshot.py --method ProtoNet_Cosine

# 第 4 步：不确定性加权评估
python eval_uwt.py

# === CWRU 实验 ===

# 第 1 步：数据预处理
python step1_preprocess_cwru.py

# 第 2 步：小样本训练
python step3_train_fewshot.py --config configs/cwru.yaml --method ProtoNet_Cosine
```

### 超参数

**风机数据**（`configs/optimized.yaml`）：
- ways: 10, shot: 5, query: 5
- episodes: 3000, lr: 0.0001
- sep_weight: 0.05
- base_filters: 32, use_se: true
- 数据增强：noise=0.02, mask=0.15, scale=0.03

**最佳直推参数**（风机 5w5s）：
- num_steps: 3, tau: 0.3, mix_ratio: 0.8, beta: 3.0

---

## 📁 项目结构

```
Fan_Fsl_Xai/
├── configs/
│   ├── baseline.yaml          # 基线配置
│   ├── optimized.yaml         # 风机实验配置
│   └── cwru.yaml              # CWRU 实验配置
├── src/
│   ├── config.py              # 配置管理
│   ├── data/
│   │   ├── dataset.py         # 数据集 + 采样器
│   │   ├── preprocess.py      # 风机数据预处理
│   │   └── augmentation.py    # 数据增强
│   ├── models/
│   │   ├── encoder.py         # CNN / ResNet1D / 多尺度编码器
│   │   └── prototypical.py    # ProtoNet 损失 + 直推推理 + 不确定性加权
│   ├── training/
│   │   └── train_fewshot.py   # 训练循环
│   └── interpret/             # 可解释性工具
├── data/                      # 原始 .mat 文件（不上传）
├── data_cwru/                 # CWRU 预处理数据（不上传）
├── outputs/                   # 风机模型权重（不上传）
├── outputs_cwru/              # CWRU 模型权重（不上传）
├── step1_preprocess.py        # 风机预处理入口
├── step1_preprocess_cwru.py   # CWRU 预处理入口
├── step2_pretrain_simclr.py   # 对比学习预训练
├── step3_train_fewshot.py     # 小样本训练
├── step5_experiments.py       # 批量实验
├── step6_tsne.py              # t-SNE 可视化
├── eval_uwt.py                # 不确定性加权评估
└── README.md                  # 本文件
```

---

## 🔮 待办

- [ ] 完整消融实验（SE / 增强 / 10-way / 直推 / 加权）
- [ ] 基线方法对比（MAML、RelationNet、1D-CNN、SVM）
- [ ] t-SNE 特征可视化
- [ ] IG 归因分析（高熵 vs 低熵样本）
- [ ] 混淆矩阵分析

---

## 📝 说明

- **数据和模型权重不上传**到此仓库（详见 `.gitignore`）。
- 不确定性加权贡献的准确率提升有限，但能提升**原型鲁棒性**（通过直推推理过程中的熵分析验证）。
- 直推式推理是主要的准确率驱动因素；余弦度量和 10‑way 训练提供了扎实的基线提升。

---

## 📄 许可证

MIT
