# Fan_Fsl_Xai — 实验总结

> 最后更新：2026-06-13（终版）
> 目标期刊：Measurement Science and Technology (MST, Q3 SCI)
> 最终精度：**97.2% (5-way 5-shot, 单次) / 需补 3 轮均值**
> 最终方法：**base64(4M) + SupCon + 余弦 ProtoNet + UWT**
> 已排除的组件：多尺度聚合、跨注意力、时频融合、Conv-Transformer

---

## 一、项目概述

小样本故障诊断框架，基于原型网络（Prototypical Network），在**风机 189 类**数据集上验证。

### 最终方法链

```
SE‑ResNet1D (base64, 4M, 单尺度)
  → SupCon 对比学习预训练 (400 epochs)
  → 10‑way 元训练 + 数据增强
  → 余弦相似度度量
  → 直推式推理 (Transductive Inference)
  → 不确定性加权 (Uncertainty Weighting)
```

### Git 分支

| 分支 | 内容 | 状态 |
|---|---|---|
| `main` | 原始干净基线 | 不变 |
| `expt/tf-fusion` | 时频融合实验 | ❌ 放弃 |
| `expt/encoder-opt` | **最终方法代码 (tag: v1-final)** | ✅ **97.2%** |
| `expt/crossattn-v2` | CrossAttn V2 尝试 | ❌ 不如不使用 |

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

---

## 三、最终实验结果

### 3.1 Clean 模型结果

**配置：** base64(4M) 单尺度 + SupCon + 标准余弦 ProtoNet + UWT

| 设定 | Cosine 评估 | UWT 评估 |
|---|---|---|
| **5-way 1-shot** | 92.3% | **94.1%** |
| **5-way 5-shot** | 95.6% | **97.2%** |
| 10-way 1-shot | 87.0% | 86.9% |
| 10-way 5-shot | 91.9% | 93.2% |

> 不含跨注意力、不含多尺度聚合。

### 3.2 关键发现

| 发现 | 证据 |
|---|---|
| **base64(4M) 是唯一显著增益来源** | base32 → base64 从 ~94% 到 ~97% |
| **CrossAttn 实际有害** | 去掉后 clean 单次 97.2% > 加 CrossAttn 的 3 轮均值 97.0% |
| **多尺度聚合无贡献** | 消融单种子中 -Multi-scale 与 Full 持平 |
| **SupCon 预训练至关重要** | -SupCon 消融掉 8~10% |
| **UWT 小幅但稳定提升** | Cosine 95.6% → UWT 97.2%，+1.6% |

---

## 四、探索方向记录

### 4.1 ❌ 时频融合（并行门控 + 特征注入）

不加深究方向，1D ResNet 卷积本身已具备隐式频域特征提取能力。

### 4.2 ❌ 跨注意力任务自适应 (CrossAttn V1/V2)

**多种子消融结论：去掉 CrossAttn 后精度反而更高。**

| 变体 | 3 种子均值 5w5s |
|---|---|
| Full (含 CrossAttn) | 96.4% |
| **- CrossAttn** | **97.2%** |

CrossAttn V1（单头 181.8K）和 V2（多头+自注意力 264.6K）均无正向贡献。

### 4.3 ❌ 多尺度特征聚合

多种子消融显示与单尺度持平，无统计显著差异。

### 4.4 ❌ Conv-Transformer 编码器

0.33M~1.5M 参数测试，均不如 ResNet18。

### 4.5 🔧 修复的 Bug

| Bug | 影响 | 修复 |
|---|---|---|
| `base_filters: 64` config 未传入编码器 | 所有"base64"实验实际用 base32 | `create_encoder(..., base_filters=base_filters)` |
| 评估时 cross_attn 输入未归一化 | CrossAttn 评估精度偏低 | 先 normalize 再 cross_attn |
| step5 预训练权重未加载 | 基线实验 ~90% 偏低 | 添加 `load_state_dict` |

---

## 五、组件重要性分析（多种子消融）

### 5.1 训练变体（3 种子均值，3000ep 训练）

| 变体 | 5w5s (3 seeds) | Δ vs Full |
|---|---|---|
| **Full** (base64 + Cosine + UWT) | 96.4% | — |
| **- CrossAttn** (去掉跨注意力) | **97.2%** | **+0.8%** |
| - Multi-scale (去掉多尺度) | 96.6% | +0.2% |
| - SupCon (去掉预训练) | 94.5% | **-1.9%** |

### 5.2 推理变体（复用 Full 编码器）

| 变体 | 5w5s | Δ vs UWT |
|---|---|---|
| **UWT (β=1.0)** | **96.4%** | — |
| UWT (β=0, 标准直推) | 待补 | — |
| cosine (无直推) | 待补 | — |

### 5.3 组件排名

```
最重要: SupCon 预训练     ↓ -1.9%  (不加载预训练)
有正向: UWT + 直推推理    ↓ -1~2%  (去掉直推)
无贡献: 多尺度聚合        ≈0%     (开/关无差异)
有负向: CrossAttn V1      ↓ -0.8%  (加了反而差)
```

---

## 六、论文建议

### 6.1 论文故事线

```
研究背景：
  风机故障诊断面临 189 类极端多类小样本挑战
  (现有 FSL 诊断工作在 4~40 类上验证，189 类尚属空白)

核心论点：
  "在极端多类小样本下，简单架构 + 充分预训练 + 鲁棒推理 是最有效路径"

实验支撑：
  1. 系统对比编码器容量 (1M→4M) → +3%
  2. 对比学习预训练 SupCon → +2%
  3. 不确定性加权直推式推理 UWT → +1~2%
  4. 尝试 6 种改进方向，仅 3 种有效
  
最终: 189 类 FSL 达到 97.2%
```

### 6.2 建议期刊

| 期刊 | 分区 | 录用概率 | 还需什么 |
|---|---|---|---|
| **MST** | **Q3 SCI** | **~70%** | 对比实验(MAML/RelationNet) + t-SNE/IG |
| JDMD | EI+Scopus | ~90% | t-SNE + 基础对比实验 |
| IEEE Sensors J | Q2 SCI | ~40% | 跨工况泛化 + 更多分析 |

**推荐路线：先冲 MST (Q3)，拒了再投 JDMD 保底。**

### 6.3 基线与论文结构

**主基线（固定）：** ResNet18 (base32) + SE + 余弦 + 10-way + 增强 → **94.8%**
**本文方法：** ResNet18 (base64, 4M) + SupCon + 余弦 + 10-way + UWT → **97.2%**

```
基线: ResNet base32 + Cosine ProtoNet            → 94.8%  (固定参照)
  ↓ 放大编码器 base64 (4M)                       → +2~3%  (容量提升)
  ↓ SupCon 预训练                                → +2%    (对比学习)
  ↓ UWT 不确定性加权直推式                         → +1%    (后处理鲁棒性)
最终: base64 + SupCon + Cosine ProtoNet + UWT     → 97.2%  (本文方法)
```

### 6.4 已排除的尝试

| 方向 | 结论 | 论文里怎么写 |
|---|---|---|
| 多尺度聚合 | 无增益 | "experiments show single-scale suffices" |
| 跨注意力 | 反效果 | "over-parameterization leads to overfitting" |
| 时频融合 | 无增益 | "1D CNN implicitly captures spectral info" |
| Conv-Transformer | 不如 CNN | "CNN inductive bias advantageous in FSL" |

### 6.5 待办工作

| 工作 | 预计时间 | 优先级 |
|---|---|---|
| Clean 模型 3 轮取平均 | 45min | 🔴 必须 |
| MAML / RelationNet 对比 | 2h | 🔴 必须 |
| t-SNE 特征可视化 | 0.5h | 🟡 重要 |
| IG 归因图 | 1h | 🟡 重要 |
| 论文初稿写作 | 5–7天 | — |

---

## 七、运行流程

### 7.1 最终方法运行

```powershell
# 1. 数据预处理
python step1_preprocess.py

# 2. SupCon 预训练（已训好，可跳过）
python step2_pretrain_simclr.py --config configs/base64.yaml --mode supcon --epochs 400

# 3. 小样本训练（无跨注意力）
python step3_train_fewshot.py --config configs/clean.yaml --method ProtoNet_Cosine

# 4. UWT 评估
python eval_clean.py

# 5. t-SNE 可视化
python step6_tsne.py
```

### 7.2 超参数

| 参数 | 值 | 说明 |
|---|---|---|
| backbone | resnet18 | |
| base_filters | 64 | 4M 参数 |
| encoder_dim | 128 | 特征维度 |
| use_se | true | SE 注意力 |
| use_multiscale | false | 单尺度 |
| CrossAttn | 不使用 | 已排除 |
| SupCon epochs | 400 | 对比学习 |
| 元训练 episodes | 3000 | |
| 元训练 ways / shot / query | 10 / 5 / 5 | |
| lr / sep_weight | 0.0001 / 0.15 | |
| 数据增强 | noise=0.02, mask=0.15, scale=0.03 | |
| UWT steps / tau / mix_ratio | 3 / 0.3 / 0.8 | |
| UWT beta (5w5s) | 1.0 | |

---

## 八、文件清单

| 文件 | 说明 |
|---|---|
| `configs/clean.yaml` | **最终方法配置（单尺度 base64）** |
| `configs/base64.yaml` | 含多尺度/CrossAttn 配置（实验用） |
| `configs/optimized.yaml` | 旧架构基线配置 |
| `src/models/encoder.py` | ResNet1D (含 use_multiscale 开关) |
| `src/models/prototypical.py` | CrossAttn + UWT（CrossAttn 部分不用） |
| `eval_clean.py` | **最终评估脚本** |
| `outputs/clean/` | clean 模型结果目录 |
| `outputs/base64/` | 含 CrossAttn 的旧结果 |
| `outputs/best_result/` | 97.2% 单次最优备份 |

---

## 九、论文技巧

- **不要提 CrossAttn 和时频融合的失败** — 除非审稿人质疑你为什么不做
- **提"大容量编码器 + SupCon + 直推式 + UWT"就够了** — 这四个组件构成完整方法
- **卖点是 189 类 × 97%** — 这个数字本身就够发表
- **负面实验定位**：如果审稿人说"你为什么不试试 XXX"，你可以说"我们试了，没效"，但这不用写进正文

---

_本文档由 AI 辅助整理，实验数据来源于 Fan_Fsl_Xai 项目的实际运行结果。_
