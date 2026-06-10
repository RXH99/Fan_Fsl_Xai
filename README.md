# Fan_Fsl_Xai

**Few-Shot Fault Diagnosis with Uncertainty-Weighted Transductive Prototypical Network**

A prototypical network framework for few-shot fault diagnosis on vibration signals, tested on **189-class fan data** and **CWRU bearing data**.

---

## ✨ Features

- **SE‑ResNet1D** feature extractor with squeeze-and-excitation attention
- **Cosine similarity** metric replacing standard Euclidean distance
- **10‑way meta-training** with online data augmentation (noise / time masking / random scaling)
- **Transductive inference** — iteratively refines prototypes using unlabeled query samples
- **Uncertainty-weighted transductive** — suppresses high-entropy (ambiguous) query samples during prototype refinement
- **Two datasets** — industrial fan data (189 classes, 20 samples/class) + CWRU bearing data (40 classes)

---

## 🧠 Method Overview

```
Raw Signal (1024 points)
       │
       ▼
  SE‑ResNet1D Encoder
       │
       ├── Time branch (1D convolutions)
       └── SE attention (channel-wise recalibration)
       │
       ▼
  L2‑normalized feature embedding
       │
       ▼
  Cosine similarity vs class prototypes
       │
       ▼
  [Optional] Transductive refinement
       │   └── Uncertainty weighting (ours)
       │
       ▼
  Prediction
```

### Uncertainty-Weighted Transductive Inference

Standard transductive inference treats all query samples equally when updating prototypes. However, boundary samples with high prediction uncertainty can pull prototypes in the wrong direction.

**Our fix:** compute softmax entropy per query and down-weight high-entropy samples:

```python
entropy = -(soft * torch.log(soft + 1e-8)).sum(dim=1)
entropy_norm = entropy / log(ways)
weight = exp(-entropy_norm * beta)
weighted_soft = soft * weight.unsqueeze(1)
```

This is a **regularization mechanism**, not a major accuracy booster — it improves prototype robustness rather than pushing raw accuracy.

---

## 📊 Results

### Fan Dataset — 189 Classes (153 base / 18 novel / 18 val)

| Method | 5w1s | 5w5s | 10w1s | 10w5s |
|---|---|---|---|---|
| ProtoNet_CNN (baseline) | 72.0% | 80.5% | 54.5% | 65.3% |
| ProtoNet_ResNet18 | 88.9% | 94.0% | 79.7% | 87.8% |
| **ProtoNet_Cosine** (10-way train) | 90.8% | 95.3% | 83.1% | 90.5% |
| + Transductive inference | 93.2% | 96.3% | 86.3% | 92.5% |
| + **Uncertainty‑weighted (ours)** | **93.4%** | **96.6%** | **86.7%** | **92.5%** |

**Key observations:**
- Cosine metric + 10‑way training + augmentation: **+1.3–1.9%** over ResNet baseline
- Transductive inference: **+1.0–2.4%** (primary contributor)
- Uncertainty weighting: **+0.2–0.4%** (robustness mechanism, not a performance driver)

### CWRU Dataset — 40 Classes (20 base / 10 novel / 10 val)

| Method | 5w1s | 5w5s | 10w1s | 10w5s |
|---|---|---|---|---|
| **ProtoNet_Cosine** | **99.7%** | **99.9%** | **99.4%** | **99.8%** |

> CWRU classes are well-separated with abundant samples (~235–947/class). Performance saturates near ceiling.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- PyTorch 2.0+
- `scipy`, `numpy`, `pyyaml`
- CUDA-capable GPU recommended

### Data Preparation

**1. Fan dataset** — place your `.mat` files in:
```
data/
├── train/   (153 classes)
├── test/    (18 classes)
└── val/     (18 classes)
```

**2. CWRU dataset** — place CWRU `.mat` files in:
```
data/CWRU data/12k Drive End Bearing Fault Data/
data/CWRU data/Normal Baseline/
```

### Run Experiments

```powershell
# === Fan Dataset ===

# Step 1: Preprocess
python step1_preprocess.py

# Step 2: SupCon pretraining (optional)
python step2_pretrain_simclr.py --mode supcon --epochs 200

# Step 3: Few-shot training
python step3_train_fewshot.py --method ProtoNet_Cosine

# Step 4: Uncertainty-weighted evaluation
python eval_uwt.py

# === CWRU Dataset ===

# Step 1: Preprocess
python step1_preprocess_cwru.py

# Step 2: Few-shot training
python step3_train_fewshot.py --config configs/cwru.yaml --method ProtoNet_Cosine
```

### Hyperparameters

**Fan dataset** (`configs/optimized.yaml`):
- ways: 10, shot: 5, query: 5
- episodes: 3000, lr: 0.0001
- sep_weight: 0.05
- base_filters: 32, use_se: true
- Augmentation: noise=0.02, mask=0.15, scale=0.03

**Best transductive params** (fan 5w5s):
- num_steps: 3, tau: 0.3, mix_ratio: 0.8, beta: 3.0

---

## 📁 Project Structure

```
Fan_Fsl_Xai/
├── configs/
│   ├── baseline.yaml          # Baseline config
│   ├── optimized.yaml         # Fan experiment config
│   └── cwru.yaml              # CWRU experiment config
├── src/
│   ├── config.py              # Config manager
│   ├── data/
│   │   ├── dataset.py         # FaultDataset + EpisodicSampler
│   │   ├── preprocess.py      # Fan data preprocessing
│   │   └── augmentation.py    # Data augmentation
│   ├── models/
│   │   ├── encoder.py         # CNN / ResNet1D / MultiScale encoders
│   │   └── prototypical.py    # ProtoNet loss + transductive + uncertainty
│   ├── training/
│   │   └── train_fewshot.py   # Training loop
│   └── interpret/             # Interpretability tools
├── data/                      # Raw .mat files (not tracked)
├── data_cwru/                 # CWRU preprocessed data (not tracked)
├── outputs/                   # Fan model weights (not tracked)
├── outputs_cwru/              # CWRU model weights (not tracked)
├── step1_preprocess.py        # Fan preprocessing entry
├── step1_preprocess_cwru.py   # CWRU preprocessing entry
├── step2_pretrain_simclr.py   # Contrastive pretraining
├── step3_train_fewshot.py     # Few-shot training
├── step5_experiments.py       # Batch experiments
├── step6_tsne.py              # t-SNE visualization
├── eval_uwt.py                # Uncertainty-weighted evaluation
└── README.md
```

---

## 🔮 TODO

- [ ] Full ablation study (SE / augmentation / 10-way / transductive / weighting)
- [ ] Baseline comparisons (MAML, RelationNet, 1D-CNN, SVM)
- [ ] t-SNE feature visualization
- [ ] IG attribution analysis (high-entropy vs low-entropy samples)
- [ ] Confusion matrix analysis

---

## 📝 Notes

- **Data and model weights are not tracked** in this repository (see `.gitignore`).
- Uncertainty weighting contributes marginal accuracy gain but improves **prototype robustness**, verified via entropy analysis during transductive refinement.
- The transductive inference step is the primary accuracy driver; cosine metric and 10‑way training provide solid baseline improvements.

---

## 📄 License

MIT
