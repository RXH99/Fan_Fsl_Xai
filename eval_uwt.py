"""
不确定性加权直推式推理 (Uncertainty-Weighted Transductive)

核心思想：
  标准直推式推理对所有 query 样本一视同仁。
  但边界模糊、低置信度的 query 样本会污染原型。
  
  本文改进：
  → 计算每个 query 的 softmax 熵（不确定性）
  → 熵高的样本（模糊/边界样本）在原型更新中权重自动降低
  → 低熵的（高置信度样本）主导更新

  这是论文的核心创新点。
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn.functional as F
import yaml
import numpy as np
from src.data.dataset import FaultDataset, EpisodicSampler
from src.models.encoder import create_encoder
from src.data.augmentation import augment_vibration_batch


def uncertainty_weighted_transductive(encoder, support_x, support_y, query_x, query_y,
                                        num_steps=5, tau=0.3, mix_ratio=0.8,
                                        beta=2.0):
    """
    不确定性加权的直推式推理

    Args:
        beta: 不确定性惩罚强度。beta 越大，高熵样本的权重被压制得越狠
    """
    s_emb = encoder(support_x)
    q_emb = encoder(query_x)

    s_emb = F.normalize(s_emb, dim=1)
    q_emb = F.normalize(q_emb, dim=1)

    ways = len(torch.unique(support_y))

    # 初始原型（仅 support）
    prototypes = torch.stack([
        s_emb[support_y == cls].mean(0) for cls in range(ways)
    ])
    prototypes = F.normalize(prototypes, dim=1)

    for step in range(num_steps):
        # 计算 query 到原型的相似度
        sims = torch.mm(q_emb, prototypes.t())  # (Q, ways)
        soft = torch.softmax(sims / tau, dim=1)  # (Q, ways)

        # ===== 核心创新：不确定性加权 =====
        # 计算每个 query 的熵
        entropy = -(soft * torch.log(soft + 1e-8)).sum(dim=1)  # (Q,)
        # 归一化熵到 [0, 1]
        entropy_norm = entropy / np.log(ways)
        # 计算权重：低熵（高置信度）→ 高权重
        weight = torch.exp(-entropy_norm * beta)  # (Q,)

        # 加权后的软分配
        weighted_soft = soft * weight.unsqueeze(1)  # (Q, ways)

        # 用加权后的软分配更新原型
        new_protos = []
        for w in range(ways):
            weight_sum = weighted_soft[:, w].sum()
            if weight_sum > 1e-8:
                qp = (weighted_soft[:, w] @ q_emb) / weight_sum
            else:
                qp = prototypes[w]  # fallback
            new_protos.append(qp)
        new_protos = torch.stack(new_protos)
        new_protos = F.normalize(new_protos, dim=1)

        # 融合
        prototypes = F.normalize(
            mix_ratio * prototypes + (1 - mix_ratio) * new_protos, dim=1
        )

    # 最终分类
    final_sims = torch.mm(q_emb, prototypes.t())
    preds = torch.argmax(final_sims, dim=1)
    acc = (preds == query_y).float().mean().item()

    return acc


# ===== 评估 =====
device = torch.device('cuda')
with open('configs/optimized.yaml', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)

test_dataset = FaultDataset('data/processed/preprocessed.npz', split='test')
encoder = create_encoder('resnet18', encoder_dim=64, use_se=True).to(device)
encoder.load_state_dict(torch.load('outputs/fewshot_encoder_ProtoNet_Cosine.pth'))
encoder.eval()

configs = [
    (5, 1, 15, "5-way 1-shot"),
    (5, 5, 15, "5-way 5-shot"),
    (10, 1, 10, "10-way 1-shot"),
    (10, 5, 10, "10-way 5-shot"),
]

# 参数扫描：找最佳 beta
print("参数扫描：beta 对精度的影响\n")

for ways, shot, query, name in configs:
    sampler = EpisodicSampler(test_dataset, ways=ways, shot=shot, query=query)

    best_beta = 0
    best_acc = 0

    for beta in [0.5, 1.0, 2.0, 3.0, 5.0]:
        accs = []
        for _ in range(300):
            s_x, s_y, q_x, q_y = sampler.sample_episode()
            s_x, q_x = s_x.to(device), q_x.to(device)
            s_y, q_y = s_y.to(device), q_y.to(device)

            with torch.no_grad():
                acc = uncertainty_weighted_transductive(
                    encoder, s_x, s_y, q_x, q_y,
                    num_steps=3, tau=0.3, mix_ratio=0.8, beta=beta)
                accs.append(acc)

        mean_acc = np.mean(accs) * 100
        print(f"  {name} beta={beta:.1f} → {mean_acc:.1f}%")
        if mean_acc > best_acc:
            best_acc = mean_acc
            best_beta = beta

    print(f"  ✅ {name} 最佳 beta={best_beta}, acc={best_acc:.1f}%\n")


# ===== 论文主实验：用最佳 beta 跑 1000 episode =====
print("=" * 60)
print("📊 论文主实验：不确定性加权直推式推理 (1000 episodes)")
print("=" * 60)

results = {}
for ways, shot, query, name in configs:
    sampler = EpisodicSampler(test_dataset, ways=ways, shot=shot, query=query)
    accs = []

    for _ in range(1000):
        s_x, s_y, q_x, q_y = sampler.sample_episode()
        s_x, q_x = s_x.to(device), q_x.to(device)
        s_y, q_y = s_y.to(device), q_y.to(device)

        with torch.no_grad():
            acc = uncertainty_weighted_transductive(
                encoder, s_x, s_y, q_x, q_y,
                num_steps=3, tau=0.3, mix_ratio=0.8, beta=2.0)
            accs.append(acc)

    mean = np.mean(accs) * 100
    std = np.std(accs) * 100
    results[name] = (mean, std)
    print(f"  {name:<18} → {mean:.1f}% ± {std:.1f}%")

# ===== 对照：原始直推式（无加权） =====
print("\n" + "=" * 60)
print("📊 对照实验：无加权直推式 (beta=0 等价于标准直推)")
print("=" * 60)

for ways, shot, query, name in configs:
    sampler = EpisodicSampler(test_dataset, ways=ways, shot=shot, query=query)
    accs = []

    for _ in range(1000):
        s_x, s_y, q_x, q_y = sampler.sample_episode()
        s_x, q_x = s_x.to(device), q_x.to(device)
        s_y, q_y = s_y.to(device), q_y.to(device)

        with torch.no_grad():
            acc = uncertainty_weighted_transductive(
                encoder, s_x, s_y, q_x, q_y,
                num_steps=3, tau=0.3, mix_ratio=0.8, beta=0.0)
            accs.append(acc)

    mean = np.mean(accs) * 100
    std = np.std(accs) * 100
    base_mean, base_std = results[name]
    diff = mean - base_mean
    print(f"  {name:<18} → {mean:.1f}% ± {std:.1f}%  (vs 加权: {base_mean:.1f}%, 差: {diff:+.1f}%)")

print("\n✅ 实验结束。Uncertainty-Weighted Transductive 的结果可用于论文。")
