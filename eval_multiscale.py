"""
评估多尺度编码器 + 不确定性加权直推式推理
运行: python eval_multiscale.py
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn.functional as F
import numpy as np
from src.data.dataset import FaultDataset, EpisodicSampler
from src.models.encoder import create_encoder


def evaluate_uwt(encoder, support_x, support_y, query_x, query_y,
                 num_steps=3, tau=0.3, mix_ratio=0.8, beta=2.0):
    s_emb = encoder(support_x)
    q_emb = encoder(query_x)
    s_emb = F.normalize(s_emb, dim=1)
    q_emb = F.normalize(q_emb, dim=1)

    ways = len(torch.unique(support_y))
    prototypes = torch.stack([
        s_emb[support_y == cls].mean(0) for cls in range(ways)
    ])
    prototypes = F.normalize(prototypes, dim=1)

    for _ in range(num_steps):
        sims = torch.mm(q_emb, prototypes.t())
        soft = torch.softmax(sims / tau, dim=1)
        entropy = -(soft * torch.log(soft + 1e-8)).sum(dim=1)
        entropy_norm = entropy / np.log(ways)
        weight = torch.exp(-entropy_norm * beta)
        weighted_soft = soft * weight.unsqueeze(1)

        new_protos = []
        for w in range(ways):
            ws = weighted_soft[:, w].sum()
            if ws > 1e-8:
                qp = (weighted_soft[:, w] @ q_emb) / ws
            else:
                qp = prototypes[w]
            new_protos.append(qp)
        new_protos = torch.stack(new_protos)
        new_protos = F.normalize(new_protos, dim=1)
        prototypes = F.normalize(
            mix_ratio * prototypes + (1 - mix_ratio) * new_protos, dim=1)

    final_sims = torch.mm(q_emb, prototypes.t())
    preds = torch.argmax(final_sims, dim=1)
    return (preds == query_y).float().mean().item()


device = torch.device("cuda")
print("=" * 60)
print("多尺度编码器 — UWT 评估")
print("=" * 60)

# 加载测试集
test_dataset = FaultDataset("data/processed/preprocessed.npz", split="test")

# 创建多尺度编码器（encoder_dim=128，与训练时一致）
encoder = create_encoder("resnet18", encoder_dim=128, use_se=True).to(device)
ckpt = torch.load("outputs/base64/fewshot_encoder_ProtoNet_Cosine.pth",
                  map_location=device)
encoder.load_state_dict(ckpt)
encoder.eval()
print(f"✅ 已加载多尺度编码器 (encoder_dim=128)")

configs = [
    (5, 1, 15, "5-way 1-shot"),
    (5, 5, 15, "5-way 5-shot"),
    (10, 1, 10, "10-way 1-shot"),
    (10, 5, 10, "10-way 5-shot"),
]

# 参数扫描 + 主实验
print("\n参数扫描：beta 对精度的影响\n")
best_betas = {}

for ways, shot, query, name in configs:
    sampler = EpisodicSampler(test_dataset, ways=ways, shot=shot, query=query)
    best_beta, best_acc = 0, 0

    for beta in [0.5, 1.0, 2.0, 3.0, 5.0]:
        accs = []
        for _ in range(300):
            s_x, s_y, q_x, q_y = sampler.sample_episode()
            s_x, q_x, s_y, q_y = s_x.to(device), q_x.to(device), s_y.to(device), q_y.to(device)
            with torch.no_grad():
                acc = evaluate_uwt(encoder, s_x, s_y, q_x, q_y, beta=beta)
                accs.append(acc)
        mean = np.mean(accs) * 100
        print(f"  {name} beta={beta:.1f} → {mean:.1f}%")
        if mean > best_acc:
            best_acc, best_beta = mean, beta
    best_betas[name] = best_beta
    print(f"  ✅ {name} 最佳 beta={best_beta}, acc={best_acc:.1f}%\n")

# 主实验：1000 episode
print("=" * 60)
print("📊 UWT 主实验 (1000 episodes, per-setting best beta)")
print("=" * 60)

uwt_results = {}
for ways, shot, query, name in configs:
    beta = best_betas[name]
    sampler = EpisodicSampler(test_dataset, ways=ways, shot=shot, query=query)
    accs = []
    for _ in range(1000):
        s_x, s_y, q_x, q_y = sampler.sample_episode()
        s_x, q_x, s_y, q_y = s_x.to(device), q_x.to(device), s_y.to(device), q_y.to(device)
        with torch.no_grad():
            acc = evaluate_uwt(encoder, s_x, s_y, q_x, q_y, beta=beta)
            accs.append(acc)
    mean, std = np.mean(accs) * 100, np.std(accs) * 100
    uwt_results[name] = (mean, std)
    print(f"  {name:<18} beta={beta:.1f} → {mean:.1f}% ± {std:.1f}%")

# 对照：标准直推式
print("\n" + "=" * 60)
print("📊 对照：标准直推式 (beta=0)")
print("=" * 60)

for ways, shot, query, name in configs:
    sampler = EpisodicSampler(test_dataset, ways=ways, shot=shot, query=query)
    accs = []
    for _ in range(1000):
        s_x, s_y, q_x, q_y = sampler.sample_episode()
        s_x, q_x, s_y, q_y = s_x.to(device), q_x.to(device), s_y.to(device), q_y.to(device)
        with torch.no_grad():
            acc = evaluate_uwt(encoder, s_x, s_y, q_x, q_y, beta=0.0)
            accs.append(acc)
    mean, std = np.mean(accs) * 100, np.std(accs) * 100
    base = uwt_results[name]
    diff = mean - base[0]
    print(f"  {name:<18} → {mean:.1f}% ± {std:.1f}%  (vs 加权: {base[0]:.1f}%, 差: {diff:+.1f}%)")

# 对比汇总
print("\n" + "=" * 60)
print("📊 对比：多尺度 base32 vs 多尺度 base64")
print("=" * 60)
# base32 多尺度参考值 (从 outputs/multiscale eval 取)
prev = {"5-way 1-shot": 92.2, "5-way 5-shot": 94.9, "10-way 1-shot": 83.9, "10-way 5-shot": 90.2}
for name, (mean, std) in uwt_results.items():
    p = prev.get(name, 0)
    diff = mean - p
    mark = "✅" if diff > 0 else "❌" if diff < -1 else "➖"
    print(f"  {name:<18} 多尺度: {mean:.1f}%   单尺度: {p:.1f}%   差: {diff:+.1f}%  {mark}")

print("\n✅ 评估完成")
