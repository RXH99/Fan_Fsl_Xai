"""
最终 3 轮实验：base64 + CrossAttn V1 + UWT
输出均值±标准差，论文可用
运行: python run_final.py
"""

import os, sys, subprocess, time, json
from datetime import datetime

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import yaml
import numpy as np
from src.data.dataset import FaultDataset, EpisodicSampler
from src.models.encoder import create_encoder
from src.models.prototypical import CrossAttentionModule


# ===== UWT 评估函数（同 eval_final.py） =====
def evaluate_uwt(encoder, cross_attn, support_x, support_y, query_x, query_y,
                 num_steps=3, tau=0.3, mix_ratio=0.8, beta=1.0):
    import torch.nn.functional as F
    s_emb = encoder(support_x)
    q_emb = encoder(query_x)
    # 先归一化再跨注意力（跟训练一致）
    s_emb = F.normalize(s_emb, dim=1)
    q_emb = F.normalize(q_emb, dim=1)
    if cross_attn is not None:
        q_emb = cross_attn(q_emb, s_emb)
        q_emb = F.normalize(q_emb, dim=1)
    ways = len(torch.unique(support_y))
    prototypes = torch.stack([s_emb[support_y == cls].mean(0) for cls in range(ways)])
    prototypes = F.normalize(prototypes, dim=1)
    for _ in range(num_steps):
        sims = torch.mm(q_emb, prototypes.t())
        soft = torch.softmax(sims / tau, dim=1)
        entropy = -(soft * torch.log(soft + 1e-8)).sum(dim=1)
        weight = torch.exp(-entropy / np.log(ways) * beta)
        weighted_soft = soft * weight.unsqueeze(1)
        new_protos = []
        for w in range(ways):
            ws = weighted_soft[:, w].sum()
            qp = (weighted_soft[:, w] @ q_emb) / ws if ws > 1e-8 else prototypes[w]
            new_protos.append(qp)
        new_protos = torch.stack(new_protos)
        new_protos = F.normalize(new_protos, dim=1)
        prototypes = F.normalize(mix_ratio * prototypes + (1 - mix_ratio) * new_protos, dim=1)
    final_sims = torch.mm(q_emb, prototypes.t())
    preds = torch.argmax(final_sims, dim=1)
    return (preds == query_y).float().mean().item()


# ===== 训练函数（直接调用 step3） =====
def train_round(round_idx, seed):
    print(f"\n{'='*60}")
    print(f"🏃 Round {round_idx}/3 (seed={seed})")
    print(f"{'='*60}")

    # 设置随机种子（一定程度上控制随机性）
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 直接执行 step3 训练（实时输出）
    result = subprocess.run([
        "python", "step3_train_fewshot.py",
        "--config", "configs/base64.yaml",
        "--method", "ProtoNet_CrossAttn",
    ])

    if result.returncode != 0:
        print(f"  [FAIL] 训练失败 (exit code={result.returncode})")
        return None

    return True


# ===== 评估函数 =====
def evaluate_round(round_idx):
    device = torch.device("cuda")
    test_dataset = FaultDataset("data/processed/preprocessed.npz", split="test")

    encoder = create_encoder("resnet18", encoder_dim=128, use_se=True,
                         base_filters=64).to(device)
    encoder.load_state_dict(torch.load("outputs/base64/fewshot_encoder_ProtoNet_CrossAttn.pth",
                                        map_location=device))
    encoder.eval()

    cross_attn = CrossAttentionModule(d_model=128).to(device)
    cross_attn.load_state_dict(torch.load("outputs/base64/crossattn_ProtoNet_CrossAttn.pth",
                                           map_location=device))
    cross_attn.eval()

    configs = [
        (5, 1, 15, "5-way 1-shot", 3.0),
        (5, 5, 15, "5-way 5-shot", 1.0),
        (10, 1, 10, "10-way 1-shot", 0.5),
        (10, 5, 10, "10-way 5-shot", 1.0),
    ]

    results = {}
    for ways, shot, query, name, beta in configs:
        sampler = EpisodicSampler(test_dataset, ways=ways, shot=shot, query=query)
        accs = []
        with torch.no_grad():
            for _ in range(1000):
                s_x, s_y, q_x, q_y = sampler.sample_episode()
                s_x, q_x, s_y, q_y = s_x.to(device), q_x.to(device), s_y.to(device), q_y.to(device)
                acc = evaluate_uwt(encoder, cross_attn, s_x, s_y, q_x, q_y, beta=beta)
                accs.append(acc)
        results[name] = (np.mean(accs) * 100, np.std(accs) * 100)

    return results


# ===== 主流程 =====
seeds = [42, 123, 999]
all_results = {s: [] for s in ["5-way 1-shot", "5-way 5-shot",
                                 "10-way 1-shot", "10-way 5-shot"]}

for i in range(3):
    train_round(i + 1, seeds[i])
    results = evaluate_round(i + 1)
    print(f"\n  Round {i+1} 结果:")
    for name, (mean, std) in results.items():
        all_results[name].append(mean)
        print(f"    {name:<18} → {mean:.1f}% ± {std:.1f}%")

# ===== 汇总 =====
print(f"\n\n{'='*70}")
print("📊 最终 3 轮汇总 — CrossAttn V1 + UWT")
print(f"{'='*70}")

for s in ["5-way 1-shot", "5-way 5-shot", "10-way 1-shot", "10-way 5-shot"]:
    accs = all_results[s]
    mean = np.mean(accs)
    std = np.std(accs)
    print(f"  {s:<18} → {mean:.1f}% ± {std:.1f}%  (3 runs)")

# 保存
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
save_path = f"outputs/base64/final_results_{timestamp}.txt"
with open(save_path, "w") as f:
    f.write(f"Fan_Fsl_Xai 最终结果 ({timestamp})\n")
    f.write("base64 + 多尺度聚合 + CrossAttn V1 + UWT\n")
    f.write("3 轮随机种子: {seeds}\n\n")
    for s in ["5-way 1-shot", "5-way 5-shot", "10-way 1-shot", "10-way 5-shot"]:
        accs = all_results[s]
        f.write(f"{s}: {np.mean(accs):.1f}% ± {np.std(accs):.1f}%  (rounds: {[f'{a:.1f}' for a in accs]})\n")

print(f"\n✅ 已保存: {save_path}")
print(f"{'='*70}")
print("🏆 论文可用的最终数据")
print(f"{'='*70}")
