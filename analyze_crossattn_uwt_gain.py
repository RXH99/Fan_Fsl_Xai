"""
CrossAttn vs UWT 增益分析 - 完整版
基于已训练的 Full 模型权重进行推断变体评估
用法: python analyze_crossattn_uwt_gain.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn.functional as F
import numpy as np
from src.data.dataset import FaultDataset, EpisodicSampler
from src.models.encoder import create_encoder
from src.models.prototypical import CrossAttentionModule

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"设备: {device}\n")

# 加载测试集
test_dataset = FaultDataset("data/processed/preprocessed.npz", split="test")
print(f"加载数据 [test]: {len(test_dataset)} 样本\n")

def evaluate_mode(encoder, cross_attn, mode='uwt', beta=1.0, episodes=500):
    """评估不同推理模式"""
    ways, shot, query = 5, 5, 15
    sampler = EpisodicSampler(test_dataset, ways=ways, shot=shot, query=query)
    
    encoder.eval()
    if cross_attn is not None:
        cross_attn.eval()
    
    accs = []
    with torch.no_grad():
        for _ in range(episodes):
            sx, sy, qx, qy = sampler.sample_episode()
            sx, qx, sy, qy = sx.to(device), qx.to(device), sy.to(device), qy.to(device)
            
            se = F.normalize(encoder(sx), dim=1)
            qe = F.normalize(encoder(qx), dim=1)
            
            if mode == 'cosine':
                # Cosine + CrossAttn
                if cross_attn is not None:
                    qe = F.normalize(cross_attn(qe, se), dim=1)
            elif mode == 'cosine_only':
                # 纯 Cosine（无 CrossAttn）
                pass
            elif mode == 'uwt':
                # UWT
                if cross_attn is not None:
                    qe = F.normalize(cross_attn(qe, se), dim=1)
                
                ways_k = len(torch.unique(sy))
                p = F.normalize(torch.stack([se[sy == c].mean(0) for c in range(ways_k)]), dim=1)
                
                for _ in range(3):
                    sft = torch.softmax(torch.mm(qe, p.t()) / 0.3, dim=1)
                    wt = torch.exp(-(sft * torch.log(sft + 1e-8)).sum(1) / np.log(ways_k) * beta)
                    ws = sft * wt.unsqueeze(1)
                    np_ = []
                    for c in range(ways_k):
                        wsc = ws[:, c].sum()
                        np_.append((ws[:, c] @ qe) / wsc if wsc > 1e-8 else p[c])
                    p = F.normalize(0.8 * p + 0.2 * torch.stack(np_), dim=1)
                
                _, pred = torch.max(torch.mm(qe, p.t()), 1)
                accs.append((pred == qy).float().mean().item())
                continue
            
            # Cosine 分类
            ways_k = len(torch.unique(sy))
            pk = F.normalize(torch.stack([se[sy == c].mean(0) for c in range(ways_k)]), dim=1)
            _, pred = torch.max(torch.mm(qe, pk.t()), 1)
            accs.append((pred == qy).float().mean().item())
    
    return np.mean(accs) * 100, np.std(accs) * 100

# 加载 Full 模型权重（最后一个种子 seed=999）
encoder_path = "outputs/base64/fewshot_encoder_ProtoNet_CrossAttn.pth"
crossattn_path = "outputs/base64/crossattn_ProtoNet_CrossAttn.pth"

if not os.path.exists(encoder_path):
    print(f"❌ 找不到权重文件: {encoder_path}")
    sys.exit(1)

print(f"{'='*70}")
print(f"📌 加载 Full 模型 (Seed 999)")
print(f"{'='*70}")

encoder = create_encoder("resnet18", encoder_dim=128, use_se=True,
                         base_filters=64, use_multiscale=True).to(device)
cross_attn = CrossAttentionModule(d_model=128).to(device)

encoder.load_state_dict(torch.load(encoder_path, map_location=device))
cross_attn.load_state_dict(torch.load(crossattn_path, map_location=device))

print(f"✅ Encoder: {sum(p.numel() for p in encoder.parameters())/1e6:.2f}M params")
print(f"✅ CrossAttn: {sum(p.numel() for p in cross_attn.parameters())/1e3:.1f}K params\n")

# 评估三种模式
print(f"{'='*70}")
print(f"📊 推断变体评估（500 Episodes, 5-way 5-shot）")
print(f"{'='*70}")

cosine_ca_acc, cosine_ca_std = evaluate_mode(encoder, cross_attn, mode='cosine', episodes=500)
cosine_only_acc, cosine_only_std = evaluate_mode(encoder, cross_attn, mode='cosine_only', episodes=500)
uwt_ca_acc, uwt_ca_std = evaluate_mode(encoder, cross_attn, mode='uwt', beta=1.0, episodes=500)

print(f"\n  Cosine (有CrossAttn):     {cosine_ca_acc:.1f}% ± {cosine_ca_std:.1f}%")
print(f"  Cosine (无CrossAttn):     {cosine_only_acc:.1f}% ± {cosine_only_std:.1f}%")
print(f"  UWT (有CrossAttn):        {uwt_ca_acc:.1f}% ± {uwt_ca_std:.1f}%")

print(f"\n{'='*70}")
print("📈 增益分析")
print(f"{'='*70}")

crossattn_contribution = cosine_ca_acc - cosine_only_acc
uwt_gain_with_ca = uwt_ca_acc - cosine_ca_acc
uwt_gain_without_ca = uwt_ca_acc - cosine_only_acc

print(f"\n  → CrossAttn 贡献:         {crossattn_contribution:+.1f}%")
print(f"  → UWT 增益 (有CrossAttn):  {uwt_gain_with_ca:+.1f}%")
print(f"  → UWT 增益 (无CrossAttn):  {uwt_gain_without_ca:+.1f}%")

print(f"\n{'='*70}")
print("💡 结论")
print(f"{'='*70}")

if abs(uwt_gain_with_ca) < 0.5:
    print(f"  ✅ CrossAttn 与 UWT 功能冗余（UWT 增益 < 0.5%）")
    print(f"  ✅ 选择 UWT 的理由：零参数、不改变训练、即插即用")
else:
    print(f"  ⚠️ UWT 在有 CrossAttn 时仍有显著增益 ({uwt_gain_with_ca:+.1f}%)")

# 与 Clean 模型对比
print(f"\n{'='*70}")
print("📊 与 Clean 模型（无 CrossAttn）对比")
print(f"{'='*70}")
print(f"  Clean + UWT:              97.1% ± 0.4%  (来自 eval_clean.py)")
print(f"  Full + UWT:               {uwt_ca_acc:.1f}% ± {uwt_ca_std:.1f}%")
print(f"  → 差异:                     {97.1 - uwt_ca_acc:+.1f}%")

print(f"\n[OK] 分析完成！")
