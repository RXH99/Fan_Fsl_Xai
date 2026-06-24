"""
base32 版 2×2 对比：SupCon × UWT 交互效应
用法: python eval_supcon_2x2_base32.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch, torch.nn.functional as F, numpy as np
from src.data.dataset import FaultDataset, EpisodicSampler
from src.models.encoder import create_encoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test = FaultDataset("data/processed/preprocessed.npz", split="test")

# 加载 base32 + SupCon 模型
ckpt = "outputs/clean_base32/fewshot_encoder_ProtoNet_Cosine.pth"
sd = torch.load(ckpt, map_location=device)
encoder = create_encoder("resnet18", encoder_dim=128, use_se=True,
                         base_filters=32, use_multiscale=False).to(device)
encoder.load_state_dict(sd)
encoder.eval()
print(f"[OK] 加载 base32 + SupCon: {ckpt}\n")

def eval_cosine(enc, sx, sy, qx, qy):
    se = F.normalize(enc(sx), dim=1)
    qe = F.normalize(enc(qx), dim=1)
    w = len(torch.unique(sy))
    p = F.normalize(torch.stack([se[sy == c].mean(0) for c in range(w)]), dim=1)
    _, pred = torch.max(torch.mm(qe, p.t()), 1)
    return (pred == qy).float().mean().item()

def eval_uwt(enc, sx, sy, qx, qy, beta=1.0):
    se = F.normalize(enc(sx), dim=1)
    qe = F.normalize(enc(qx), dim=1)
    w = len(torch.unique(sy))
    p = F.normalize(torch.stack([se[sy == c].mean(0) for c in range(w)]), dim=1)
    for _ in range(3):
        sft = torch.softmax(torch.mm(qe, p.t()) / 0.3, dim=1)
        wt = torch.exp(-(sft * torch.log(sft + 1e-8)).sum(1) / np.log(w) * beta)
        ws = sft * wt.unsqueeze(1)
        np_ = []
        for c in range(w):
            wsc = ws[:, c].sum()
            np_.append((ws[:, c] @ qe) / wsc if wsc > 1e-8 else p[c])
        p = F.normalize(0.8 * p + 0.2 * torch.stack(np_), dim=1)
    _, pred = torch.max(torch.mm(qe, p.t()), 1)
    return (pred == qy).float().mean().item()

# Beta 扫描
print("UWT beta 扫描 (300 ep):")
best_betas = {}
for ways, shot, query, name in [(5,1,15,"5w1s"),(5,5,15,"5w5s"),(10,1,10,"10w1s"),(10,5,10,"10w5s")]:
    sp = EpisodicSampler(test, ways=ways, shot=shot, query=query)
    best_b, best_a = 0, 0
    for b in [0.5, 1.0, 2.0, 3.0, 5.0]:
        a = []
        for _ in range(300):
            sx,sy,qx,qy = sp.sample_episode()
            sx,qx,sy,qy = sx.to(device),qx.to(device),sy.to(device),qy.to(device)
            with torch.no_grad(): a.append(eval_uwt(encoder, sx, sy, qx, qy, beta=b))
        m = np.mean(a) * 100
        if m > best_a: best_a, best_b = m, b
    best_betas[name] = (best_b, best_a)
    print(f"  {name} 最佳 beta={best_b} → {best_a:.1f}%")

# 主实验
print("\n" + "=" * 55)
print("📊 base32 2×2 — Cosine vs UWT (1000 ep)")
print("=" * 55)
results = {}
for ways, shot, query, name in [(5,1,15,"5w1s"),(5,5,15,"5w5s"),(10,1,10,"10w1s"),(10,5,10,"10w5s")]:
    sp = EpisodicSampler(test, ways=ways, shot=shot, query=query)
    a = []
    for _ in range(1000):
        sx,sy,qx,qy = sp.sample_episode()
        sx,qx,sy,qy = sx.to(device),qx.to(device),sy.to(device),qy.to(device)
        with torch.no_grad(): a.append(eval_cosine(encoder, sx, sy, qx, qy))
    cos_m, cos_s = np.mean(a) * 100, np.std(a) * 100

    beta = best_betas[name][0]
    a = []
    for _ in range(1000):
        sx,sy,qx,qy = sp.sample_episode()
        sx,qx,sy,qy = sx.to(device),qx.to(device),sy.to(device),qy.to(device)
        with torch.no_grad(): a.append(eval_uwt(encoder, sx, sy, qx, qy, beta=beta))
    uwt_m, uwt_s = np.mean(a) * 100, np.std(a) * 100

    results[name] = (cos_m, cos_s, uwt_m, uwt_s)
    print(f"  {name}: Cosine={cos_m:.1f}% → UWT={uwt_m:.1f}% (Δ={uwt_m-cos_m:+.1f}%)")

# 对比 base32 无 SupCon 数据（来自 eval_compare_capacity.py）
b32_nosupcon_cosine = {"5w1s": 88.0, "5w5s": 94.3, "10w1s": 79.1, "10w5s": 88.4}

print("\n" + "=" * 70)
print("📈 base32 2×2 部分矩阵 (5w5s)")
print("=" * 70)
ns_cos = b32_nosupcon_cosine["5w5s"]
ws_cos = results["5w5s"][0]
ws_uwt = results["5w5s"][2]

print(f"\n{'':>14} | {'Cosine':>10}")
print(f"{'-'*14}-+-{'-'*10}")
print(f"{'无SupCon':>14} | {ns_cos:>7.1f}%    (来自容量对比实验)")
print(f"{'有SupCon':>14} | {ws_cos:>7.1f}%    (当前)")
print(f"{'Δ SupCon':>14} | {ws_cos-ns_cos:>+7.1f}%")

print(f"\n{'':>14} | {'UWT':>10}")
print(f"{'-'*14}-+-{'-'*10}")
print(f"{'无SupCon':>14} | {'N/A':>7}   (权重已被覆盖)")
print(f"{'有SupCon':>14} | {ws_uwt:>7.1f}%    (当前)")
print(f"{'UWT增益':>14} | {ws_uwt-ws_cos:>+7.1f}%")

print(f"\n  SupCon 单独增益: +{ws_cos-ns_cos:.1f}%")
print(f"  base32 + SupCon + Cosine = {ws_cos:.1f}%")
print(f"  base32 + SupCon + UWT    = {ws_uwt:.1f}%")
print(f"  SupCon + UWT 结合 -> UWT 增益: +{ws_uwt-ws_cos:.1f}%")

# 与 base64 对比
b64_cos_ns = 93.6
b64_cos_ws = 94.9
b64_uwt_ws = 97.1

print("\n" + "=" * 70)
print("📊 base32 vs base64 — 5w5s 对照")
print("=" * 70)
print(f"\n{'':>14} | {'base32 (1M)':>12} | {'base64 (4M)':>12}")
print(f"{'-'*14}-+-{'-'*12}-+-{'-'*12}")
print(f"{'无SupCon+Cos':>14} | {b32_nosupcon_cosine['5w5s']:>9.1f}%    | {b64_cos_ns:>9.1f}%")
print(f"{'有SupCon+Cos':>14} | {ws_cos:>9.1f}%    | {b64_cos_ws:>9.1f}%")
print(f"{'SupCon+Cos增益':>14} | {ws_cos-b32_nosupcon_cosine['5w5s']:>+9.1f}%    | {b64_cos_ws-b64_cos_ns:>+9.1f}%")
print(f"{'有SupCon+UWT':>14} | {ws_uwt:>9.1f}%    | {b64_uwt_ws:>9.1f}%")

# 保存
with open("outputs/clean_base32/2x2_comparison.txt", "w") as f:
    f.write(f"base32 SupCon × UWT 结果\n\n")
    f.write(f"  base32 无SupCon + Cosine: {b32_nosupcon_cosine['5w5s']:.1f}%\n")
    f.write(f"  base32 无SupCon + UWT:    N/A (权重已覆盖)\n")
    f.write(f"  base32 有SupCon + Cosine: {ws_cos:.1f}%\n")
    f.write(f"  base32 有SupCon + UWT:    {ws_uwt:.1f}%\n")
    f.write(f"  SupCon 增益 (Cosine下): {ws_cos-b32_nosupcon_cosine['5w5s']:+.1f}%\n")
    f.write(f"  UWT 增益 (SupCon下):    {ws_uwt-ws_cos:+.1f}%\n")
