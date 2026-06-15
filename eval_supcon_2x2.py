"""
2×2 对比实验：评估 SupCon × UWT 交互效应
用法: python eval_supcon_2x2.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch, torch.nn.functional as F, numpy as np
from src.data.dataset import FaultDataset, EpisodicSampler
from src.models.encoder import create_encoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"设备: {device}")

# ===== 1. 加载 无 SupCon 模型 =====
ckpt = "outputs/clean_nopretrain/fewshot_encoder_ProtoNet_Cosine.pth"
if not os.path.exists(ckpt):
    print(f"[FAIL] 未找到 {ckpt}，请先运行训练: python step3_train_fewshot.py --config configs/clean_nopretrain.yaml --method ProtoNet_Cosine --no_pretrain")
    sys.exit(1)
sd = torch.load(ckpt, map_location=device)
fc_shape = sd.get('fc.weight', torch.zeros(0)).shape
use_ms = len(fc_shape) == 2 and fc_shape[1] == 960
encoder = create_encoder("resnet18", encoder_dim=128, use_se=True,
                         base_filters=64, use_multiscale=use_ms).to(device)
encoder.load_state_dict(sd)
encoder.eval()
print(f"[OK] 加载权重: {ckpt}")

# ===== 2. 准备测试集 =====
test = FaultDataset("data/processed/preprocessed.npz", split="test")

def eval_cosine(enc, sx, sy, qx, qy):
    se = F.normalize(enc(sx), dim=1)
    qe = F.normalize(enc(qx), dim=1)
    ways = len(torch.unique(sy))
    p = F.normalize(torch.stack([se[sy == c].mean(0) for c in range(ways)]), dim=1)
    _, pred = torch.max(torch.mm(qe, p.t()), 1)
    return (pred == qy).float().mean().item()

def eval_uwt(enc, sx, sy, qx, qy, beta=1.0, steps=3, tau=0.3, mix_ratio=0.8):
    se = F.normalize(enc(sx), dim=1)
    qe = F.normalize(enc(qx), dim=1)
    ways = len(torch.unique(sy))
    p = F.normalize(torch.stack([se[sy == c].mean(0) for c in range(ways)]), dim=1)
    for _ in range(steps):
        sft = torch.softmax(torch.mm(qe, p.t()) / tau, dim=1)
        wt = torch.exp(-(sft * torch.log(sft + 1e-8)).sum(1) / np.log(ways) * beta)
        ws = sft * wt.unsqueeze(1)
        np_ = []
        for c in range(ways):
            wsc = ws[:, c].sum()
            np_.append((ws[:, c] @ qe) / wsc if wsc > 1e-8 else p[c])
        p = F.normalize(mix_ratio * p + (1 - mix_ratio) * torch.stack(np_), dim=1)
    _, pred = torch.max(torch.mm(qe, p.t()), 1)
    return (pred == qy).float().mean().item()

# ===== 3. 参数扫描（UWT beta）=====
print("\n" + "=" * 50)
print("UWT 参数扫描 (300 ep)")
print("=" * 50)
best_betas = {}
for ways, shot, query, name in [(5,1,15,"5w1s"),(5,5,15,"5w5s"),
                                 (10,1,10,"10w1s"),(10,5,10,"10w5s")]:
    sp = EpisodicSampler(test, ways=ways, shot=shot, query=query)
    best_b, best_a = 0, 0
    for b in [0.5, 1.0, 2.0, 3.0, 5.0]:
        a = []
        for _ in range(300):
            sx,sy,qx,qy = sp.sample_episode()
            sx,qx,sy,qy = sx.to(device),qx.to(device),sy.to(device),qy.to(device)
            with torch.no_grad():
                a.append(eval_uwt(encoder, sx, sy, qx, qy, beta=b))
        m = np.mean(a) * 100
        print(f"  {name} beta={b:.1f} → {m:.1f}%")
        if m > best_a:
            best_a, best_b = m, b
    best_betas[name] = (best_b, best_a)
    print(f"  ✅ {name} 最佳 beta={best_b} → {best_a:.1f}%\n")

# ===== 4. 主实验：Cosine vs UWT =====
print("\n" + "=" * 50)
print("📊 2×2 对比矩阵 — 无 SupCon 模型")
print("=" * 50)
results = {}
for ways, shot, query, name in [(5,1,15,"5w1s"),(5,5,15,"5w5s"),
                                 (10,1,10,"10w1s"),(10,5,10,"10w5s")]:
    sp = EpisodicSampler(test, ways=ways, shot=shot, query=query)

    # Cosine
    a = []
    for _ in range(1000):
        sx,sy,qx,qy = sp.sample_episode()
        sx,qx,sy,qy = sx.to(device),qx.to(device),sy.to(device),qy.to(device)
        with torch.no_grad(): a.append(eval_cosine(encoder, sx, sy, qx, qy))
    cos_m, cos_s = np.mean(a) * 100, np.std(a) * 100

    # UWT
    beta = best_betas[name][0]
    a = []
    for _ in range(1000):
        sx,sy,qx,qy = sp.sample_episode()
        sx,qx,sy,qy = sx.to(device),qx.to(device),sy.to(device),qy.to(device)
        with torch.no_grad(): a.append(eval_uwt(encoder, sx, sy, qx, qy, beta=beta))
    uwt_m, uwt_s = np.mean(a) * 100, np.std(a) * 100

    results[name] = (cos_m, cos_s, uwt_m, uwt_s)
    print(f"  {name}: Cosine={cos_m:.1f}%±{cos_s:.1f}% → UWT={uwt_m:.1f}%±{uwt_s:.1f}%  (Δ={uwt_m-cos_m:+.1f}%)")

# ===== 5. 与有 SupCon 的结果对比对照表 =====
print("\n" + "=" * 70)
print("📈 完整 2×2 矩阵 — 5w5s")
print("=" * 70)
# 无 SupCon 结果
ns_cos = results["5w5s"][0]
ns_uwt = results["5w5s"][2]
# 有 SupCon 结果（来自已有数据）
ws_cos = 94.9  # 3 种子均值
ws_uwt = 97.1  # 3 种子均值

print(f"\n{'':>12} | {'Cosine':>8} | {'UWT':>8} | {'Δ UWT':>8}")
print(f"{'-'*12}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")
print(f"{'无SupCon':>12} | {ns_cos:>7.1f}% | {ns_uwt:>7.1f}% | {ns_uwt-ns_cos:>+7.1f}%")
print(f"{'有SupCon':>12} | {ws_cos:>7.1f}% | {ws_uwt:>7.1f}% | {ws_uwt-ws_cos:>+7.1f}%")
print(f"{'-'*12}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")
print(f"{'Δ SupCon':>12} | {ws_cos-ns_cos:>+7.1f}% | {ws_uwt-ns_uwt:>+7.1f}% |")

# 判断交互效应
supcon_gain_cos = ws_cos - ns_cos
supcon_gain_uwt = ws_uwt - ns_uwt
interaction = supcon_gain_uwt - supcon_gain_cos

print(f"\n📌 分析:")
print(f"  SupCon 增益 (Cosine): +{supcon_gain_cos:.1f}%")
print(f"  SupCon 增益 (UWT):    +{supcon_gain_uwt:.1f}%")
if abs(interaction) < 0.5:
    print(f"  → 交互效应: {interaction:+.1f}% (几乎可加，各自独立)")
elif interaction > 0.5:
    print(f"  → 交互效应: {interaction:+.1f}% (协同放大，SupCon+UWT > 各自单独)")
else:
    print(f"  → 交互效应: {interaction:+.1f}% (替代效应，有部分冗余)")

# 保存结果
out = os.path.dirname(ckpt)
sp = os.path.join(out, "2x2_comparison.txt")
with open(sp, "w") as f:
    f.write(f"2×2 对比矩阵 — 无 SupCon 模型\n\n")
    for name, (cm, cs, um, us) in results.items():
        f.write(f"  {name}: Cosine={cm:.1f}%±{cs:.1f}% → UWT={um:.1f}%±{us:.1f}% (Δ={um-cm:+.1f}%)\n")
    f.write(f"\n对照 有SupCon 数据:\n")
    f.write(f"  5w5s 无SupCon Cosine={ns_cos:.1f}%  UWT={ns_uwt:.1f}%\n")
    f.write(f"  5w5s 有SupCon Cosine={ws_cos:.1f}%  UWT={ws_uwt:.1f}%\n")
    f.write(f"  SupCon增益(Cosine)={ws_cos-ns_cos:+.1f}%\n")
    f.write(f"  SupCon增益(UWT)={ws_uwt-ns_uwt:+.1f}%\n")
print(f"\n[OK] 保存: {sp}")
