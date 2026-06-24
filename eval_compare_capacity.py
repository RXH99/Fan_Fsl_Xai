"""
统一公平对比：base32 (1M) vs base64 (4M)
相同评估参数：1000 episodes, Cosine
用法: python eval_compare_capacity.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch, torch.nn.functional as F, numpy as np
from src.data.dataset import FaultDataset, EpisodicSampler
from src.models.encoder import create_encoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test = FaultDataset("data/processed/preprocessed.npz", split="test")

def eval_cosine(enc, sx, sy, qx, qy):
    se = F.normalize(enc(sx), dim=1)
    qe = F.normalize(enc(qx), dim=1)
    ways = len(torch.unique(sy))
    p = F.normalize(torch.stack([se[sy == c].mean(0) for c in range(ways)]), dim=1)
    _, pred = torch.max(torch.mm(qe, p.t()), 1)
    return (pred == qy).float().mean().item()

configs = [
    ("base32 (1M)", "outputs/clean_base32/fewshot_encoder_ProtoNet_Cosine.pth", 32),
    ("base64 (4M)", "outputs/clean_nopretrain/fewshot_encoder_ProtoNet_Cosine.pth", 64),
]

print("=" * 55)
print("📊 容量对比：base32 vs base64 (统一 1000 episodes)")
print("=" * 55)

all_results = {}
for label, ckpt_path, bf in configs:
    if not os.path.exists(ckpt_path):
        print(f"  ⚠️ {label}: 找不到 {ckpt_path}，跳过")
        continue

    encoder = create_encoder("resnet18", encoder_dim=128, use_se=True,
                             base_filters=bf, use_multiscale=False).to(device)
    sd = torch.load(ckpt_path, map_location=device)
    encoder.load_state_dict(sd)
    encoder.eval()
    print(f"\n--- {label} ---")

    results = {}
    for ways, shot, query, name in [(5,1,15,"5w1s"),(5,5,15,"5w5s"),
                                     (10,1,10,"10w1s"),(10,5,10,"10w5s")]:
        sp = EpisodicSampler(test, ways=ways, shot=shot, query=query)
        accs = []
        for _ in range(1000):
            sx,sy,qx,qy = sp.sample_episode()
            sx,qx,sy,qy = sx.to(device),qx.to(device),sy.to(device),qy.to(device)
            with torch.no_grad():
                accs.append(eval_cosine(encoder, sx, sy, qx, qy))
        m, s = np.mean(accs) * 100, np.std(accs) * 100
        results[name] = (m, s)
        print(f"  {name}: {m:.1f}% ± {s:.1f}%")
    all_results[label] = results

print("\n" + "=" * 55)
print("📈 对比汇总")
print("=" * 55)
names = ["5w1s", "5w5s", "10w1s", "10w5s"]
header = f"{'Setting':<12}"
for l in all_results: header += f" {l:>20}"
print(header)
print("-" * len(header))
for n in names:
    line = f"{n:<12}"
    vals = []
    for l in all_results:
        m, s = all_results[l][n]
        line += f" {m:>5.1f}% ± {s:.1f}%      "
        vals.append(m)
    if len(vals) == 2:
        line += f"  Δ={vals[1]-vals[0]:+.1f}%"
    print(line)
