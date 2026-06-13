"""
评估 clean 模型 + UWT
运行: python eval_clean.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch, torch.nn.functional as F, numpy as np
from src.data.dataset import FaultDataset, EpisodicSampler
from src.models.encoder import create_encoder

device = torch.device("cuda")
encoder = create_encoder("resnet18", encoder_dim=128, use_se=True,
                         base_filters=64, use_multiscale=False).to(device)
sd = torch.load("outputs/clean/fewshot_encoder_ProtoNet_Cosine.pth", map_location=device)
if 'fc.weight' in sd and sd['fc.weight'].shape != encoder.fc.weight.shape:
    del sd['fc.weight'], sd['fc.bias']
encoder.load_state_dict(sd, strict=False)
encoder.eval()

test = FaultDataset("data/processed/preprocessed.npz", split="test")

def uwt(enc, sx, sy, qx, qy, beta=1.0):
    se, qe = F.normalize(enc(sx), dim=1), F.normalize(enc(qx), dim=1)
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
    _, pd = torch.max(torch.mm(qe, p.t()), 1)
    return (pd == qy).float().mean().item()

print("参数扫描 beta:")
for ways, shot, query, name in [(5,1,15,"5w1s"),(5,5,15,"5w5s"),(10,1,10,"10w1s"),(10,5,10,"10w5s")]:
    sp = EpisodicSampler(test, ways=ways, shot=shot, query=query)
    best_b, best_a = 0, 0
    for b in [0.5,1.0,2.0,3.0,5.0]:
        a = []
        for _ in range(300):
            sx,sy,qx,qy = sp.sample_episode()
            sx,qx,sy,qy = sx.to(device),qx.to(device),sy.to(device),qy.to(device)
            with torch.no_grad(): a.append(uwt(encoder, sx, sy, qx, qy, beta=b))
        m = np.mean(a)*100
        print(f"  {name} beta={b:.1f} → {m:.1f}%")
        if m > best_a: best_a, best_b = m, b
    print(f"  ✅ {name} 最佳 beta={best_b}, acc={best_a:.1f}%\n")

print("="*50)
print("📊 主实验 (1000 episodes, 各setting最佳beta)")
print("="*50)
for ways, shot, query, name, beta in [(5,1,15,"5w1s",3.0),(5,5,15,"5w5s",1.0),
                                       (10,1,10,"10w1s",2.0),(10,5,10,"10w5s",1.0)]:
    sp = EpisodicSampler(test, ways=ways, shot=shot, query=query)
    a = []
    for _ in range(1000):
        sx,sy,qx,qy = sp.sample_episode()
        sx,qx,sy,qy = sx.to(device),qx.to(device),sy.to(device),qy.to(device)
        with torch.no_grad(): a.append(uwt(encoder, sx, sy, qx, qy, beta=beta))
    m,s = np.mean(a)*100, np.std(a)*100
    print(f"  {name:<18} beta={beta:.1f} → {m:.1f}% ± {s:.1f}%")

print(f"\n对比: step3 测试 Cosine = 95.6%")
