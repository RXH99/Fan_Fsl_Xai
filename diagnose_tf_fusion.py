"""
检查训练好的时频融合模型：门控值、时域/频域单独表现
运行: python diagnose_tf_fusion.py
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import yaml
import numpy as np
from src.models.encoder import create_encoder
from src.data.dataset import FaultDataset, EpisodicSampler
from src.models.prototypical import prototypical_loss


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("configs/tf_fusion.yaml", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

# 创建编码器并加载权重
encoder = create_encoder(
    "time_frequency", encoder_dim=64, use_se=True,
    freeze_time_encoder=False,
    freq_branch_config=cfg["model"]["freq_branch"],
).to(device)

ckpt = torch.load("outputs/tf_fusion/fewshot_encoder_ProtoNet_Cosine.pth",
                  map_location=device)
encoder.load_state_dict(ckpt, strict=False)
encoder.eval()

# 1. 门控值
gate = torch.sigmoid(encoder.gate_logit).item()
print(f"门控值 (时域权重): {gate:.4f}")
print(f"  → 如果接近 1.0: 模型忽略频域，频域没贡献")
print(f"  → 如果接近 0.5: 融合没收敛，时域/频域互相干扰")
print(f"  → 如果接近 0.0: 模型依赖频域（不太可能）")

# 2. 单独测试时域分支
test_dataset = FaultDataset("data/processed/preprocessed.npz", split="test")

# 创建一个只包含 time_encoder 的包装器，模拟原始 ResNet
class TimeOnlyWrapper(torch.nn.Module):
    def __init__(self, time_encoder):
        super().__init__()
        self.time_encoder = time_encoder
    def forward(self, x):
        return self.time_encoder(x)

# 创建只含频域分支的包装器
class FreqOnlyWrapper(torch.nn.Module):
    def __init__(self, freq_branch):
        super().__init__()
        self.freq_branch = freq_branch
    def forward(self, x):
        return self.freq_branch(x)

time_only = TimeOnlyWrapper(encoder.time_encoder).to(device)
freq_only = FreqOnlyWrapper(encoder.freq_branch).to(device)

test_configs = [
    (5, 1, 15, "5-way 1-shot"),
    (5, 5, 15, "5-way 5-shot"),
    (10, 1, 10, "10-way 1-shot"),
    (10, 5, 10, "10-way 5-shot"),
]

print(f"\n{'='*60}")
print("时域分支 (单独):")
for ways, shot, query, name in test_configs:
    sampler = EpisodicSampler(test_dataset, ways=ways, shot=shot, query=query)
    accs = []
    with torch.no_grad():
        for _ in range(500):
            s_x, s_y, q_x, q_y = sampler.sample_episode()
            s_x, q_x, s_y, q_y = s_x.to(device), q_x.to(device), s_y.to(device), q_y.to(device)
            _, acc = prototypical_loss(time_only, s_x, s_y, q_x, q_y, device, method='cosine')
            accs.append(acc)
    print(f"  {name:<18} → {np.mean(accs)*100:.1f}% ± {np.std(accs)*100:.1f}%")

print(f"\n频域分支 (单独):")
for ways, shot, query, name in test_configs:
    sampler = EpisodicSampler(test_dataset, ways=ways, shot=shot, query=query)
    accs = []
    with torch.no_grad():
        for _ in range(500):
            s_x, s_y, q_x, q_y = sampler.sample_episode()
            s_x, q_x, s_y, q_y = s_x.to(device), q_x.to(device), s_y.to(device), q_y.to(device)
            _, acc = prototypical_loss(freq_only, s_x, s_y, q_x, q_y, device, method='cosine')
            accs.append(acc)
    print(f"  {name:<18} → {np.mean(accs)*100:.1f}% ± {np.std(accs)*100:.1f}%")

print(f"\n时频融合 (完整):")
for ways, shot, query, name in test_configs:
    sampler = EpisodicSampler(test_dataset, ways=ways, shot=shot, query=query)
    accs = []
    with torch.no_grad():
        for _ in range(500):
            s_x, s_y, q_x, q_y = sampler.sample_episode()
            s_x, q_x, s_y, q_y = s_x.to(device), q_x.to(device), s_y.to(device), q_y.to(device)
            _, acc = prototypical_loss(encoder, s_x, s_y, q_x, q_y, device, method='cosine')
            accs.append(acc)
    print(f"  {name:<18} → {np.mean(accs)*100:.1f}% ± {np.std(accs)*100:.1f}%")
