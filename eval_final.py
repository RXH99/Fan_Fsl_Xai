"""
终极评估：直推式推理 + 测试时增强集成

直接从已有模型评估，不训练。
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import yaml
import numpy as np
from src.data.dataset import FaultDataset, EpisodicSampler
from src.models.encoder import create_encoder
from src.models.prototypical import prototypical_loss
from src.data.augmentation import augment_vibration_batch

device = torch.device('cuda')
with open('configs/optimized.yaml', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)

test_dataset = FaultDataset('data/processed/preprocessed.npz', split='test')
encoder = create_encoder('resnet18', encoder_dim=64, use_se=True).to(device)
encoder.load_state_dict(torch.load('outputs/fewshot_encoder_ProtoNet_Cosine.pth'))
encoder.eval()

for ways, shot, query, name in [(5, 1, 15, '5-way 1-shot'),
                                 (5, 5, 15, '5-way 5-shot'),
                                 (10, 1, 10, '10-way 1-shot'),
                                 (10, 5, 10, '10-way 5-shot')]:
    sampler = EpisodicSampler(test_dataset, ways=ways, shot=shot, query=query)

    # === 1. 直推式推理 ===
    print(f'\n【{name}】直推式推理...')
    trans_accs = []
    for _ in range(500):
        s_x, s_y, q_x, q_y = sampler.sample_episode()
        _, acc = prototypical_loss(
            encoder, s_x.to(device), s_y.to(device),
            q_x.to(device), q_y.to(device), device,
            method='transductive', num_steps=10, tau=0.5, mix_ratio=0.7)
        trans_accs.append(acc)
    print(f'  直推式: {np.mean(trans_accs)*100:.1f}% ± {np.std(trans_accs)*100:.1f}%')

    # === 2. 直推式 + 测试时增强集成 ===
    print(f'  直推式 + 5次增强集成...')
    ensemble_accs = []
    for _ in range(500):
        s_x, s_y, q_x, q_y = sampler.sample_episode()
        s_x, q_x = s_x.to(device), q_x.to(device)
        s_y, q_y = s_y.to(device), q_y.to(device)

        # 对 query 做 5 种不同增强
        all_q = [q_x]
        for _ in range(4):
            all_q.append(augment_vibration_batch(
                q_x, noise_std=0.02, mask_ratio=0.1, scale_std=0.03))

        # 每个增强版本分别做直推式推理
        all_preds = []
        with torch.no_grad():
            for q_aug in all_q:
                s_emb = encoder(s_x)
                q_emb = encoder(q_aug)

                s_emb = s_emb / s_emb.norm(dim=1, keepdim=True)
                q_emb = q_emb / q_emb.norm(dim=1, keepdim=True)

                prototypes = torch.stack([
                    s_emb[s_y == cls].mean(0) for cls in range(ways)
                ])
                prototypes = prototypes / prototypes.norm(dim=1, keepdim=True)

                # 直推式 5 步
                for step in range(5):
                    sims = torch.mm(q_emb, prototypes.t()) / 0.5
                    soft = torch.softmax(sims, dim=1)
                    new_protos = []
                    for w in range(ways):
                        qp = (soft[:, w] @ q_emb) / (soft[:, w].sum() + 1e-8)
                        new_protos.append(qp)
                    new_protos = torch.stack(new_protos)
                    new_protos = new_protos / new_protos.norm(dim=1, keepdim=True)
                    prototypes = 0.7 * prototypes + 0.3 * new_protos
                    prototypes = prototypes / prototypes.norm(dim=1, keepdim=True)

                final_sim = torch.mm(q_emb, prototypes.t())
                all_preds.append(final_sim)

        # 对 5 个版本取平均
        avg_sim = torch.stack(all_preds).mean(0)
        preds = torch.argmax(avg_sim, dim=1)
        acc = (preds == q_y).float().mean().item()
        ensemble_accs.append(acc)

    print(f'  ✅ 集成直推式: {np.mean(ensemble_accs)*100:.1f}% ± {np.std(ensemble_accs)*100:.1f}%')
