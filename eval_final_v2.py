"""
终极评估 v2：最佳直推式参数 + 测试时增强集成
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import yaml
import numpy as np
from src.data.dataset import FaultDataset, EpisodicSampler
from src.models.encoder import create_encoder
from src.data.augmentation import augment_vibration_batch

device = torch.device('cuda')
with open('configs/optimized.yaml', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)

test_dataset = FaultDataset('data/processed/preprocessed.npz', split='test')
encoder = create_encoder('resnet18', encoder_dim=64, use_se=True).to(device)
encoder.load_state_dict(torch.load('outputs/fewshot_encoder_ProtoNet_Cosine.pth'))
encoder.eval()

# === 最佳参数 ===
BEST_PARAMS = {'num_steps': 3, 'tau': 0.3, 'mix_ratio': 0.8}

for ways, shot, query, name in [(5, 1, 15, '5-way 1-shot'),
                                 (5, 5, 15, '5-way 5-shot')]:

    sampler = EpisodicSampler(test_dataset, ways=ways, shot=shot, query=query)

    # 直推式推理（最佳参数）
    print(f'\n【{name}】直推式 (best params)...')
    accs = []
    for _ in range(1000):
        s_x, s_y, q_x, q_y = sampler.sample_episode()
        s_x, q_x = s_x.to(device), q_x.to(device)
        s_y, q_y = s_y.to(device), q_y.to(device)
        with torch.no_grad():
            s_emb = encoder(s_x)
            q_emb = encoder(q_x)
            s_emb = s_emb / s_emb.norm(dim=1, keepdim=True)
            q_emb = q_emb / q_emb.norm(dim=1, keepdim=True)
            prototypes = torch.stack([s_emb[s_y == cls].mean(0) for cls in range(ways)])
            prototypes = prototypes / prototypes.norm(dim=1, keepdim=True)
            for _ in range(BEST_PARAMS['num_steps']):
                sims = torch.mm(q_emb, prototypes.t()) / BEST_PARAMS['tau']
                soft = torch.softmax(sims, dim=1)
                new_protos = []
                for w in range(ways):
                    qp = (soft[:, w] @ q_emb) / (soft[:, w].sum() + 1e-8)
                    new_protos.append(qp)
                new_protos = torch.stack(new_protos)
                new_protos = new_protos / new_protos.norm(dim=1, keepdim=True)
                prototypes = BEST_PARAMS['mix_ratio'] * prototypes + (1 - BEST_PARAMS['mix_ratio']) * new_protos
                prototypes = prototypes / prototypes.norm(dim=1, keepdim=True)
            final_sim = torch.mm(q_emb, prototypes.t())
            preds = torch.argmax(final_sim, dim=1)
            accs.append((preds == q_y).float().mean().item())
    mean = np.mean(accs) * 100
    std = np.std(accs) * 100
    print(f'  直推式: {mean:.1f}% ± {std:.1f}%')

    # 直推式 + 10次增强集成
    print(f'  直推式 + 10次增强集成...')
    ensemble_accs = []
    n_aug = 10
    for _ in range(1000):
        s_x, s_y, q_x, q_y = sampler.sample_episode()
        s_x, q_x = s_x.to(device), q_x.to(device)
        s_y, q_y = s_y.to(device), q_y.to(device)

        with torch.no_grad():
            s_emb = encoder(s_x)
            s_emb = s_emb / s_emb.norm(dim=1, keepdim=True)
            s_proto = torch.stack([s_emb[s_y == cls].mean(0) for cls in range(ways)])
            s_proto = s_proto / s_proto.norm(dim=1, keepdim=True)

            all_logits = []
            for i in range(n_aug):
                if i == 0:
                    q_aug = q_x
                else:
                    q_aug = augment_vibration_batch(
                        q_x, noise_std=0.02, mask_ratio=0.1, scale_std=0.03)

                q_emb = encoder(q_aug)
                q_emb = q_emb / q_emb.norm(dim=1, keepdim=True)

                prototypes = s_proto.clone()
                for _ in range(BEST_PARAMS['num_steps']):
                    sims = torch.mm(q_emb, prototypes.t()) / BEST_PARAMS['tau']
                    soft = torch.softmax(sims, dim=1)
                    new_protos = []
                    for w in range(ways):
                        qp = (soft[:, w] @ q_emb) / (soft[:, w].sum() + 1e-8)
                        new_protos.append(qp)
                    new_protos = torch.stack(new_protos)
                    new_protos = new_protos / new_protos.norm(dim=1, keepdim=True)
                    prototypes = BEST_PARAMS['mix_ratio'] * prototypes + (1 - BEST_PARAMS['mix_ratio']) * new_protos
                    prototypes = prototypes / prototypes.norm(dim=1, keepdim=True)

                all_logits.append(torch.mm(q_emb, prototypes.t()))

            avg_logits = torch.stack(all_logits).mean(0)
            preds = torch.argmax(avg_logits, dim=1)
            ensemble_accs.append((preds == q_y).float().mean().item())

    mean_e = np.mean(ensemble_accs) * 100
    std_e = np.std(ensemble_accs) * 100
    print(f'  ✅ 集成直推式: {mean_e:.1f}% ± {std_e:.1f}%')
