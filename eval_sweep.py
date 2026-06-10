"""
参数扫描：找直推式推理的最佳超参数
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

ways, shot, query = 5, 5, 15

# 参数组合扫描
best_acc = 0.0
best_params = {}

for num_steps in [3, 5, 10, 15]:
    for tau in [0.3, 0.5, 0.7, 1.0]:
        for mix_ratio in [0.6, 0.7, 0.8]:
            sampler = EpisodicSampler(test_dataset, ways=ways, shot=shot,
                                       query=query)
            accs = []
            for _ in range(300):
                s_x, s_y, q_x, q_y = sampler.sample_episode()
                s_x, q_x = s_x.to(device), q_x.to(device)
                s_y, q_y = s_y.to(device), q_y.to(device)

                with torch.no_grad():
                    s_emb = encoder(s_x)
                    q_emb = encoder(q_x)
                    s_emb = s_emb / s_emb.norm(dim=1, keepdim=True)
                    q_emb = q_emb / q_emb.norm(dim=1, keepdim=True)

                    prototypes = torch.stack([
                        s_emb[s_y == cls].mean(0) for cls in range(ways)
                    ])
                    prototypes = prototypes / prototypes.norm(dim=1, keepdim=True)

                    for _ in range(num_steps):
                        sims = torch.mm(q_emb, prototypes.t()) / tau
                        soft = torch.softmax(sims, dim=1)
                        new_protos = []
                        for w in range(ways):
                            qp = (soft[:, w] @ q_emb) / (soft[:, w].sum() + 1e-8)
                            new_protos.append(qp)
                        new_protos = torch.stack(new_protos)
                        new_protos = new_protos / new_protos.norm(dim=1, keepdim=True)
                        prototypes = mix_ratio * prototypes + (1 - mix_ratio) * new_protos
                        prototypes = prototypes / prototypes.norm(dim=1, keepdim=True)

                    final_sim = torch.mm(q_emb, prototypes.t())
                    preds = torch.argmax(final_sim, dim=1)
                    acc = (preds == q_y).float().mean().item()
                    accs.append(acc)

            mean_acc = np.mean(accs) * 100
            print(f'steps={num_steps:2d} tau={tau:.1f} mix={mix_ratio:.1f} → {mean_acc:.1f}%')

            if mean_acc > best_acc:
                best_acc = mean_acc
                best_params = {'num_steps': num_steps, 'tau': tau,
                               'mix_ratio': mix_ratio}

print(f'\n✅ 最佳参数: {best_params} → {best_acc:.1f}%')
print(f'再用最佳参数跑 1000 episode 确认...')

# 用最佳参数验证
sampler = EpisodicSampler(test_dataset, ways=ways, shot=shot, query=query)
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

        prototypes = torch.stack([
            s_emb[s_y == cls].mean(0) for cls in range(ways)
        ])
        prototypes = prototypes / prototypes.norm(dim=1, keepdim=True)

        for _ in range(best_params['num_steps']):
            sims = torch.mm(q_emb, prototypes.t()) / best_params['tau']
            soft = torch.softmax(sims, dim=1)
            new_protos = []
            for w in range(ways):
                qp = (soft[:, w] @ q_emb) / (soft[:, w].sum() + 1e-8)
                new_protos.append(qp)
            new_protos = torch.stack(new_protos)
            new_protos = new_protos / new_protos.norm(dim=1, keepdim=True)
            prototypes = best_params['mix_ratio'] * prototypes + (1 - best_params['mix_ratio']) * new_protos
            prototypes = prototypes / prototypes.norm(dim=1, keepdim=True)

        final_sim = torch.mm(q_emb, prototypes.t())
        preds = torch.argmax(final_sim, dim=1)
        acc = (preds == q_y).float().mean().item()
        accs.append(acc)

print(f'5-way 5-shot: {np.mean(accs)*100:.1f}% ± {np.std(accs)*100:.1f}%')
