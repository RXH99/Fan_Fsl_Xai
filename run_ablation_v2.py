"""
新架构消融实验：base64 + 多尺度 + CrossAttn V1 + UWT
运行: python run_ablation_v2.py
"""

import os, sys, copy
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse, yaml
import numpy as np
import torch
import torch.nn.functional as F
from datetime import datetime
from src.data.dataset import FaultDataset, EpisodicSampler
from src.models.encoder import create_encoder
from src.models.prototypical import CrossAttentionModule, prototypical_loss_crossattn
from src.training.train_fewshot import train_fewshot


# ===== UWT 评估函数 =====
def evaluate_uwt(encoder, cross_attn, support_x, support_y, query_x, query_y,
                 num_steps=3, tau=0.3, mix_ratio=0.8, beta=1.0):
    s_emb = encoder(support_x)
    q_emb = encoder(query_x)
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
        ws = soft * weight.unsqueeze(1)
        new_protos = []
        for w in range(ways):
            wsum = ws[:, w].sum()
            qp = (ws[:, w] @ q_emb) / wsum if wsum > 1e-8 else prototypes[w]
            new_protos.append(qp)
        new_protos = torch.stack(new_protos)
        new_protos = F.normalize(new_protos, dim=1)
        prototypes = F.normalize(mix_ratio * prototypes + (1 - mix_ratio) * new_protos, dim=1)
    preds = torch.argmax(torch.mm(q_emb, prototypes.t()), dim=1)
    return (preds == query_y).float().mean().item()


def evaluate_standard(encoder, cross_attn, support_x, support_y, query_x, query_y, device, method='cosine'):
    """标准 Cosine 或 Transductive 评估"""
    if method == 'cosine':
        from src.models.prototypical import prototypical_loss
        _, acc = prototypical_loss(encoder, support_x, support_y, query_x, query_y, device,
                                    sep_weight=0, method='cosine')
        return acc
    elif method == 'transductive':
        from src.models.prototypical import prototypical_loss
        _, acc = prototypical_loss(encoder, support_x, support_y, query_x, query_y, device,
                                    sep_weight=0, method='transductive',
                                    num_steps=5, tau=0.5, mix_ratio=0.7)
        return acc
    elif method == 'crossattn_cosine':
        s_emb = F.normalize(encoder(support_x), dim=1)
        q_emb = F.normalize(encoder(query_x), dim=1)
        q_emb = F.normalize(cross_attn(q_emb, s_emb), dim=1)
        ways = len(torch.unique(support_y))
        prototypes = torch.stack([s_emb[support_y == cls].mean(0) for cls in range(ways)])
        prototypes = F.normalize(prototypes, dim=1)
        sims = torch.mm(q_emb, prototypes.t())
        _, preds = torch.max(sims, dim=1)
        return (preds == query_y).float().mean().item()


def deep_merge(base, overrides):
    result = copy.deepcopy(base)
    for key, val in overrides.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = deep_merge(result[key], val)
        else:
            result[key] = copy.deepcopy(val)
    return result


def count_params(m):
    return sum(p.numel() for p in m.parameters())


# ===== 消融定义 =====
# 每个变体返回 (name, config_overrides, use_aug, no_pretrain, method, use_crossattn, eval_mode)
# eval_mode: 'uwt' | 'crossattn_cosine' | 'transductive' | 'cosine'
# 消融变体定义
# (name, config_overrides, use_aug, no_pretrain, method, use_crossattn, eval_mode, beta)
ABLATIONS = [
    ("Full (Our Method)",
     {}, True, False, "ProtoNet_CrossAttn", True, "uwt", 1.0),

    ("  - CrossAttn",
     {}, True, False, "ProtoNet_Cosine", False, "uwt", 1.0),

    ("  - Multi-scale",
     {"model": {"resnet": {"use_multiscale": False}}},
     True, False, "ProtoNet_CrossAttn", True, "uwt", 1.0),

    ("  - SE",
     {"model": {"resnet": {"use_se": False}}},
     True, False, "ProtoNet_CrossAttn", True, "uwt", 1.0),

    ("  - Aug",
     {}, False, False, "ProtoNet_CrossAttn", True, "uwt", 1.0),

    ("  - SupCon",
     {}, True, True, "ProtoNet_CrossAttn", True, "uwt", 1.0),
]

# ===== 推断变体（复用 Full 编码器） =====
# 推断变体：复用 Full 编码器，仅改变推理方式
# (name, eval_mode, beta)
INFERENCE_ABLATIONS = [
    ("  - UWT (beta=0, 无加权)", "uwt_std", 0.0),
    ("  - Transductive (余弦, 无直推)", "crossattn_cosine", 0.0),
]


def evaluate(encoder, cross_attn, test_dataset, device, eval_mode, episodes=500, beta=1.0):
    configs = [(5, 1, 15, "5-way 1-shot"), (5, 5, 15, "5-way 5-shot"),
               (10, 1, 10, "10-way 1-shot"), (10, 5, 10, "10-way 5-shot")]
    results = {}
    for ways, shot, query, name in configs:
        sampler = EpisodicSampler(test_dataset, ways=ways, shot=shot, query=query)
        accs = []
        encoder.eval()
        if cross_attn is not None:
            cross_attn.eval()
        with torch.no_grad():
            for _ in range(episodes):
                s_x, s_y, q_x, q_y = sampler.sample_episode()
                s_x, q_x, s_y, q_y = s_x.to(device), q_x.to(device), s_y.to(device), q_y.to(device)

                if eval_mode == 'uwt':
                    acc = evaluate_uwt(encoder, cross_attn, s_x, s_y, q_x, q_y, beta=beta)
                elif eval_mode == 'uwt_std':
                    acc = evaluate_uwt(encoder, cross_attn, s_x, s_y, q_x, q_y, beta=0.0)
                elif eval_mode == 'crossattn_cosine':
                    acc = evaluate_standard(encoder, cross_attn, s_x, s_y, q_x, q_y, device, method='crossattn_cosine')
                elif eval_mode == 'cosine':
                    acc = evaluate_standard(encoder, cross_attn, s_x, s_y, q_x, q_y, device, method='cosine')
                elif eval_mode == 'transductive':
                    acc = evaluate_standard(encoder, cross_attn, s_x, s_y, q_x, q_y, device, method='transductive')
                accs.append(acc)
        results[name] = (np.mean(accs) * 100, np.std(accs) * 100)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base64.yaml")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--eval_episodes", type=int, default=500)
    parser.add_argument("--output", default="outputs/base64/ablation_v2")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)

    if args.quick:
        args.eval_episodes = 100
        base_cfg["training"]["fewshot"]["episodes"] = 200
        print("[QUICK] 训练减至 200 episodes")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")
    if args.quick:
        print(f"新架构消融实验 (QUICK) — {base_cfg['training']['fewshot']['episodes']} 训练ep, {args.eval_episodes} 评估ep\n")
    else:
        print(f"新架构消融实验 — {base_cfg['training']['fewshot']['episodes']} 训练ep, {args.eval_episodes} 评估ep\n")

    npz_path = os.path.join(base_cfg["data"]["processed_dir"], "preprocessed.npz")
    test_dataset = FaultDataset(npz_path, split="test")
    train_dataset = FaultDataset(npz_path, split="train")
    val_dataset = FaultDataset(npz_path, split="val")

    os.makedirs(args.output, exist_ok=True)

    eval_modes = ["uwt", "uwt_std", "crossattn_cosine"]
    metric_names = ["5-way 1-shot", "5-way 5-shot", "10-way 1-shot", "10-way 5-shot"]
    all_results = {m: {} for m in metric_names}

    full_encoder = None       # 供推断变体复用
    full_cross_attn = None

    for ablation in ABLATIONS:
        name, overrides, use_aug, no_pretrain, method, use_crossattn, eval_mode, beta = ablation
        print(f"\n{'='*60}")
        print(f"📌 {name}")
        print(f"{'='*60}")

        # 固定种子：所有变体训练使用相同的 episode 序列
        import random
        torch.manual_seed(42)
        np.random.seed(42)
        random.seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)

        cfg = deep_merge(base_cfg, overrides)
        backbone = cfg["model"]["backbone"]
        encoder_dim = cfg["model"]["encoder_dim"]
        use_se = cfg["model"]["resnet"].get("use_se", True)
        use_ms = cfg["model"]["resnet"].get("use_multiscale", True)

        # 创建编码器
        encoder = create_encoder(backbone, encoder_dim=encoder_dim, use_se=use_se,
                                 base_filters=cfg["model"]["resnet"]["base_filters"],
                                 use_multiscale=use_ms).to(device)

        # 加载预训练（跳过 fc 维度不匹配）
        if not no_pretrain:
            candidates = [
                os.path.join(cfg["paths"]["output_dir"], f"pretrained_{backbone}_encoder.pth"),
                os.path.join("outputs", f"pretrained_{backbone}_encoder.pth"),
                os.path.join("outputs/base64", f"pretrained_{backbone}_encoder.pth"),
            ]
            loaded = False
            for p in candidates:
                if os.path.exists(p):
                    sd = torch.load(p, map_location=device)
                    if 'fc.weight' in sd and sd['fc.weight'].shape != encoder.fc.weight.shape:
                        del sd['fc.weight']
                        if 'fc.bias' in sd:
                            del sd['fc.bias']
                    encoder.load_state_dict(sd, strict=False)
                    print(f"   [OK] 加载预训练: {p}")
                    loaded = True
                    break
            if not loaded:
                print("   [INFO] 未找到预训练权重，从头训练")

        # 跨注意力模块
        cross_attn = None
        if use_crossattn:
            cross_attn = CrossAttentionModule(d_model=encoder_dim).to(device)
            print(f"   [NEW] CrossAttn V1 ({count_params(cross_attn)/1e3:.1f}K)")

        total_p = count_params(encoder) / 1e6
        trainable_p = count_params(encoder) / 1e6
        print(f"   参数: {total_p:.2f}M")

        # 训练（非推断变体需要训练）
        ca_kwargs = {"cross_attn": cross_attn} if cross_attn is not None else {}
        sep_weight = cfg["training"].get("sep_weight", 0.15)

        # 每轮权重存到不同路径，避免覆盖
        variant_dir = os.path.join(args.output, "weights", name.strip().replace("  ", "").replace(" ", "_"))
        cfg["paths"]["output_dir"] = variant_dir
        os.makedirs(variant_dir, exist_ok=True)

        try:
            encoder, best_val = train_fewshot(
                encoder, train_dataset, val_dataset, cfg, device,
                method=method, sep_weight=sep_weight,
                use_augmentation=use_aug, **ca_kwargs)
            print(f"   验证集 Best: {best_val:.1f}%")
        except Exception as e:
            print(f"   [FAIL] 训练失败: {e}")
            continue

        # 如果是 Full，缓存编码器
        if name == "Full (Our Method)":
            full_encoder = encoder
            full_cross_attn = cross_attn

        # 评估
        print(f"\n   评估模式: {eval_mode}")
        results = evaluate(encoder, cross_attn, test_dataset, device, eval_mode, args.eval_episodes, beta=beta)
        for s in metric_names:
            mean, std = results[s]
            all_results[s][name] = (mean, std)
            print(f"     {s:<18} → {mean:.1f}% ± {std:.1f}%")

    # ===== 推断变体 =====
    if full_encoder is not None:
        for name, eval_mode, beta in INFERENCE_ABLATIONS:
            print(f"\n{'='*60}")
            print(f"📌 {name} (复用 Full 编码器)")
            print(f"{'='*60}")
            results = evaluate(full_encoder, full_cross_attn, test_dataset, device, eval_mode, args.eval_episodes, beta=beta)
            for s in metric_names:
                mean, std = results[s]
                all_results[s][name] = (mean, std)
                print(f"     {s:<18} → {mean:.1f}% ± {std:.1f}%")

    # ===== 汇总表 =====
    print(f"\n\n{'='*70}")
    print("📊 消融实验汇总 (新架构)")
    print(f"{'='*70}")

    all_variants = []
    for ab in ABLATIONS:
        all_variants.append(ab[0])
    for inf in INFERENCE_ABLATIONS:
        all_variants.append(inf[0])

    header = f"{'Setting':<20}"
    for v in all_variants:
        header += f" {v:>22}"
    print(header)
    print("-" * len(header))

    for s in metric_names:
        line = f"{s:<20}"
        full_mean = all_results[s].get("Full (Our Method)", (0, 0))[0]
        for v in all_variants:
            if v in all_results[s]:
                mean, std = all_results[s][v]
                diff = mean - full_mean
                line += f" {mean:>6.1f}±{std:.1f}%  "
            else:
                line += f" {'N/A':>22}"
        print(line)

    # 保存
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(args.output, f"ablation_v2_{timestamp}.txt")
    with open(save_path, "w") as f:
        f.write(f"Fan_Fsl_Xai 新架构消融实验 ({timestamp})\n")
        f.write(f"作者: Task\n\n")
        for s in metric_names:
            f.write(f"{s}:\n")
            for v in all_variants:
                if v in all_results[s]:
                    mean, std = all_results[s][v]
                    f.write(f"  {v:<22} → {mean:.1f}% ± {std:.1f}%\n")
            f.write("\n")
    print(f"\n[OK] 已保存: {save_path}")


if __name__ == "__main__":
    main()
