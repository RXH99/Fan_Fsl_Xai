"""
多种子消融实验：对关键变体跑多个种子取平均
用法:
    python run_seeded_ablation.py                          # 完整
    python run_seeded_ablation.py --quick                   # 快速
    python run_seeded_ablation.py --seeds 42 123 999        # 自定义种子
"""

import os, sys, copy, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse, yaml, random
import numpy as np
import torch
import torch.nn.functional as F
from datetime import datetime
from src.data.dataset import FaultDataset, EpisodicSampler
from src.models.encoder import create_encoder
from src.models.prototypical import CrossAttentionModule, prototypical_loss_crossattn
from src.training.train_fewshot import train_fewshot


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_uwt(encoder, cross_attn, support_x, support_y, query_x, query_y, beta=1.0):
    s_emb = encoder(support_x)
    q_emb = encoder(query_x)
    s_emb = F.normalize(s_emb, dim=1)
    q_emb = F.normalize(q_emb, dim=1)
    if cross_attn is not None:
        q_emb = cross_attn(q_emb, s_emb)
        q_emb = F.normalize(q_emb, dim=1)
    ways = len(torch.unique(support_y))
    p = torch.stack([s_emb[support_y == c].mean(0) for c in range(ways)])
    p = F.normalize(p, dim=1)
    for _ in range(3):
        sft = torch.softmax(torch.mm(q_emb, p.t()) / 0.3, dim=1)
        w = torch.exp(-(sft * torch.log(sft + 1e-8)).sum(1) / np.log(ways) * beta)
        ws = sft * w.unsqueeze(1)
        np_ = []
        for c in range(ways):
            ws_c = ws[:, c].sum()
            np_.append((ws[:, c] @ q_emb) / ws_c if ws_c > 1e-8 else p[c])
        p = F.normalize(0.8 * p + 0.2 * torch.stack(np_), dim=1)
    _, pred = torch.max(torch.mm(q_emb, p.t()), 1)
    return (pred == query_y).float().mean().item()


def evaluate(encoder, cross_attn, test_dataset, device, mode='uwt', beta=1.0, ep=500):
    cfgs = [(5, 1, 15, "5-way 1-shot"), (5, 5, 15, "5-way 5-shot"),
            (10, 1, 10, "10-way 1-shot"), (10, 5, 10, "10-way 5-shot")]
    r = {}
    encoder.eval()
    if cross_attn is not None:
        cross_attn.eval()
    for ways, shot, query, name in cfgs:
        sp = EpisodicSampler(test_dataset, ways=ways, shot=shot, query=query)
        accs = []
        with torch.no_grad():
            for _ in range(ep):
                sx, sy, qx, qy = sp.sample_episode()
                sx, qx, sy, qy = sx.to(device), qx.to(device), sy.to(device), qy.to(device)
                if mode == 'uwt':
                    accs.append(evaluate_uwt(encoder, cross_attn, sx, sy, qx, qy, beta=beta))
                elif mode == 'cosine':
                    se = F.normalize(encoder(sx), dim=1)
                    qe = F.normalize(encoder(qx), dim=1)
                    if cross_attn is not None:
                        qe = F.normalize(cross_attn(qe, se), dim=1)
                    ways_k = len(torch.unique(sy))
                    pk = F.normalize(torch.stack([se[sy == c].mean(0) for c in range(ways_k)]), dim=1)
                    _, pd_ = torch.max(torch.mm(qe, pk.t()), 1)
                    accs.append((pd_ == qy).float().mean().item())
                elif mode == 'transductive':
                    _, acc_ = prototypical_loss_crossattn(encoder, cross_attn, sx, sy, qx, qy, device, sep_weight=0) \
                        if cross_attn is not None else (None, None)
                    if acc_ is None:
                        from src.models.prototypical import prototypical_loss as pl
                        _, acc_ = pl(encoder, sx, sy, qx, qy, device, method='transductive',
                                     num_steps=5, tau=0.5, mix_ratio=0.7)
                    accs.append(acc_)
        r[name] = (np.mean(accs) * 100, np.std(accs) * 100)
    return r


def deep_merge(b, o):
    r = copy.deepcopy(b)
    for k, v in o.items():
        if k in r and isinstance(r[k], dict) and isinstance(v, dict):
            r[k] = deep_merge(r[k], v)
        else:
            r[k] = copy.deepcopy(v)
    return r


def make_encoder(cfg, use_crossattn, device):
    base_f = cfg["model"]["resnet"]["base_filters"]
    edim = cfg["model"]["encoder_dim"]
    use_se = cfg["model"]["resnet"].get("use_se", True)
    use_ms = cfg["model"]["resnet"].get("use_multiscale", True)
    encoder = create_encoder("resnet18", encoder_dim=edim, use_se=use_se,
                             base_filters=base_f, use_multiscale=use_ms).to(device)
    cross_attn = CrossAttentionModule(d_model=edim).to(device) if use_crossattn else None
    return encoder, cross_attn


# === 消融定义 ===
ABLATIONS = [
    ("Full (Our Method)",           {},         True,  False, "ProtoNet_CrossAttn", True),
    ("- CrossAttn",                 {},         True,  False, "ProtoNet_Cosine",   False),
    ("- Multi-scale",               {"model": {"resnet": {"use_multiscale": False}}},
                                                      True,  False, "ProtoNet_CrossAttn", True),
    ("- SupCon",                    {},         True,  True,  "ProtoNet_CrossAttn", True),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base64.yaml")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 999])
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--eval_episodes", type=int, default=500)
    parser.add_argument("--output", default="outputs/base64/ablation_v2")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)

    if args.quick:
        args.eval_episodes = 100
        base_cfg["training"]["fewshot"]["episodes"] = 200

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")
    print(f"种子: {args.seeds}")
    print(f"训练: {base_cfg['training']['fewshot']['episodes']}ep, 评估: {args.eval_episodes}ep\n")

    npz = os.path.join(base_cfg["data"]["processed_dir"], "preprocessed.npz")
    test_dataset = FaultDataset(npz, split="test")
    train_dataset = FaultDataset(npz, split="train")
    val_dataset = FaultDataset(npz, split="val")
    os.makedirs(args.output, exist_ok=True)

    metrics = ["5-way 1-shot", "5-way 5-shot", "10-way 1-shot", "10-way 5-shot"]
    # all_seed_results[name][seed][metric] = (mean, std)
    all_seed_results = {}

    for variant_name, overrides, use_aug, no_pretrain, method, use_crossattn in ABLATIONS:
        print(f"\n{'='*60}")
        print(f"📌 {variant_name}")
        print(f"{'='*60}")
        all_seed_results[variant_name] = {}
        cfg = deep_merge(base_cfg, overrides)

        full_encoder, full_cross = None, None

        for seed in args.seeds:
            print(f"\n  --- Seed {seed} ---")
            set_seed(seed)

            encoder, cross_attn = make_encoder(cfg, use_crossattn, device)

            # 加载预训练
            if not no_pretrain:
                for p in [os.path.join(cfg["paths"]["output_dir"], "pretrained_resnet18_encoder.pth"),
                          "outputs/pretrained_resnet18_encoder.pth",
                          "outputs/base64/pretrained_resnet18_encoder.pth"]:
                    if os.path.exists(p):
                        sd = torch.load(p, map_location=device)
                        if 'fc.weight' in sd and sd['fc.weight'].shape != encoder.fc.weight.shape:
                            del sd['fc.weight']
                            if 'fc.bias' in sd:
                                del sd['fc.bias']
                        encoder.load_state_dict(sd, strict=False)
                        print(f"    [OK] 加载预训练")
                        break

            # 训练
            ca_kw = {"cross_attn": cross_attn} if cross_attn is not None else {}
            sep_w = cfg["training"].get("sep_weight", 0.15)
            try:
                encoder, best_val = train_fewshot(
                    encoder, train_dataset, val_dataset, cfg, device,
                    method=method, sep_weight=sep_w,
                    use_augmentation=use_aug, **ca_kw)
                print(f"    Val Best: {best_val:.1f}%")
            except Exception as e:
                print(f"    [FAIL] {e}")
                continue

            # 评估 (UWT)
            beta = 1.0
            results = evaluate(encoder, cross_attn, test_dataset, device,
                               mode='uwt', beta=beta, ep=args.eval_episodes)
            for m in metrics:
                all_seed_results[variant_name].setdefault(m, []).append(results[m][0])

            # 如果是 Full，缓存用于推断变体
            if variant_name == "Full (Our Method)":
                full_encoder, full_cross = encoder, cross_attn

        # 打印本变体所有种子结果
        print(f"\n  [{variant_name}] 所有种子:")
        for m in metrics:
            vals = all_seed_results[variant_name].get(m, [])
            if vals:
                print(f"    {m}: {[f'{v:.1f}' for v in vals]}  →  {np.mean(vals):.1f}% ± {np.std(vals):.1f}%")

    # === 推断变体（复用 Full 编码器） ===
    if full_encoder is not None:
        for name, mode, beta, label in [
            ("- UWT (beta=0, 无加权)", "uwt", 0.0, ""),
            ("- Transductive (余弦)", "cosine", 0.0, ""),
        ]:
            print(f"\n{'='*60}")
            print(f"📌 {name} (复用 Full 编码器)")
            print(f"{'='*60}")
            for seed in args.seeds[:1]:  # 只需评估一次，编码器一样
                results = evaluate(full_encoder, full_cross, test_dataset, device,
                                   mode=mode, beta=beta, ep=args.eval_episodes)
                for m in metrics:
                    all_seed_results.setdefault(name, {}).setdefault(m, []).append(results[m][0])
                    print(f"    {m}: {results[m][0]:.1f}%")

    # === 汇总表 ===
    print(f"\n\n{'='*70}")
    print("📊 多种子消融汇总")
    print(f"{'='*70}")
    variants = [a[0] for a in ABLATIONS] + ["- UWT (beta=0, 无加权)", "- Transductive (余弦)"]

    header = f"{'Setting':<20}"
    for v in variants:
        header += f" {v:>26}"
    print(header)
    print("-" * len(header))

    for m in metrics:
        line = f"{m:<20}"
        fmean = np.mean(all_seed_results.get(variants[0], {}).get(m, [0]))
        for v in variants:
            vals = all_seed_results.get(v, {}).get(m, [])
            if vals:
                mean, std = np.mean(vals), np.std(vals)
                diff = mean - fmean if v != variants[0] else 0
                line += f" {mean:>6.1f}±{std:.1f}%  "
            else:
                line += f" {'N/A':>26}"
        print(line)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sp = os.path.join(args.output, f"seeded_ablation_{ts}.txt")
    with open(sp, "w") as f:
        f.write(f"多种子消融 ({ts})\n种子: {args.seeds}\n\n")
        for m in metrics:
            f.write(f"{m}:\n")
            for v in variants:
                vals = all_seed_results.get(v, {}).get(m, [])
                if vals:
                    f.write(f"  {v:<28} {np.mean(vals):.1f}% ± {np.std(vals):.1f}%\n")
            f.write("\n")
    print(f"\n[OK] 保存: {sp}")


if __name__ == "__main__":
    main()
