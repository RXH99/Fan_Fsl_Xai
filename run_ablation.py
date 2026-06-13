"""
消融实验脚本 (Ablation Study)

依次跑 5 个训练变体 + 3 种推理模式 → 共 7 组结果
输出汇总表格，保存到 outputs/ablation/

运行:
    python run_ablation.py                        # 完整实验
    python run_ablation.py --quick                 # 快速验证 (2 runs × 100 episodes)
    python run_ablation.py --skip_train            # 仅用已有权重评估
"""

import os, sys, copy, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse, yaml
import numpy as np
import torch
from datetime import datetime

from src.data.dataset import FaultDataset, EpisodicSampler
from src.models.encoder import create_encoder
from src.models.prototypical import (
    prototypical_loss,
    transductive_inference,
)
from src.training.train_fewshot import train_fewshot
from src.data.augmentation import augment_vibration_batch


# ============ 工具函数 ============

def deep_merge(base, overrides):
    """递归合并字典"""
    result = copy.deepcopy(base)
    for key, val in overrides.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = deep_merge(result[key], val)
        else:
            result[key] = copy.deepcopy(val)
    return result


def evaluate(encoder, test_dataset, device, method='cosine',
             uwt_kwargs=None, episodes=500):
    """
    通用评估函数

    method: 'cosine' | 'transductive' | 'uwt'
    uwt_kwargs: 仅 uwt 模式有效
    """
    experiments = [
        ("5-way 1-shot",  5, 1, 15),
        ("5-way 5-shot",  5, 5, 15),
        ("10-way 1-shot", 10, 1, 10),
        ("10-way 5-shot", 10, 5, 10),
    ]
    results = {}

    for name, ways, shot, query in experiments:
        sampler = EpisodicSampler(test_dataset, ways=ways, shot=shot, query=query)
        encoder.eval()
        accs = []

        with torch.no_grad():
            for _ in range(episodes):
                s_x, s_y, q_x, q_y = sampler.sample_episode()
                s_x, q_x, s_y, q_y = (
                    s_x.to(device), q_x.to(device),
                    s_y.to(device), q_y.to(device),
                )

                if method == 'cosine':
                    _, acc = prototypical_loss(
                        encoder, s_x, s_y, q_x, q_y, device,
                        sep_weight=0, method='cosine')
                elif method == 'transductive':
                    _, acc = prototypical_loss(
                        encoder, s_x, s_y, q_x, q_y, device,
                        sep_weight=0, method='transductive',
                        num_steps=5, tau=0.5, mix_ratio=0.7)
                elif method == 'uwt':
                    acc = evaluate_uwt(
                        encoder, s_x, s_y, q_x, q_y,
                        **(uwt_kwargs or {}))
                else:
                    raise ValueError(f"Unknown method: {method}")

                accs.append(acc)

        results[name] = (np.mean(accs) * 100, np.std(accs) * 100)

    return results


def evaluate_uwt(encoder, support_x, support_y, query_x, query_y,
                 num_steps=3, tau=0.3, mix_ratio=0.8, beta=2.0):
    """不确定性加权直推式推理（从 eval_uwt.py 提取）"""
    s_emb = encoder(support_x)
    q_emb = encoder(query_x)
    s_emb = torch.nn.functional.normalize(s_emb, dim=1)
    q_emb = torch.nn.functional.normalize(q_emb, dim=1)

    ways = len(torch.unique(support_y))
    prototypes = torch.stack([
        s_emb[support_y == cls].mean(0) for cls in range(ways)
    ])
    prototypes = torch.nn.functional.normalize(prototypes, dim=1)

    for _ in range(num_steps):
        sims = torch.mm(q_emb, prototypes.t())
        soft = torch.softmax(sims / tau, dim=1)
        entropy = -(soft * torch.log(soft + 1e-8)).sum(dim=1)
        entropy_norm = entropy / np.log(ways)
        weight = torch.exp(-entropy_norm * beta)
        weighted_soft = soft * weight.unsqueeze(1)

        new_protos = []
        for w in range(ways):
            ws = weighted_soft[:, w].sum()
            if ws > 1e-8:
                qp = (weighted_soft[:, w] @ q_emb) / ws
            else:
                qp = prototypes[w]
            new_protos.append(qp)
        new_protos = torch.stack(new_protos)
        new_protos = torch.nn.functional.normalize(new_protos, dim=1)
        prototypes = torch.nn.functional.normalize(
            mix_ratio * prototypes + (1 - mix_ratio) * new_protos, dim=1)

    final_sims = torch.mm(q_emb, prototypes.t())
    preds = torch.argmax(final_sims, dim=1)
    return (preds == query_y).float().mean().item()


def count_params(encoder):
    total = sum(p.numel() for p in encoder.parameters())
    trainable = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    return total / 1e6, trainable / 1e6


# ============ 消融定义 ============

ABLATIONS = [
    {
        "name": "Full (Our Method)",
        "desc": "10-way + SE + Aug + SupCon → UWT eval",
        "config_overrides": {},
        "use_aug": True,
        "no_pretrain": False,
        "eval_modes": ["uwt"],      # 完整方法只用 UWT 评估
    },
    {
        "name": "  - Transductive",
        "desc": "相同训练 → 标准 Cosine 评估 (无直推)",
        "config_overrides": {},
        "use_aug": True,
        "no_pretrain": False,
        "eval_modes": ["cosine"],   # 复用 Full 的编码器
        "reuse_encoder": "Full (Our Method)",
    },
    {
        "name": "  - UWT",
        "desc": "相同训练 → 标准直推 (β=0, 无加权)",
        "config_overrides": {},
        "use_aug": True,
        "no_pretrain": False,
        "eval_modes": ["transductive"],
        "reuse_encoder": "Full (Our Method)",
    },
    {
        "name": "  - SE",
        "desc": "去掉 SE 注意力模块",
        "config_overrides": {"model": {"resnet": {"use_se": False}}},
        "use_aug": True,
        "no_pretrain": False,
        "eval_modes": ["uwt"],
    },
    {
        "name": "  - Aug",
        "desc": "训练时关闭数据增强",
        "config_overrides": {},
        "use_aug": False,
        "no_pretrain": False,
        "eval_modes": ["uwt"],
    },
    {
        "name": "  - 10-way (改用 5-way)",
        "desc": "训练时用 5-way episode 替代 10-way",
        "config_overrides": {"training": {"fewshot": {"ways": 5}}},
        "use_aug": True,
        "no_pretrain": False,
        "eval_modes": ["uwt"],
    },
    {
        "name": "  - SupCon (从头训练)",
        "desc": "不加载 SimCLR 预训练权重",
        "config_overrides": {},
        "use_aug": True,
        "no_pretrain": True,
        "eval_modes": ["uwt"],
    },
]


# ============ 主流程 ============

def parse_args():
    parser = argparse.ArgumentParser(description="消融实验")
    parser.add_argument("--config", default="configs/optimized.yaml")
    parser.add_argument("--quick", action="store_true", help="快速验证")
    parser.add_argument("--runs", type=int, default=3, help="每个变体跑几轮")
    parser.add_argument("--eval_episodes", type=int, default=500)
    parser.add_argument("--skip_train", action="store_true",
                        help="跳过训练，仅用已有权重评估")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.runs = 2
        args.eval_episodes = 100

    with open(args.config, "r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)

    if args.quick:
        base_cfg["training"]["fewshot"]["episodes"] = 500

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 设备: {device}")
    print(f"📊 消融实验 — 每轮 {args.eval_episodes} episodes × {args.runs} runs")
    if args.quick:
        print(f"⚡ Quick 模式: 训练减至 {base_cfg['training']['fewshot']['episodes']} episodes")

    npz_path = os.path.join(base_cfg["data"]["processed_dir"], "preprocessed.npz")
    test_dataset = FaultDataset(npz_path, split="test")
    train_dataset = FaultDataset(npz_path, split="train")
    val_dataset = FaultDataset(npz_path, split="val")

    # 输出目录
    output_dir = os.path.join(base_cfg["paths"]["output_dir"], "ablation")
    os.makedirs(output_dir, exist_ok=True)

    # 存储每次评估结果: results[setting][variant] = [acc_run1, acc_run2, ...]
    settings = ["5-way 1-shot", "5-way 5-shot",
                 "10-way 1-shot", "10-way 5-shot"]
    results = {s: {} for s in settings}

    # 缓存已训练的编码器（避免重复训练 Full）
    encoder_cache = {}

    for ablation in ABLATIONS:
        name = ablation["name"]
        desc = ablation["desc"]
        print(f"\n{'='*70}")
        print(f"📌 {name}")
        print(f"   {desc}")
        print(f"{'='*70}")

        # 如果复用其他编码器
        if "reuse_encoder" in ablation:
            source = ablation["reuse_encoder"]
            if source not in encoder_cache:
                print(f"   ⚠️ 源 '{source}' 未训练，跳过")
                continue
            trainable_params = encoder_cache[source]["params"]
            print(f"   复用 '{source}' 的编码器 ({trainable_params:.2f}M 参数)")
            encoders = encoder_cache[source]["encoders"]
        else:
            # 训练
            encoders = []
            for run_idx in range(1, args.runs + 1):
                print(f"\n   --- Run {run_idx}/{args.runs} ---")

                # 创建配置
                cfg = deep_merge(base_cfg, ablation.get("config_overrides", {}))
                backbone = cfg.get("model", {}).get("backbone", "resnet18")
                encoder_dim = cfg.get("model", {}).get("encoder_dim", 64)
                use_se = cfg.get("model", {}).get("resnet", {}).get("use_se", True)

                # 创建编码器
                encoder = create_encoder(
                    backbone, encoder_dim=encoder_dim, use_se=use_se
                ).to(device)

                # 加载预训练权重
                if not ablation["no_pretrain"]:
                    pretrain_path = os.path.join(
                        base_cfg["paths"]["output_dir"],
                        "pretrained_resnet18_encoder.pth")
                    if os.path.exists(pretrain_path):
                        sd = torch.load(pretrain_path, map_location=device)
                        miss, unexp = encoder.load_state_dict(sd, strict=False)
                        print(f"   ✅ 加载预训练权重")
                        if miss:
                            print(f"      缺失键: {len(miss)}")
                        if unexp:
                            print(f"      意外键: {len(unexp)} (SE 相关，无害)")
                    else:
                        print(f"   ℹ️ 未找到预训练权重，从头训练")

                total_m, trainable_m = count_params(encoder)
                print(f"   参数: {total_m:.2f}M 总, {trainable_m:.2f}M 可训练")

                # 训练
                sep_weight = cfg.get("training", {}).get("sep_weight", 0.15)
                encoder, best_val_acc = train_fewshot(
                    encoder, train_dataset, val_dataset, cfg, device,
                    method="ProtoNet_Cosine", sep_weight=sep_weight,
                    use_augmentation=ablation["use_aug"])

                encoders.append(encoder)

            # 缓存
            encoder_cache[name] = {
                "encoders": encoders,
                "params": trainable_m,
            }

        # 评估
        for eval_mode in ablation["eval_modes"]:
            uwt_kwargs = {"beta": 0.0} if eval_mode == "transductive" else \
                         {"beta": 2.0} if eval_mode == "uwt" else None
            mode_label = "UWT" if eval_mode == "uwt" else \
                         "Transductive" if eval_mode == "transductive" else "Cosine"

            print(f"\n   🔍 评估模式: {mode_label}")

            # 每轮单独评估，收集所有结果
            run_results = {s: [] for s in settings}
            for run_idx, encoder in enumerate(encoders):
                res = evaluate(encoder, test_dataset, device,
                               method=eval_mode, uwt_kwargs=uwt_kwargs,
                               episodes=args.eval_episodes)
                for s in settings:
                    run_results[s].append(res[s][0])

            # 汇总
            for s in settings:
                accs = run_results[s]
                mean_acc = np.mean(accs)
                std_acc = np.std(accs)
                print(f"     {s:<18} → {mean_acc:.1f}% ± {std_acc:.1f}%")

                # 存入总表
                col_name = name if eval_mode == "uwt" else f"{name} ({mode_label})"
                if col_name not in results[s]:
                    results[s][col_name] = []
                results[s][col_name].extend(run_results[s])

    # ===== 汇总表 =====
    print(f"\n\n{'='*80}")
    print("📊 消融实验汇总表")
    print(f"{'='*80}")

    # 收集所有列名，保持顺序
    all_variants = []
    for ablation in ABLATIONS:
        for mode in ablation["eval_modes"]:
            name = ablation["name"]
            if mode != "uwt":
                name = f"{name} ({'UWT' if mode == 'uwt' else 'Cosine' if mode == 'cosine' else 'Transductive'})"
            if name not in all_variants:
                all_variants.append(name)

    # 表头
    header = f"{'Setting':<20}"
    for v in all_variants:
        header += f" {v:>20}"
    print(header)
    print("-" * len(header))

    table_data = {}
    for s in settings:
        line = f"{s:<20}"
        row = []
        for v in all_variants:
            accs = results[s].get(v, [])
            if accs:
                mean = np.mean(accs)
                std = np.std(accs)
                line += f" {mean:>6.1f}±{std:.1f}%  "
                row.append(f"{mean:.1f}±{std:.1f}")
            else:
                line += f" {'N/A':>20}"
                row.append("N/A")
        print(line)
        table_data[s] = row

    # 保存
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(output_dir, f"ablation_{timestamp}.txt")
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("Fan_Fsl_Xai — 消融实验\n")
        f.write(f"{'='*80}\n\n")
        f.write(f"配置源文件: {args.config}\n")
        f.write(f"每个变体: {args.runs} runs × {args.eval_episodes} episodes\n\n")
        f.write(f"{'Setting':<20}")
        for v in all_variants:
            f.write(f" {v:>20}")
        f.write("\n" + "-" * (20 + 21 * len(all_variants)) + "\n")
        for s in settings:
            f.write(f"{s:<20}")
            for v in all_variants:
                accs = results[s].get(v, [])
                if accs:
                    mean = np.mean(accs)
                    std = np.std(accs)
                    f.write(f" {mean:>6.1f}±{std:.1f}%  ")
                else:
                    f.write(f" {'N/A':>20}")
            f.write("\n")

    print(f"\n✅ 已保存: {save_path}")
    print(f"\n💡 实验耗时较长，运行前准备好:\n"
          f"   完整 ({args.runs} runs × 不跳过训练) → 约 {7 * args.runs * 15} 分钟\n"
          f"   跳过训练 ({args.skip_train}) → 约 5 分钟")


if __name__ == "__main__":
    main()
