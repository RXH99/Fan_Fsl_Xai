"""
MAML 对比实验评估 — 纯 PyTorch，无外部依赖

加载已训练的 MAML 编码器，在测试集上做全面评估。

评估协议与 ProtoNet 一致：
  - 3 seeds × 1000 episodes
  - 5-way / 10-way, 1-shot / 5-shot

运行:
  python eval_maml.py                                    # 评估训练好的 MAML
  python eval_maml.py --encoder_path outputs/maml/maml_encoder_final.pth
  python eval_maml.py --seeds 3 --episodes 1000          # 完整评估
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import yaml
import torch
import torch.nn.functional as F
import numpy as np

from src.data.dataset import FaultDataset, EpisodicSampler
from src.models.encoder import create_encoder
from src.models.maml import MAMLModel, maml_evaluate


def parse_args():
    parser = argparse.ArgumentParser(description="MAML 对比评估")
    parser.add_argument("--config", default="configs/clean.yaml")
    parser.add_argument("--model_path", default="outputs/maml/maml_model_final.pth",
                        help="MAML 完整模型权重（含分类头）")
    parser.add_argument("--encoder_path", default=None,
                        help="MAML 编码器权重（可选，与 model_path 二选一）")
    parser.add_argument("--inner_steps", type=int, default=5,
                        help="评估时内循环步数")
    parser.add_argument("--inner_lr", type=float, default=0.01,
                        help="评估时内循环学习率")
    parser.add_argument("--episodes", type=int, default=1000,
                        help="每个 setting 的评估 episode 数")
    parser.add_argument("--seeds", type=int, default=3,
                        help="重复 seed 次数")
    parser.add_argument("--first_order", action="store_true", default=True,
                        help="一阶 MAML")
    return parser.parse_args()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    args = parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    print(f"🔧 设备: {device}")
    print(f"📊 MAML 对比评估")
    print(f"   评估: {args.episodes} episodes × {args.seeds} seeds")
    print(f"   内循环: {args.inner_steps} steps × lr={args.inner_lr}")
    print(f"   模式: {'FOMAML' if args.first_order else '二阶 MAML'}")

    if args.model_path and os.path.exists(args.model_path):
        print(f"   模型权重: {args.model_path}")
    elif args.encoder_path and os.path.exists(args.encoder_path):
        print(f"   编码器权重: {args.encoder_path}")
    else:
        print("   ⚠️ 权重文件不存在，确保先运行 python train_maml.py")
        return

    npz_path = os.path.join(cfg["data"]["processed_dir"], "preprocessed.npz")
    test_dataset = FaultDataset(npz_path, split="test")

    experiments = [
        ("5-way 1-shot",  5, 1, 15),
        ("5-way 5-shot",  5, 5, 15),
        ("10-way 1-shot", 10, 1, 10),
        ("10-way 5-shot", 10, 5, 10),
    ]

    # 收集所有 seed 的结果
    all_results = {name: [] for name, _, _, _ in experiments}

    for seed_idx in range(1, args.seeds + 1):
        print(f"\n{'='*50}")
        print(f"🎲 Seed {seed_idx}/{args.seeds}")
        print(f"{'='*50}")

        torch.manual_seed(seed_idx)
        np.random.seed(seed_idx)

        # 创建模型
        use_ms = cfg.get("model", {}).get("resnet", {}).get("use_multiscale", True)
        encoder = create_encoder(
            cfg["model"]["backbone"],
            encoder_dim=cfg["model"]["encoder_dim"],
            use_se=cfg["model"]["resnet"]["use_se"],
            base_filters=cfg["model"]["resnet"]["base_filters"],
            in_channels=1,
            use_multiscale=use_ms,
        ).to(device)

        model = MAMLModel(encoder, n_way=cfg["training"]["fewshot"]["ways"]).to(device)

        # 加载权重
        if args.model_path and os.path.exists(args.model_path):
            model.load_state_dict(torch.load(args.model_path, map_location=device))
        elif args.encoder_path and os.path.exists(args.encoder_path):
            model.encoder.load_state_dict(
                torch.load(args.encoder_path, map_location=device))
            # 分类头随机初始化（评估时会重新适应）
        else:
            continue

        model.eval()
        print(f"   ✅ 模型加载完成 ({sum(p.numel() for p in model.parameters())/1e6:.2f}M)")

        for name, ways, shot, query in experiments:
            # 重新创建分类头（匹配当前 n_way）
            if ways != cfg["training"]["fewshot"]["ways"] and args.encoder_path:
                # 编码器模式需要重建分类头
                model.classifier = torch.nn.Linear(
                    model.encoder.output_dim, ways).to(device)

            sampler = EpisodicSampler(test_dataset, ways=ways, shot=shot, query=query)
            accs = []

            for ep_idx in range(args.episodes):
                s_x, s_y, q_x, q_y = sampler.sample_episode()
                s_x, q_x, s_y, q_y = (
                    s_x.to(device), q_x.to(device),
                    s_y.to(device), q_y.to(device),
                )

                with torch.no_grad():
                    acc = maml_evaluate(
                        model, s_x, s_y, q_x, q_y,
                        inner_steps=args.inner_steps,
                        inner_lr=args.inner_lr,
                        first_order=args.first_order,
                    )
                accs.append(acc)

            mean_acc = np.mean(accs) * 100
            all_results[name].append(mean_acc)
            print(f"  {name:<18} → {mean_acc:.1f}%")

    # 跨 seed 汇总
    print(f"\n{'='*60}")
    print(f"📊 MAML 对比实验 — 最终结果 ({args.seeds} seeds)")
    print(f"{'='*60}")
    print(f"  {'Setting':<20} {'Accuracy':>12}")
    print(f"  {'-'*32}")

    final_results = {}
    for name, _, _, _ in experiments:
        accs = all_results[name]
        if accs:
            mean = np.mean(accs)
            std = np.std(accs)
            final_results[name] = (mean, std)
            print(f"  {name:<20} {mean:.1f}% ± {std:.1f}%")

    # 对比表格
    print(f"\n{'='*70}")
    print(f"📋 与 ProtoNet + SupCon 对比（参考值）")
    print(f"{'='*70}")
    print(f"  {'Method':<30} {'5w1s':>8} {'5w5s':>8} {'10w1s':>8} {'10w5s':>8}")
    print(f"  {'-'*70}")
    print(f"  {'ProtoNet + SupCon + UWT':<30} {'84.0':>8} {'97.2':>8} {'74.0':>8} {'89.0':>8}")
    print(f"  {'ProtoNet + SupCon (Cosine)':<30} {'82.0':>8} {'95.6':>8} {'72.0':>8} {'87.0':>8}")

    maml_vals = ""
    for name, _, _, _ in experiments:
        if name in final_results:
            maml_vals += f" {final_results[name][0]:>6.1f}%"
        else:
            maml_vals += f" {'N/A':>8}"
    print(f"  {'MAML (无 SupCon 预训练)':<30}{maml_vals}")

    print(f"\n{'='*70}")
    print(f"✅ 完成！")


if __name__ == "__main__":
    main()
