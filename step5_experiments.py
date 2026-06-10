"""
Step 5: 完整对比实验（论文数据）— 升级版

新增测试:
  - ProtoNet_Cosine:        余弦相似度（推荐）
  - ProtoNet_Transductive:  直推式推理
  - ProtoNet_ResNet:        保留旧基线
  - ProtoNet_CNN:           保留旧基线

用法:
  python step5_experiments.py                              # 所有方法
  python step5_experiments.py --methods ProtoNet_Cosine     # 指定方法
  python step5_experiments.py --methods ProtoNet_Cosine ProtoNet_Transductive
  python step5_experiments.py --runs 3                     # 多次运行
  python step5_experiments.py --no_aug                     # 关闭增强
  python step5_experiments.py --quick                      # 快速验证
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse
import numpy as np
import torch
import yaml
from datetime import datetime

from src.data.dataset import FaultDataset, EpisodicSampler
from src.models.encoder import create_encoder
from src.models.prototypical import prototypical_loss


def evaluate_method(encoder, test_dataset, config, device,
                    method='cosine', eval_episodes=500):
    """
    在测试集的 4 个 setting 上评估

    Args:
        method: 'euclidean' | 'cosine' | 'transductive'
    """
    experiments = [
        ("5-way 1-shot",  5, 1, 15),
        ("5-way 5-shot",  5, 5, 15),
        ("10-way 1-shot", 10, 1, 10),
        ("10-way 5-shot", 10, 5, 10),
    ]
    results = {}

    for name, ways, shot, query in experiments:
        sampler = EpisodicSampler(test_dataset, ways=ways, shot=shot,
                                   query=query)
        encoder.eval()
        accs = []

        trans_kwargs = {}
        if method == 'transductive':
            trans_kwargs = {'num_steps': 10, 'tau': 0.5, 'mix_ratio': 0.7}

        with torch.no_grad():
            for _ in range(eval_episodes):
                s_x, s_y, q_x, q_y = sampler.sample_episode()
                s_x = s_x.to(device)
                q_x = q_x.to(device)
                s_y = s_y.to(device)
                q_y = q_y.to(device)

                _, acc = prototypical_loss(
                    encoder, s_x, s_y, q_x, q_y, device,
                    sep_weight=0, method=method, **trans_kwargs)
                accs.append(acc)

        mean_acc = np.mean(accs) * 100
        std_acc = np.std(accs) * 100
        results[name] = (mean_acc, std_acc)

    return results


def parse_args():
    parser = argparse.ArgumentParser(description="完整对比实验")
    parser.add_argument("--config", default="configs/optimized.yaml")
    parser.add_argument("--methods", nargs="+",
                        default=["ProtoNet_CNN", "ProtoNet_ResNet",
                                 "ProtoNet_Cosine", "ProtoNet_Transductive"])
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--eval_episodes", type=int, default=500)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--no_aug", action="store_true",
                        help="关闭数据增强")
    parser.add_argument("--max_episodes", type=int, default=None,
                        help="限制训练 episode 数（快速调试用）")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.runs = 2
        args.eval_episodes = 100

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 设备: {device}")

    npz_path = os.path.join(cfg["data"]["processed_dir"], "preprocessed.npz")
    test_dataset = FaultDataset(npz_path, split="test")
    train_dataset = FaultDataset(npz_path, split="train")
    val_dataset = FaultDataset(npz_path, split="val")

    backbone = cfg.get("model", {}).get("backbone", "resnet18")
    encoder_dim = cfg.get("model", {}).get("encoder_dim", 64)

    # 方法 → 内部方法名 映射
    method_to_proto = {
        'ProtoNet_CNN': 'euclidean',
        'ProtoNet_ResNet': 'cosine',
        'ProtoNet_Cosine': 'cosine',
        'ProtoNet_CosineT': 'cosine',
        'ProtoNet_Transductive': 'transductive',
        'ProtoNet_Consistency': 'cosine',
    }

    all_results = {}

    for method in args.methods:
        print(f"\n{'#'*60}")
        print(f"# 方法: {method}")
        print(f"{'#'*60}")

        proto_method = method_to_proto.get(method, 'cosine')
        method_results = {s: [] for s in
                         ["5-way 1-shot", "5-way 5-shot",
                          "10-way 1-shot", "10-way 5-shot"]}

        for run_idx in range(1, args.runs + 1):
            print(f"\n--- Run {run_idx}/{args.runs} ---")

            # 创建 encoder
            if method == "ProtoNet_CNN":
                encoder = create_encoder("cnn").to(device)
            else:
                encoder = create_encoder(backbone, encoder_dim=encoder_dim,
                                         use_se=True).to(device)

            if not args.skip_train:
                from src.training.train_fewshot import train_fewshot
                sep_weight = cfg.get("training", {}).get("sep_weight", 0.05)

                # 如果有 max_episodes，临时修改配置
                if args.max_episodes:
                    old_episodes = cfg["training"]["fewshot"]["episodes"]
                    cfg["training"]["fewshot"]["episodes"] = args.max_episodes

                encoder, _ = train_fewshot(
                    encoder, train_dataset, val_dataset, cfg, device,
                    method=method, sep_weight=sep_weight,
                    use_augmentation=not args.no_aug)

                if args.max_episodes:
                    cfg["training"]["fewshot"]["episodes"] = old_episodes
            else:
                model_path = os.path.join(
                    cfg["paths"]["output_dir"],
                    f"fewshot_encoder_{method.replace('/', '_')}.pth")
                if os.path.exists(model_path):
                    encoder.load_state_dict(
                        torch.load(model_path, map_location=device))
                    print(f"✅ 加载已有模型: {model_path}")
                else:
                    print(f"⚠️ 未找到 {model_path}，跳过")
                    continue

            # 评估
            results = evaluate_method(encoder, test_dataset, cfg, device,
                                      method=proto_method,
                                      eval_episodes=args.eval_episodes)

            for setting, (mean_acc, std_acc) in results.items():
                method_results[setting].append(mean_acc)
                print(f"  {setting:<18} → {mean_acc:.1f}% ± {std_acc:.1f}%")

        all_results[method] = method_results

    # ===== 汇总表格 =====
    settings = ["5-way 1-shot", "5-way 5-shot",
                 "10-way 1-shot", "10-way 5-shot"]

    print(f"\n\n{'='*70}")
    print("📊 实验汇总")
    print(f"{'='*70}")

    header = f"{'方法':<24}"
    for s in settings:
        header += f" {s:>24}"
    print(header)
    print("-" * len(header))

    for method in args.methods:
        if method not in all_results:
            continue
        line = f"{method:<24}"
        for s in settings:
            accs = all_results[method][s]
            if accs:
                mean = np.mean(accs)
                std = np.std(accs)
                line += f" {mean:>6.1f}%±{std:.1f}%        "
            else:
                line += f" {'N/A':>24}"
        print(line)

    # 保存
    output_dir = cfg["paths"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(output_dir, f"experiment_results_{timestamp}.txt")
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(f"Fan_Fsl_Xai 对比实验 ({timestamp})\n")
        f.write(f"配置文件: {args.config}\n")
        f.write(f"每个方法跑 {args.runs} 次，每次 {args.eval_episodes} episodes\n")
        if not args.no_aug:
            f.write("✅ 数据增强已启用\n")
        f.write("\n")
        for method in args.methods:
            if method not in all_results:
                continue
            f.write(f"{method}\n")
            for s in settings:
                accs = all_results[method][s]
                if accs:
                    mean = np.mean(accs)
                    std = np.std(accs)
                    f.write(f"  {s}: {mean:.1f}% ± {std:.1f}%\n")
            f.write("\n")

    print(f"\n✅ 结果已保存: {save_path}")


if __name__ == "__main__":
    main()
