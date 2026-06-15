"""
MAML 元训练脚本 — 纯 PyTorch，无外部依赖

与 ProtoNet 共享相同 SE-ResNet1D(base64) 编码器架构，但不使用 SupCon 预训练，
而是用 MAML 从零开始元训练（end-to-end meta-learning）。

显存管理（RTX 3060 6GB）:
  - first_order=True (FOMAML) → ~3-3.5 GB ✅
  - first_order=False (二阶)  → ~5-6 GB ⚠️ 可能边缘
  - inner_steps=5 → 标准步数

运行:
  python train_maml.py                          # 默认 FOMAML
  python train_maml.py --first_order False      # 二阶 MAML
  python train_maml.py --config configs/clean.yaml
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import yaml
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from datetime import datetime

from src.data.dataset import FaultDataset, EpisodicSampler
from src.models.encoder import create_encoder
from src.models.maml import MAMLModel, maml_forward, maml_evaluate
from src.data.augmentation import augment_vibration_batch


def parse_args():
    parser = argparse.ArgumentParser(description="MAML 元训练")
    parser.add_argument("--config", default="configs/compare_maml.yaml")
    parser.add_argument("--episodes", type=int, default=10000,
                        help="元训练 episode 总数")
    parser.add_argument("--val_interval", type=int, default=200,
                        help="验证间隔")
    parser.add_argument("--inner_lr", type=float, default=0.01,
                        help="内循环学习率")
    parser.add_argument("--meta_lr", type=float, default=0.001,
                        help="外循环（元）学习率")
    parser.add_argument("--inner_steps", type=int, default=5,
                        help="内循环步数")
    parser.add_argument("--first_order", action="store_true", default=True,
                        help="一阶 MAML (FOMAML)，省显存")
    parser.add_argument("--val_episodes", type=int, default=200,
                        help="验证 episode 数")
    parser.add_argument("--eval_episodes", type=int, default=500,
                        help="最终评估 episode 数")
    parser.add_argument("--early_stop", type=int, default=15,
                        help="早停 patience（×val_interval）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--no_aug", action="store_true",
                        help="关闭数据增强")
    parser.add_argument("--no_pretrain", action="store_true",
                        help="不使用 SupCon 预训练权重（从零开始）")
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 设备: {device}")
    print(f"🧪 MAML 元训练")
    print(f"   内循环: {args.inner_steps} steps × lr={args.inner_lr}")
    print(f"   外循环: lr={args.meta_lr}")
    print(f"   方式: {'一阶 FOMAML' if args.first_order else '二阶 MAML'}")
    print(f"   编码器: {cfg['model']['backbone']}, dim={cfg['model']['encoder_dim']}")

    # 1. 数据
    npz_path = os.path.join(cfg["data"]["processed_dir"], "preprocessed.npz")
    train_dataset = FaultDataset(npz_path, split="train")
    val_dataset = FaultDataset(npz_path, split="val")
    test_dataset = FaultDataset(npz_path, split="test")

    n_way = cfg["training"]["fewshot"]["ways"]
    n_shot = cfg["training"]["fewshot"]["shot"]
    n_query = cfg["training"]["fewshot"]["query"]

    train_sampler = EpisodicSampler(train_dataset, ways=n_way, shot=n_shot, query=n_query)
    val_sampler = EpisodicSampler(val_dataset, ways=n_way, shot=n_shot, query=n_query)

    # 2. 创建编码器
    use_multiscale = cfg.get("model", {}).get("resnet", {}).get("use_multiscale", True)
    encoder = create_encoder(
        cfg["model"]["backbone"],
        encoder_dim=cfg["model"]["encoder_dim"],
        use_se=cfg["model"]["resnet"]["use_se"],
        base_filters=cfg["model"]["resnet"]["base_filters"],
        in_channels=1,
        use_multiscale=use_multiscale,
    ).to(device)

    # 可选：加载 SupCon 预训练权重
    if not args.no_pretrain:
        candidates = [
            os.path.join(cfg["paths"]["output_dir"], f"pretrained_{cfg['model']['backbone']}_encoder.pth"),
            os.path.join("outputs", f"pretrained_{cfg['model']['backbone']}_encoder.pth"),
            os.path.join("outputs/base64", f"pretrained_{cfg['model']['backbone']}_encoder.pth"),
        ]
        for p in candidates:
            if os.path.exists(p):
                sd = torch.load(p, map_location=device)
                if 'fc.weight' in sd and sd['fc.weight'].shape != encoder.fc.weight.shape:
                    del sd['fc.weight']
                    if 'fc.bias' in sd:
                        del sd['fc.bias']
                miss, unexp = encoder.load_state_dict(sd, strict=False)
                print(f"   [OK] 加载 SupCon 预训练权重: {p}")
                print(f"       缺失 {len(miss)} 键, 意外 {len(unexp)} 键 (fc 随机初始化)")
                break
        else:
            print("   [INFO] 未找到预训练权重，从零开始训练")
    else:
        print("   (编码器从零开始训练，无 SupCon 预训练)")

    # 3. 创建 MAML 模型（编码器 + 分类头）
    model = MAMLModel(encoder, n_way=n_way).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   总参数: {total_params/1e6:.2f}M, 可训练: {trainable_params/1e6:.2f}M")

    # 4. 元优化器（只更新原始参数）
    meta_optimizer = optim.Adam(model.parameters(), lr=args.meta_lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        meta_optimizer, mode='max', factor=0.5, patience=5,
        min_lr=1e-6, verbose=True)

    # 5. 训练循环
    print(f"\n{'='*60}")
    print(f"🚀 MAML 元训练开始")
    print(f"{'='*60}")
    print(f"Episodes: {args.episodes} | Val interval: {args.val_interval}")
    print(f"Outer lr: {args.meta_lr} | Inner lr: {args.inner_lr}")
    print()

    best_val_acc = 0.0
    best_step = 0
    best_weights = None
    no_improve = 0

    for step in range(1, args.episodes + 1):
        model.train()
        meta_loss = 0.0

        # 采样一个 episode
        s_x, s_y, q_x, q_y = train_sampler.sample_episode()
        s_x, q_x, s_y, q_y = (
            s_x.to(device), q_x.to(device),
            s_y.to(device), q_y.to(device),
        )

        # 数据增强（可选）
        if not args.no_aug:
            s_x = augment_vibration_batch(
                s_x, noise_std=0.02, mask_ratio=0.1, scale_std=0.03)
            q_x = augment_vibration_batch(
                q_x, noise_std=0.02, mask_ratio=0.0, scale_std=0.03)

        # MAML：内循环适应 → 外循环损失
        query_pred = maml_forward(
            model, s_x, s_y, q_x,
            inner_steps=args.inner_steps,
            inner_lr=args.inner_lr,
            first_order=args.first_order,
        )

        outer_loss = F.cross_entropy(query_pred, q_y)

        # 外循环优化
        meta_optimizer.zero_grad()
        outer_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        meta_optimizer.step()

        # 训练准确率
        _, preds = torch.max(query_pred, dim=1)
        train_acc = (preds == q_y).float().mean().item()

        # 验证
        if step % args.val_interval == 0 or step == 1:
            model.eval()
            val_accs = []

            for _ in range(args.val_episodes):
                s_x, s_y, q_x, q_y = val_sampler.sample_episode()
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
                    val_accs.append(acc)

            mean_val_acc = np.mean(val_accs) * 100
            scheduler.step(mean_val_acc)

            if mean_val_acc > best_val_acc:
                best_val_acc = mean_val_acc
                best_step = step
                best_weights = {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                }
                no_improve = 0
            else:
                no_improve += 1

            print(f"Step {step:5d}/{args.episodes} | "
                  f"Loss: {outer_loss.item():.4f} | "
                  f"Train: {train_acc*100:.1f}% | "
                  f"Val: {mean_val_acc:.1f}% | "
                  f"Best: {best_val_acc:.1f}% (ep {best_step}) | "
                  f"LR: {meta_optimizer.param_groups[0]['lr']:.2e}")

            if no_improve >= args.early_stop:
                print(f"\n⏹️ 早停: {args.early_stop} 次验证无提升")
                break

    # 恢复最佳权重
    if best_weights is not None:
        model.load_state_dict(best_weights)

    # 6. 保存权重
    output_dir = os.path.join(cfg["paths"]["output_dir"], "maml")
    os.makedirs(output_dir, exist_ok=True)

    # 保存完整模型
    model_path = os.path.join(output_dir, "maml_model_final.pth")
    torch.save(model.state_dict(), model_path)

    # 仅保存编码器（便于与 ProtoNet 公平比较）
    encoder_path = os.path.join(output_dir, "maml_encoder_final.pth")
    torch.save(model.encoder.state_dict(), encoder_path)

    print(f"\n{'='*60}")
    print(f"✅ MAML 训练完成!")
    print(f"   最佳验证准确率: {best_val_acc:.1f}% (step {best_step})")
    print(f"   完整模型: {model_path}")
    print(f"   编码器: {encoder_path}")

    # 7. 测试集快速评估
    print(f"\n{'='*60}")
    print(f"📊 测试集快速评估 (500 episodes)")
    print(f"{'='*60}")
    test_configs = [
        (5, 1, 15, "5-way 1-shot"),
        (5, 5, 15, "5-way 5-shot"),
        (10, 1, 10, "10-way 1-shot"),
        (10, 5, 10, "10-way 5-shot"),
    ]
    for ways, shot, query, name in test_configs:
        sampler = EpisodicSampler(test_dataset, ways=ways, shot=shot, query=query)
        model.eval()
        accs = []
        for _ in range(args.eval_episodes):
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
        mean = np.mean(accs) * 100
        std = np.std(accs) * 100
        print(f"  {name:<18} → {mean:.1f}% ± {std:.1f}%")

    # 保存日志
    log_path = os.path.join(output_dir, "train_log.txt")
    with open(log_path, "w") as f:
        f.write(f"MAML Training - {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"Config: {args.config}\n")
        f.write(f"Episodes: {args.episodes}\n")
        f.write(f"First order: {args.first_order}\n")
        f.write(f"Inner steps: {args.inner_steps}, Inner lr: {args.inner_lr}\n")
        f.write(f"Meta lr: {args.meta_lr}\n")
        f.write(f"Best val acc: {best_val_acc:.1f}% (step {best_step})\n")
    print(f"\n   日志已保存: {log_path}")
    print(f"   💡 完整评估见: python eval_maml.py")


if __name__ == "__main__":
    main()
