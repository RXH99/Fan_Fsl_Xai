"""
Step 2: 对比学习预训练

提供两种模式：
  1. SimCLR (默认): 无监督对比学习，适合无标签场景
  2. SupCon: 监督对比学习，利用 153 类的标签避免类碰撞

⚠️ 你当前数据每类仅 20 个样本，SimCLR 会把同类样本当作负样本推开
   （class collision），推荐使用 SupCon 模式。

用法:
  python step2_pretrain_simclr.py                          # SimCLR 模式
  python step2_pretrain_simclr.py --mode supcon            # SupCon 模式（推荐）
  python step2_pretrain_simclr.py --mode simclr --epochs 200
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
import numpy as np

from src.data.dataset import FaultDataset
from src.models.encoder import create_encoder
from src.data.augmentation import SimCLRAugment


# ============ 对比学习模型 ============
class ContrastiveModel(nn.Module):
    """编码器 + 投影头"""
    def __init__(self, encoder, projection_dim=128):
        super().__init__()
        self.encoder = encoder
        self.projection = nn.Sequential(
            nn.Linear(encoder.output_dim, encoder.output_dim),
            nn.ReLU(),
            nn.Linear(encoder.output_dim, projection_dim),
        )

    def forward(self, x):
        h = self.encoder(x)
        z = self.projection(h)
        return F.normalize(z, dim=1)


# ============ 损失函数 ============
def nt_xent_loss(z1, z2, temperature=0.1):
    """
    SimCLR 无监督对比损失（InfoNCE）
    ⚠️ 同类样本会被当作负样本推开（class collision 问题）
    """
    batch_size = z1.shape[0]
    device = z1.device

    z = torch.cat([z1, z2], dim=0)              # (2B, D)
    sim = torch.mm(z, z.t()) / temperature       # (2B, 2B)

    mask = ~torch.eye(2 * batch_size, dtype=torch.bool, device=device)

    pos_indices = torch.cat([
        torch.arange(batch_size, device=device) + batch_size,
        torch.arange(batch_size, device=device),
    ])
    batch_indices = torch.arange(2 * batch_size, device=device)

    neg_sim = sim[mask].view(2 * batch_size, -1)       # (2B, 2B-1)
    pos_sim = sim[batch_indices, pos_indices].unsqueeze(1)  # (2B, 1)

    logits = torch.cat([pos_sim, neg_sim], dim=1)      # (2B, 2B)
    labels = torch.zeros(2 * batch_size, dtype=torch.long, device=device)

    return F.cross_entropy(logits, labels)


def supcon_loss(z, labels, temperature=0.1):
    """
    Supervised Contrastive Loss (SupCon)
    利用标签信息，同一类的样本互为正样本，避免 class collision

    Args:
        z: (B, D) 归一化的投影向量
        labels: (B,) 类别标签
        temperature: 温度系数
    """
    batch_size = z.shape[0]
    device = z.device

    # 余弦相似度矩阵
    sim = torch.mm(z, z.t()) / temperature       # (B, B)

    # 正样本掩码：同类样本（排除自身）
    labels = labels.unsqueeze(1)                  # (B, 1)
    pos_mask = (labels == labels.t())             # (B, B)，同类为 True
    pos_mask.fill_diagonal_(False)                # 排除自身

    # 负样本掩码：不同类
    neg_mask = ~pos_mask                          # (B, B)
    neg_mask.fill_diagonal_(False)

    # 对每个样本计算对比损失
    loss = 0.0
    for i in range(batch_size):
        pos_i = sim[i][pos_mask[i]]               # 正样本相似度
        neg_i = sim[i][neg_mask[i]]               # 负样本相似度
        if len(pos_i) == 0:
            continue                              # 该 batch 只有 1 个该类的样本

        logits = torch.cat([pos_i.unsqueeze(0), neg_i.unsqueeze(0)], dim=0).T  # (N_pos + N_neg,)
        # logits 的格式：先把 pos_i 放前面，然后拼接 neg_i
        # 实际上 cross_entropy 需要 (N, C) 形状
        pos_count = len(pos_i)
        all_logits = torch.cat([pos_i, neg_i], dim=0).unsqueeze(0)  # (1, N_pos+N_neg)
        one_label = torch.zeros(1, dtype=torch.long, device=device)
        loss += F.cross_entropy(all_logits, one_label)

    return loss / batch_size


def supcon_loss_batch(z, labels, temperature=0.1):
    """
    SupCon 的向量化实现（更快）
    """
    batch_size = z.shape[0]
    device = z.device

    # 相似度矩阵
    sim = torch.mm(z, z.t()) / temperature       # (B, B)

    # 正样本掩码
    labels = labels.unsqueeze(1)
    pos_mask = (labels == labels.t()).float()     # (B, B)
    pos_mask.fill_diagonal_(0)                    # 排除自身

    # 每个样本的正样本数
    pos_counts = pos_mask.sum(dim=1)              # (B,)

    # 数值稳定性：减去最大值
    sim_max = sim.max(dim=1, keepdim=True)[0]
    sim_stable = sim - sim_max.detach()           # (B, B)

    # exp 后求和（分母 = 所有同一样本 + 所有负样本）
    exp_sim = torch.exp(sim_stable)               # (B, B)
    denominator = exp_sim.sum(dim=1)              # (B,)

    # 正样本的 numerator
    numerator = (exp_sim * pos_mask).sum(dim=1)   # (B,)

    # loss = -log( numerator / denominator ) = -(log numerator - log denominator)
    log_prob = torch.log(numerator + 1e-8) - torch.log(denominator + 1e-8)

    # 忽略没有正样本的样本
    valid = pos_counts > 0
    loss = -(log_prob * valid).sum() / (valid.sum() + 1e-8)

    return loss


# ============ 主函数 ============
def parse_args():
    parser = argparse.ArgumentParser(description="对比学习预训练")
    parser.add_argument("--config", default="configs/optimized.yaml")
    parser.add_argument("--mode", default="supcon",
                        choices=["simclr", "supcon"],
                        help="simclr=无监督, supcon=监督对比（推荐）")
    parser.add_argument("--epochs", type=int, default=200,
                        help="SupCon 建议 200 epoch, SimCLR 建议 300-500")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--workers", type=int, default=0,
                        help="DataLoader num_workers (Windows 建议 0)")
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")
    print(f"模式: {args.mode}")
    print(f"配置文件: {args.config}")

    npz_path = os.path.join(cfg["data"]["processed_dir"], "preprocessed.npz")
    train_dataset = FaultDataset(npz_path, split="train")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, drop_last=True,
                              num_workers=args.workers)

    backbone = cfg.get("model", {}).get("backbone", "resnet18")
    encoder_dim = cfg.get("model", {}).get("encoder_dim", 64)
    base_filters = cfg.get("model", {}).get("resnet", {}).get("base_filters", 32)

    encoder = create_encoder(backbone, encoder_dim=encoder_dim,
                             use_se=True, in_channels=1,
                             base_filters=base_filters).to(device)
    model = ContrastiveModel(encoder, projection_dim=128).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # 余弦退火 + 线性预热
    warmup_epochs = min(10, args.epochs // 10)
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / (args.epochs - warmup_epochs)
            return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # 增强器
    augmenter = SimCLRAugment(noise_std=0.03, mask_ratio=0.2, scale_std=0.05)

    print(f"\n{'='*60}")
    print(f"{'SupCon' if args.mode == 'supcon' else 'SimCLR'} 对比学习预训练")
    print(f"{'='*60}")
    print(f"数据: {len(train_dataset)} 样本 / {len(train_dataset.y.unique())} 类")
    print(f"Epochs: {args.epochs}, Batch: {args.batch_size}, LR: {args.lr}")
    print(f"温度: {args.temperature}, 预热: {warmup_epochs} epochs")

    best_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # 两种增强
            x1, x2 = augmenter(inputs)

            if args.mode == 'supcon':
                # SupCon: 两次增强拼接后算损失
                x_aug = torch.cat([x1, x2], dim=0)   # (2B, 1, 1024)
                z = model(x_aug)                      # (2B, D)
                loss = supcon_loss_batch(z, torch.cat([labels, labels], dim=0),
                                         temperature=args.temperature)
            else:
                # SimCLR: 标准 NT-Xent
                z1 = model(x1)
                z2 = model(x2)
                loss = nt_xent_loss(z1, z2, temperature=args.temperature)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        scheduler.step()

        if avg_loss < best_loss:
            best_loss = avg_loss

        if epoch % 20 == 0 or epoch == 1 or epoch == args.epochs:
            lr_now = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch:3d}/{args.epochs} | "
                  f"Loss: {avg_loss:.4f} | Best: {best_loss:.4f} | LR: {lr_now:.2e}")

    print(f"\n✅ 预训练完成！最小损失: {best_loss:.4f}")

    # 保存 encoder 权重
    output_dir = cfg["paths"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    encoder_path = os.path.join(output_dir, f"pretrained_{backbone}_encoder.pth")
    torch.save(model.encoder.state_dict(), encoder_path)
    print(f"Encoder 权重已保存: {encoder_path}")

    full_path = os.path.join(output_dir, f"{args.mode}_{backbone}_full.pth")
    torch.save(model.state_dict(), full_path)
    print(f"完整模型已保存: {full_path}")


if __name__ == "__main__":
    main()
