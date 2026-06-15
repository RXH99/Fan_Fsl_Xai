"""
RelationNet 对比实验

在共享 SupCon 编码器（clean 模型）基础上，训练 Relation Module 替代余弦相似度。

流程:
  1. 加载预训练的 SupCon 编码器 (clean 模型)
  2. 冻结编码器，训练 Relation Module（500 episodes）
  3. 在各种 way/shot 配置下全面评估

运行:
  python eval_relationnet.py
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
from src.models.relationnet import RelationModule, predict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser(description="RelationNet 对比实验")
    parser.add_argument("--config", default="configs/clean.yaml",
                        help="配置文件（用于编码器结构）")
    parser.add_argument("--encoder_path", default="outputs/clean/fewshot_encoder_ProtoNet_Cosine.pth",
                        help="共享 SupCon 编码器权重路径")
    parser.add_argument("--relnet_episodes", type=int, default=1000,
                        help="Relation Module 训练 episode 数")
    parser.add_argument("--relnet_lr", type=float, default=0.001,
                        help="Relation Module 学习率")
    parser.add_argument("--eval_episodes", type=int, default=1000,
                        help="评估 episode 数")
    return parser.parse_args()


def train_relation_module(encoder, relnet, train_dataset, args):
    """
    冻结编码器，训练 Relation Module

    Args:
        encoder: 预训练编码器（冻结）
        relnet: RelationModule 实例
        train_dataset: 训练集（153 类）
        args: 命令行参数
    """
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    ways = cfg["training"]["fewshot"]["ways"]
    shot = cfg["training"]["fewshot"]["shot"]
    query = cfg["training"]["fewshot"]["query"]

    sampler = EpisodicSampler(train_dataset, ways=ways, shot=shot, query=query)
    optimizer = torch.optim.Adam(relnet.parameters(), lr=args.relnet_lr)

    encoder.eval()  # 冻结
    relnet.train()

    print(f"\n{'='*50}")
    print(f"🧪 训练 Relation Module (编码器冻结)")
    print(f"{'='*50}")
    print(f"  配置: {ways}-way {shot}-shot, {args.relnet_episodes} episodes")
    print(f"  Relation Module 参数: {sum(p.numel() for p in relnet.parameters()):,}")
    print(f"  LR: {args.relnet_lr}")

    log_interval = 200
    best_loss = float("inf")

    for ep in range(1, args.relnet_episodes + 1):
        s_x, s_y, q_x, q_y = sampler.sample_episode()
        s_x, q_x, s_y, q_y = (
            s_x.to(device), q_x.to(device),
            s_y.to(device), q_y.to(device),
        )

        # RelationNet 前向
        scores, _ = predict(encoder, relnet, s_x, s_y, q_x, device)

        # MSE 损失: 正确类目标为 1, 其余为 0
        target = F.one_hot(q_y, num_classes=ways).float().to(device)
        loss = F.mse_loss(scores, target)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(relnet.parameters(), max_norm=5.0)
        optimizer.step()

        if ep % log_interval == 0 or ep == 1:
            preds = torch.argmax(scores, dim=1)
            acc = (preds == q_y).float().mean().item() * 100
            print(f"  Ep {ep:4d}/{args.relnet_episodes} | Loss: {loss.item():.4f} | Train Acc: {acc:.1f}%")
            if loss.item() < best_loss:
                best_loss = loss.item()

    print(f"  ✅ 训练完成, 最佳 Loss: {best_loss:.4f}\n")
    return relnet


def evaluate(encoder, relnet, test_dataset, args, with_uwt=True):
    """
    全面评估 RelationNet

    Args:
        with_uwt: 是否在 RelationNet 基础上叠加 UWT
    """
    experiments = [
        ("5-way 1-shot",  5, 1, 15),
        ("5-way 5-shot",  5, 5, 15),
        ("10-way 1-shot", 10, 1, 10),
        ("10-way 5-shot", 10, 5, 10),
    ]

    print(f"\n{'='*50}")
    print(f"{'📊 RelationNet 对比评估' if not with_uwt else '📊 RelationNet + UWT 对比评估'}")
    print(f"{'='*50}")

    for name, ways, shot, query in experiments:
        sampler = EpisodicSampler(test_dataset, ways=ways, shot=shot, query=query)
        accs = []

        for _ in range(args.eval_episodes):
            s_x, s_y, q_x, q_y = sampler.sample_episode()
            s_x, q_x, s_y, q_y = (
                s_x.to(device), q_x.to(device),
                s_y.to(device), q_y.to(device),
            )

            with torch.no_grad():
                if with_uwt:
                    acc = _relnet_uwt(encoder, relnet, s_x, s_y, q_x, q_y, device)
                else:
                    scores, _ = predict(encoder, relnet, s_x, s_y, q_x, device)
                    preds = torch.argmax(scores, dim=1)
                    acc = (preds == q_y).float().mean().item()
                accs.append(acc)

        mean = np.mean(accs) * 100
        std = np.std(accs) * 100
        print(f"  {name:<18} → {mean:.1f}% ± {std:.1f}%")

    return


def _relnet_uwt(encoder, relnet, support_x, support_y, query_x, query_y, device,
                num_steps=3, tau=0.3, mix_ratio=0.8, beta=2.0):
    """
    RelationNet + UWT (不确定性加权直推式推理)

    与 ProtoNet + UWT 类似, 但用 Relation Module 的 softmax 分数做权重
    """
    s_feat = F.normalize(encoder(support_x), dim=1)
    q_feat = F.normalize(encoder(query_x), dim=1)

    ways = len(torch.unique(support_y))
    support_y_dev = support_y.to(device)

    prototypes = torch.stack([
        s_feat[support_y_dev == c].mean(0) for c in range(ways)
    ])
    prototypes = F.normalize(prototypes, dim=1)

    for _ in range(num_steps):
        # 用 Relation Module 分数作为 soft assignment
        q_exp = q_feat.repeat_interleave(ways, dim=0)
        p_exp = prototypes.repeat(q_feat.shape[0], 1)
        concat = torch.cat([q_exp, p_exp], dim=1)
        scores = relnet(concat).view(q_feat.shape[0], ways)

        # UWT 流程
        soft = torch.softmax(scores / tau, dim=1)
        entropy = -(soft * torch.log(soft + 1e-8)).sum(dim=1)
        entropy_norm = entropy / np.log(ways)
        weight = torch.exp(-entropy_norm * beta)
        weighted_soft = soft * weight.unsqueeze(1)

        new_protos = []
        for w in range(ways):
            ws = weighted_soft[:, w].sum()
            if ws > 1e-8:
                qp = (weighted_soft[:, w] @ q_feat) / ws
            else:
                qp = prototypes[w]
            new_protos.append(qp)
        new_protos = torch.stack(new_protos)
        new_protos = F.normalize(new_protos, dim=1)
        prototypes = F.normalize(
            mix_ratio * prototypes + (1 - mix_ratio) * new_protos, dim=1)

    # 最终分类: 用 Relation Module 在新原型上打分
    q_exp = q_feat.repeat_interleave(ways, dim=0)
    p_exp = prototypes.repeat(q_feat.shape[0], 1)
    concat = torch.cat([q_exp, p_exp], dim=1)
    final_scores = relnet(concat).view(q_feat.shape[0], ways)

    preds = torch.argmax(final_scores, dim=1)
    return (preds == query_y).float().mean().item()


def main():
    args = parse_args()
    print(f"🔧 设备: {device}")
    print(f"📄 编码器: {args.encoder_path}")
    print(f"📋 RelationNet episodes: {args.relnet_episodes}")

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # 1. 加载数据集
    npz_path = os.path.join(cfg["data"]["processed_dir"], "preprocessed.npz")
    train_dataset = FaultDataset(npz_path, split="train")
    test_dataset = FaultDataset(npz_path, split="test")

    # 2. 加载预训练编码器 — 从权重自动推断 use_multiscale
    sd = torch.load(args.encoder_path, map_location=device)
    fc_shape = sd.get('fc.weight', torch.zeros(0)).shape
    # base_filters=64 时: 多尺度 → 960, 单尺度 → 512
    use_ms = len(fc_shape) == 2 and fc_shape[1] == 960

    encoder = create_encoder("resnet18", encoder_dim=128, use_se=True,
                             base_filters=64, use_multiscale=use_ms).to(device)
    miss, unexp = encoder.load_state_dict(sd, strict=False)
    encoder.eval()
    print(f"  ✅ 编码器加载完成 ({sum(p.numel() for p in encoder.parameters())/1e6:.2f}M)")
    print(f"     use_multiscale={'是' if use_ms else '否'} | fc.shape={tuple(fc_shape)}")

    # 3. 创建 Relation Module
    relnet = RelationModule(feat_dim=128).to(device)

    # 4. 训练 Relation Module
    relnet = train_relation_module(encoder, relnet, train_dataset, args)

    # 5. 保存权重
    os.makedirs("outputs/relationnet", exist_ok=True)
    torch.save(relnet.state_dict(), "outputs/relationnet/relation_module.pth")
    print(f"  ✅ Relation Module 已保存: outputs/relationnet/relation_module.pth")

    # 6. 评估 (不含 UWT)
    evaluate(encoder, relnet, test_dataset, args, with_uwt=False)

    # 7. 评估 (含 UWT)
    evaluate(encoder, relnet, test_dataset, args, with_uwt=True)

    # 8. 与 ProtoNet 对比
    print("\n" + "=" * 50)
    print("📋 与 ProtoNet (Cosine) 对比 (引用 eval_clean 结果)")
    print("=" * 50)
    print(f"  {'Method':<30} {'5w1s':>8} {'5w5s':>8} {'10w1s':>8} {'10w5s':>8}")
    print(f"  {'ProtoNet + UWT (参考)':<30} {'93.3':>8} {'97.1':>8} {'87.3':>8} {'94.2':>8}")
    print(f"  {'RelationNet (当前)':<30} {'TBD':>8} {'TBD':>8} {'TBD':>8} {'TBD':>8}")
    print(f"  {'RelationNet + UWT (当前)':<30} {'TBD':>8} {'TBD':>8} {'TBD':>8} {'TBD':>8}")
    print("=" * 50)

    print("\n✅ 完成!")


if __name__ == "__main__":
    main()
