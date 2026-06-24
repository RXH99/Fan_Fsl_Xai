"""
Step 3: 小样本训练 — 升级版

新增方法:
  - ProtoNet_Cosine:    余弦相似度（默认，推荐）
  - ProtoNet_Transductive: 直推式推理
保留旧方法:
  - ProtoNet_CNN:       欧氏距离
  - ProtoNet_ResNet:    旧名 → 映射到 Cosine

用法:
  python step3_train_fewshot.py --method ProtoNet_Cosine
  python step3_train_fewshot.py --method ProtoNet_Transductive
  python step3_train_fewshot.py --method ProtoNet_CNN
  python step3_train_fewshot.py --no_aug          # 关闭数据增强
  python step3_train_fewshot.py --no_pretrain     # 不加载 SimCLR 权重
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse
import torch
import yaml
import numpy as np

from src.data.dataset import FaultDataset, EpisodicSampler
from src.models.encoder import create_encoder
from src.training.train_fewshot import train_fewshot
from src.models.prototypical import (
    prototypical_loss,
    CrossAttentionModule,
    CrossAttentionModuleV2,
    prototypical_loss_crossattn,
)


def parse_args():
    parser = argparse.ArgumentParser(description="小样本训练")
    parser.add_argument("--config", default="configs/clean.yaml",
                        help="配置文件路径（默认clean最终方法配置）")
    parser.add_argument("--method", default="ProtoNet_Cosine",
                        choices=["ProtoNet_CNN", "ProtoNet_ResNet",
                                 "ProtoNet_Cosine", "ProtoNet_CosineT",
                                 "ProtoNet_Transductive", "ProtoNet_CrossAttn",
                                 "ProtoNet_CrossAttnV2"])
    parser.add_argument("--no_aug", action="store_true",
                        help="关闭数据增强")
    parser.add_argument("--no_pretrain", action="store_true",
                        help="不加载 SimCLR 预训练权重")
    parser.add_argument("--sep", type=float, default=None,
                        help="类间分离损失权重（覆盖 config 值）")
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    npz_path = os.path.join(cfg["data"]["processed_dir"], "preprocessed.npz")
    train_dataset = FaultDataset(npz_path, split="train")
    val_dataset = FaultDataset(npz_path, split="val")
    test_dataset = FaultDataset(npz_path, split="test")

    backbone = cfg.get("model", {}).get("backbone", "resnet18")
    encoder_dim = cfg.get("model", {}).get("encoder_dim", 64)
    base_filters = cfg.get("model", {}).get("resnet", {}).get("base_filters", 32)
    use_multiscale = cfg.get("model", {}).get("resnet", {}).get("use_multiscale", True)

    method = args.method
    print(f"\n{'#'*60}")
    print(f"# 方法: {method}")
    print(f"{'#'*60}")

    cross_attn = None

    if method == "ProtoNet_CNN":
        encoder = create_encoder("cnn").to(device)
    elif method == "ProtoNet_CrossAttn":
        encoder = create_encoder(backbone, encoder_dim=encoder_dim,
                                 use_se=True,
                                 base_filters=base_filters,
                                 use_multiscale=use_multiscale).to(device)
        cross_attn = CrossAttentionModule(d_model=encoder_dim).to(device)
        ca_params = sum(p.numel() for p in cross_attn.parameters())
        print(f"[NEW] 跨注意力 V1 (参数: {ca_params/1e3:.1f}K)")
    elif method == "ProtoNet_CrossAttnV2":
        encoder = create_encoder(backbone, encoder_dim=encoder_dim,
                                 use_se=True,
                                 base_filters=base_filters,
                                 use_multiscale=use_multiscale).to(device)
        cross_attn = CrossAttentionModuleV2(
            d_model=encoder_dim, nhead=4, dropout=0.3,
            use_self_attn=True).to(device)
        ca_params = sum(p.numel() for p in cross_attn.parameters())
        print(f"[NEW] 跨注意力 V2 多头+自注意力 (参数: {ca_params/1e3:.1f}K)")
    else:
        encoder = create_encoder(backbone, encoder_dim=encoder_dim,
                                 use_se=True,
                                 base_filters=base_filters,
                                 use_multiscale=use_multiscale).to(device)

    # 加载 SimCLR 预训练权重
    if not args.no_pretrain and method != "ProtoNet_CNN":
        # 先查 config 指定路径，再 fallback 到默认 outputs/
        cand_paths = [
            os.path.join(cfg["paths"]["output_dir"],
                         f"pretrained_{backbone}_encoder.pth"),
            os.path.join("outputs", f"pretrained_{backbone}_encoder.pth"),
            os.path.join("outputs/base64", f"pretrained_{backbone}_encoder.pth"),
        ]
        pretrain_path = None
        for p in cand_paths:
            if os.path.exists(p):
                pretrain_path = p
                break

        if pretrain_path:
            sd = torch.load(pretrain_path, map_location=device)
            # 检查 fc 层维度是否匹配（处理 encoder_dim 变化）
            fc_key = 'fc.weight'
            if fc_key in sd and sd[fc_key].shape != encoder.fc.weight.shape:
                print(f"   [WARN] fc 维度不匹配: 预训练 {sd[fc_key].shape} → 模型 {encoder.fc.weight.shape}，跳过 fc")
                del sd[fc_key]
                if 'fc.bias' in sd:
                    del sd['fc.bias']
            miss, unexp = encoder.load_state_dict(sd, strict=False)
            print(f"[OK] 加载 SimCLR 预训练权重: {pretrain_path}")
            if miss:
                print(f"   缺失键: {len(miss)} (fc 层随机初始化)")
            if unexp:
                print(f"   意外键: {len(unexp)}")
        else:
            print(f"[INFO]  未找到 SimCLR 预训练权重，从头训练")

    sep_weight = args.sep if args.sep is not None else \
        cfg.get("training", {}).get("sep_weight", 0.05)

    ca_kwargs = {"cross_attn": cross_attn} if cross_attn is not None else {}
    encoder, best_val_acc = train_fewshot(
        encoder, train_dataset, val_dataset, cfg, device,
        method=method, sep_weight=sep_weight,
        use_augmentation=not args.no_aug, **ca_kwargs)

    # ===== 测试集全面评估 =====
    print(f"\n{'='*50}")
    print("📊 测试集评估")
    print(f"{'='*50}")

    # 方法映射
    method_map = {
        'ProtoNet_CNN': 'euclidean',
        'ProtoNet_ResNet': 'cosine',
        'ProtoNet_Cosine': 'cosine',
        'ProtoNet_CosineT': 'cosine',
        'ProtoNet_Transductive': 'transductive',
        'ProtoNet_CrossAttn': 'crossattn',
        'ProtoNet_CrossAttnV2': 'crossattn',
    }
    proto_method = method_map.get(method, 'cosine')

    test_configs = [
        (5, 1, 15, "5-way 1-shot"),
        (5, 5, 15, "5-way 5-shot"),
        (10, 1, 10, "10-way 1-shot"),
        (10, 5, 10, "10-way 5-shot"),
    ]

    for ways, shot, query, name in test_configs:
        sampler = EpisodicSampler(test_dataset, ways=ways, shot=shot,
                                   query=query)
        encoder.eval()
        if cross_attn is not None:
            cross_attn.eval()
        accs = []

        # Transductive / CrossAttn 评估参数
        trans_kwargs = {}
        if proto_method == 'transductive':
            trans_kwargs = {'num_steps': 10, 'tau': 0.5, 'mix_ratio': 0.7}

        with torch.no_grad():
            for _ in range(500):
                s_x, s_y, q_x, q_y = sampler.sample_episode()
                s_x = s_x.to(device)
                q_x = q_x.to(device)
                s_y = s_y.to(device)
                q_y = q_y.to(device)

                if cross_attn is not None:
                    _, acc = prototypical_loss_crossattn(
                        encoder, cross_attn, s_x, s_y, q_x, q_y,
                        device, sep_weight=0)
                else:
                    _, acc = prototypical_loss(
                        encoder, s_x, s_y, q_x, q_y, device,
                        sep_weight=0, method=proto_method, **trans_kwargs)
                accs.append(acc)

        mean_acc = np.mean(accs) * 100
        std_acc = np.std(accs) * 100
        print(f"  {name:<18} → {mean_acc:.1f}% ± {std_acc:.1f}%")


if __name__ == "__main__":
    main()
