"""
小样本训练模块 — 升级版

新增:
  1. 数据增强（episode 采样时对 support/query 做增强）
  2. 余弦相似度（默认）替换欧氏距离
  3. 直推式推理训练模式
  4. ReduceLROnPlateau 调度
  5. 早停（Early Stopping）
"""

import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os

from ..data.dataset import EpisodicSampler
from ..data.augmentation import augment_vibration_batch
from ..models.prototypical import (
    prototypical_loss,
)


def train_fewshot(encoder, train_dataset, val_dataset, config, device,
                  method="ProtoNet_Cosine", sep_weight=0.05,
                  use_augmentation=True, use_early_stop=True,
                  patience=30):
    """
    训练入口

    Args:
        method:
          - 'ProtoNet_Cosine' (默认): 余弦相似度
          - 'ProtoNet_CNN': 欧氏距离
          - 'ProtoNet_Transductive': 直推式推理
          - 'ProtoNet_ResNet': 旧名兼容 → 映射到 Cosine
          - 'ProtoNet_CosineT': 旧名兼容
        use_augmentation: 是否在 episode 中做数据增强
        use_early_stop: 早停
        patience: 早停 patience
    """
    fs_cfg = config["training"]["fewshot"]
    episodes = fs_cfg["episodes"]
    val_episodes = fs_cfg["val_episodes"]
    ways = fs_cfg["ways"]
    shot = fs_cfg["shot"]
    query = fs_cfg["query"]
    lr = fs_cfg["lr"]

    # 方法映射（保持旧名兼容）
    method_map = {
        'ProtoNet_CNN': 'euclidean',
        'ProtoNet_ResNet': 'cosine',
        'ProtoNet_Cosine': 'cosine',
        'ProtoNet_CosineT': 'cosine',
        'ProtoNet_Transductive': 'transductive',
        'ProtoNet_Consistency': 'cosine',
    }
    proto_method = method_map.get(method, 'cosine')

    train_sampler = EpisodicSampler(train_dataset, ways=ways, shot=shot,
                                     query=query)
    val_sampler = EpisodicSampler(val_dataset, ways=ways, shot=shot,
                                   query=query)

    optimizer = optim.Adam(encoder.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=15,
        min_lr=1e-6, verbose=True)

    print(f"\n{'='*60}")
    print(f"小样本训练 — {method} (backbone={proto_method})")
    print(f"{'='*60}")
    print(f"配置: {ways}-way {shot}-shot, {episodes} episodes, lr={lr}")
    if use_augmentation:
        print(f"✅ 数据增强已启用")
    if sep_weight > 0:
        print(f"类间分离权重: {sep_weight}")
    if proto_method == 'transductive':
        print(f"直推式推理模式")

    best_val_acc = 0.0
    best_epoch = 0
    best_weights = None
    no_improve = 0

    log_interval = 200

    for ep in range(1, episodes + 1):
        encoder.train()

        # ===== 采样 episode =====
        s_x, s_y, q_x, q_y = train_sampler.sample_episode()

        s_x = s_x.to(device)
        q_x = q_x.to(device)
        s_y = s_y.to(device)
        q_y = q_y.to(device)

        # ===== 数据增强 =====
        if use_augmentation:
            s_x = augment_vibration_batch(
                s_x, noise_std=0.02, mask_ratio=0.1, scale_std=0.03)
            q_x = augment_vibration_batch(
                q_x, noise_std=0.02, mask_ratio=0.0, scale_std=0.03)

        # ===== 前向 =====
        trans_kwargs = {}
        if proto_method == 'transductive':
            trans_kwargs = {'num_steps': 5, 'tau': 0.5, 'mix_ratio': 0.7}

        loss, train_acc = prototypical_loss(
            encoder, s_x, s_y, q_x, q_y, device,
            sep_weight=sep_weight, method=proto_method, **trans_kwargs)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=5.0)
        optimizer.step()

        # ===== 验证 =====
        if ep % log_interval == 0 or ep == 1:
            encoder.eval()
            val_accs = []

            with torch.no_grad():
                for _ in range(val_episodes):
                    s_x, s_y, q_x, q_y = val_sampler.sample_episode()
                    s_x = s_x.to(device)
                    q_x = q_x.to(device)
                    s_y = s_y.to(device)
                    q_y = q_y.to(device)

                    # 验证时不加增强
                    _, acc = prototypical_loss(
                        encoder, s_x, s_y, q_x, q_y, device,
                        sep_weight=0, method=proto_method)

                    val_accs.append(acc)

            mean_val_acc = np.mean(val_accs) * 100

            # 学习率调度
            scheduler.step(mean_val_acc)

            if mean_val_acc > best_val_acc:
                best_val_acc = mean_val_acc
                best_epoch = ep
                best_weights = encoder.state_dict().copy()
                no_improve = 0
            else:
                no_improve += 1

            print(f"Ep {ep:4d}/{episodes} | "
                  f"Loss: {loss.item():.4f} | "
                  f"Train: {train_acc*100:.1f}% | "
                  f"Val: {mean_val_acc:.1f}% | "
                  f"Best: {best_val_acc:.1f}% (ep {best_epoch}) | "
                  f"LR: {optimizer.param_groups[0]['lr']:.2e}")

            # 早停
            if use_early_stop and no_improve >= patience:
                print(f"\n⏹️ 早停触发: {patience} 个 log 间隔无提升")
                break

    # 恢复最佳权重
    if best_weights is not None:
        encoder.load_state_dict(best_weights)

    output_dir = config["paths"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(
        output_dir, f"fewshot_encoder_{method.replace('/', '_')}.pth")
    torch.save(encoder.state_dict(), model_path)
    print(f"\n✅ {method} 训练完成！最佳验证准确率: {best_val_acc:.1f}%")
    print(f"   最佳 epoch: {best_epoch}, 模型保存: {model_path}")

    return encoder, best_val_acc
