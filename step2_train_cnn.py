"""
Step 2: 全量训练分类器（预训练用）

在 153 类基类上训练完整分类器，
训练好的 encoder 用于初始化小样本训练。

用法:
  python step2_train_cnn.py                    # 默认 ResNet
  python step2_train_cnn.py --backbone cnn      # 原始 CNN
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml

from src.data.dataset import FaultDataset
from src.models.classifier import Classifier


def parse_args():
    parser = argparse.ArgumentParser(description="全量预训练")
    parser.add_argument("--config", default="configs/optimized.yaml")
    parser.add_argument("--backbone", default=None,
                        choices=["cnn", "resnet18", "multiscale_cnn"])
    parser.add_argument("--epochs", type=int, default=50)
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}")
    print(f"配置文件: {args.config}")

    npz_path = os.path.join(cfg["data"]["processed_dir"], "preprocessed.npz")

    # 加载 .npz 原始数据
    all_data = np.load(npz_path)
    X_all = torch.tensor(all_data["X_train"]).unsqueeze(1).float()
    y_all = torch.tensor(all_data["y_train"])

    # 从训练集 153 类中按类别划分 train/val
    classes = sorted(y_all.unique().tolist())
    random.shuffle(classes)
    split_idx = int(len(classes) * 0.9)
    train_classes = classes[:split_idx]
    val_classes = classes[split_idx:]
    print(f"划分: {len(train_classes)} 训练类 / {len(val_classes)} 验证类")

    # 按类别筛选 + 标签重映射
    train_mask = torch.tensor([l.item() in train_classes for l in y_all])
    val_mask = torch.tensor([l.item() in val_classes for l in y_all])

    train_y_orig = y_all[train_mask]
    val_y_orig = y_all[val_mask]

    train_label_map = {orig: i for i, orig in enumerate(sorted(train_y_orig.unique().tolist()))}
    val_label_map = {orig: i for i, orig in enumerate(sorted(val_y_orig.unique().tolist()))}

    train_dataset = FaultDataset(npz_path, split="train")  # dummy
    train_dataset.X = X_all[train_mask]
    train_dataset.y = torch.tensor([train_label_map[l.item()] for l in train_y_orig])

    val_dataset = FaultDataset(npz_path, split="train")    # dummy
    val_dataset.X = X_all[val_mask]
    val_dataset.y = torch.tensor([val_label_map[l.item()] for l in val_y_orig])

    num_classes = len(train_classes)
    print(f"标签映射: {num_classes} 类 (0~{num_classes-1})")
    print(f"数据: {len(train_dataset)} 训练, {len(val_dataset)} 验证")

    backbone = args.backbone or cfg.get("model", {}).get("backbone", "resnet18")
    encoder_dim = cfg.get("model", {}).get("encoder_dim", 64)

    print(f"\n全量分类器预训练: {backbone}, {num_classes} 类")

    # 创建模型
    model = Classifier(num_classes=num_classes, encoder_dim=encoder_dim,
                       backbone=backbone).to(device)

    batch_size = cfg["training"]["cnn"]["batch_size"]
    epochs = args.epochs or cfg["training"]["cnn"]["epochs"]
    lr = cfg["training"]["cnn"]["lr"]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    best_acc = 0.0
    best_weights = None

    print(f"\n{'='*50}")
    print(f"训练 {epochs} epochs, batch_size={batch_size}, lr={lr}")
    print(f"{'='*50}")

    for epoch in range(1, epochs + 1):
        # 训练
        model.train()
        loss_sum = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()

        avg_loss = loss_sum / len(train_loader)

        # 验证
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                _, preds = torch.max(model(inputs), 1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()

        acc = 100.0 * correct / total
        if acc > best_acc:
            best_acc = acc
            best_weights = model.state_dict().copy()

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:2d}/{epochs} | Loss: {avg_loss:.4f} | Val Acc: {acc:.2f}% | LR: {scheduler.get_last_lr()[0]:.2e}")

        scheduler.step()

    print(f"\n✅ 预训练完成！最高验证准确率: {best_acc:.2f}%")

    # 保存整个分类器
    output_dir = cfg["paths"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    # 保存完整模型（含分类头）
    model_path = os.path.join(output_dir, f"pretrained_{backbone}_full.pth")
    torch.save(best_weights, model_path)
    print(f"完整模型 (含头) 已保存: {model_path}")

    # 单独保存 encoder 权重（给小样本用）
    model.load_state_dict(best_weights)
    encoder_path = os.path.join(output_dir, f"pretrained_{backbone}_encoder.pth")
    torch.save(model.encoder.state_dict(), encoder_path)
    print(f"Encoder 权重已保存: {encoder_path}")


if __name__ == "__main__":
    main()
