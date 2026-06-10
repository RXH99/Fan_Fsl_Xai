"""
Step 6: t-SNE 特征空间可视化

展示编码器提取的特征在降维后各类别的聚类效果
训练类（153 类）和测试类（18 类）分开画

运行: python step6_tsne.py
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# -------------------------- 新增：解决中文显示问题 --------------------------
plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"] # 兼容不同系统的中文字体
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题
# ---------------------------------------------------------------------------

from src.config import Config
from src.data.dataset import FaultDataset
from src.models.encoder import create_encoder


def extract_embeddings(encoder, dataset, device):
    """提取整个数据集的编码器嵌入特征"""
    encoder.eval()
    all_emb = []
    all_labels = []

    with torch.no_grad():
        batch_size = 64
        for i in range(0, len(dataset), batch_size):
            batch_x = dataset.X[i:i+batch_size].to(device)
            batch_y = dataset.y[i:i+batch_size]
            emb = encoder(batch_x)
            all_emb.append(emb.cpu().numpy())
            all_labels.extend(batch_y.tolist())

    return np.concatenate(all_emb, axis=0), np.array(all_labels)


def plot_tsne(emb, labels, title, save_path, max_points=1000):
    """画 t-SNE 图（优化标注显示）"""
    # 如果数据太多，随机采样
    if len(emb) > max_points:
        idx = np.random.choice(len(emb), max_points, replace=False)
        emb = emb[idx]
        labels = labels[idx]

    # t-SNE 降维到 2D
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=500)
    emb_2d = tsne.fit_transform(emb)

    # 画图
    plt.figure(figsize=(12, 9))  # 增大画布，提升标注显示空间
    unique_labels = np.unique(labels)
    n_classes = len(unique_labels)

    # 适配颜色映射（类别数超过20时扩展颜色）
    if n_classes <= 20:
        colors = plt.cm.tab20(np.linspace(0, 1, n_classes))
    else:
        colors = plt.cm.tab20c(np.linspace(0, 1, n_classes))

    # 绘制散点图（调整标记大小，提升可读性）
    for i, label in enumerate(unique_labels):
        mask = (labels == label)
        color = colors[i % len(colors)]
        plt.scatter(emb_2d[mask, 0], emb_2d[mask, 1],
                   color=color, s=20, alpha=0.8, label=f'类#{label}',
                   edgecolors='none')

    # 设置标题和坐标轴（优化字体大小）
    plt.title(title, fontsize=16, pad=20)
    plt.xlabel("t-SNE 维度1", fontsize=12, labelpad=10)
    plt.ylabel("t-SNE 维度2", fontsize=12, labelpad=10)
    
    # -------------------------- 优化图例显示 --------------------------
    # 类别数过多时，图例分多列显示，避免超出画布
    n_cols = min(3, max(1, n_classes // 20))  # 每20类分一列，最多3列
    plt.legend(
        bbox_to_anchor=(1.02, 1), 
        loc='upper left',
        ncol=n_cols,  # 分栏显示
        markerscale=1.2, 
        fontsize=8 if n_classes > 50 else 10,  # 类别多则缩小字体
        frameon=False,  # 去掉图例边框，节省空间
        labelspacing=0.5  # 减小图例项间距
    )
    # -------------------------------------------------------------------

    plt.tight_layout()
    # 增加保存时的边距，确保图例不被截断
    plt.savefig(save_path, dpi=200, bbox_inches='tight', pad_inches=0.5)
    plt.close()
    print(f"  ✅ 已保存: {save_path}")


def run():
    config_path = "configs/baseline.yaml"
    cfg = Config(config_path)
    device = cfg.device

    npz_path = os.path.join(cfg.cfg["data"]["processed_dir"], "preprocessed.npz")
    encoder_path = os.path.join(cfg.cfg["paths"]["output_dir"], "fewshot_encoder.pth")

    if not os.path.exists(encoder_path):
        print(f"❌ 找不到 {encoder_path}，请先运行 step3_train_fewshot.py")
        sys.exit(1)

    print(f"设备: {device}")

    encoder = create_encoder("cnn").to(device)
    encoder.load_state_dict(torch.load(encoder_path, map_location=device))
    print(f"✅ 加载编码器")

    output_dir = os.path.join(cfg.cfg["paths"]["output_dir"], "tsne")
    os.makedirs(output_dir, exist_ok=True)

    # 1. 训练集特征（153 类）
    print("\n提取训练集特征...")
    train_dataset = FaultDataset(npz_path, split="train")
    train_emb, train_labels = extract_embeddings(encoder, train_dataset, device)
    print(f"  训练集: {train_emb.shape[0]} 样本, {len(np.unique(train_labels))} 类")

    # 训练集类太多（153），优化绘图参数
    plot_tsne(train_emb, train_labels,
              "训练集特征空间 (153 类)",
              os.path.join(output_dir, "tsne_train.png"),
              max_points=1500)

    # 2. 测试集特征（18 类）
    print("\n提取测试集特征...")
    test_dataset = FaultDataset(npz_path, split="test")
    test_emb, test_labels = extract_embeddings(encoder, test_dataset, device)
    print(f"  测试集: {test_emb.shape[0]} 样本, {len(np.unique(test_labels))} 类")

    plot_tsne(test_emb, test_labels,
              "测试集特征空间 (18 类)",
              os.path.join(output_dir, "tsne_test.png"),
              max_points=360)

    # 3. 训练 + 测试混合
    print("\n生成对照图...")
    all_emb = np.concatenate([train_emb, test_emb], axis=0)
    all_type = np.concatenate([
        np.zeros(len(train_emb)),    # 0 = 训练集
        np.ones(len(test_emb))       # 1 = 测试集
    ])

    # 降维
    print("  t-SNE 降维中（需要几秒）...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=50, max_iter=500)
    all_2d = tsne.fit_transform(all_emb)

    # 增大对比图画布，优化标注显示
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    # 左：按训练/测试标色（优化标记和字体）
    ax1.scatter(all_2d[all_type==0, 0], all_2d[all_type==0, 1],
               color='steelblue', s=10, alpha=0.6, label='训练 (153 类)')
    ax1.scatter(all_2d[all_type==1, 0], all_2d[all_type==1, 1],
               color='coral', s=25, alpha=0.9, label='测试 (18 类)')
    ax1.set_title("训练集 vs 测试集", fontsize=14, pad=15)
    ax1.legend(fontsize=11, frameon=False)
    ax1.set_xlabel("t-SNE 维度1", fontsize=11)
    ax1.set_ylabel("t-SNE 维度2", fontsize=11)

    # 右：按标签标色（只显示测试集，优化标注）
    test_colors = plt.cm.tab10(np.linspace(0, 1, 18))
    test_2d = all_2d[all_type==1]
    test_lbl = test_labels
    unique_test_labels = np.unique(test_lbl)
    
    for i, label in enumerate(unique_test_labels):
        mask = (test_lbl == label)
        ax2.scatter(test_2d[mask, 0], test_2d[mask, 1],
                   color=test_colors[i % 10], s=30, alpha=0.9,
                   label=f'类#{label}', edgecolors='black', linewidth=0.5)

    ax2.set_title("测试集各类别分布 (18 类)", fontsize=14, pad=15)
    # 优化测试集图例显示
    ax2.legend(
        bbox_to_anchor=(1.02, 1), 
        loc='upper left',
        fontsize=9, 
        markerscale=1.5,
        frameon=False,
        labelspacing=0.3
    )
    ax2.set_xlabel("t-SNE 维度1", fontsize=11)

    plt.tight_layout()
    # 确保对比图图例不被截断
    plt.savefig(os.path.join(output_dir, "tsne_comparison.png"), dpi=200, bbox_inches='tight', pad_inches=0.5)
    plt.close()
    print(f"  ✅ 已保存: {output_dir}/tsne_comparison.png")

    print(f"\n✅ t-SNE 可视化完成，文件在 {output_dir}/")


if __name__ == "__main__":
    run()