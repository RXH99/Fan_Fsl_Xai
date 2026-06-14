"""
Relation Network 模块

在共享的 SupCon 编码器基础上，用可学习的 Relation Module 替代余弦相似度。
编码器冻结，只训练 Relation Module。

Relation Module:
  拼接 [query_feat, prototype_feat] → MLP → Sigmoid → 关系分数 [0,1]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RelationModule(nn.Module):
    """
    关系网络模块

    输入: 拼接的 [query_feat, prototype_feat] → 输出: 关系分数 [0,1]
    架构: MLP: 2*feat_dim → feat_dim → feat_dim//2 → 1 → Sigmoid
    """
    def __init__(self, feat_dim=128, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = feat_dim
        self.fc = nn.Sequential(
            nn.Linear(feat_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, concat_features):
        """
        Args:
            concat_features: (B, feat_dim*2) 拼接后的特征
        Returns:
            scores: (B, 1) 关系分数 [0,1]
        """
        return self.fc(concat_features)


def predict(encoder, relation_module, support_x, support_y, query_x, device):
    """
    RelationNet 前向推理

    Args:
        encoder: 预训练编码器（冻结）
        relation_module: 关系网络模块
        support_x: (n_way*n_shot, 1, 1024)
        support_y: (n_way*n_shot,)
        query_x: (n_query, 1, 1024)
        device: torch device
    Returns:
        scores: (n_query, n_way) 关系分数
        prototypes: (n_way, feat_dim) 原型向量
    """
    encoder.eval()
    relation_module.eval()

    with torch.no_grad():
        s_feat = F.normalize(encoder(support_x.to(device)), dim=1)
        q_feat = F.normalize(encoder(query_x.to(device)), dim=1)

    ways = len(torch.unique(support_y))
    support_y_dev = support_y.to(device)

    prototypes = torch.stack([
        s_feat[support_y_dev == c].mean(0) for c in range(ways)
    ])  # (n_way, feat_dim)

    n_query = q_feat.shape[0]

    # 拼接: 每个 query 与每个 prototype
    # q_feat: (n_query, D) → repeat_interleave(ways) → (n_query*ways, D)
    # prototypes: (ways, D) → repeat(n_query, 1) → (n_query*ways, D)
    q_expanded = q_feat.repeat_interleave(ways, dim=0)
    p_expanded = prototypes.repeat(n_query, 1)
    concat = torch.cat([q_expanded, p_expanded], dim=1)  # (n_query*ways, 2D)

    scores = relation_module(concat).view(n_query, ways)  # (n_query, ways)

    return scores, prototypes
