"""
原型网络核心逻辑 — 升级版

新增功能:
  1. 余弦相似度 + 可学习温度（默认）
  2. 直推式推理（Transductive Inference）
  3. 类间分离损失（已增强）
  4. 保持与旧代码兼容
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ============ 工具函数 ============
def _compute_prototypes(support_emb, support_y, ways):
    """计算每个类的原型向量"""
    prototypes = []
    for cls in range(ways):
        cls_mask = (support_y == cls)
        proto = support_emb[cls_mask].mean(dim=0)
        prototypes.append(proto)
    return torch.stack(prototypes)


# ============ 欧氏距离 ProtoNet（原始版，保留兼容） ============
def prototypical_loss_euclidean(encoder, support_x, support_y,
                                 query_x, query_y, device, sep_weight=0.0,
                                 margin=10.0):
    """
    原始欧氏距离 ProtoNet 损失
    sep_weight > 0: 拉远不同类原型的距离
    """
    support_emb = encoder(support_x.to(device))
    query_emb = encoder(query_x.to(device))

    ways = len(torch.unique(support_y))
    prototypes = _compute_prototypes(support_emb, support_y, ways)

    dists = torch.cdist(query_emb.unsqueeze(1),
                        prototypes.unsqueeze(0)).squeeze(1)
    ce_loss = F.cross_entropy(-dists, query_y.to(device))

    # 类间分离损失
    if sep_weight > 0:
        proto_dists = torch.cdist(prototypes, prototypes)
        n = proto_dists.shape[0]
        mask = ~torch.eye(n, dtype=torch.bool, device=proto_dists.device)
        mean_dist = proto_dists[mask].mean()
        sep_loss = torch.clamp(margin - mean_dist, min=0)
        loss = ce_loss + sep_weight * sep_loss
    else:
        loss = ce_loss

    _, preds = torch.min(dists, dim=1)
    acc = (preds == query_y.to(device)).float().mean().item()

    return loss, acc


# ============ 余弦相似度 ProtoNet（推荐） ============
class CosineProtoNet(nn.Module):
    """
    余弦相似度 ProtoNet，带可学习温度

    用法:
        model = CosineProtoNet(encoder, init_temp=10.0)
        loss, acc = model(s_x, s_y, q_x, q_y)
    """
    def __init__(self, encoder, init_temp=10.0, temp_trainable=True):
        super().__init__()
        self.encoder = encoder
        if temp_trainable:
            self.temperature = nn.Parameter(torch.tensor(init_temp))
        else:
            self.register_buffer('temperature', torch.tensor(init_temp))

    def forward(self, support_x, support_y, query_x, query_y,
                sep_weight=0.0, margin=10.0):
        support_emb = F.normalize(self.encoder(support_x), dim=1)
        query_emb = F.normalize(self.encoder(query_x), dim=1)

        ways = len(torch.unique(support_y))
        prototypes = _compute_prototypes(support_emb, support_y, ways)

        # 余弦相似度
        sims = torch.mm(query_emb, prototypes.t())  # (Q, ways)
        loss = F.cross_entropy(sims * self.temperature, query_y)

        # 类间分离损失（提高原型可区分性）
        if sep_weight > 0:
            proto_sim = torch.mm(prototypes, prototypes.t())
            n = proto_sim.shape[0]
            mask = ~torch.eye(n, dtype=torch.bool, device=proto_sim.device)
            # 原型之间余弦相似度应低（即角度大）
            cos_sim = proto_sim[mask].mean()
            sep_loss = torch.clamp(cos_sim - (-margin/10.0), min=0)
            loss = loss + sep_weight * sep_loss

        _, preds = torch.max(sims, dim=1)
        acc = (preds == query_y).float().mean().item()

        return loss, acc


# ============ 直推式推断 + 余弦相似度 ============
def transductive_inference(encoder, support_x, support_y, query_x, query_y,
                            device, num_steps=5, tau=0.5, mix_ratio=0.7,
                            sep_weight=0.0):
    """
    直推式推理 Transductive ProtoNet

    用 query 样本迭代优化原型（self-training / soft assignment）

    Args:
        encoder: 特征提取器
        num_steps: 迭代步数
        tau: 温度（soft assignment 用）
        mix_ratio: support 原型权重（1-mix_ratio = query 贡献)
    Returns:
        loss, acc
    """
    support_emb = encoder(support_x.to(device))
    query_emb = encoder(query_x.to(device))

    # 归一化
    support_emb = F.normalize(support_emb, dim=1)
    query_emb = F.normalize(query_emb, dim=1)

    ways = len(torch.unique(support_y))

    # 初始原型（仅 support）
    prototypes = _compute_prototypes(support_emb, support_y, ways)
    prototypes = F.normalize(prototypes, dim=1)

    for step in range(num_steps):
        # query → 原型 的余弦相似度
        sims = torch.mm(query_emb, prototypes.t())  # (Q, ways)
        # soft assignment
        soft_assign = F.softmax(sims / tau, dim=1)  # (Q, ways)

        # 用 query 加权更新原型
        query_protos = []
        for w in range(ways):
            query_proto = (soft_assign[:, w] @ query_emb) / (soft_assign[:, w].sum() + 1e-8)
            query_protos.append(query_proto)
        query_protos = torch.stack(query_protos)  # (ways, D)
        query_protos = F.normalize(query_protos, dim=1)

        # 融合: mix_ratio * support_proto + (1-mix_ratio) * query_proto
        prototypes = F.normalize(
            mix_ratio * prototypes + (1 - mix_ratio) * query_protos, dim=1
        )

    # 最终分类
    final_sims = torch.mm(query_emb, prototypes.t())
    loss = F.cross_entropy(final_sims * 10.0, query_y.to(device))

    # 类间分离
    if sep_weight > 0:
        proto_sim = torch.mm(prototypes, prototypes.t())
        n = proto_sim.shape[0]
        mask = ~torch.eye(n, dtype=torch.bool, device=proto_sim.device)
        cos_sim = proto_sim[mask].mean()
        sep_loss = torch.clamp(cos_sim - 0.2, min=0)
        loss = loss + sep_weight * sep_loss

    _, preds = torch.max(final_sims, dim=1)
    acc = (preds == query_y.to(device)).float().mean().item()

    return loss, acc


# ============ 统一接口 ============
def prototypical_loss(encoder, support_x, support_y, query_x, query_y, device,
                       sep_weight=0.0, method='cosine', **kwargs):
    """
    统一损失函数入口

    Args:
        method: 'cosine' (默认，推荐), 'euclidean', 'transductive'
    """
    if method == 'euclidean':
        return prototypical_loss_euclidean(
            encoder, support_x, support_y, query_x, query_y, device,
            sep_weight=sep_weight)

    elif method == 'transductive':
        return transductive_inference(
            encoder, support_x, support_y, query_x, query_y, device,
            sep_weight=sep_weight, **kwargs)

    else:  # 'cosine' (default)
        support_emb = F.normalize(encoder(support_x.to(device)), dim=1)
        query_emb = F.normalize(encoder(query_x.to(device)), dim=1)

        ways = len(torch.unique(support_y))
        prototypes = _compute_prototypes(support_emb, support_y, ways)

        # 余弦相似度（温度=10.0）
        sims = torch.mm(query_emb, prototypes.t())
        loss = F.cross_entropy(sims * 10.0, query_y.to(device))

        if sep_weight > 0:
            proto_sim = torch.mm(prototypes, prototypes.t())
            n = proto_sim.shape[0]
            mask = ~torch.eye(n, dtype=torch.bool, device=proto_sim.device)
            cos_sim = proto_sim[mask].mean()
            sep_loss = torch.clamp(cos_sim + 0.5, min=0)
            loss = loss + sep_weight * sep_loss

        _, preds = torch.max(sims, dim=1)
        acc = (preds == query_y.to(device)).float().mean().item()

        return loss, acc


# 向后兼容
prototypical_loss_cosine = prototypical_loss
