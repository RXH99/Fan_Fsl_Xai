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


# ============ 跨注意力任务自适应模块 ============
class CrossAttentionModule(nn.Module):
    """
    轻量跨注意力：query 特征参照 support 样本做自适应

    让每个 query 样本通过注意力机制整合 support 集信息，
    输出 task-adapted query 特征。

    Architecture:
        Q (query) → Q_proj
        K, V (support) → K_proj, V_proj
        attn = softmax(Q × K^T / sqrt(d))
        out = attn × V
        out = LayerNorm(query + out)
        out = LayerNorm(out + FFN(out))
    """
    def __init__(self, d_model=128, dropout=0.1):
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        self.attn_dropout = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, query, support):
        """
        query:   (N_q, D)  query embeddings
        support: (N_s, D)  support embeddings
        Returns: (N_q, D)  adapted query embeddings
        """
        Q = self.q_proj(query)
        K = self.k_proj(support)
        V = self.v_proj(support)

        # 注意力权重
        attn = torch.mm(Q, K.t()) / (Q.size(-1) ** 0.5)  # (N_q, N_s)
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        # 加权求和 support 特征
        out = torch.mm(attn, V)  # (N_q, D)

        # 残差 + LayerNorm
        out = self.norm1(query + out)

        # FFN
        out = self.norm2(out + self.ffn(out))

        return out


# ============ 跨注意力 V2（多头 + 可选自注意力） ============
class CrossAttentionModuleV2(nn.Module):
    """
    增强版跨注意力模块

    改进 V1:
      1. 多头注意力（4 heads）→ 多角度同时关注 support
      2. 可选自注意力层 → query 间互相对齐特征
      3. 更高 dropout（0.3）→ 缓解过拟合

    Architecture:
        query → MultiheadCrossAttn(query, support) → LayerNorm
             → [可选] SelfAttn(query, query) → LayerNorm
             → FFN → LayerNorm
    """
    def __init__(self, d_model=128, nhead=4, dropout=0.3, use_self_attn=True):
        super().__init__()
        # 多头交叉注意力
        self.cross_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)

        # 可选自注意力
        self.use_self_attn = use_self_attn
        if use_self_attn:
            self.self_attn = nn.MultiheadAttention(
                d_model, nhead, dropout=dropout, batch_first=True)
            self.norm_self = nn.LayerNorm(d_model)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, support):
        # query: (N_q, D), support: (N_s, D)
        # MultiheadAttention 需要 (B, N, D) 格式
        q = query.unsqueeze(0)     # (1, N_q, D)
        s = support.unsqueeze(0)   # (1, N_s, D)

        # 交叉注意力
        out, _ = self.cross_attn(q, s, s)                 # (1, N_q, D)
        out = self.norm1(query + out.squeeze(0))          # (N_q, D)
        out = self.dropout(out)

        # 自注意力
        if self.use_self_attn:
            out_u = out.unsqueeze(0)                      # (1, N_q, D)
            out2, _ = self.self_attn(out_u, out_u, out_u) # (1, N_q, D)
            out = self.norm_self(out + out2.squeeze(0))   # (N_q, D)
            out = self.dropout(out)

        # FFN
        out = self.norm2(out + self.ffn(out))

        return out


# ============ 跨注意力 V1 损失 ============
def prototypical_loss_crossattn(encoder, cross_attn, support_x, support_y,
                                 query_x, query_y, device, sep_weight=0.0):
    """
    跨注意力增强的原型网络损失

    流程:
        1. 编码器提取 support / query 特征
        2. cross_attn 根据 support 集自适应调整 query 特征
        3. 余弦相似度分类
    """
    support_emb = F.normalize(encoder(support_x.to(device)), dim=1)
    query_emb = F.normalize(encoder(query_x.to(device)), dim=1)

    # 跨注意力：query 参照 support 做自适应
    adapted_query = cross_attn(query_emb, support_emb)
    adapted_query = F.normalize(adapted_query, dim=1)

    ways = len(torch.unique(support_y))
    prototypes = torch.stack([
        support_emb[support_y.to(device) == cls].mean(0)
        for cls in range(ways)
    ])
    prototypes = F.normalize(prototypes, dim=1)

    # 余弦相似度
    sims = torch.mm(adapted_query, prototypes.t())
    loss = F.cross_entropy(sims * 10.0, query_y.to(device))

    _, preds = torch.max(sims, dim=1)
    acc = (preds == query_y.to(device)).float().mean().item()

    return loss, acc


# ============ 评估包装 ============
def evaluate_crossattn(encoder, cross_attn, support_x, support_y,
                        query_x, query_y, device):
    """
    跨注意力评估（无梯度）
    """
    support_emb = F.normalize(encoder(support_x), dim=1)
    query_emb = F.normalize(encoder(query_x), dim=1)
    adapted_query = F.normalize(cross_attn(query_emb, support_emb), dim=1)

    ways = len(torch.unique(support_y))
    prototypes = torch.stack([
        support_emb[support_y == cls].mean(0) for cls in range(ways)
    ])
    prototypes = F.normalize(prototypes, dim=1)

    sims = torch.mm(adapted_query, prototypes.t())
    _, preds = torch.max(sims, dim=1)
    acc = (preds == query_y).float().mean().item()
    return acc


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
