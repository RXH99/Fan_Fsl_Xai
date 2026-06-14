"""
MAML 元学习 — 纯 PyTorch 实现，无外部依赖

不使用 learn2learn，使用 PyTorch 自带的 torch.func.functional_call 实现。
兼容 Python 3.13，无需 C++ 编译工具。

实现原理:
  1. 用 functional_call 在不修改 model.parameters() 的情况下做前向
  2. 内循环: support set 上迭代计算梯度 → 更新副本参数
  3. 外循环: query set 上计算损失 → 二阶梯度回传

配置（RTX 3060 6GB）:
  - first_order=True (FOMAML): 约 3-3.5 GB
  - first_order=False (二阶): 约 5-6 GB
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch.func import functional_call
    HAS_FUNCTIONAL_CALL = True
except ImportError:
    HAS_FUNCTIONAL_CALL = False


class MAMLModel(nn.Module):
    """
    MAML 全模型：编码器 + 分类头

    编码器结构与 ProtoNet 的 SE-ResNet1D(base64) 完全一致，
    分类头使用无偏置线性层。
    """
    def __init__(self, encoder, n_way):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(encoder.output_dim, n_way)

    def forward(self, x):
        features = self.encoder(x)
        logits = self.classifier(features)
        return logits


def _get_params_dict(model):
    """获取模型的可训练参数作为扁平的 dict"""
    return {k: v for k, v in model.named_parameters() if v.requires_grad}


def _get_buffers_dict(model):
    """获取模型的 buffer (BN running stats 等)"""
    return dict(model.named_buffers())


def maml_adapt(model, support_x, support_y, params, buffers,
               inner_steps=5, inner_lr=0.01, first_order=True):
    """
    MAML 内循环适应

    在给定的 support set 上迭代优化参数副本，不修改原始模型参数。

    Args:
        model: MAMLModel 实例
        support_x: (n_way*n_shot, 1, 1024)
        support_y: (n_way*n_shot,)
        params: 当前参数字典
        buffers: 当前 buffer 字典
        inner_steps: 内循环步数
        inner_lr: 内循环学习率
        first_order: 一阶近似（FOMAML），省显存
    Returns:
        adapted_params: 适应后的参数字典
    """
    adapted = {k: v.clone() for k, v in params.items()}

    for _ in range(inner_steps):
        pred = functional_call(model, (adapted, buffers), support_x)
        loss = F.cross_entropy(pred, support_y)

        # 计算梯度（一阶不需要 create_graph）
        grads = torch.autograd.grad(
            loss, adapted.values(),
            create_graph=not first_order,
        )

        # SGD 更新副本参数
        adapted = {
            k: p - inner_lr * g
            for (k, p), g in zip(adapted.items(), grads)
        }

    return adapted


def maml_forward(model, support_x, support_y, query_x,
                 inner_steps=5, inner_lr=0.01, first_order=True):
    """
    MAML 完整前向：内循环适应 → 外循环预测

    Args:
        model: MAMLModel
        support_x, support_y: 支撑集
        query_x: 查询集
    Returns:
        query_pred: (n_query, n_way) 预测 logits
    """
    params = _get_params_dict(model)
    buffers = _get_buffers_dict(model)

    adapted = maml_adapt(
        model, support_x, support_y,
        params, buffers,
        inner_steps=inner_steps,
        inner_lr=inner_lr,
        first_order=first_order,
    )

    query_pred = functional_call(model, (adapted, buffers), query_x)
    return query_pred


def maml_evaluate(model, support_x, support_y, query_x, query_y,
                  inner_steps=5, inner_lr=0.01, first_order=True):
    """
    MAML 评估单 episode 准确率

    不跟踪梯度（eval 模式）。
    """
    params = _get_params_dict(model)
    buffers = _get_buffers_dict(model)

    with torch.set_grad_enabled(True):
        adapted = maml_adapt(
            model, support_x, support_y,
            params, buffers,
            inner_steps=inner_steps,
            inner_lr=inner_lr,
            first_order=first_order,
        )

    with torch.no_grad():
        query_pred = functional_call(model, (adapted, buffers), query_x)
        _, preds = torch.max(query_pred, dim=1)
        acc = (preds == query_y).float().mean().item()

    return acc
