"""
特征提取器（Encoder）

提供三种 Backbone:
  - CNNEncoder:      原有简单 CNN（基线）
  - ResNet1DEncoder: 1D ResNet-18 + SE 注意力（优化版）
  - MultiScaleCNN:   多尺度卷积（3/5/7 核并行，优化版备选）

输入: (B, 1, 1024)
输出: (B, D) 特征向量，D = encoder_dim
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============ SE 注意力模块 ============
class SEBlock(nn.Module):
    """Squeeze-and-Excitation 通道注意力"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, channels // reduction, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(channels // reduction, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: (B, C, L)
        w = self.fc(x)  # (B, C, 1)
        return x * w


# ============ ResNet1D 基础模块 ============
class BasicBlock1D(nn.Module):
    """1D 残差块，支持 SE"""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, use_se=False):
        super().__init__()
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.se = SEBlock(planes) if use_se else None

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_planes, planes * self.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * self.expansion),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.se is not None:
            out = self.se(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


# ============ ResNet1D-18 ============
class ResNet1DEncoder(nn.Module):
    """
    1D ResNet-18 特征提取器
    输入: (B, 1, 1024) → 输出: (B, encoder_dim)
    """
    def __init__(self, in_channels=1, base_filters=32, encoder_dim=64,
                 use_se=True, use_multiscale=True):
        super().__init__()
        self.in_planes = base_filters

        self.conv1 = nn.Conv1d(in_channels, base_filters,
                               kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(base_filters)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(base_filters, 2, stride=1, use_se=use_se)
        self.layer2 = self._make_layer(base_filters * 2, 2, stride=2, use_se=use_se)
        self.layer3 = self._make_layer(base_filters * 4, 2, stride=2, use_se=use_se)
        self.layer4 = self._make_layer(base_filters * 8, 2, stride=2, use_se=use_se)

        if use_multiscale:
            feat_dim = base_filters * 15  # layer1~4: (1+2+4+8) * base_f
        else:
            feat_dim = base_filters * 8   # 仅 layer4
        self.use_multiscale = use_multiscale
        self.ms_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(feat_dim, encoder_dim)
        self.output_dim = encoder_dim

    def _make_layer(self, planes, num_blocks, stride, use_se):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock1D(self.in_planes, planes, s, use_se=use_se))
            self.in_planes = planes * BasicBlock1D.expansion
        return nn.Sequential(*layers)

    def forward(self, x, return_features=False):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        f1 = self.layer1(out)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)

        if self.use_multiscale:
            p1 = self.ms_pool(f1).squeeze(-1)
            p2 = self.ms_pool(f2).squeeze(-1)
            p3 = self.ms_pool(f3).squeeze(-1)
            p4 = self.ms_pool(f4).squeeze(-1)
            out = torch.cat([p1, p2, p3, p4], dim=1)
        else:
            out = self.ms_pool(f4).squeeze(-1)

        if return_features:
            return out
        out = self.fc(out)
        return out


# ============ 原有简单 CNN（保留用作基线对比） ============
class CNNEncoder(nn.Module):
    """
    1D-CNN 特征提取器（与原项目完全一致）
    输入: (B, 1, 1024) → 输出: (B, 64)
    """
    def __init__(self, in_channels=1, hidden_dims=[16, 32, 64],
                 kernel_sizes=[15, 7, 3]):
        super().__init__()
        layers = []

        layers.extend([
            nn.Conv1d(in_channels, hidden_dims[0],
                      kernel_size=kernel_sizes[0], stride=2, padding=7),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(),
            nn.MaxPool1d(2),
        ])
        layers.extend([
            nn.Conv1d(hidden_dims[0], hidden_dims[1],
                      kernel_size=kernel_sizes[1], stride=1, padding=3),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.ReLU(),
            nn.MaxPool1d(2),
        ])
        layers.extend([
            nn.Conv1d(hidden_dims[1], hidden_dims[2],
                      kernel_size=kernel_sizes[2], stride=1, padding=1),
            nn.BatchNorm1d(hidden_dims[2]),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        ])

        self.features = nn.Sequential(*layers)
        self.output_dim = hidden_dims[2]

    def forward(self, x):
        return self.features(x).squeeze(-1)


# ============ 多尺度 CNN（备选） ============
class MultiScaleCNN(nn.Module):
    """
    多尺度 1D-CNN（核大小 3/5/7），输出 concat
    输入: (B, 1, 1024) → 输出: (B, encoder_dim)
    """
    def __init__(self, in_channels=1, base_filters=32, encoder_dim=64):
        super().__init__()

        def _conv_branch(kernel_size):
            padding = kernel_size // 2
            return nn.Sequential(
                nn.Conv1d(in_channels, base_filters,
                          kernel_size=kernel_size, stride=2, padding=padding),
                nn.BatchNorm1d(base_filters),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Conv1d(base_filters, base_filters * 2,
                          kernel_size=3, stride=1, padding=1),
                nn.BatchNorm1d(base_filters * 2),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
            )

        self.branch3 = _conv_branch(3)
        self.branch5 = _conv_branch(5)
        self.branch7 = _conv_branch(7)

        self.fc = nn.Linear(base_filters * 2 * 3, encoder_dim)
        self.output_dim = encoder_dim

    def forward(self, x):
        f3 = self.branch3(x).squeeze(-1)
        f5 = self.branch5(x).squeeze(-1)
        f7 = self.branch7(x).squeeze(-1)
        out = torch.cat([f3, f5, f7], dim=1)  # (B, 3 * base_filters * 2)
        out = self.fc(out)
        return out


# ============ 工厂方法 ============
def create_encoder(encoder_type="cnn", encoder_dim=64, use_se=True, **kwargs):
    """工厂方法"""
    if encoder_type == "cnn":
        return CNNEncoder(**kwargs)
    elif encoder_type == "resnet18":
        return ResNet1DEncoder(
            encoder_dim=encoder_dim, use_se=use_se, **kwargs)
    elif encoder_type == "multiscale_cnn":
        return MultiScaleCNN(encoder_dim=encoder_dim, **kwargs)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")
