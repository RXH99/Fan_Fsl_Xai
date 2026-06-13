"""
验证多尺度聚合编码器
运行: python test_multiscale.py
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from src.models.encoder import create_encoder

e = create_encoder("resnet18", encoder_dim=128, use_se=True)
x = torch.randn(4, 1, 1024)
out = e(x)

# 前向验证
assert out.shape == (4, 128), f"形状错误: {out.shape}"

# 中间特征
with torch.no_grad():
    feat = e(x, return_features=True)
    print(f"多尺度聚合特征: {feat.shape}  ✅")
    assert feat.shape[1] == 32 * 15, f"特征维度错误: {feat.shape}"

total = sum(p.numel() for p in e.parameters())
print(f"输出形状: {out.shape}")
print(f"中间特征: {feat.shape} (480维 = 32+64+128+256)")
print(f"总参数:   {total/1e3:.1f}K")
print(f"✅ 前向验证通过")
