"""
快速验证：时频注入编码器前向传播
运行: python test_tf_inject.py
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from src.models.encoder import create_encoder, TimeFrequencyInjectEncoder


def main():
    print("=" * 50)
    print("🔍 时频注入编码器 — 前向验证")
    print("=" * 50)

    e = create_encoder(
        "time_frequency_inject",
        encoder_dim=64,
        use_se=True,
        freeze_time_encoder=False,
        freq_branch_config=dict(
            n_fft=128, hop_length=64, win_length=128, base_filters=32,
        ),
    )

    x = torch.randn(4, 1, 1024)
    out = e(x)

    total = sum(p.numel() for p in e.parameters())
    trainable = sum(p.numel() for p in e.parameters() if p.requires_grad)

    print(f"\n✅ 前向通过")
    print(f"   输入形状: {x.shape}")
    print(f"   输出形状: {out.shape}")
    print(f"   总参数:    {total/1e3:.1f}K")
    print(f"   可训练:    {trainable/1e3:.1f}K")
    print(f"   (注入前 256 维 + 频域 64 维 → concat 320 → FC → 64)")

    assert out.shape == (4, 64), f"形状错误: {out.shape}"

    # 验证：时域分支 return_features 路径
    with torch.no_grad():
        feat = e.time_encoder(x, return_features=True)
        assert feat.shape == (4, 256), f"时域特征 shape: {feat.shape}"
        print(f"   时域中间特征: {feat.shape} ✅ (256 维 avgpool)")

    print(f"\n{'=' * 50}")
    print("🎉 通过！")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
