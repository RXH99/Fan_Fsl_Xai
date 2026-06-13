"""
快速验证：时频融合编码器前向传播
运行: python test_tf_fusion.py
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from src.models.encoder import create_encoder


def main():
    print("=" * 50)
    print("🔍 时频融合编码器 — 前向验证")
    print("=" * 50)

    # 1. 创建编码器（时域分支冻结）
    e = create_encoder(
        "time_frequency",
        encoder_dim=64,
        use_se=True,
        freeze_time_encoder=True,
        freq_branch_config=dict(
            n_fft=128,
            hop_length=64,
            win_length=128,
            base_filters=32,
            gate_init=0.5,
        ),
    )

    # 2. 前向
    x = torch.randn(4, 1, 1024)
    out = e(x)

    # 3. 统计
    total = sum(p.numel() for p in e.parameters())
    trainable = sum(p.numel() for p in e.parameters() if p.requires_grad)
    frozen = total - trainable
    gate = torch.sigmoid(e.gate_logit).item()

    print(f"\n✅ 前向通过")
    print(f"   输入形状: {x.shape}")
    print(f"   输出形状: {out.shape}")
    print(f"   总参数:    {total/1e3:.1f}K")
    print(f"   可训练:    {trainable/1e3:.1f}K  (频域分支 + 融合层)")
    print(f"   冻结:      {frozen/1e3:.1f}K  (时域分支)")
    print(f"   门控初值:  {gate:.3f}  (时域权重, 预期 ≈0.50)")

    # 4. 检查梯度状态
    assert out.shape == (4, 64), f"形状错误: {out.shape}"
    assert not e.time_encoder.conv1.weight.requires_grad, "时域分支应被冻结"
    assert e.freq_branch.conv[0].weight.requires_grad, "频域分支应可训练"
    assert e.gate_logit.requires_grad, "门控参数应可训练"
    print(f"   梯度状态:  ✅ 时域冻结 / 频域可训 / 门控可训")

    # 5. STFT 尺寸检查
    spec = torch.stft(
        x.squeeze(1), n_fft=128, hop_length=64, win_length=128,
        window=torch.hann_window(128), return_complex=True,
    )
    print(f"   STFT 形状: {spec.shape}  (B, F={spec.shape[1]}, T={spec.shape[2]})")

    print(f"\n{'=' * 50}")
    print("🎉 全部通过！可以继续下一步训练")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
