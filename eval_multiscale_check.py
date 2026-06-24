"""
快速验证：多尺度 vs 单尺度 — 完全公平对比

使用 return_features=True 跳过 fc 层，直接在卷积特征层上对比。
骨干网络完全相同，只差在特征聚合方式。

运行: python eval_multiscale_check.py --config configs/clean.yaml
"""

import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch, torch.nn.functional as F
import numpy as np
from src.data.dataset import FaultDataset, EpisodicSampler
from src.models.encoder import create_encoder
from src.utils import load_yaml_config, load_model_weights, check_gpu_availability

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/clean.yaml", help="配置文件路径")
    return parser.parse_args()

def main():
    args = parse_args()
    cfg = load_yaml_config(args.config)
    
    device = check_gpu_availability()
    CKPT = os.path.join(cfg['paths']['output_dir'], "fewshot_encoder_ProtoNet_Cosine.pth")
    npz_path = os.path.join(cfg['data']['processed_dir'], "preprocessed.npz")
    
    print(f"权重文件: {CKPT}")
    print(f"数据文件: {npz_path}\n")

    def evaluate_features(use_multiscale, beta=1.0):
        """
        使用 return_features 跳过 fc 层，直接在 conv 特征上做 ProtoNet
        这样骨干权重完全一样，只差在聚合方式
        """
        encoder = create_encoder("resnet18", encoder_dim=128, use_se=True,
                                 base_filters=64, use_multiscale=use_multiscale).to(device)
        
        try:
            sd = torch.load(CKPT, map_location=device, weights_only=True)
            if 'fc.weight' in sd and sd['fc.weight'].shape != encoder.fc.weight.shape:
                del sd['fc.weight']
                if 'fc.bias' in sd:
                    del sd['fc.bias']
            missing, unexpected = load_model_weights(encoder, CKPT, device, strict=False)
        except (FileNotFoundError, RuntimeError) as e:
            print(str(e))
            return None
        
        encoder.eval()

        test = FaultDataset(npz_path, split="test")
        feat_dim = 960 if use_multiscale else 512  # 特征维度

        def uwt_features(enc, sx, sy, qx, qy, beta=1.0):
            se = F.normalize(enc(sx, return_features=True), dim=1)
            qe = F.normalize(enc(qx, return_features=True), dim=1)
            w = len(torch.unique(sy))
            p = F.normalize(torch.stack([se[sy == c].mean(0) for c in range(w)]), dim=1)
            for _ in range(3):
                sft = torch.softmax(torch.mm(qe, p.t()) / 0.3, dim=1)
                wt = torch.exp(-(sft * torch.log(sft + 1e-8)).sum(1) / np.log(w) * beta)
                ws = sft * wt.unsqueeze(1)
                np_ = []
                for c in range(w):
                    wsc = ws[:, c].sum()
                    np_.append((ws[:, c] @ qe) / wsc if wsc > 1e-8 else p[c])
            p = F.normalize(0.8 * p + 0.2 * torch.stack(np_), dim=1)
            _, pd = torch.max(torch.mm(qe, p.t()), 1)
            return (pd == qy).float().mean().item()

        configs = [(5,1,15,"5-way 1-shot",3.0),(5,5,15,"5-way 5-shot",1.0),
                   (10,1,10,"10-way 1-shot",2.0),(10,5,10,"10-way 5-shot",1.0)]
        results = {}
        label = f"{'多尺度' if use_multiscale else '单尺度'} (conv特征 {feat_dim}维)"
        print(f"\n--- {label} ---")
        for ways, shot, query, name, b in configs:
            sp = EpisodicSampler(test, ways=ways, shot=shot, query=query)
            accs = []
            for _ in range(500):
                sx,sy,qx,qy = sp.sample_episode()
                sx,qx,sy,qy = sx.to(device), qx.to(device), sy.to(device), qy.to(device)
                with torch.no_grad():
                    accs.append(uwt_features(encoder, sx, sy, qx, qy, beta=b))
            m,s = np.mean(accs)*100, np.std(accs)*100
            results[name] = (m,s)
            print(f"  {name:<18} → {m:.1f}% ± {s:.1f}%")
        return results

    print("="*50)
    print("🧪 多尺度 vs 单尺度 — 公平对比")
    print("   同一骨干权重，return_features 跳过 fc")
    print("   只差在：多尺度=pool(f1~f4) / 单尺度=pool(f4)")
    print(f"权重: {CKPT}")
    print("="*50)

    r_ms = evaluate_features(use_multiscale=True)
    r_ss = evaluate_features(use_multiscale=False)

    print("\n" + "="*50)
    print("📊 对比汇总")
    print("="*50)
    print(f"  {'Setting':<18} {'多尺度':>13} {'单尺度':>13} {'差值':>8}")
    print(f"  {'-'*52}")
    for name in r_ms:
        ms_m, ms_s = r_ms[name]
        ss_m, ss_s = r_ss[name]
        diff = ms_m - ss_m
        print(f"  {name:<18} {ms_m:>6.1f}±{ms_s:.1f}%  {ss_m:>6.1f}±{ss_s:.1f}%  {diff:>+5.1f}%")
    print(f"  {'-'*52}")
    diffs = [r_ms[n][0]-r_ss[n][0] for n in r_ms]
    avg_diff = np.mean(diffs)
    print(f"  平均差值: {avg_diff:+.2f}%")
    if abs(avg_diff) < 0.5:
        print("  ✅ 多尺度无显著影响 (所有 setting 差值 < 0.5%)")
    elif abs(avg_diff) < 1.0:
        print("  ⚠️ 有微小差异 (0.5-1.0%)，需关注方向")
    else:
        print(f"  🔴 差异显著 ({avg_diff:.1f}%)，多尺度不能忽略")

if __name__ == "__main__":
    main()
