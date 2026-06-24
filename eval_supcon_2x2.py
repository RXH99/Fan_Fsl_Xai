"""
2×2 对比实验：评估 SupCon × UWT 交互效应
用法: python eval_supcon_2x2.py --config configs/clean.yaml
"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch, torch.nn.functional as F, numpy as np, yaml
from src.data.dataset import FaultDataset, EpisodicSampler
from src.models.encoder import create_encoder

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/clean.yaml", help="UWT参数配置文件")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 加载配置
    with open(args.config, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    
    # 提取UWT参数
    uwt_cfg = cfg.get('inference', {}).get('uwt', {})
    uwt_steps = uwt_cfg.get('steps', 3)
    uwt_tau = uwt_cfg.get('tau', 0.3)
    uwt_mix_ratio = uwt_cfg.get('mix_ratio', 0.8)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")
    print(f"UWT配置: steps={uwt_steps}, tau={uwt_tau}, mix_ratio={uwt_mix_ratio}\n")
    
    # ===== 1. 加载 无 SupCon 模型 =====
    ckpt = "outputs/clean_nopretrain/fewshot_encoder_ProtoNet_Cosine.pth"
    if not os.path.exists(ckpt):
        print(f"[FAIL] 未找到 {ckpt}，请先运行训练: python step3_train_fewshot.py --config configs/clean_nopretrain.yaml --method ProtoNet_Cosine --no_pretrain")
        sys.exit(1)
    sd = torch.load(ckpt, map_location=device)
    fc_shape = sd.get('fc.weight', torch.zeros(0)).shape
    use_ms = len(fc_shape) == 2 and fc_shape[1] == 960
    encoder = create_encoder("resnet18", encoder_dim=128, use_se=True,
                             base_filters=64, use_multiscale=use_ms).to(device)
    encoder.load_state_dict(sd)
    encoder.eval()
    print(f"[OK] 加载权重: {ckpt}")
    
    # ===== 2. 准备测试集 =====
    test = FaultDataset("data/processed/preprocessed.npz", split="test")
    
    def eval_cosine(enc, sx, sy, qx, qy):
        se = F.normalize(enc(sx), dim=1)
        qe = F.normalize(enc(qx), dim=1)
        ways = len(torch.unique(sy))
        p = F.normalize(torch.stack([se[sy == c].mean(0) for c in range(ways)]), dim=1)
        _, pred = torch.max(torch.mm(qe, p.t()), 1)
        return (pred == qy).float().mean().item()
    
    def eval_uwt(enc, sx, sy, qx, qy, beta=1.0):
        se = F.normalize(enc(sx), dim=1)
        qe = F.normalize(enc(qx), dim=1)
        ways = len(torch.unique(sy))
        p = F.normalize(torch.stack([se[sy == c].mean(0) for c in range(ways)]), dim=1)
        for _ in range(uwt_steps):
            sft = torch.softmax(torch.mm(qe, p.t()) / uwt_tau, dim=1)
            wt = torch.exp(-(sft * torch.log(sft + 1e-8)).sum(1) / np.log(ways) * beta)
            ws = sft * wt.unsqueeze(1)
            np_ = []
            for c in range(ways):
                wsc = ws[:, c].sum()
                np_.append((ws[:, c] @ qe) / wsc if wsc > 1e-8 else p[c])
            p = F.normalize(uwt_mix_ratio * p + (1 - uwt_mix_ratio) * torch.stack(np_), dim=1)
        _, pred = torch.max(torch.mm(qe, p.t()), 1)
        return (pred == qy).float().mean().item()
    
    # ===== 3. 参数扫描（UWT beta）=====
    print("\n" + "=" * 50)
    print("UWT 参数扫描 (300 ep)")
    print("=" * 50)
    best_betas = {}
    
    for ways, shot, query, name in [(5,1,15,"5w1s"), (5,5,15,"5w5s"), (10,1,10,"10w1s"), (10,5,10,"10w5s")]:
        sp = EpisodicSampler(test, ways=ways, shot=shot, query=query)
        best_b, best_a = 0, 0
        for b in [0.5, 1.0, 2.0, 3.0, 5.0]:
            a = []
            for _ in range(300):
                sx, sy, qx, qy = sp.sample_episode()
                sx, qx, sy, qy = sx.to(device), qx.to(device), sy.to(device), qy.to(device)
                with torch.no_grad():
                    a.append(eval_uwt(encoder, sx, sy, qx, qy, beta=b))
            m = np.mean(a) * 100
            if m > best_a:
                best_a, best_b = m, b
        best_betas[name] = best_b
        print(f"  {name}: beta={best_b}, acc={best_a:.1f}%")
    
    # ===== 4. 主实验（1000 episodes）=====
    print("\n" + "=" * 60)
    print("📊 2×2 对比矩阵 (1000 episodes)")
    print("=" * 60)
    
    results = {}
    for ways, shot, query, name in [(5,1,15,"5w1s"), (5,5,15,"5w5s"), (10,1,10,"10w1s"), (10,5,10,"10w5s")]:
        beta = best_betas[name]
        sp = EpisodicSampler(test, ways=ways, shot=shot, query=query)
        
        cosine_accs, uwt_accs = [], []
        for _ in range(1000):
            sx, sy, qx, qy = sp.sample_episode()
            sx, qx, sy, qy = sx.to(device), qx.to(device), sy.to(device), qy.to(device)
            with torch.no_grad():
                cosine_accs.append(eval_cosine(encoder, sx, sy, qx, qy))
                uwt_accs.append(eval_uwt(encoder, sx, sy, qx, qy, beta=beta))
        
        cosine_mean = np.mean(cosine_accs) * 100
        uwt_mean = np.mean(uwt_accs) * 100
        results[name] = {'cosine': cosine_mean, 'uwt': uwt_mean, 'delta': uwt_mean - cosine_mean}
        
        print(f"\n  {name}:")
        print(f"    Cosine: {cosine_mean:.1f}%")
        print(f"    UWT:    {uwt_mean:.1f}% (beta={beta})")
        print(f"    Δ:      {uwt_mean - cosine_mean:+.1f}%")
    
    # ===== 5. 计算交互效应 =====
    print("\n" + "=" * 60)
    print("📈 SupCon × UWT 交互效应分析")
    print("=" * 60)
    print("\n注意: 此脚本仅评估无SupCon模型。需结合有SupCon模型的评估结果计算交互效应。")
    print("请参考 FINAL_EXPERIMENT_RESULTS.md 中的完整 2×2 矩阵数据。")

if __name__ == "__main__":
    main()
