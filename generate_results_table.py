"""
自动生成实验结果汇总表

从多种子权重文件读取评估结果，生成 Markdown 表格
用法: python generate_results_table.py
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn.functional as F
import numpy as np
from src.data.dataset import FaultDataset, EpisodicSampler
from src.models.encoder import create_encoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_model(encoder_path, test_dataset, episodes=1000):
    """评估单个模型"""
    sd = torch.load(encoder_path, map_location=device)
    fc_shape = sd.get('fc.weight', torch.zeros(0)).shape
    use_ms = len(fc_shape) == 2 and fc_shape[1] == 960
    
    encoder = create_encoder("resnet18", encoder_dim=128, use_se=True,
                             base_filters=64, use_multiscale=use_ms).to(device)
    encoder.load_state_dict(sd)
    encoder.eval()
    
    def eval_cosine(sx, sy, qx, qy):
        se = F.normalize(encoder(sx), dim=1)
        qe = F.normalize(encoder(qx), dim=1)
        ways = len(torch.unique(sy))
        p = F.normalize(torch.stack([se[sy == c].mean(0) for c in range(ways)]), dim=1)
        _, pred = torch.max(torch.mm(qe, p.t()), 1)
        return (pred == qy).float().mean().item()
    
    def eval_uwt(sx, sy, qx, qy, beta=1.0):
        se = F.normalize(encoder(sx), dim=1)
        qe = F.normalize(encoder(qx), dim=1)
        ways = len(torch.unique(sy))
        p = F.normalize(torch.stack([se[sy == c].mean(0) for c in range(ways)]), dim=1)
        for _ in range(3):
            sft = torch.softmax(torch.mm(qe, p.t()) / 0.3, dim=1)
            wt = torch.exp(-(sft * torch.log(sft + 1e-8)).sum(1) / np.log(ways) * beta)
            ws = sft * wt.unsqueeze(1)
            np_ = []
            for c in range(ways):
                wsc = ws[:, c].sum()
                np_.append((ws[:, c] @ qe) / wsc if wsc > 1e-8 else p[c])
            p = F.normalize(0.8 * p + 0.2 * torch.stack(np_), dim=1)
        _, pred = torch.max(torch.mm(qe, p.t()), 1)
        return (pred == qy).float().mean().item()
    
    configs = [
        (5, 1, 15, "5-way 1-shot", 3.0),
        (5, 5, 15, "5-way 5-shot", 1.0),
        (10, 1, 10, "10-way 1-shot", 2.0),
        (10, 5, 10, "10-way 5-shot", 1.0),
    ]
    
    results = {}
    for ways, shot, query, name, beta in configs:
        sampler = EpisodicSampler(test_dataset, ways=ways, shot=shot, query=query)
        accs_cosine = []
        accs_uwt = []
        
        for _ in range(episodes):
            sx, sy, qx, qy = sampler.sample_episode()
            sx, qx, sy, qy = sx.to(device), qx.to(device), sy.to(device), qy.to(device)
            
            with torch.no_grad():
                accs_cosine.append(eval_cosine(sx, sy, qx, qy))
                accs_uwt.append(eval_uwt(sx, sy, qx, qy, beta=beta))
        
        results[name] = {
            'cosine': (np.mean(accs_cosine) * 100, np.std(accs_cosine) * 100),
            'uwt': (np.mean(accs_uwt) * 100, np.std(accs_uwt) * 100),
        }
    
    return results


def main():
    print("=" * 70)
    print("📊 自动生成实验结果汇总表")
    print("=" * 70)
    
    test_dataset = FaultDataset("data/processed/preprocessed.npz", split="test")
    
    # 检查多种子权重是否存在
    seed_paths = [
        "outputs/clean/fewshot_encoder_seed42.pth",
        "outputs/clean/fewshot_encoder_seed123.pth",
        "outputs/clean/fewshot_encoder_seed999.pth",
    ]
    
    if not all(os.path.exists(p) for p in seed_paths):
        print("⚠️  未找到多种子权重文件，请先运行:")
        print("   python run_seeded_ablation.py --seeds 42 123 999")
        print("\n或者使用默认单种子权重进行演示...")
        # 如果多种子不存在，尝试使用主权重
        seed_paths = ["outputs/clean/fewshot_encoder_ProtoNet_Cosine.pth"]
    
    # 评估每个种子
    all_results = []
    for seed_path in seed_paths:
        if not os.path.exists(seed_path):
            print(f"  ⚠️  跳过: {seed_path} (不存在)")
            continue
        print(f"\n评估: {os.path.basename(seed_path)}")
        results = evaluate_model(seed_path, test_dataset, episodes=500)  # 快速评估用500 episodes
        all_results.append(results)
    
    if not all_results:
        print("❌ 没有可评估的权重文件")
        return
    
    # 计算均值和标准差
    settings = ["5-way 1-shot", "5-way 5-shot", "10-way 1-shot", "10-way 5-shot"]
    
    print("\n" + "=" * 70)
    print("📋 Clean 模型结果" + (f" ({len(all_results)} seeds 均值)" if len(all_results) > 1 else ""))
    print("=" * 70)
    print(f"| Setting | Cosine | UWT | Δ |")
    print(f"|---|---|---|---|")
    
    for setting in settings:
        cosine_vals = [r[setting]['cosine'][0] for r in all_results]
        uwt_vals = [r[setting]['uwt'][0] for r in all_results]
        
        cosine_mean = np.mean(cosine_vals)
        cosine_std = np.std(cosine_vals)
        uwt_mean = np.mean(uwt_vals)
        uwt_std = np.std(uwt_vals)
        delta = uwt_mean - cosine_mean
        
        short_name = setting.replace("-way ", "w").replace("-shot", "s")
        if len(all_results) > 1:
            print(f"| **{short_name}** | {cosine_mean:.1f}% ± {cosine_std:.1f}% | **{uwt_mean:.1f}% ± {uwt_std:.1f}%** | +{delta:.1f}% |")
        else:
            print(f"| **{short_name}** | {cosine_mean:.1f}% | **{uwt_mean:.1f}%** | +{delta:.1f}% |")
    
    # 保存为 Markdown 文件
    output_file = "outputs/clean/results_table.md"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# Clean 模型实验结果\n\n")
        if len(all_results) > 1:
            f.write(f"> {len(all_results)} seeds × 500 episodes 均值 ± 标准差\n\n")
        else:
            f.write("> 单次运行结果（500 episodes）\n\n")
        f.write("| Setting | Cosine | UWT | Δ |\n")
        f.write("|---|---|---|---|\n")
        
        for setting in settings:
            cosine_vals = [r[setting]['cosine'][0] for r in all_results]
            uwt_vals = [r[setting]['uwt'][0] for r in all_results]
            
            cosine_mean = np.mean(cosine_vals)
            cosine_std = np.std(cosine_vals)
            uwt_mean = np.mean(uwt_vals)
            uwt_std = np.std(uwt_vals)
            delta = uwt_mean - cosine_mean
            
            short_name = setting.replace("-way ", "w").replace("-shot", "s")
            if len(all_results) > 1:
                f.write(f"| **{short_name}** | {cosine_mean:.1f}% ± {cosine_std:.1f}% | **{uwt_mean:.1f}% ± {uwt_std:.1f}%** | +{delta:.1f}% |\n")
            else:
                f.write(f"| **{short_name}** | {cosine_mean:.1f}% | **{uwt_mean:.1f}%** | +{delta:.1f}% |\n")
    
    print(f"\n✅ 结果表已保存: {output_file}")
    print(f"💡 可直接复制到 README.md 中")


if __name__ == "__main__":
    main()
