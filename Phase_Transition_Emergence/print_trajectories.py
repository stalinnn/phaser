import numpy as np
import json
from pathlib import Path
import torch

def calc_cov(run_dir):
    ckpt_path = Path(run_dir) / "checkpoint.pt"
    data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    euc_emb = data["euc_emb"]
    euc_emb_centered = euc_emb - euc_emb.mean(dim=0, keepdim=True)
    cov = (euc_emb_centered.T @ euc_emb_centered) / euc_emb.size(0)
    return float(torch.linalg.matrix_norm(cov, ord='fro').item())

def load_data(run_dir):
    jsonl_path = Path(run_dir) / "telm_readings.jsonl"
    ckpt_path = Path(run_dir) / "checkpoint.pt"
    
    data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    
    steps = []
    eff_ranks = []
    T_sys = []
    
    with open(jsonl_path, "r") as f:
        for line in f:
            if not line.strip(): continue
            rec = json.loads(line)
            steps.append(rec["step"])
            eff_ranks.append(rec["effective_rank"])
            T_sys.append(rec["physics_T"])
            
    return np.array(steps), np.array(eff_ranks) / 16.0, np.array(T_sys), calc_cov(run_dir), data["config"]

experiments = [
    {"arch": "mlp", "dir": "runs/arch_mlp_l2_d64"},
    {"arch": "transformer (L=2)", "dir": "runs/arch_tf_l2_d64"},
    {"arch": "transformer (L=3)", "dir": "runs/arch_tf_l3_d64"},
    {"arch": "mamba (L=3)", "dir": "runs/arch_mamba_l3_d64"},
]

print("="*80)
print(f"{'Architecture':<20} | {'Step':<5} | {'Norm Rank':<10} | {'T_sys':<10} | {'Log10(Lambda)':<15}")
print("="*80)

for exp in experiments:
    steps, ranks, Ts, cov, cfg = load_data(exp["dir"])
    
    # 理论 Lambda
    lambda_t = cov / (Ts * 16.0) 
    log_lambda = np.log10(lambda_t)
    
    # 选取关键的降落点打印
    for i in [0, 5, 10, 15, -1]:
        idx = min(i, len(steps)-1)
        if idx < 0: idx = len(steps) - 1
        print(f"{exp['arch']:<20} | {steps[idx]:<5} | {ranks[idx]:<10.4f} | {Ts[idx]:<10.4f} | {log_lambda[idx]:<15.4f}")
    print("-" * 80)
