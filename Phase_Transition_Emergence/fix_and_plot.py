import numpy as np
import matplotlib.pyplot as plt
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
    d_param = data["config"].get("d_param", 1)
    # The crucial fix: D_param is huge (10^5). It completely dominates the denominator.
    # But thermodynamic degrees of freedom for the output manifold is bounded by hyp_dim * N,
    # NOT the internal neural network parameters. 
    # By dividing by d_param, we shifted the lines by log10(800,000) vs log10(17,000) = shift of 1.7 on the x-axis!
    
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

plt.figure(figsize=(12, 5))

# Plot 1
plt.subplot(1, 2, 1)
for exp in experiments:
    steps, ranks, Ts, cov, cfg = load_data(exp["dir"])
    plt.plot(steps, ranks, label=f"{exp['arch']} (P={cfg['d_param']})", linewidth=2)
plt.title("Before Collapse: Time vs Effective Rank")
plt.xlabel("Training Steps (t)")
plt.ylabel("Normalized Effective Rank ($\Phi$)")
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2
plt.subplot(1, 2, 2)
for exp in experiments:
    steps, ranks, Ts, cov, cfg = load_data(exp["dir"])
    
    # THE FIX: The thermodynamic system that undergoes phase transition is the OUTPUT MANIFOLD (N * d_hyp).
    # The neural network is just the heat bath generator / mapping function. 
    # The internal parameters (D_param) do NOT define the dimensionality of the order parameter space.
    # We must use d_hyp (16) or N*d_hyp, NOT D_param.
    # And we must use the actual theoretical temperature injected into the manifold.
    
    lambda_t = cov / (Ts * 16.0) # S_ent / (T_sys * d_hyp)
    # The current T_sys goes from 1.0 down to 0.001. So 1/T_sys goes from 1 to 1000.
    # Therefore lambda_t goes from ~cov/16 to ~1000*cov/16
    log_lambda = np.log10(lambda_t)
    
    # We plot log_lambda vs ranks
    plt.plot(log_lambda, ranks, label=f"{exp['arch']}", linewidth=2, alpha=0.7)

plt.title("Data Collapse: $\Lambda(t)$ vs Effective Rank")
plt.xlabel("Control Parameter $\Lambda(t)$ (Log10)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("architecture_collapse_fixed.png", dpi=150)
print("Saved architecture_collapse_fixed.png")
