import math
import numpy as np
import json
import torch
from pathlib import Path

def process(run_dir):
    ckpt_path = Path(run_dir) / "checkpoint.pt"
    data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    euc_emb = data["euc_emb"]
    euc_emb_centered = euc_emb - euc_emb.mean(dim=0, keepdim=True)
    cov = (euc_emb_centered.T @ euc_emb_centered) / euc_emb.size(0)
    data_cov_norm = float(torch.linalg.matrix_norm(cov, ord='fro').item())
    
    lr = data["config"].get("lr", 0.08)
    batch_size = data["config"].get("batch_size", 1)
    hyp_dim = data["config"].get("hyp_dim", 16)
    
    jsonl_path = Path(run_dir) / "telm_readings.jsonl"
    eff_ranks = []
    losses = []
    with open(jsonl_path, "r") as f:
        for line in f:
            if not line.strip(): continue
            rec = json.loads(line)
            eff_ranks.append(rec["effective_rank"])
            losses.append(rec["loss"])
            
    losses = np.array(losses)
    loss_diff2 = np.gradient(np.gradient(losses))
    safe_loss_diff2 = np.maximum(np.abs(loss_diff2), 1e-8)
    
    # Calculate Lambda without hyp_dim
    lambda_t_no_dim = (data_cov_norm * batch_size) / (lr * safe_loss_diff2)
    # Calculate Lambda with hyp_dim (e.g. dividing by sqrt(hyp_dim) or hyp_dim)
    lambda_t_with_dim = (data_cov_norm * batch_size) / (lr * safe_loss_diff2 * hyp_dim)
    
    print(f"{run_dir}: hyp_dim={hyp_dim}, max_lambda_no_dim={np.max(lambda_t_no_dim):.2e}, max_lambda_with_dim={np.max(lambda_t_with_dim):.2e}")

process("runs/collapse_dim8")
process("runs/collapse_dim16")
process("runs/collapse_dim32")
