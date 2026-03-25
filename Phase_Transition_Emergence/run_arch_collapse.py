#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
架构与参数量的数据坍缩实验 (Architecture & Parameter Scale Collapse Experiment)

验证：不同架构（MLP / Transformer / Mamba-Proxy）、不同层数和宽度导致的参数量极大差异，
在引入完整的热力学常数后，是否依然符合二阶相变普适类。
"""

import os
import subprocess
import json
import math
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def calculate_data_covariance_norm(euc_emb_path: str) -> float:
    try:
        import torch
        data = torch.load(euc_emb_path, map_location="cpu", weights_only=False)
    except Exception:
        import torch
        data = torch.load(euc_emb_path, map_location="cpu")
    euc_emb = data["euc_emb"]
    euc_emb_centered = euc_emb - euc_emb.mean(dim=0, keepdim=True)
    cov = (euc_emb_centered.T @ euc_emb_centered) / euc_emb.size(0)
    return float(torch.linalg.matrix_norm(cov, ord='fro').item())

def process_run(run_dir: str):
    run_path = Path(run_dir)
    jsonl_path = run_path / "telm_readings.jsonl"
    ckpt_path = run_path / "checkpoint.pt"
    
    if not jsonl_path.exists() or not ckpt_path.exists():
        return None
        
    try:
        import torch
        data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except:
        import torch
        data = torch.load(ckpt_path, map_location="cpu")
        
    config = data["config"]
    lr = config.get("lr", 0.01)
    batch_size = config.get("batch_size", 1)
    d_param = config.get("d_param", 1)
    d_hyp = config.get("hyp_dim", 16)
    
    data_cov_norm = calculate_data_covariance_norm(ckpt_path)
    
    steps = []
    eff_ranks = []
    losses = []
    
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            rec = json.loads(line)
            steps.append(rec["step"])
            eff_ranks.append(rec["effective_rank"])
            losses.append(rec["loss"])
            
    eff_ranks = np.array(eff_ranks)
    losses = np.array(losses)
    
    loss_diff2 = np.gradient(np.gradient(losses))
    safe_loss_diff2 = np.maximum(np.abs(loss_diff2), 1e-8)
    
    # T_sys = (lr / batch_size) * Trace(H) * D_param
    # D_param represent the dimensionality of the noise exploration space.
    lambda_t = (data_cov_norm * batch_size) / (lr * safe_loss_diff2 * d_param)
    log_lambda = np.log10(lambda_t)
    
    norm_eff_rank = eff_ranks / d_hyp
    
    return {
        "arch": config.get("arch", "unknown"),
        "layers": config.get("num_layers", 0),
        "d_param": d_param,
        "steps": steps,
        "lambda": log_lambda,
        "eff_rank": norm_eff_rank
    }

def main():
    base_dir = Path(__file__).resolve().parent
    experiments = [
        {"arch": "mlp", "layers": 2, "dim": 64, "lr": 0.01, "dir": "runs/arch_mlp_l2_d64"},
        {"arch": "transformer", "layers": 2, "dim": 64, "lr": 0.005, "dir": "runs/arch_tf_l2_d64"},
        {"arch": "transformer", "layers": 4, "dim": 128, "lr": 0.005, "dir": "runs/arch_tf_l4_d128"},
        {"arch": "mamba", "layers": 3, "dim": 64, "lr": 0.01, "dir": "runs/arch_mamba_l3_d64"},
    ]
    
    print(">>> 开始运行不同架构与参数量的坍缩实验 <<<")
    for exp in experiments:
        run_dir = base_dir / exp["dir"]
        if not (run_dir / "checkpoint.pt").exists():
            print(f"正在运行实验 {exp['arch']}, layers={exp['layers']}, dim={exp['dim']}...")
            cmd = [
                "python", "emerge_arch_holographic.py",
                "--arch", exp["arch"],
                "--num_layers", str(exp["layers"]),
                "--hidden_dim", str(exp["dim"]),
                "--num_nodes", "256",
                "--hyp_dim", "16",
                "--epochs", "200",
                "--lr", str(exp["lr"]),
                "--batch_size", "1",
                "--run_dir", str(run_dir),
                "--synthetic_euclidean"
            ]
            subprocess.run(cmd, check=True)
        else:
            print(f"跳过实验 {exp['arch']}, layers={exp['layers']}, dim={exp['dim']}，已存在。")
            
    print("\n>>> 开始生成架构坍缩图 (Architecture Collapse) <<<")
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    for exp in experiments:
        data = process_run(base_dir / exp["dir"])
        if data:
            label = f"{data['arch'].upper()} (L={data['layers']}, P={data['d_param']})"
            plt.plot(data["steps"], data["eff_rank"], label=label, linewidth=2)
    plt.title("Before Collapse: Time vs Effective Rank")
    plt.xlabel("Training Steps (t)")
    plt.ylabel("Normalized Effective Rank ($\Phi$)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    for exp in experiments:
        data = process_run(base_dir / exp["dir"])
        if data:
            label = f"{data['arch'].upper()} (L={data['layers']}, P={data['d_param']})"
            plt.plot(data["lambda"], data["eff_rank"], label=label, linewidth=2, alpha=0.7)
    
    plt.title("Data Collapse: $\Lambda(t)$ vs Effective Rank")
    plt.xlabel("Control Parameter $\Lambda(t)$ (Log10)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    out_path = base_dir / "architecture_collapse.png"
    plt.savefig(out_path, dpi=150)
    print(f"\n已生成架构参数规模坍缩图: {out_path}")

if __name__ == "__main__":
    main()
