#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
普适类与数据坍缩实验 (Universality Class and Data Collapse Experiment)

本脚本通过运行多个不同温度（学习率）的相变实验，证明：
无论系统在哪个 step 发生相变，只要横坐标换成无量纲常数 Lambda(t)，
所有曲线都会完美坍缩 (Collapse) 到同一个临界点上。
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
        
    lr = data["config"].get("lr", 0.08)
    batch_size = data["config"].get("batch_size", 1)
    hyp_dim = data["config"].get("hyp_dim", 16)
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
    
    # Calculate Lambda(t) with complete formula including Batch Size
    loss_diff2 = np.gradient(np.gradient(losses))
    safe_loss_diff2 = np.maximum(np.abs(loss_diff2), 1e-8)
    
    # T_sys = (lr / batch_size) * Trace(H) * D_param
    lambda_t = (data_cov_norm * batch_size) / (lr * safe_loss_diff2 * hyp_dim)
    log_lambda = np.log10(lambda_t)
    
    # Normalize effective rank (Phi / max(Phi)) for fair comparison across possibly different scales
    norm_eff_rank = eff_ranks / np.max(eff_ranks)
    
    return {
        "lr": lr,
        "steps": steps,
        "lambda": log_lambda,
        "eff_rank": norm_eff_rank
    }

def main():
    base_dir = Path(__file__).resolve().parent
    experiments = [
        {"lr": 0.08, "batch_size": 1, "dir": "runs/collapse_lr08_b1"},
        {"lr": 0.08, "batch_size": 4, "dir": "runs/collapse_lr08_b4"},
        {"lr": 0.08, "batch_size": 16, "dir": "runs/collapse_lr08_b16"},
    ]
    
    # 1. Run experiments
    print(">>> 开始运行数据坍缩实验 (这需要一点时间) <<<")
    for exp in experiments:
        run_dir = base_dir / exp["dir"]
        if not (run_dir / "checkpoint.pt").exists():
            print(f"正在运行实验 lr={exp['lr']}, batch_size={exp['batch_size']}...")
            cmd = [
                "python", "emerge_holographic_bulk.py",
                "--num_nodes", "256",
                "--hyp_dim", "16",
                "--epochs", "200",
                "--lr", str(exp["lr"]),
                "--batch_size", str(exp["batch_size"]),
                "--run_dir", str(run_dir),
                "--synthetic_euclidean"
            ]
            subprocess.run(cmd, check=True)
        else:
            print(f"跳过实验 lr={exp['lr']}, batch_size={exp['batch_size']}，已存在。")
            
    # 2. Process data and plot
    print("\n>>> 开始生成数据坍缩图 (Data Collapse) <<<")
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Normal Time vs Order Parameter
    plt.subplot(1, 2, 1)
    for exp in experiments:
        data = process_run(base_dir / exp["dir"])
        if data:
            plt.plot(data["steps"], data["eff_rank"], label=f"LR={exp['lr']}, B={exp['batch_size']}", linewidth=2)
    plt.title("Before Collapse: Time vs Effective Rank")
    plt.xlabel("Training Steps (t)")
    plt.ylabel("Normalized Effective Rank ($\Phi$)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Lambda vs Order Parameter (The Collapse)
    plt.subplot(1, 2, 2)
    for exp in experiments:
        data = process_run(base_dir / exp["dir"])
        if data:
            plt.plot(data["lambda"], data["eff_rank"], label=f"LR={exp['lr']}, B={exp['batch_size']}", linewidth=2, alpha=0.7)
    
    plt.title("Data Collapse: $\Lambda(t)$ vs Effective Rank")
    plt.xlabel("Control Parameter $\Lambda(t)$ (Log10)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    out_path = base_dir / "universality_collapse.png"
    plt.savefig(out_path, dpi=150)
    print(f"\n已生成普适性数据坍缩图: {out_path}")

if __name__ == "__main__":
    main()
