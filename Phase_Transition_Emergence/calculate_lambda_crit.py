#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
相变临界常数计算器 (Phase Transition Critical Constant Calculator)

基于大统一物理 AI 理论 (极简相变公式):
Lambda = S / T

其中：
1. S (结构化引力, Structure Gravity) -> 用边界语料的协方差矩阵范数代理 ||Data_Covariance||_F。代表将节点拉入逻辑树的纠缠力量。
2. T (系统热力学温度, Temperature) -> 由学习率 \eta 和损失流形曲率 Trace(H) 决定，T = \eta * Trace(H)。代表把节点推向无序的涨落力量。

本脚本尝试在有效秩(eff_rank)发生最大一阶导数（即断崖下跌）的时刻，测定这个动态比值。
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt


def calculate_data_covariance_norm(euc_emb: torch.Tensor) -> float:
    """计算边界CFT（欧氏）特征协方差矩阵的 Frobenius 范数"""
    # 减去均值
    euc_emb_centered = euc_emb - euc_emb.mean(dim=0, keepdim=True)
    # 协方差矩阵 (近似)
    cov = (euc_emb_centered.T @ euc_emb_centered) / euc_emb.size(0)
    return float(torch.linalg.matrix_norm(cov, ord='fro').item())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, default="runs/gpu_hotpotqa")
    args = parser.parse_args()
    run_dir = Path(args.run_dir)

    jsonl_path = run_dir / "telm_readings.jsonl"
    ckpt_path = run_dir / "checkpoint.pt"

    if not jsonl_path.exists() or not ckpt_path.exists():
        print(f"Error: 找不到数据 {run_dir}")
        return

    # 1. 加载边界纠缠数据（计算分子引力项）
    try:
        data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        data = torch.load(ckpt_path, map_location="cpu")
    
    euc_emb = data["euc_emb"]
    config = data["config"]
    lr = config.get("lr", 0.08)
    c = config.get("c", 1.0)
    hyp_dim = config.get("hyp_dim", 16)
    
    # ||Data_Cov_F|| 
    data_cov_norm = calculate_data_covariance_norm(euc_emb)
    print(f"[{run_dir}] 边界纠缠强度 (Data Covariance Norm): {data_cov_norm:.4f}")

    # 2. 读取演化序列
    steps = []
    eff_ranks = []
    grad_norms = []
    losses = []
    
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            rec = json.loads(line)
            steps.append(rec["step"])
            eff_ranks.append(rec["effective_rank"])
            grad_norms.append(rec["grad_norm"])
            losses.append(rec["loss"])
            
    eff_ranks = np.array(eff_ranks)
    grad_norms = np.array(grad_norms)
    losses = np.array(losses)

    # 3. 寻找相变点（有效秩的一阶导数最小值，即跌得最狠的地方）
    # 使用中心差分
    rank_diff = np.gradient(eff_ranks)
    phase_transition_idx = np.argmin(rank_diff)
    pt_step = steps[phase_transition_idx]
    
    print(f"\n>>>> 侦测到拓扑相变点 (Phase Transition Triggered) <<<<")
    print(f"相变发生 Step : {pt_step}")
    print(f"有效秩跳水   : {eff_ranks[max(0, phase_transition_idx-1)]:.2f} -> {eff_ranks[phase_transition_idx]:.2f}")
    
    # 4. 计算临界公式各项
    # 理论公式优化: Lambda(t) = S_{ent} / T
    # 分子: S_{ent} 边界语义纠缠熵 (用数据协方差范数代理)
    numerator = data_cov_norm
    
    # 分母: T 系统热力学温度 (由学习率和曲率决定, T ~ lr * Tr(H))
    # Hessian trace 约等于 loss 的二阶差分
    loss_diff2 = np.gradient(np.gradient(losses))
    approx_hessian_trace = abs(loss_diff2[phase_transition_idx])
    # 避免除以 0
    approx_hessian_trace = max(approx_hessian_trace, 1e-6)
    
    batch_size = config.get("batch_size", 1)
    temperature = (lr / batch_size) * approx_hessian_trace * hyp_dim
    denominator = temperature
    
    lambda_t = numerator / denominator
    
    print("\n[相变物理参量拆解]")
    print(f"引力项 (边界语义纠缠 S_ent): {numerator:.6f}")
    print(f"热力学涨落 (系统温度 T): {denominator:.6f}")
    print(f"==================================================")
    print(f"推算的临界比值 (引力/涨落) Lambda_crit = {math.log10(lambda_t):.4f} (Log10 scale)")
    print(f"==================================================")
    
    # 5. 绘制相变动力学公式曲线
    # 计算整个过程的 Lambda 演化 (Log10)
    safe_loss_diff2 = np.maximum(np.abs(loss_diff2), 1e-8)
    temperatures = (lr / batch_size) * safe_loss_diff2 * hyp_dim
    lambda_evolution = np.log10(numerator / temperatures)
    
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:red'
    ax1.set_xlabel('Training Steps / Holographic Depth (z)')
    ax1.set_ylabel('Effective Rank (Order Parameter \u03a6)', color=color)
    ax1.plot(steps, eff_ranks, color=color, linewidth=2, label="Effective Rank")
    ax1.tick_params(axis='y', labelcolor=color)
    
    # 标注相变点
    ax1.axvline(x=pt_step, color='k', linestyle='--', alpha=0.5, label='Phase Transition Cliff')

    ax2 = ax1.twinx()  
    color = 'tab:blue'
    ax2.set_ylabel('Control Parameter $\Lambda(t)$ (Log10)', color=color)  
    ax2.plot(steps, lambda_evolution, color=color, linewidth=1.5, alpha=0.7, label="$\Lambda(t)$")
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.title(f"Holographic Phase Transition Dynamics (Critical Step={pt_step})")
    out_path = run_dir / "lambda_critical_dynamics.png"
    plt.savefig(out_path, dpi=150)
    print(f"\n已生成临界常数演化图: {out_path}")

if __name__ == "__main__":
    main()
