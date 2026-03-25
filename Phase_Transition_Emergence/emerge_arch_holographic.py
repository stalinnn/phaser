#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
架构普适性与全息相变涌现训练：
测试不同架构（MLP, Transformer, Mamba-Proxy）和层数是否均能发生相变并坍缩到同一主曲线。
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_EXPERIMENT_ROOT = Path(__file__).resolve().parent
if str(_EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENT_ROOT))

import geoopt

from core.hyperbolic_space import clamp_to_ball, make_poincare_ball
from core.metric_alignment import (
    cosine_similarity_matrix,
    map_similarity_to_entanglement_proxy,
    metric_alignment_loss,
    upper_triangle_mask,
)
from probes.telm_monitor import collect_reading
from emerge_holographic_bulk import set_seed, load_texts, encode_euclidean, build_target_proxy

class MambaProxyLayer(nn.Module):
    """
    一个极简的 Mamba / SSM 代理层。
    包含 1D 卷积和门控线性单元 (GLU)，模拟 Mamba 的选择性状态空间机制在序列上的局部混合和门控特性。
    """
    def __init__(self, d_model):
        super().__init__()
        self.conv1d = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.linear_x = nn.Linear(d_model, d_model)
        self.linear_gate = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        # x: (B, L, D)
        residual = x
        x = self.norm(x)
        # Conv1d expects (B, D, L)
        x_conv = self.conv1d(x.transpose(1, 2)).transpose(1, 2)
        gate = torch.sigmoid(self.linear_gate(x))
        x_proj = self.linear_x(x_conv)
        out = gate * x_proj
        return self.out_proj(out) + residual

class HolographicArchitecture(nn.Module):
    def __init__(self, arch_type, input_dim, hidden_dim, hyp_dim, num_layers, c=1.0):
        super().__init__()
        self.arch_type = arch_type
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList()
        
        if arch_type == "mlp":
            for _ in range(num_layers):
                self.layers.append(nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.GELU(),
                    nn.LayerNorm(hidden_dim)
                ))
        elif arch_type == "transformer":
            layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim, 
                nhead=4, 
                dim_feedforward=hidden_dim*4, 
                batch_first=True, 
                norm_first=True
            )
            self.layers.append(nn.TransformerEncoder(layer, num_layers=num_layers))
        elif arch_type == "mamba":
            for _ in range(num_layers):
                self.layers.append(MambaProxyLayer(hidden_dim))
        else:
            raise ValueError(f"Unknown arch: {arch_type}")
            
        self.out_proj = nn.Linear(hidden_dim, hyp_dim)
        self.manifold = geoopt.PoincareBall(c=c)
        
    def forward(self, x_euc):
        # x_euc: (N, input_dim) -> Treat N nodes as a sequence of length N
        h = self.input_proj(x_euc).unsqueeze(0) # (1, N, hidden_dim)
        
        if self.arch_type == "mlp":
            for layer in self.layers:
                h = layer(h) + h # residual connection
        elif self.arch_type == "transformer":
            h = self.layers[0](h)
        elif self.arch_type == "mamba":
            for layer in self.layers:
                h = layer(h)
                
        h = h.squeeze(0) # (N, hidden_dim)
        z_euc = self.out_proj(h) * 0.1 # Initialize small to stay near origin
        
        # 将欧氏空间输出映射到庞加莱球 (以原点为切空间)
        z_hyp = self.manifold.expmap0(z_euc)
        return clamp_to_ball(z_hyp)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_arch(
    arch: str,
    num_layers: int,
    hidden_dim: int,
    num_nodes: int,
    hyp_dim: int,
    epochs: int,
    lr: float,
    batch_size: int,
    telm_every: int,
    data_path: str | None,
    run_dir: Path,
    device: torch.device,
    c: float = 1.0,
    seed: int = 42,
    force_synthetic: bool = False,
):
    set_seed(seed)
    run_dir.mkdir(parents=True, exist_ok=True)

    texts = load_texts(data_path, num_nodes)
    
    euc_emb = encode_euclidean(texts, device, force_synthetic=force_synthetic)
    target = build_target_proxy(euc_emb)
    n = target.size(0)
    input_dim = euc_emb.size(1)
    mask = upper_triangle_mask(n, device=device)

    model = HolographicArchitecture(
        arch_type=arch,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        hyp_dim=hyp_dim,
        num_layers=num_layers,
        c=c
    ).to(device)
    
    d_param = count_parameters(model)
    print(f"架构: {arch.upper()}, 层数: {num_layers}, 隐藏维数: {hidden_dim}")
    print(f"节点数: {n}, 参数量 D_param = {d_param}")
    
    # 因为参数现在是标准的欧氏权重，我们直接使用普通的 Adam 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history: list[dict] = []
    jsonl_path = run_dir / "telm_readings.jsonl"

    print("开始基于网络架构的全息涌现训练...")
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        z_hyp = model(euc_emb)
        loss = metric_alignment_loss(model.manifold, z_hyp, target, mask=mask)
        loss.backward()
        optimizer.step()

        step = epoch + 1
        if step % telm_every == 0 or step == 1:
            # 记录此时的 z_hyp 状态，注意我们要监控网络参数的梯度，而不仅是 z_hyp
            reading = collect_reading(
                step=step,
                loss=loss.item(),
                z_hyp=z_hyp.detach(), # 仅用于计算几何特征和秩
                params_with_grad=list(model.parameters()), # 计算整个架构的梯度范数
            )
            rec = reading.to_json_dict()
            rec["epoch"] = epoch + 1
            history.append(rec)
            with open(jsonl_path, "a", encoding="utf-8") as fj:
                fj.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(
                f"epoch {step:4d}  loss={loss.item():.6f}  "
                f"eff_rank={reading.effective_rank:.4f}  "
                f"grad_norm={reading.grad_norm:.6f}  "
                f"||x||_mean={reading.hyp_coord_norm_mean:.4f}"
            )

    torch.save(
        {
            "z_hyp": z_hyp.detach().cpu(),
            "target_proxy": target.cpu(),
            "euc_emb": euc_emb.detach().cpu(),
            "texts": texts,
            "config": {
                "arch": arch,
                "num_layers": num_layers,
                "hidden_dim": hidden_dim,
                "d_param": d_param,
                "num_nodes": n,
                "hyp_dim": hyp_dim,
                "epochs": epochs,
                "lr": lr,
                "batch_size": batch_size,
                "c": c,
            },
        },
        run_dir / "checkpoint.pt",
    )
    with open(run_dir / "history_summary.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    print(f"完成。日志: {jsonl_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", type=str, default="transformer", choices=["mlp", "transformer", "mamba"])
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_nodes", type=int, default=128)
    parser.add_argument("--hyp_dim", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--telm_every", type=int, default=10)
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--synthetic_euclidean", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_arch(
        arch=args.arch,
        num_layers=args.num_layers,
        hidden_dim=args.hidden_dim,
        num_nodes=args.num_nodes,
        hyp_dim=args.hyp_dim,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        telm_every=args.telm_every,
        data_path=None,
        run_dir=Path(args.run_dir),
        device=device,
        force_synthetic=args.synthetic_euclidean,
    )

if __name__ == "__main__":
    main()
