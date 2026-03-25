"""
TELM (Training Energy Landscape Monitor) 精简探针：有效秩、谱熵、梯度范数。

有效秩 (Effective Rank, Roy & Vetterli):
  设奇异值 σ_i，p_i = σ_i^2 / Σ σ_j^2，则 effective_rank = exp(-Σ p_i log p_i)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Any

import torch


@dataclass
class TELMReading:
    step: int
    loss: float
    effective_rank: float
    spectral_entropy: float
    grad_norm: float
    hyp_coord_norm_mean: float
    hyp_coord_norm_max: float

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


def effective_rank_from_matrix(z: torch.Tensor, eps: float = 1e-12) -> tuple[float, float]:
    """
    对矩阵 z [N, d] 计算有效秩与谱熵（基于奇异值能量分布）。

    Returns:
        eff_rank: 有效秩（连续值，约在 1..min(N,d)）
        spec_entropy: 归一化奇异值平方的 Shannon 熵
    """
    with torch.no_grad():
        # SVD 在 CPU 上更稳；小矩阵可放 GPU
        zf = z.detach().float()
        if zf.numel() == 0:
            return 0.0, 0.0
        try:
            _, s, _ = torch.linalg.svd(zf, full_matrices=False)
        except RuntimeError:
            # 回退：对称阵特征值
            s = torch.linalg.svdvals(zf)

        s2 = s**2
        total = s2.sum().clamp_min(eps)
        p = (s2 / total).clamp_min(eps)
        entropy = -(p * torch.log(p)).sum().item()
        eff = float(math.exp(entropy))
        return eff, float(entropy)


def gradient_l2_norm(parameters: list[torch.nn.Parameter] | Any) -> float:
    """所有可训练参数梯度 L2 范数平方和开根。"""
    if parameters is None:
        return 0.0
    total_sq = 0.0
    for p in parameters:
        if p.grad is None:
            continue
        g = p.grad.data
        total_sq += float(g.float().pow(2).sum().item())
    return total_sq**0.5


def hyperbolic_coord_stats(x: torch.Tensor, eps: float = 1e-8) -> tuple[float, float]:
    """庞加莱球坐标范数均值与最大值（用于监测是否贴边）。"""
    with torch.no_grad():
        norms = x.detach().float().norm(dim=-1)
        return float(norms.mean().item()), float(norms.max().item())


def collect_reading(
    step: int,
    loss: float,
    z_hyp: torch.Tensor,
    params_with_grad: list[torch.nn.Parameter],
) -> TELMReading:
    eff, sent = effective_rank_from_matrix(z_hyp)
    gn = gradient_l2_norm(params_with_grad)
    mn, mx = hyperbolic_coord_stats(z_hyp)
    return TELMReading(
        step=step,
        loss=float(loss),
        effective_rank=eff,
        spectral_entropy=sent,
        grad_norm=gn,
        hyp_coord_norm_mean=mn,
        hyp_coord_norm_max=mx,
    )
