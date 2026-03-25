"""
庞加莱球 (Poincaré Ball) 双曲空间封装。
基于 geoopt，提供测地线距离与安全的范数钳位。
"""
from __future__ import annotations

import torch
import geoopt


def make_poincare_ball(c: float = 1.0) -> geoopt.PoincareBall:
    """创建曲率参数为 c 的庞加莱球流形（geoopt 约定）。"""
    return geoopt.PoincareBall(c=c)


def clamp_to_ball(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """将向量范数钳位到单位球内部，避免 arcosh / dist 数值爆炸。"""
    norm = x.norm(dim=-1, keepdim=True).clamp_min(eps)
    max_norm = 1.0 - eps
    scale = torch.where(norm > max_norm, max_norm / norm, torch.ones_like(norm))
    return x * scale


def pairwise_geodesic_dist(
    manifold: geoopt.PoincareBall,
    x: torch.Tensor,
    chunk_size: int | None = None,
) -> torch.Tensor:
    """
    计算 x 中所有节点两两之间的庞加莱测地线距离。

    Args:
        manifold: geoopt PoincareBall
        x: [N, d]，需在球内
        chunk_size: 若不为 None，按行分块计算以节省显存

    Returns:
        D: [N, N]，D[i,j] = d(x[i], x[j])
    """
    n = x.size(0)
    device, dtype = x.device, x.dtype
    dmat = torch.zeros(n, n, device=device, dtype=dtype)

    if chunk_size is None or n <= chunk_size:
        for i in range(n):
            xi = x[i : i + 1]  # [1, d]
            dmat[i, :] = manifold.dist(xi, x)
        return dmat

    for i in range(0, n, chunk_size):
        end_i = min(i + chunk_size, n)
        xi = x[i:end_i]  # [B, d]
        for j in range(0, n, chunk_size):
            end_j = min(j + chunk_size, n)
            xj = x[j:end_j]  # [B2, d]
            # dist: [B, 1, d] vs [1, B2, d] -> [B, B2]
            d_block = manifold.dist(
                xi.unsqueeze(1),
                xj.unsqueeze(0),
            )
            dmat[i:end_i, j:end_j] = d_block
    return dmat
