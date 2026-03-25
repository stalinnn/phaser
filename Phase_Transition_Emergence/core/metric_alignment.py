"""
全息度规对齐损失 (RT 映射的离散近似)。

边界 (欧氏): 语义余弦相似度矩阵 M，取值约在 [-1,1]，训练时映射到 [0,1] 作为“纠缠强度”代理。
体空间 (双曲): 测地线距离 D，用 exp(-D) 与 M 对齐，使高相似度对应小双曲距离。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
import geoopt

from .hyperbolic_space import pairwise_geodesic_dist


def cosine_similarity_matrix(embeddings: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    对行 L2 归一化后计算余弦相似度矩阵。

    Args:
        embeddings: [N, D] 欧氏嵌入（未归一化也可）

    Returns:
        M: [N, N]，M_ij = cos(e_i, e_j)
    """
    z = F.normalize(embeddings, p=2, dim=-1, eps=eps)
    return z @ z.t()


def map_similarity_to_entanglement_proxy(
    M: torch.Tensor,
    shift_positive: bool = True,
) -> torch.Tensor:
    """
    将余弦相似度转为 [0,1] 的“边界纠缠代理”，用于与 exp(-d_hyp) 对齐。

    shift_positive=True 时: M' = (M + 1) / 2
    """
    if shift_positive:
        return (M + 1.0) * 0.5
    return M.clamp(0.0, 1.0)


def metric_alignment_loss(
    manifold: geoopt.PoincareBall,
    x_hyp: torch.Tensor,
    target_proxy: torch.Tensor,
    mask: torch.Tensor | None = None,
    chunk_dist: int | None = None,
) -> torch.Tensor:
    """
    重整化损失: mean( (exp(-D_ij) - T_ij)^2 )。

    Args:
        manifold: 庞加莱球
        x_hyp: [N, d] 可微双曲坐标（ManifoldParameter）
        target_proxy: [N, N] 目标矩阵，与 exp(-D) 同量纲，元素在 [0,1]
        mask: 可选 [N,N] bool，仅对 True 的位置求平均（如上三角去重）
        chunk_dist: 传给 pairwise_geodesic_dist 的分块大小
    """
    dmat = pairwise_geodesic_dist(manifold, x_hyp, chunk_size=chunk_dist)
    pred = torch.exp(-dmat)
    diff = (pred - target_proxy) ** 2
    if mask is not None:
        m = mask.float()
        return (diff * m).sum() / m.sum().clamp_min(1.0)
    return diff.mean()


def upper_triangle_mask(n: int, device: torch.device, diagonal: bool = False) -> torch.Tensor:
    """上三角掩码（不含对角），避免 (i,j)(j,i) 重复计数。"""
    idx = torch.triu_indices(n, n, offset=1 if not diagonal else 0, device=device)
    mask = torch.zeros(n, n, dtype=torch.bool, device=device)
    mask[idx[0], idx[1]] = True
    return mask
