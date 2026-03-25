import numpy as np
from itertools import combinations
import json
import torch

def compute_gromov_hyperbolicity(distance_matrix):
    """
    计算给定距离矩阵的 Gromov \delta-hyperbolicity.
    \delta 越小，说明空间越像一棵树（双曲性越强）。
    \delta 越大，说明空间越像平坦的欧氏空间。
    
    公式：
    对于任意 4 个点 x, y, z, w，定义：
    S1 = d(x,y) + d(z,w)
    S2 = d(x,z) + d(y,w)
    S3 = d(x,w) + d(y,z)
    
    将这三个和按降序排列：L >= M >= S
    \delta(x,y,z,w) = (L - M) / 2
    整个空间的 \delta 是所有四点组合中最大的 \delta(x,y,z,w)
    
    为了计算效率，我们随机采样 N 个点进行估计。
    """
    n = distance_matrix.shape[0]
    if n < 4:
        return 0.0
        
    # 随机采样以加快计算速度（O(N^4) 太慢）
    sample_size = min(n, 50)
    indices = np.random.choice(n, sample_size, replace=False)
    
    max_delta = 0.0
    sum_delta = 0.0
    count = 0
    
    for x, y, z, w in combinations(indices, 4):
        s1 = distance_matrix[x, y] + distance_matrix[z, w]
        s2 = distance_matrix[x, z] + distance_matrix[y, w]
        s3 = distance_matrix[x, w] + distance_matrix[y, z]
        
        sums = sorted([s1, s2, s3], reverse=True)
        delta = (sums[0] - sums[1]) / 2.0
        
        if delta > max_delta:
            max_delta = delta
        sum_delta += delta
        count += 1
        
    return {
        "max_delta": max_delta,
        "mean_delta": sum_delta / count if count > 0 else 0.0
    }

def analyze_dataset_manifold(embeddings, num_samples=1000):
    """
    分析一批 Embedding 的流形曲率。
    返回建议使用的 RAG 模式。
    """
    # 1. 采样计算距离矩阵
    n = embeddings.shape[0]
    sample_size = min(n, num_samples)
    indices = np.random.choice(n, sample_size, replace=False)
    sampled_embs = embeddings[indices]
    
    # 将 NumPy 转换为 PyTorch 计算欧氏距离
    tensor_embs = torch.tensor(sampled_embs, dtype=torch.float32)
    dist_matrix = torch.cdist(tensor_embs, tensor_embs, p=2).numpy()
    
    # 2. 计算 Gromov Delta
    delta_stats = compute_gromov_hyperbolicity(dist_matrix)
    mean_delta = delta_stats["mean_delta"]
    
    # 3. 启发式路由逻辑
    # 注意：这里的阈值 (0.1, 0.5) 只是示例，在真实实验中需要通过分布拟合来确定
    if mean_delta < 0.5:
        topology = "Tree-like (Highly Hyperbolic)"
        recommended_mode = "Hyperbolic RAG (Poincaré Ball)"
        weight_hyperbolic = 0.8
        weight_euclidean = 0.2
    elif mean_delta < 1.0:
        topology = "Mixed (Graph-like)"
        recommended_mode = "Holo-Hybrid RAG (RRF Fusion)"
        weight_hyperbolic = 0.5
        weight_euclidean = 0.5
    else:
        topology = "Flat (Euclidean / Euclidean Mesh)"
        recommended_mode = "Standard RAG (Cosine)"
        weight_hyperbolic = 0.0  # 切断双曲信道，单跳事实不需要多跳逻辑
        weight_euclidean = 1.0
        
    return {
        "mean_delta": mean_delta,
        "max_delta": delta_stats["max_delta"],
        "topology": topology,
        "recommended_mode": recommended_mode,
        "fusion_weights": (weight_hyperbolic, weight_euclidean)
    }

if __name__ == "__main__":
    # 测试一下探测器
    print("Testing Manifold Curvature Detector...")
    
    # 模拟树状数据 (双曲)
    tree_embs = np.random.randn(100, 64) * 0.1
    for i in range(1, 100):
        tree_embs[i] = tree_embs[i//2] + np.random.randn(64) * 0.5
    
    # 模拟平坦数据 (欧氏)
    flat_embs = np.random.rand(100, 64) * 10
    
    print("\n--- Tree-like Data ---")
    res_tree = analyze_dataset_manifold(tree_embs)
    print(json.dumps(res_tree, indent=2))
    
    print("\n--- Flat Data ---")
    res_flat = analyze_dataset_manifold(flat_embs)
    print(json.dumps(res_flat, indent=2))
