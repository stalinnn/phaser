import torch
import torch.nn as nn
import torch.nn.functional as F

class EntropyDetector:
    """
    TELM Probe C: 局部熵探测器 / 信息磁畴雷达 (Entropy Detector / Information Magnetic Domain Radar)
    
    原理 (物理视角):
    在相变点附近，系统会出现强烈的涨落，不同阶段(相)的边界被称为"磁畴壁"。
    在序列模型(如Transformer/Mamba)中，当我们处理混合了高价值信息(Needle)和噪声(Haystack)的数据时，
    模型的信息处理状态应该发生突变：
    - 噪声区(平庸相)：信息流平稳，局部熵低（或者说是高乱度低确信度导致输出趋于均匀分布）。
    - 核心信息区(凝聚相)：注意力高度集中，熵极低（高度确信）。
    - 相变界面：当模型从扫描模式切换到精确匹配模式时，隐藏状态的分布会发生剧烈变化。
    
    这个探针通过监测模型中间层激活(Activations)或者注意力权重(Attention Weights)的**香农熵(Shannon Entropy)**
    和**局部方差(Local Variance)**，来可视化这种“信息磁畴”的边界。
    """
    def __init__(self, eps=1e-8):
        self.eps = eps
        
    def compute_attention_entropy(self, attn_weights):
        """
        计算注意力分布的熵。
        attn_weights: [Batch, Heads, SeqLen_Q, SeqLen_K]
        
        如果熵很高：说明模型处于“发散”状态，四处张望 (扫描模式 / 探索相)。
        如果熵很低：说明模型“凝聚”在少数几个 Token 上 (精确提取模式 / 剥削相)。
        """
        # H(P) = -sum(P * log(P))
        # 在序列维度 (SeqLen_K) 上计算熵
        entropy = -torch.sum(attn_weights * torch.log(attn_weights + self.eps), dim=-1)
        
        # 返回每个 Query Token 的平均熵 [Batch, SeqLen_Q]
        # 对 Head 取平均
        return entropy.mean(dim=1)
        
    def compute_activation_roughness(self, hidden_states, window_size=3):
        """
        计算隐藏状态在序列上的“粗糙度” (局部方差/梯度)。
        用于探测信息流的“相变界面” (Domain Walls)。
        
        hidden_states: [Batch, SeqLen, D_model]
        """
        B, L, D = hidden_states.shape
        
        # 我们测量相邻 Token 之间的表示差异 (即时间倒数或者说空间梯度)
        # diff: [Batch, SeqLen-1, D_model]
        diff = hidden_states[:, 1:, :] - hidden_states[:, :-1, :]
        
        # 局部曲率/粗糙度，使用梯度的范数
        roughness = torch.norm(diff, dim=-1) # [Batch, SeqLen-1]
        
        # 为了保持长度一致，前面 pad 一个 0
        roughness = F.pad(roughness, (1, 0), value=0.0) # [Batch, SeqLen]
        
        # 可选：平滑处理 (Rolling Window)
        if window_size > 1:
            roughness = roughness.unsqueeze(1) # [B, 1, L]
            pool_kernel = torch.ones(1, 1, window_size, device=hidden_states.device) / window_size
            # 使用反射 pad 保证长度不变
            roughness = F.pad(roughness, (window_size//2, window_size//2), mode='reflect')
            roughness = F.conv1d(roughness, pool_kernel).squeeze(1)
            
        return roughness
        
    def analyze_domain_walls(self, hidden_states, threshold=None):
        """
        探测“磁畴壁”(Domain Walls)，即粗糙度极高的区域。
        返回可能的相变点。
        """
        roughness = self.compute_activation_roughness(hidden_states)
        
        if threshold is None:
            # 动态阈值：均值 + 2倍标准差
            mean_r = roughness.mean(dim=-1, keepdim=True)
            std_r = roughness.std(dim=-1, keepdim=True)
            threshold = mean_r + 2 * std_r
            
        # 标记出超过阈值的点作为 Domain Wall
        domain_walls = (roughness > threshold).float()
        
        return {
            "roughness": roughness,
            "domain_walls": domain_walls,
            "threshold": threshold
        }
