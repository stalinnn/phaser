import torch
import torch.nn as nn
import torch.nn.functional as F

class U1GaugeLinear(nn.Module):
    """
    U(1) Gauge Linear Layer (复数域实现)
    基于理论文档 9.1 节：利用 U(1) 对称性（相位旋转），用复数张量高效实现 NGF。
    输入张量的幅值代表特征强度，相位代表局部坐标系的旋转角度。
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # 权重 W 是复数矩阵
        # 使用极坐标形式初始化：幅值 r 和 相位 theta
        self.weight_r = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_theta = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight_r, a=math.sqrt(5))
        nn.init.uniform_(self.weight_theta, -math.pi, math.pi)

    def forward(self, z_in):
        """
        z_in: [batch_size, in_features] (Complex Tensor)
        计算公式: z_j^{out} = \sum_i W_{ji} (z_i^{in} e^{i \theta_{ji}})
        在 PyTorch 中可以直接利用复数乘法：W_complex @ z_in
        """
        # 构建复数权重
        W_complex = torch.polar(self.weight_r, self.weight_theta)
        
        # [batch_size, out_features]
        z_out = F.linear(z_in, W_complex)
        return z_out


class DynamicLowRankGaugeConnection(nn.Module):
    """
    动态低秩规范场联络 (Dynamic Low-Rank Gauge Field)
    基于论文草稿 3.1 节：为解决全秩旋转矩阵的显存爆炸，提出低秩分解。
    U_t(h_t) = I + \alpha_t A_t B_t^T
    """
    def __init__(self, hidden_dim, rank=8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.rank = rank
        
        # 【新增：训练稳定性保护】
        # Qwen 等大模型的 hidden state 范数可能非常大 (高达百级别)。
        # 直接生成 A 和 B 会导致 A * B^T * h 产生 h^3 级别的数值爆炸，引发 NaN。
        # 因此我们在进入 Hypernetwork 前加入 LayerNorm，确保生成器的输入尺度恒定。
        self.norm = nn.LayerNorm(hidden_dim)
        
        # Hypernetwork：从输入状态生成 A 和 B 矩阵
        self.gen_A = nn.Linear(hidden_dim, hidden_dim * rank)
        self.gen_B = nn.Linear(hidden_dim, hidden_dim * rank)
        
        # 【初始化缩放】防止 A 矩阵初始方差过大
        nn.init.normal_(self.gen_A.weight, std=0.001)
        if self.gen_A.bias is not None:
            nn.init.zeros_(self.gen_A.bias)
        
        # 【零初始化】让初始的规范场是一个完美的恒等映射 (Identity Mapping)
        # 这样在训练初期，网络不会破坏基座模型原有的强大特征
        nn.init.zeros_(self.gen_B.weight)
        if self.gen_B.bias is not None:
            nn.init.zeros_(self.gen_B.bias)
        
        # 熵驱动门控 (Entropy-Driven Gating) 参数
        # 简化版：这里用一个线性层学习标量门控系数 alpha
        self.entropy_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, h):
        """
        h: [batch_size, seq_len, hidden_dim]
        返回: h_transformed [batch_size, seq_len, hidden_dim]
        """
        B, L, D = h.shape
        
        # 使用 LayerNorm 保护生成器的输入 (转换成 float32 保证 Norm 的数值稳定，再转回原类型)
        h_norm = self.norm(h.to(torch.float32)).to(h.dtype)
        
        # 生成低秩矩阵 A 和 B
        # A: [B, L, D, R]
        A = self.gen_A(h_norm).view(B, L, D, self.rank)
        # B: [B, L, D, R]
        B_mat = self.gen_B(h_norm).view(B, L, D, self.rank)
        
        # 计算门控系数 \alpha_t
        # alpha: [B, L, 1]
        alpha = self.entropy_gate(h_norm)
        
        # 计算低秩联络的微扰部分: A * B^T * h
        # 为了高效计算，避免显式实例化 [D, D] 矩阵，我们先算 B^T * h，再算 A * (B^T * h)
        # B_mat (transpose): [B, L, R, D], h.unsqueeze: [B, L, D, 1]
        # B_h: [B, L, R, 1]
        B_h = torch.matmul(B_mat.transpose(-1, -2), h.unsqueeze(-1))
        
        # A_B_h: [B, L, D, 1]
        # 【新增：缩放因子】除以 sqrt(rank) 进一步防止特征方差在矩阵乘法中随 rank 增大而爆炸
        import math
        A_B_h = torch.matmul(A, B_h).squeeze(-1) / math.sqrt(self.rank)
        
        # U(h) = h + \alpha * (A * B^T * h)
        h_out = h + alpha * A_B_h
        return h_out


class GaugeCovariantMessagePassing(nn.Module):
    """
    规范协变消息传递层 (Gauge-Covariant Layer for Graph/Point Cloud)
    基于理论文档 5.2 节神经元更新公式：
    h_y = \sigma ( \sum_{x \in N(y)} W * (U_{y \leftarrow x} h_x) )
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.W = nn.Linear(hidden_dim, hidden_dim)
        
        # 预测边上的联络 (Connection / Parallel Transporter)
        # 输入：节点特征对 [h_x, h_y]
        self.connection_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            # 预测出一个正交变换矩阵的生成元（这里简化为动态生成一个权重，并利用正交投影）
            nn.Linear(hidden_dim, hidden_dim * hidden_dim)
        )

    def _get_orthogonal_matrix(self, A):
        """
        将任意矩阵投影到正交群 SO(d) 上 (简化实现，实际可用 expm 或 Cayley 变换)
        """
        # A: [E, D, D]
        # 一种简单方法是利用 A - A^T 得到反对称矩阵，然后进行矩阵指数运算（计算量较大）
        # 这里为了演示，我们假设 A 已经是某种旋转的近似（可以通过 SVD 等方式投影，此处略过复杂计算）
        # 返回近似的平行移动算子
        return A

    def forward(self, x, edge_index):
        """
        x: [num_nodes, hidden_dim] 节点特征
        edge_index: [2, num_edges] 邻接关系 (source -> target)
        """
        num_nodes = x.size(0)
        source, target = edge_index[0], edge_index[1]
        
        h_x = x[source]
        h_y = x[target]
        
        # 1. 预测平行移动算子 U_{y <- x}
        edge_features = torch.cat([h_x, h_y], dim=-1)
        # [num_edges, D*D] -> [num_edges, D, D]
        A = self.connection_predictor(edge_features).view(-1, self.hidden_dim, self.hidden_dim)
        U_yx = self._get_orthogonal_matrix(A) # 实际工程中需严格约束为正交矩阵
        
        # 2. 对邻居特征进行几何校正（平行移动）
        # h_x_aligned = U_{y <- x} * h_x
        # U_yx: [num_edges, D, D], h_x: [num_edges, D, 1]
        h_x_aligned = torch.bmm(U_yx, h_x.unsqueeze(-1)).squeeze(-1)
        
        # 3. 聚合与非线性激活
        out = torch.zeros_like(x)
        # 简单使用 scatter_add 进行求和聚合
        out.scatter_add_(0, target.unsqueeze(-1).expand_as(h_x_aligned), h_x_aligned)
        
        # 4. 乘以权重并激活
        return F.relu(self.W(out))


# 测试代码
if __name__ == "__main__":
    import math
    
    print("Testing U(1) Gauge Linear Layer...")
    # 模拟复数输入特征 [batch=2, features=4]
    z_in = torch.randn(2, 4, dtype=torch.cfloat)
    u1_layer = U1GaugeLinear(4, 8)
    z_out = u1_layer(z_in)
    print(f"U(1) Layer Output Shape: {z_out.shape} (Expected: 2, 8)")
    print("-" * 30)
    
    print("Testing Dynamic Low-Rank Gauge Connection...")
    # 模拟序列输入 [batch=2, seq_len=10, dim=64]
    h_seq = torch.randn(2, 10, 64)
    low_rank_conn = DynamicLowRankGaugeConnection(64, rank=8)
    h_transformed = low_rank_conn(h_seq)
    print(f"Low-Rank Connection Output Shape: {h_transformed.shape} (Expected: 2, 10, 64)")
    print("-" * 30)
    
    print("Testing Gauge-Covariant Message Passing...")
    # 模拟图结构: 5个节点，6条边
    x_nodes = torch.randn(5, 16)
    edges = torch.tensor([[0, 1, 1, 2, 3, 4],
                          [1, 0, 2, 1, 4, 3]], dtype=torch.long)
    gc_layer = GaugeCovariantMessagePassing(16)
    x_updated = gc_layer(x_nodes, edges)
    print(f"Gauge-Covariant GNN Output Shape: {x_updated.shape} (Expected: 5, 16)")
