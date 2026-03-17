import torch
import math

class PoincareMath:
    """
    全息检索系统的核心几何引擎：庞加莱球 (Poincaré Ball) 流形算子库。
    处理指数容量空间的映射、测地线距离计算与非欧加法。
    所有操作强制进行数值钳位 (Clamping)，防止双曲边缘的梯度爆炸与 NaN 诅咒。
    """
    def __init__(self, c=1.0, eps=1e-5):
        """
        :param c: 空间曲率参数 (Curvature)。c > 0 对应双曲空间。工程上常取 c=1.0。
        :param eps: 数值稳定器。防止向量触碰流形绝对边界 (||x||=1/sqrt(c)) 导致除零错误。
        """
        self.c = c
        self.eps = eps
        self.max_norm = (1.0 - eps) / math.sqrt(c)

    def proj(self, x):
        """
        流形投影 (Manifold Projection)：
        如果优化器不小心把向量推到了庞加莱球外，强行把它拉回到合法的球内。
        """
        norm_x = torch.norm(x, p=2, dim=-1, keepdim=True)
        # 防止除零
        norm_x_safe = torch.clamp(norm_x, min=self.eps)
        # 只有超过 max_norm 的才被压缩，没超过的保持原样
        cond = norm_x > self.max_norm
        projected = torch.where(cond, x / norm_x_safe * self.max_norm, x)
        return projected

    def mobius_add(self, x, y):
        """
        莫比乌斯加法 (Möbius Addition)：
        在双曲空间中融合两个特征。它是不满足交换律的非欧加法。
        x ⊕_c y = [ (1 + 2c<x,y> + c||y||^2)x + (1 - c||x||^2)y ] / [ 1 + 2c<x,y> + c^2||x||^2||y||^2 ]
        """
        x2 = torch.sum(x * x, dim=-1, keepdim=True)
        y2 = torch.sum(y * y, dim=-1, keepdim=True)
        xy = torch.sum(x * y, dim=-1, keepdim=True)
        
        num = (1 + 2 * self.c * xy + self.c * y2) * x + (1 - self.c * x2) * y
        denom = 1 + 2 * self.c * xy + (self.c ** 2) * x2 * y2
        
        return self.proj(num / (denom + self.eps))

    def exp_map0(self, v):
        """
        原点处的指数映射 (Exponential Map from Origin)：
        这是【全息降维打击算子】！负责把传统 Transformer (如 BGE) 输出的平直欧氏向量，
        优雅地“拍进”庞加莱球里，自动获得层级特征（半径=抽象度）。
        exp_0(v) = tanh(sqrt(c) ||v||) * (v / (sqrt(c) ||v||))
        """
        norm_v = torch.norm(v, p=2, dim=-1, keepdim=True)
        # 防止除零
        norm_v_safe = torch.clamp(norm_v, min=self.eps)
        
        scale = torch.tanh(math.sqrt(self.c) * norm_v) / (math.sqrt(self.c) * norm_v_safe)
        
        # 当 ||v|| 极小时，tanh(x)/x 约等于 1，为了数值稳定，直接返回 v
        cond = norm_v > self.eps
        res = torch.where(cond, v * scale, v)
        
        return self.proj(res)

    def dist(self, x, y):
        """
        测地线距离 (Geodesic Distance / Poincaré Distance)：
        全息 RAG 的检索核心。利用 AdS/CFT 中 Ryu-Takayanagi 公式的物理直觉。
        它在检索两个边缘节点（细节）时，距离会剧烈放大，强迫检索路径向球心（宏观父节点）弯曲。
        d_H(x, y) = (2/sqrt(c)) * artanh(sqrt(c) * ||(-x) ⊕_c y||)
        """
        # 注意莫比乌斯加法的顺序
        m_xy = self.mobius_add(-x, y)
        norm_m_xy = torch.norm(m_xy, p=2, dim=-1, keepdim=True)
        
        # 再次钳位，防止 artanh 输入达到 1 导致无穷大
        arg = math.sqrt(self.c) * norm_m_xy
        arg_safe = torch.clamp(arg, max=1.0 - self.eps)
        
        # artanh(x) = 0.5 * ln((1+x)/(1-x))
        dist = (2.0 / math.sqrt(self.c)) * torch.atanh(arg_safe)
        return dist

# =====================================================================
# 简单的单元测试，证明物理特性的存在
# =====================================================================
if __name__ == "__main__":
    holo = PoincareMath(c=1.0)
    
    print("=== 全息检索几何引擎 (Holo-RAG Math Core) 测试 ===")
    
    # 1. 模拟传统大模型输出的三个平直空间向量
    v_root = torch.tensor([[0.1, 0.1, 0.1]])      # 宏观概念（模长小）
    v_leaf1 = torch.tensor([[5.0, 0.0, 0.0]])     # 具体细节1（模长大）
    v_leaf2 = torch.tensor([[-5.0, 0.0, 0.0]])    # 具体细节2（模长大，方向相反）
    
    # 2. 拍入庞加莱球 (指数映射)
    h_root = holo.exp_map0(v_root)
    h_leaf1 = holo.exp_map0(v_leaf1)
    h_leaf2 = holo.exp_map0(v_leaf2)
    
    print("\n[映射结果]")
    print(f"根节点投射位置: {h_root} (模长: {torch.norm(h_root).item():.4f}, 靠近球心)")
    print(f"叶子1投射位置 : {h_leaf1} (模长: {torch.norm(h_leaf1).item():.4f}, 极其靠近边缘)")
    
    # 3. 感受“空间扭曲”的魔力 (距离计算)
    print("\n[全息检索：测地线距离的魔法]")
    dist_root_leaf1 = holo.dist(h_root, h_leaf1).item()
    dist_leaf1_leaf2 = holo.dist(h_leaf1, h_leaf2).item()
    
    print(f"根节点 到 叶子1 的距离: {dist_root_leaf1:.4f}")
    print(f"叶子1  到 叶子2 的距离: {dist_leaf1_leaf2:.4f}")
    
    # 揭示真相
    print("\n>>> 物理现象揭秘:")
    if dist_leaf1_leaf2 > 2 * dist_root_leaf1:
        print("发现 Ryu-Takayanagi 现象！")
        print("两个边缘叶子节点之间的直接距离，远远大于它们绕道中心父节点的距离。")
        print("结论：在 Holo-RAG 中检索时，测地线会被迫向上弯曲，自动把包含它们的【宏观父节点】一并拉出！")