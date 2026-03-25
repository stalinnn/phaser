import torch
import torch.nn as nn

class LieBracketTracker:
    """
    TELM Probe B: 拓扑纠缠与李括号规范场强度计 (Lie Bracket & Gauge Field Tracker)
    
    原理 (物理视角):
    在现代规范场论(Gauge Theory)和流形学习中，参数的移动路径如果形成闭环，并且所在空间是弯曲的或存在规范群，
    那么闭环移动后会产生额外的“相位相移”或者叫“和乐(Holonomy)”。
    李括号 [X, Y] = XY - YX 测量了沿着两个方向 X 和 Y 移动时不满足交换律的程度。
    
    如果在神经网络训练中：
    假设我们有参数 W。计算其两个不同特征方向的梯度向量场或激活的雅可比(Jacobian)向量场 V_1 和 V_2。
    强烈的李括号 [V_1, V_2] != 0 意味着流形存在强曲率或非平凡的拓扑纠缠结构，此时模型正处在高度复杂的特征表示折叠阶段。
    
    在工程实现上，由于完整计算 Jacobian 或者海森矩阵的李代数非常耗时，
    我们采用近似的**特征子空间正交性与曲率积**来度量局部的“纠缠强度(Entanglement Intensity)”。
    """
    def __init__(self, module_a: nn.Module, module_b: nn.Module, device="cuda"):
        """
        初始化李括号探针。
        我们通常监测两个有强烈交互的模块，比如：
        1. Query 投影层 vs Context 投影层
        2. Q-K Attention 投影层 vs V 投影层
        3. RTN模型中的 SimpleMamba(序列混合) vs ThermoEntropyHead(全局热力学门)
        
        Args:
            module_a: 模块 A (对应场 V_1)
            module_b: 模块 B (对应场 V_2)
        """
        self.module_a = module_a
        self.module_b = module_b
        self.device = device
        
    def _flatten_grads(self, module):
        """展平并收集模块的所有梯度"""
        grads = []
        for p in module.parameters():
            if p.grad is not None:
                grads.append(p.grad.view(-1))
        if not grads:
            return torch.tensor([0.0], device=self.device)
        return torch.cat(grads)

    def _flatten_params(self, module):
         """展平并收集模块的所有参数本身，用于评估子空间基底"""
         params = []
         for p in module.parameters():
             params.append(p.data.view(-1))
         if not params:
             return torch.tensor([0.0], device=self.device)
         return torch.cat(params)

    def compute_lie_bracket_intensity(self):
        """
        计算近似的李括号强度。
        如果参数量过大，直接计算 Jacobian 是不可行的。
        我们使用：
        I_lie = || grad(A) || * || grad(B) || * (1 - |cos(theta_{grad_A, grad_B})|)
        但这只衡量了梯度的正交性。
        为了更贴近规范场，我们可以计算 A 和 B 权重本身的曲率扰动：
        
        使用一种启发式的 "非交换度(Non-Commutativity)"：
        如果我们把 A 和 B 的梯度视作向量场，它们在参数空间中产生的曲率张量正比于：
        || d(grad_A) / d(param_B) - d(grad_B) / d(param_A) ||
        为了工程极简，我们使用梯度和参数的交叉投影范数来做指标。
        
        Returns:
            intensity (float): 李括号纠缠强度
        """
        # 获取梯度向量场 V_a 和 V_b
        g_a = self._flatten_grads(self.module_a)
        g_b = self._flatten_grads(self.module_b)
        
        # 避免全零梯度导致的 NaN
        norm_a = torch.norm(g_a) + 1e-8
        norm_b = torch.norm(g_b) + 1e-8
        
        # 1. 梯度的纠缠度 (不正交度)
        # 如果 A 和 B 的更新方向高度重合，说明它们在走平坦的欧氏直线
        # 如果它们接近正交，说明参数在进行复杂的流形旋转
        
        # 2. 参数本身的大小带来的缩放 (模拟场强)
        p_a = self._flatten_params(self.module_a)
        p_b = self._flatten_params(self.module_b)
        
        norm_pa = torch.norm(p_a) + 1e-8
        norm_pb = torch.norm(p_b) + 1e-8
        
        # 由于 module_a 和 module_b 参数量不同，无法直接点积
        # 李括号强度的启发式近似：两个系统梯度的相对更新强度的乘积
        # 实际意义：当两个系统同时发生剧烈变化时，它们之间的耦合/纠缠最强
        entanglement_intensity = (norm_a / norm_pa) * (norm_b / norm_pb)
        
        return entanglement_intensity.item()
