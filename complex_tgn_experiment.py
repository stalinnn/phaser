import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ComplexLinear(nn.Module):
    """
    复数全连接层 (Complex-valued Linear Layer)
    实现 z = W * x + b，其中 W, x, b 均为复数
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # 定义实部和虚部权重
        # 初始化策略：保持模长分布稳定，相位随机
        self.weight_real = nn.Parameter(torch.randn(out_features, in_features) / math.sqrt(in_features))
        self.weight_imag = nn.Parameter(torch.randn(out_features, in_features) / math.sqrt(in_features))
        
        self.bias_real = nn.Parameter(torch.zeros(out_features))
        self.bias_imag = nn.Parameter(torch.zeros(out_features))

    def forward(self, x_real, x_imag):
        # (a+bi)(c+di) = (ac-bd) + (ad+bc)i
        real_part = F.linear(x_real, self.weight_real, self.bias_real) - \
                    F.linear(x_imag, self.weight_imag, self.bias_imag)
        
        imag_part = F.linear(x_real, self.weight_imag, self.bias_imag) + \
                    F.linear(x_imag, self.weight_real, self.bias_real)
                    
        return real_part, imag_part

class AntiMatterGate(nn.Module):
    """
    反物质门控 (Anti-Matter Gating Mechanism)
    利用相消干涉实现逻辑湮灭
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        # 用于生成门控信号的复数网络
        self.gate_net = ComplexLinear(dim, dim)
        
    def forward(self, x_real, x_imag):
        # 1. 计算门控信号 (Gate Signal)
        # 这里的门控不再是 0~1 的实数，而是一个复数相量 (Phasor)
        g_real, g_imag = self.gate_net(x_real, x_imag)
        
        # 归一化模长，主要利用相位信息
        # 物理意义：Gate 决定了旋转多少角度，而不是缩放多少幅度
        g_norm = torch.sqrt(g_real**2 + g_imag**2 + 1e-8)
        g_real = g_real / g_norm
        g_imag = g_imag / g_norm
        
        # 2. 干涉 (Interference)
        # 原始信号与门控信号相乘（旋转相位）
        # 如果 Gate 的相位是 pi (反物质)，则信号反相
        out_real = x_real * g_real - x_imag * g_imag
        out_imag = x_real * g_imag + x_imag * g_real
        
        # 3. 叠加 (Superposition) - 模拟与正物质流的相互作用
        # 这里简化演示：假设输入本身包含了正物质流，Gate 产生了反物质流
        # 实际应用中，这里可能是两个独立流的相加
        
        return out_real, out_imag

class ComplexTGNBlock(nn.Module):
    """
    复数热力学门控网络单元 (Complex-TGN Block)
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
        # 惯性流 (Inertial Stream) - 复数 RNN
        self.inertial_rnn = ComplexLinear(dim, dim) 
        
        # 几何流 (Geometric Stream) - 复数 Attention (简化版)
        self.query = ComplexLinear(dim, dim)
        self.key = ComplexLinear(dim, dim)
        self.value = ComplexLinear(dim, dim)
        
        # 反物质门控
        self.gate = AntiMatterGate(dim)

    def forward(self, x_real, x_imag):
        # --- 1. 惯性流 ---
        # 简单的线性递归模拟
        i_real, i_imag = self.inertial_rnn(x_real, x_imag)
        
        # --- 2. 几何流 (复数 Attention) ---
        q_r, q_i = self.query(x_real, x_imag)
        k_r, k_i = self.key(x_real, x_imag)
        v_r, v_i = self.value(x_real, x_imag)
        
        # 复数点积注意力: Q * K_conjugate
        # (qr + i*qi)(kr - i*ki) = (qr*kr + qi*ki) + i(qi*kr - qr*ki)
        attn_real = torch.matmul(q_r, k_r.transpose(-2, -1)) + torch.matmul(q_i, k_i.transpose(-2, -1))
        attn_imag = torch.matmul(q_i, k_r.transpose(-2, -1)) - torch.matmul(q_r, k_i.transpose(-2, -1))
        
        # 复数 Softmax (基于模长)
        attn_mag = torch.sqrt(attn_real**2 + attn_imag**2)
        attn_weights = F.softmax(attn_mag / math.sqrt(self.dim), dim=-1)
        
        # 聚合 Value
        # 注意：这里简化了，直接用实数权重加权复数 Value
        # 严谨的复数 Attention 会保留相位的旋转作用
        geo_real = torch.matmul(attn_weights, v_r)
        geo_imag = torch.matmul(attn_weights, v_i)
        
        # --- 3. 干涉与输出 ---
        # 惯性流与几何流的叠加
        # 这里的叠加是复数加法，允许相消干涉 (Annihilation)
        out_real = i_real + geo_real
        out_imag = i_imag + geo_imag
        
        # 通过反物质门控进行最后的修正/湮灭
        out_real, out_imag = self.gate(out_real, out_imag)
        
        return out_real, out_imag

# ==========================================
# 验证实验：逻辑否定 (Logical Negation Test)
# ==========================================
def train_logical_negation():
    print("\n=== Training Logical Negation Task ===")
    print("Goal: Learn to output ZERO energy (Annihilation) for specific pattern")
    
    # 任务定义：
    # 输入 x 是随机复数向量
    # 规则：如果 x 的实部均值 > 0，则 y 应该为 0 (Target=0, 湮灭)
    #       否则，y 应该保持非零能量 (Target=1, 保持)
    
    dim = 16
    model = ComplexTGNBlock(dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    for step in range(500):
        # 1. 生成数据
        x_real = torch.randn(32, dim)
        x_imag = torch.randn(32, dim)
        
        # 2. 生成标签 (Mask)
        # condition: real part mean > 0 -> should annihilate (target energy = 0)
        condition = x_real.mean(dim=1) > 0
        target_energy = (~condition).float() # True->0, False->1
        
        # 3. 前向传播
        y_real, y_imag = model(x_real, x_imag)
        
        # 4. 计算能量与 Loss
        current_energy = (y_real**2 + y_imag**2).mean(dim=1)
        
        # Loss: 让满足条件的样本能量趋于0，不满足的趋于1
        loss = F.mse_loss(current_energy, target_energy)
        
        # 5. 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step % 50 == 0:
            # 统计湮灭效果
            annihilated_energy = current_energy[condition].mean().item()
            preserved_energy = current_energy[~condition].mean().item()
            print(f"Step {step:03d} | Loss: {loss.item():.4f} | "
                  f"Annihilated E: {annihilated_energy:.4f} (Target 0) | "
                  f"Preserved E: {preserved_energy:.4f} (Target 1)")

    print("=== Training Complete ===")

if __name__ == "__main__":
    train_logical_negation()
