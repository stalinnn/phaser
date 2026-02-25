import torch
import torch.nn as nn
import torch.nn.functional as F
import math

try:
    from mamba_ssm import Mamba
    HAS_MAMBA = True
except ImportError:
    HAS_MAMBA = False
    print("Warning: `mamba_ssm` library not found. Will fallback to MockMambaLayer for testing.")

class DynamicLowRankGaugeConnection(nn.Module):
    """
    动态低秩规范场联络 (Dynamic Low-Rank Gauge Field)
    从 ngf_layers.py 引入，作为 Mamba 和 Transformer 之间的流形转换器。
    U_t(h_t) = I + \alpha_t A_t B_t^T
    """
    def __init__(self, hidden_dim, rank=8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.rank = rank
        
        self.gen_A = nn.Linear(hidden_dim, hidden_dim * rank)
        self.gen_B = nn.Linear(hidden_dim, hidden_dim * rank)
        
        # 熵驱动门控 (Entropy-Driven Gating)
        self.entropy_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, h):
        B, L, D = h.shape
        
        A = self.gen_A(h).view(B, L, D, self.rank)
        B_mat = self.gen_B(h).view(B, L, D, self.rank)
        
        # alpha: [B, L, 1] 决定联络的开启强度（即“形变”的程度）
        alpha = self.entropy_gate(h)
        
        B_h = torch.matmul(B_mat.transpose(-1, -2), h.unsqueeze(-1))
        A_B_h = torch.matmul(A, B_h).squeeze(-1)
        
        # U(h) = h + \alpha * (A * B^T * h)
        h_out = h + alpha * A_B_h
        
        # 返回转换后的特征，以及门控值用于分析/正则化
        return h_out, alpha


class MockMambaLayer(nn.Module):
    """
    模拟 Mamba 层 (SSM)
    物理意义：处理时域规范 (Temporal Gauge) 下的线性动力学演化，擅长低熵序列。
    (由于实际 Mamba 需要特定的 CUDA 算子，这里用一个带门控的线性RNN结构模拟其性质)
    """
    def __init__(self, d_model):
        super().__init__()
        self.proj = nn.Linear(d_model, d_model)
        self.conv1d = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model)
        self.act = nn.SiLU()

    def forward(self, x):
        # x: [B, L, D]
        x_proj = self.proj(x)
        # Conv1d expects [B, D, L]
        x_conv = self.conv1d(x_proj.transpose(1, 2)).transpose(1, 2)
        return self.act(x_conv)


class MockTransformerLayer(nn.Module):
    """
    模拟 Transformer 层 (Attention)
    物理意义：处理频域/积分规范 (Integral Gauge) 下的全局信息交互，擅长高熵推理。
    """
    def __init__(self, d_model, n_heads=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )

    def forward(self, x):
        # Self-attention
        attn_out, _ = self.attn(x, x, x)
        x = self.norm(x + attn_out)
        # FFN
        ffn_out = self.ffn(x)
        return self.norm(x + ffn_out)


class RTNBlock(nn.Module):
    """
    递归热力学网络块 (Recursive Thermodynamic Network Block)
    核心创新：通过 Gauge Connector 解决 Mamba (球形流形) 和 Transformer (锥形流形) 的错位问题。
    """
    def __init__(self, d_model, rank=8):
        super().__init__()
        
        if HAS_MAMBA:
            # 真实云平台环境下的 Mamba
            self.mamba = Mamba(
                d_model=d_model,  # Model dimension d_model
                d_state=16,       # SSM state expansion factor
                d_conv=4,         # Local convolution width
                expand=2,         # Block expansion factor
            )
        else:
            self.mamba = MockMambaLayer(d_model)
        
        # 核心：规范场连接器
        self.gauge_connector = DynamicLowRankGaugeConnection(d_model, rank)
        
        self.transformer = MockTransformerLayer(d_model)
        
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        前向传播：
        1. 基础处理：Mamba 线性演化
        2. 流形对齐：Gauge Connector (根据熵值动态计算协变导数)
        3. 复杂推理：Transformer 注意力
        """
        # 1. 经过 Mamba 处理 (低熵/快速通道)
        h_mamba = self.mamba(x)
        
        # 2. 经过 规范场连接器 进行几何上的流形转换 (Phase Transition)
        h_aligned, entropy_alpha = self.gauge_connector(h_mamba)
        
        # 3. 只有在高熵状态下(由 alpha 隐式控制)，Transformer 才能有效处理信号
        # （这里为了模拟混合架构，我们让对齐后的特征进入 Transformer）
        h_attn = self.transformer(h_aligned)
        
        # 最终组合 (残差连接)
        out = self.norm(x + h_mamba + h_attn)
        return out, entropy_alpha


class MambaLlamaRTN(nn.Module):
    """
    模拟实验三：Mamba-Llama 异构流形对齐网络整体架构
    """
    def __init__(self, d_model=128, num_layers=4, rank=8):
        super().__init__()
        self.embedding = nn.Linear(50000, d_model) # 模拟 Vocab 词嵌入
        
        # 多层 RTN Block
        self.layers = nn.ModuleList([
            RTNBlock(d_model, rank) for _ in range(num_layers)
        ])
        
        self.lm_head = nn.Linear(d_model, 50000)

    def forward(self, input_ids):
        # 模拟 Embedding (此处直接使用连续张量代替真实 Embedding 查找，方便测试)
        x = input_ids 
        
        alphas = [] # 记录每一层的熵门控情况
        for layer in self.layers:
            x, alpha = layer(x)
            alphas.append(alpha)
            
        logits = self.lm_head(x)
        return logits, alphas


# 测试代码：模拟 CKA 流形对齐实验
if __name__ == "__main__":
    print("Initializing RTN Mamba-Llama Hybrid Model...")
    
    # 检查是否有可用的 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    d_model = 64
    seq_len = 16
    batch_size = 2
    
    model = MambaLlamaRTN(d_model=d_model, num_layers=3, rank=4).to(device)
    
    # 模拟输入 (Batch, SeqLen, D) 并移动到正确的设备上
    dummy_input = torch.randn(batch_size, seq_len, d_model, device=device)
    
    # 前向传播
    logits, alphas = model(dummy_input)
    
    print(f"\n[Forward Pass Success]")
    print(f"Input Shape: {dummy_input.shape}")
    print(f"Output Logits Shape: {logits.shape}")
    
    print("\n[Entropy-Driven Gating (Phase Transition) Analysis]")
    # 打印每一层、每一个 Token 的门控开启强度
    for i, alpha in enumerate(alphas):
        # alpha shape: [B, L, 1] -> 取均值查看层的平均“熵/热度”
        mean_alpha = alpha.mean().item()
        print(f"Layer {i+1} Average Gauge Gating (Entropy): {mean_alpha:.4f}")
        
    print("\n结论：在理论实验中，随着训练步数增加，这个 Alpha 值会发生 '相变'，引导 Mamba (球形) 和 Transformer (锥形) 之间的流形对齐 (CKA 跃升)。")
