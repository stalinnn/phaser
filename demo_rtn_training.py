import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time

# ==========================================
# 1. 核心组件实现 (Core Components)
# ==========================================

class SimpleMamba(nn.Module):
    """
    一个极简的 SSM (State Space Model) 实现，用于演示 Mamba 的行为。
    为了不依赖外部 CUDA 库，这里使用 PyTorch 原生算子实现线性递归。
    h_t = A * h_{t-1} + B * x_t
    y_t = C * h_t
    """
    def __init__(self, d_model, d_state=16):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        
        # 投影层
        self.in_proj = nn.Linear(d_model, d_model * 2)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # SSM 参数 (离散化后)
        # A: (d_model, d_state) - 对角化参数，通常初始化为 -exp(...)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_model, 1)))
        self.D = nn.Parameter(torch.ones(d_model))
        
        # B, C 由输入动态生成 (Selective Scan 简化版) -> 这里为了简化，设为静态参数演示
        self.B = nn.Parameter(torch.randn(d_model, d_state) * 0.1)
        self.C = nn.Parameter(torch.randn(d_model, d_state) * 0.1)
        
    def forward(self, x):
        # x: [Batch, SeqLen, D_model]
        B, L, D = x.shape
        
        x_and_res = self.in_proj(x)
        x_in, res = x_and_res.chunk(2, dim=-1)
        
        x_in = F.silu(x_in)
        
        # 简化的线性递归 (Parallel Scan 的串行模拟)
        # h_t = decay * h_{t-1} + x_t * B
        
        # 计算 decay rate (A): range (0, 1)
        decay = torch.exp(-torch.exp(self.A_log)) # [D, N]
        
        # 预计算输入项
        # u = x_in.unsqueeze(-1) * self.B # [B, L, D, N] - 显存太大，简化为逐位乘
        # 简化：假设 d_state 维度的混合
        
        h = torch.zeros(B, D, self.d_state, device=x.device)
        y_list = []
        
        # 串行扫描 (仅演示用，实际应使用 CUDA Kernel)
        # 注意：这里的速度会很慢，但逻辑是通的
        for t in range(L):
            xt = x_in[:, t, :] # [B, D]
            
            # 更新状态
            # h_new = h * decay + xt * B
            h = h * decay + xt.unsqueeze(-1) * self.B
            
            # 投影输出
            # yt = h * C
            yt = (h * self.C).sum(dim=-1) # [B, D]
            y_list.append(yt)
            
        y = torch.stack(y_list, dim=1)
        
        # 残差连接 + 输出投影
        y = y + x_in * self.D
        y = y * F.silu(res)
        return self.out_proj(y)

class ThermoEntropyHead(nn.Module):
    """
    热力学熵探针 (Thermometer)
    计算输入特征的局部熵/惊奇度，用于决定是否开启 Attention。
    """
    def __init__(self, d_model, chunk_size=16):
        super().__init__()
        self.chunk_size = chunk_size
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # x: [B, L, D]
        # output: [B, NumChunks, 1]
        
        # 1. Token-level entropy prediction
        token_entropy = self.net(x) # [B, L, 1]
        
        # 2. Chunk-level pooling
        B, L, _ = token_entropy.shape
        # Pad if needed
        if L % self.chunk_size != 0:
            pad_len = self.chunk_size - (L % self.chunk_size)
            token_entropy = F.pad(token_entropy, (0, 0, 0, pad_len))
            L_padded = L + pad_len
        else:
            L_padded = L
            
        # Reshape to [B, NumChunks, ChunkSize, 1]
        chunks = token_entropy.view(B, L_padded // self.chunk_size, self.chunk_size, 1)
        
        # Average pooling to get chunk entropy
        chunk_entropy = chunks.mean(dim=2) # [B, NumChunks, 1]
        
        return chunk_entropy, token_entropy

class RTNBlock(nn.Module):
    """
    递归热力学网络块 (Recursive Thermodynamic Block)
    核心机制：Mamba (System 1) + 稀疏 Attention (System 2)
    """
    def __init__(self, d_model, n_head, chunk_size=16, hard_routing=True):
        super().__init__()
        self.d_model = d_model
        self.chunk_size = chunk_size
        self.hard_routing = hard_routing
        
        self.norm1 = nn.LayerNorm(d_model)
        
        # System 1: Mamba (Always Active)
        self.mamba = SimpleMamba(d_model)
        
        # System 2: Attention (Gated)
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=True)
        
        # Gate
        self.thermo_gate = ThermoEntropyHead(d_model, chunk_size)
        
        # MLP
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )
        
    def forward(self, x):
        # x: [B, L, D]
        residual = x
        x = self.norm1(x)
        
        # --- Path 1: Mamba (Base) ---
        x_mamba = self.mamba(x)
        
        # --- Path 2: Thermodynamic Routing ---
        chunk_entropy, _ = self.thermo_gate(x) # [B, NumChunks, 1]
        
        # 生成 Routing Mask
        if self.training and self.hard_routing:
            # Gumbel-Softmax for differentiable hard gating
            # Construct logits for [Keep, Drop]
            # p_keep = chunk_entropy
            logits = torch.cat([torch.log(chunk_entropy + 1e-10), torch.log(1 - chunk_entropy + 1e-10)], dim=-1)
            gumbel_out = F.gumbel_softmax(logits, tau=1.0, hard=True)
            gate = gumbel_out[:, :, 0:1] # Keep probability (0 or 1)
        else:
            # Inference: Hard threshold
            gate = (chunk_entropy > 0.5).float()
            
        # Expand gate to token level for multiplication
        # [B, NumChunks, 1] -> [B, NumChunks, ChunkSize, 1] -> [B, L, 1]
        gate_token = gate.repeat_interleave(self.chunk_size, dim=1)
        # Crop to original length
        gate_token = gate_token[:, :x.shape[1], :]
        
        # --- Path 2: Attention (Conditional) ---
        # 优化：在实际工程中，这里应该用 index_select 选出热块进行计算
        # 演示中为了代码简单，我们全量计算 Attention 但乘以 Gate (Soft Masking)
        # 或者如果 gate 全为 0，跳过计算
        
        if gate.sum() > 0:
            x_norm = self.attn_norm(x)
            
            # 模拟稀疏 Attention:
            # Query 只来自 Hot Chunks (Gate > 0)
            # Key/Value 来自 Global Context (为了能查到 Needle)
            
            # q = x_norm * gate_token (只保留热的 Query)
            # k, v = x_norm (全局)
            
            q = x_norm * gate_token
            k = x_norm
            v = x_norm
            
            x_attn, _ = self.attn(q, k, v)
            
            # 融合：Mamba + Gate * Attention
            # x_attn 在 Cold 区域理论上应该是 0 (因为 Q=0)，但 Softmax 会导致非零
            # 所以必须再次乘 gate_token
            x_mixed = x_mamba + gate_token * x_attn
        else:
            x_mixed = x_mamba
            
        # Residual
        x = residual + x_mixed
        
        # MLP
        x = x + self.mlp(self.norm2(x))
        
        return x, chunk_entropy.mean()

class RTNModel(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_layer=4, n_head=4):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            RTNBlock(d_model, n_head) for _ in range(n_layer)
        ])
        self.out = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        x = self.emb(x)
        total_entropy = 0
        for layer in self.layers:
            x, ent = layer(x)
            total_entropy += ent
            
        return self.out(x), total_entropy / len(self.layers)

# ==========================================
# 2. 数据生成与训练循环 (Data & Training)
# ==========================================

def generate_needle_haystack(batch_size, seq_len, vocab_size, needle_id=100, trigger_id=1):
    """
    生成一个合成任务：
    大部分是随机噪声 (Mamba 擅长)，但在 Trigger ID 出现后，必须回溯去找 Needle ID。
    """
    data = torch.randint(2, vocab_size, (batch_size, seq_len))
    labels = data.clone()
    
    # 随机插入 Needle
    needle_pos = torch.randint(0, seq_len // 4, (batch_size,))
    data[torch.arange(batch_size), needle_pos] = needle_id
    
    # 随机插入 Trigger (在 Needle 之后)
    trigger_pos = torch.randint(seq_len // 2, seq_len - 1, (batch_size,))
    data[torch.arange(batch_size), trigger_pos] = trigger_id
    
    # Label: Trigger 的下一个词应该是 Needle
    target_pos = trigger_pos + 1
    labels[:] = -100 # Ignore index
    labels[torch.arange(batch_size), target_pos] = needle_id
    
    return data, labels

def train_rtn_demo():
    # Config
    VOCAB_SIZE = 1000
    D_MODEL = 64 # Small for CPU/Demo
    N_LAYER = 2
    BATCH_SIZE = 16
    SEQ_LEN = 128
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running RTN Training on: {device}")
    
    model = RTNModel(VOCAB_SIZE, D_MODEL, N_LAYER).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    print("\n--- Model Architecture ---")
    print(model)
    print(f"Total Params: {sum(p.numel() for p in model.parameters())}")
    
    print("\n--- Starting Training (Phase Transition Check) ---")
    print("Goal: Gate Activation (Entropy) should be LOW for noise, HIGH for trigger.")
    
    for step in range(201):
        # Data
        inputs, targets = generate_needle_haystack(BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Forward
        logits, avg_gate_activation = model(inputs)
        
        # Loss
        # 1. Task Loss
        task_loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), targets.view(-1))
        
        # 2. Sparsity Loss (Load Balancing)
        # 我们希望 Attention 不要一直开着，只在需要的时候开
        # Target sparsity: 10%
        sparsity_loss = (avg_gate_activation - 0.1) ** 2
        
        total_loss = task_loss + 0.5 * sparsity_loss
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        if step % 20 == 0:
            print(f"Step {step:03d} | Task Loss: {task_loss.item():.4f} | Gate Usage: {avg_gate_activation.item():.2%}")

if __name__ == "__main__":
    train_rtn_demo()
