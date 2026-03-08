import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import psutil
import os
import gc

# ==========================================
# 1. 数据生成器 (Passkey Retrieval with Distractors)
# ==========================================

def generate_passkey_data(batch_size, seq_len, vocab_size=1000):
    """
    生成大海捞针数据：
    在超长随机背景中，插入一段真实 Key 和一段干扰的 Fake Key。
    模型需要在序列末尾准确回答出真实的 Key。
    
    词表设定:
    0-9: 预留
    10: <TRUE_KEY_PREFIX>  (提示真实Key开始)
    11: <SUFFIX>           (结束符)
    12: <FAKE_KEY_PREFIX>  (提示干扰Key开始)
    13: <QUESTION>         (末尾提问：What is the True Key?)
    20-99: 候选的 Key 集合
    100-999: 无意义的随机背景噪声
    """
    data = torch.randint(100, vocab_size, (batch_size, seq_len))
    labels = torch.zeros(batch_size, dtype=torch.long)
    
    for i in range(batch_size):
        # 1. 随机位置插入 True Key (前半段)
        true_key = torch.randint(20, 100, (1,)).item()
        labels[i] = true_key
        true_pos = torch.randint(0, seq_len // 2 - 5, (1,)).item()
        data[i, true_pos] = 10
        data[i, true_pos+1] = true_key
        data[i, true_pos+2] = 11
        
        # 2. 随机位置插入 Fake Key (后半段，用来覆盖/干扰记忆)
        fake_key = torch.randint(20, 100, (1,)).item()
        # 确保 Fake Key 不等于 True Key
        while fake_key == true_key:
            fake_key = torch.randint(20, 100, (1,)).item()
            
        fake_pos = torch.randint(seq_len // 2, seq_len - 10, (1,)).item()
        data[i, fake_pos] = 12
        data[i, fake_pos+1] = fake_key
        data[i, fake_pos+2] = 11
        
        # 3. 序列末尾插入问题
        data[i, -1] = 13
        
    return data, labels

# ==========================================
# 2. 模型定义
# ==========================================

class TransformerBaseline(nn.Module):
    """传统的全量 Attention 架构"""
    def __init__(self, vocab_size, d_model=64, n_heads=4):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, 100000, d_model) * 0.01) # Max len support
        
        self.norm1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        B, L = x.shape
        h = self.emb(x) + self.pos_emb[:, :L, :]
        
        # Layer 1
        res = h
        h = self.norm1(h)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        
        # 为了不 OOM，使用 PyTorch 优化的 SDPA (FlashAttention)
        attn_out = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            is_causal=True
        ).transpose(1, 2).contiguous()
        
        h = res + self.out_proj(attn_out)
        h = h + self.mlp(self.norm2(h))
        
        # 仅取最后一个 Token 的特征进行预测
        return self.head(h[:, -1, :]), 1.0 # 1.0 表示 100% 门控利用率

try:
    from mamba_ssm import Mamba
    HAS_MAMBA_SSM = True
except ImportError:
    HAS_MAMBA_SSM = False
    print("Warning: mamba_ssm not found. Will use FFT-based simulated Mamba.")

class SSMBaseline(nn.Module):
    """
    纯 SSM / Mamba 架构。
    如果有 mamba_ssm 环境，则使用官方 Mamba；否则使用 FFT 模拟。
    """
    def __init__(self, vocab_size, d_model=64):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        
        if HAS_MAMBA_SSM:
            self.mamba = Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
        else:
            self.norm1 = nn.LayerNorm(d_model)
            self.in_proj = nn.Linear(d_model, d_model)
            self.A_log = nn.Parameter(torch.randn(d_model))
            self.out_proj = nn.Linear(d_model, d_model)
            
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        B, L = x.shape
        h = self.emb(x)
        res = h
        
        if HAS_MAMBA_SSM:
            y = self.mamba(h)
        else:
            h_norm = self.norm1(h)
            x_in = F.silu(self.in_proj(h_norm))
            decay = torch.exp(-torch.exp(self.A_log))
            t = torch.arange(L, device=x.device, dtype=torch.float32)
            filter_h = torch.exp(t.unsqueeze(1) * torch.log(decay.unsqueeze(0) + 1e-8))
            x_f = torch.fft.rfft(x_in, n=2*L, dim=1)
            h_f = torch.fft.rfft(filter_h, n=2*L, dim=0).unsqueeze(0)
            y = torch.fft.irfft(x_f * h_f, n=2*L, dim=1)[:, :L, :]
            y = self.out_proj(y)
            
        h = res + y
        h = h + self.mlp(self.norm2(h))
        
        return self.head(h[:, -1, :]), 0.0 # 0.0 表示 0% Attention 利用率

class RTNModel(nn.Module):
    """
    统一场架构 (RTN)：融合 Mamba (边界流) 与 稀疏 Attention (全息体流)
    包含：斯格明子保护 (通过相位复数表示) + 热力学门控 (熵感知)
    """
    def __init__(self, vocab_size, d_model=64, n_heads=4):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        # 为了局部 Attention 提供位置信息
        self.pos_emb = nn.Parameter(torch.randn(1, 100000, d_model) * 0.01)
        
        # --- 1. Mamba Core (System 1) ---
        if HAS_MAMBA_SSM:
            self.mamba = Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
        else:
            self.norm_ssm = nn.LayerNorm(d_model)
            self.ssm_proj = nn.Linear(d_model, d_model)
            # 引入“斯格明子”拓扑相：不仅仅是衰减，而是包含虚数旋转相位的记忆
            self.A_decay = nn.Parameter(torch.randn(d_model))
            self.A_phase = nn.Parameter(torch.randn(d_model) * 0.1) # 旋转相位
            self.ssm_out = nn.Linear(d_model, d_model)
        
        # --- 2. Thermo Gate (Maxwell's Demon) ---
        self.gate_net = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, 1)
        )
        
        # --- 3. Attention Core (System 2) ---
        self.norm_attn = nn.LayerNorm(d_model)
        self.attn_q = nn.Linear(d_model, d_model)
        self.attn_k = nn.Linear(d_model, d_model)
        self.attn_v = nn.Linear(d_model, d_model)
        self.attn_out = nn.Linear(d_model, d_model)
        
        self.norm_mlp = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        B, L = x.shape
        device = x.device
        h_orig = self.emb(x)
        
        # ==========================================
        # 路径 1: Mamba (连续扫描)
        # ==========================================
        if HAS_MAMBA_SSM:
            y_ssm = self.mamba(h_orig)
            h_out = h_orig + y_ssm
        else:
            h_ssm = self.norm_ssm(h_orig)
            x_in = F.silu(self.ssm_proj(h_ssm))
            
            decay = torch.exp(-torch.exp(self.A_decay)) # 衰减率 (0~1)
            phase = self.A_phase # 旋转角
            
            # 构建复数形式的拓扑记忆核: decay^t * e^{i * phase * t}
            t = torch.arange(L, device=device, dtype=torch.float32)
            log_term = torch.log(decay.unsqueeze(0) + 1e-8)
            # 模长衰减
            mag = torch.exp(t.unsqueeze(1) * log_term)
            # 角度旋转 (防止被后续噪声直线冲刷)
            ang = t.unsqueeze(1) * phase.unsqueeze(0)
            
            filter_real = mag * torch.cos(ang)
            
            x_f = torch.fft.rfft(x_in, n=2*L, dim=1)
            h_f = torch.fft.rfft(filter_real, n=2*L, dim=0).unsqueeze(0)
            y_ssm = torch.fft.irfft(x_f * h_f, n=2*L, dim=1)[:, :L, :]
            
            h_out = h_orig + self.ssm_out(y_ssm)
        
        # ==========================================
        # 路径 2: 热力学门控 Attention
        # ==========================================
        h_pos = h_out + self.pos_emb[:, :L, :]
        h_attn_norm = self.norm_attn(h_pos)
        
        # 计算局部熵 (Gate)
        gate_logits = self.gate_net(h_out)
        if self.training:
            gate = torch.sigmoid(gate_logits)
        else:
            gate = (torch.sigmoid(gate_logits) > 0.5).float()
            
        avg_gate = gate.mean()
        
        # 核心：计算全量 K, V，但 Q 只受 Gate 调制
        # 物理意义：全局知识都存入“潜意识”，但只有遇到“高熵节点”时，才发起主动搜索 (Query)
        Q = self.attn_q(h_attn_norm) * gate
        K = self.attn_k(h_attn_norm)
        V = self.attn_v(h_attn_norm)
        
        # 同样使用 SDPA 加速
        attn_res = F.scaled_dot_product_attention(
            Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2),
            is_causal=True
        ).transpose(1, 2).contiguous()
        
        h_out = h_out + self.attn_out(attn_res)
        
        # ==========================================
        # MLP 与 输出
        # ==========================================
        h_out = h_out + self.mlp(self.norm_mlp(h_out))
        
        return self.head(h_out[:, -1, :]), avg_gate

# ==========================================
# 3. 评测主循环
# ==========================================

def get_vram_mb():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 * 1024)
    return 0.0

def run_experiment(model_name, model, seq_len, batch_size=8, steps=1000, device='cuda'):
    print(f"\n[{model_name}] Training on Sequence Length: {seq_len} ...")
    model = model.to(device)
    # 调大一点学习率，加快收敛
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
    # 使用余弦退火学习率，帮助模型在后期稳定收敛
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    model.train()
    start_time = time.time()
    
    acc_history = []
    
    for step in range(steps):
        inputs, targets = generate_passkey_data(batch_size, seq_len)
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        logits, gate_usage = model(inputs)
        
        loss = F.cross_entropy(logits, targets)
        
        # 对于 RTN，增加稀疏正则化 (鼓励省电)，但初期权重给小一点，让它先学会做题
        if isinstance(gate_usage, torch.Tensor):
            # 前 20% 的时间不强求稀疏，先学会找 Key；后面再慢慢卡脖子
            warmup_factor = min(1.0, step / (steps * 0.2)) 
            loss = loss + 0.05 * warmup_factor * (gate_usage - 0.1)**2
            
        loss.backward()
        
        # 增加梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        
        preds = logits.argmax(dim=-1)
        acc = (preds == targets).float().mean().item()
        acc_history.append(acc)
        
        if (step + 1) % 100 == 0:
            avg_acc = sum(acc_history[-50:]) / 50
            g_u = gate_usage.item() if isinstance(gate_usage, torch.Tensor) else gate_usage
            print(f"  Step {step+1:4d} | Loss: {loss.item():.4f} | Acc: {avg_acc:.1%} | Gate: {g_u:.1%}")
            
    # 评测指标
    total_time = time.time() - start_time
    peak_vram = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0
    final_acc = sum(acc_history[-50:]) / 50 if len(acc_history) >= 50 else acc_history[-1]
    
    print(f"--> {model_name} Result:")
    print(f"    Final Accuracy : {final_acc:.1%}")
    print(f"    Peak VRAM      : {peak_vram:.0f} MB")
    print(f"    Speed          : {batch_size * steps * seq_len / total_time:.0f} tokens/sec")
    
    return final_acc, peak_vram

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"=== Holographic AI: Passkey Retrieval Long Context Test ===")
    print(f"Device: {device}")
    
    # 我们测试三个长度: 4K (短), 16K (中), 32K (长)
    # 对于 32K，Transformer 可能会爆炸，我们在 try-except 里捕捉
    seq_lengths = [4096, 16384]
    if torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory > 12*1024*1024*1024:
        seq_lengths.append(32768) # 显存够的话测 32K
        
    vocab_size = 1000
    batch_size = 8 # 设小点防止 Transformer 过早 OOM
    
    for L in seq_lengths:
        print("=" * 60)
        print(f"Testing Context Length: {L}")
        print("=" * 60)
        
        # 1. 传统 Transformer
        try:
            m_tx = TransformerBaseline(vocab_size)
            run_experiment("Transformer (FlashAttn)", m_tx, L, batch_size, steps=1000, device=device)
            del m_tx
        except RuntimeError as e:
            if 'out of memory' in str(e):
                print("--> [Transformer] OOM (Out of Memory)!")
            else:
                print(f"--> [Transformer] Error: {e}")
        torch.cuda.empty_cache()
        
        # 2. 传统 Mamba (SSM)
        try:
            m_ssm = SSMBaseline(vocab_size)
            run_experiment("Mamba / Linear SSM", m_ssm, L, batch_size, steps=1000, device=device)
            del m_ssm
        except Exception as e:
            print(f"--> [SSM] Error: {e}")
        torch.cuda.empty_cache()
        
        # 3. 您的 RTN (统一场架构)
        try:
            m_rtn = RTNModel(vocab_size)
            run_experiment("RTN (Unified Field)", m_rtn, L, batch_size, steps=1000, device=device)
            del m_rtn
        except Exception as e:
            print(f"--> [RTN] Error: {e}")
        torch.cuda.empty_cache()
