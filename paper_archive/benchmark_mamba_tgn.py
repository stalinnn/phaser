import torch
import time
import math
import torch.nn as nn
import torch.nn.functional as F
from run_cloud_sota_battle import Config, CausalSelfAttention, GeometricGate

try:
    from mamba_ssm import Mamba as FastMamba
    print(">>> ✅ Mamba-SSM (CUDA) loaded.")
except ImportError:
    print(">>> ❌ Mamba-SSM not found.")
    FastMamba = None

class MambaTGNBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        # The Upgrade: Mamba as Inertia
        self.inertia = FastMamba(
            d_model=config.n_embd,
            d_state=16,
            d_conv=4,
            expand=2
        )
        self.attn = CausalSelfAttention(config)
        self.gate = GeometricGate(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd), nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd), nn.Dropout(config.dropout)
        )

    def forward(self, x):
        # 1. Fast Inertia
        inertia_out = self.inertia(self.ln1(x))
        
        # 2. Gate
        g = self.gate(inertia_out)
        
        # 3. Geometry (Full Attention for now, simulating cost)
        # In ideal world with sparse kernel, this cost would be * 0.10
        # But here we pay full price to be honest about current implementation overhead
        attn_out = self.attn(self.ln1(x))
        
        # 4. Mix
        mixed = (1 - g) * inertia_out + g * attn_out
        
        x = x + mixed
        x = x + self.mlp(self.ln2(x))
        return x

class MambaTGN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([MambaTGNBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, idx):
        x = self.token_embedding(idx)
        x = self.drop(x)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_f(x)
        return self.head(x)

def benchmark_mamba_tgn():
    if FastMamba is None: return
    
    device = "cuda"
    config = Config(model_size='medium') # 350M
    model = MambaTGN(config).to(device).half() # FP16
    
    batch_size = 32
    seq_len = 1024
    x = torch.randint(0, 50304, (batch_size, seq_len)).to(device)
    
    print("Warmup Mamba-TGN...")
    with torch.no_grad():
        for _ in range(5): _ = model(x)
    torch.cuda.synchronize()
    
    print("Benchmarking Mamba-TGN...")
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    steps = 50
    with torch.no_grad():
        for _ in range(steps): _ = model(x)
    end.record()
    torch.cuda.synchronize()
    
    dt = start.elapsed_time(end) / 1000
    tokens = batch_size * seq_len * steps
    speed = tokens / dt
    mem = torch.cuda.max_memory_allocated() / 1024**3
    
    print(f"\n>>> Mamba-TGN Result:")
    print(f"Throughput: {speed:.2f} tok/s")
    print(f"Memory:     {mem:.2f} GB")

if __name__ == "__main__":
    benchmark_mamba_tgn()
