"""
LRA Extreme Benchmark: TGN vs Mamba (The "Mamba Killer" Experiment)
------------------------------------------------------------------
Goal: Push sequence length and nesting depth to the limit to expose Mamba's state bottleneck.

Hypothesis:
    Mamba (Finite State) will fail at extreme depths (Context Loss).
    TGN (Geometric Tunneling) will survive via Attention.

Configuration:
    Seq Len: 4096 (Extreme)
    Nesting Depth: 50 (Extreme)
    Batch Size: 4 (To fit in VRAM)
    
Usage:
    torchrun --nproc_per_node=6 code/experiment_lra_extreme.py
"""

import os
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, IterableDataset
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

# Fix for matplotlib incompatibility with numpy 2.0
if not hasattr(np, 'Inf'):
    np.Inf = np.inf

# Increase recursion limit for deep nesting
sys.setrecursionlimit(10000)

# Ensure this file's directory is importable
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.append(str(_THIS_DIR))

# Prefer official CUDA Mamba
HAS_OFFICIAL_MAMBA = False
try:
    from mamba_ssm.modules.mamba_simple import Mamba as OfficialMamba
    HAS_OFFICIAL_MAMBA = True
except Exception:
    HAS_OFFICIAL_MAMBA = False

from mamba_minimal import MambaModel

# =========================
# Extreme Configuration
# =========================
SEED = 2077
MAX_ITERS = 2000
BATCH_SIZE = 4          # TINY BATCH to survive O(L^2) memory
SEQ_LEN = 4096          # EXTREME LENGTH
NEST_DEPTH = 50         # EXTREME DEPTH (Mamba Killer)
VOCAB_SIZE = 16 
N_LAYER = 4
N_HEAD = 4
N_EMBD = 64
LR = 1e-4               # LOW LR for stability
EVAL_INTERVAL = 50
ENERGY_PENALTY = 1e-3

PAD = 15
OP_MAX = 10
OP_MIN = 11
OP_SUM = 12
TOK_OPEN = 13
TOK_CLOSE = 14

# =========================
# Data Generation (Synthetic Deep Nesting - Fast Version)
# =========================
class DeepListOpsDataset(IterableDataset):
    def __init__(self, seq_len, rank):
        self.seq_len = seq_len
        self.rank = rank
        
    def generate_sample(self):
        # FAST Iterative Generation (No Recursion)
        # Structure: [ [ [ ... [ val ] ... ] ] ]
        # Depth: NEST_DEPTH
        # Length: SEQ_LEN
        
        # 1. Generate Treasure
        val = np.random.randint(0, 10)
        
        # 2. Construct Sequence
        # Start: 50 open brackets
        tokens = [TOK_OPEN] * NEST_DEPTH
        tokens.append(val)
        
        # End: 50 close brackets
        end_tokens = [TOK_CLOSE] * NEST_DEPTH
        
        # Middle: Noise
        # Total used so far: Depth + 1 + Depth = 101
        current_len = len(tokens) + len(end_tokens)
        noise_len = self.seq_len - current_len
        
        if noise_len > 0:
            # Add random noise operators to make it hard
            noise = np.random.choice([OP_MAX, OP_MIN, OP_SUM], size=noise_len).tolist()
            tokens.extend(noise)
        
        tokens.extend(end_tokens)
        
        # Truncate or Pad (should be exact if noise_len > 0)
        tokens = tokens[:self.seq_len]
        if len(tokens) < self.seq_len:
             tokens = tokens + [PAD] * (self.seq_len - len(tokens))
             
        # Target is the value inside the deepest nest
        last_idx = len(tokens) - 1
        
        return torch.LongTensor(tokens), torch.LongTensor([val]), torch.LongTensor([last_idx])

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        seed = self.rank * 10000 + (worker_info.id if worker_info else 0)
        np.random.seed(seed)
        while True:
            yield self.generate_sample()

# =========================
# Models (Same as before)
# =========================
class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=-1)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        
        # Efficient Attention (Scaled Dot Product)
        # Using F.scaled_dot_product_attention for FlashAttention speedup if available
        # This is CRITICAL for L=4096
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)

class TGNBlock(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.ln = nn.LayerNorm(n_embd)
        self.rnn = nn.GRU(n_embd, n_embd, batch_first=True)
        self.attn = CausalSelfAttention(n_embd, n_head)
        self.gate = nn.Sequential(
            nn.Linear(n_embd, 16), nn.Tanh(), nn.Linear(16, 1), nn.Sigmoid()
        )
        self.ffn = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), nn.GELU(),
            nn.Linear(4 * n_embd, n_embd)
        )

    def forward(self, x):
        residual = x
        x_norm = self.ln(x)
        
        # Parallel Structure
        h_rnn, _ = self.rnn(x_norm)
        h_attn = self.attn(x_norm)
        
        g = self.gate(h_rnn)
        h_mixed = (1 - g) * h_rnn + g * h_attn
        
        h = residual + h_mixed
        h = h + self.ffn(self.ln(h))
        return h, g

class TGNModel(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(SEQ_LEN, n_embd)
        self.layers = nn.ModuleList([TGNBlock(n_embd, n_head) for _ in range(n_layer)])
        self.norm_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, 10)

    def forward(self, x, last_idx):
        B, T = x.shape
        pos = torch.arange(0, T, device=x.device)
        # Handle position overflow if necessary (but we set vocab size enough)
        x = self.embedding(x) + self.pos_emb(pos)
        
        gates = []
        for layer in self.layers:
            x, g = layer(x)
            gates.append(g)
            
        x = self.norm_f(x)
        idx = last_idx.view(-1, 1, 1).expand(B, 1, x.size(-1))
        last_h = x.gather(1, idx).squeeze(1)
        logits = self.head(last_h)
        return logits, gates

class MambaMinimalWrapper(MambaModel):
    def __init__(self, vocab_size, n_embd, n_layer):
        super().__init__(vocab_size, n_embd, n_layer)
        self.head = nn.Linear(n_embd, 10)

    def forward(self, x, last_idx):
        h = self.embedding(x)
        for layer in self.layers:
            h = h + layer(h)
        h = self.norm_f(h)
        B = h.size(0)
        idx = last_idx.view(-1, 1, 1).expand(B, 1, h.size(-1))
        last_h = h.gather(1, idx).squeeze(1)
        return self.head(last_h), []

class MambaOfficialWrapper(nn.Module):
    def __init__(self, vocab_size, n_embd, n_layer):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, n_embd)
        self.layers = nn.ModuleList([OfficialMamba(d_model=n_embd) for _ in range(n_layer)])
        self.norm_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, 10)

    def forward(self, x, last_idx):
        h = self.embedding(x)
        for layer in self.layers:
            h = h + layer(h)
        h = self.norm_f(h)
        B = h.size(0)
        idx = last_idx.view(-1, 1, 1).expand(B, 1, h.size(-1))
        last_h = h.gather(1, idx).squeeze(1)
        return self.head(last_h), []

# =========================
# DDP Helpers
# =========================
def setup_ddp():
    if "RANK" not in os.environ:
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12358" # New port
    dist.init_process_group(backend="nccl")
    return int(os.environ["LOCAL_RANK"])

def is_main_process(): return dist.get_rank() == 0

def run_experiment():
    local_rank = setup_ddp()
    torch.cuda.set_device(local_rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    
    if is_main_process():
        backend = "Official CUDA" if HAS_OFFICIAL_MAMBA else "Minimal PyTorch"
        print(f"Running EXTREME ListOps (L={SEQ_LEN}, D={NEST_DEPTH}): TGN vs Mamba ({backend})")
    
    mamba_cls = MambaOfficialWrapper if HAS_OFFICIAL_MAMBA else MambaMinimalWrapper
    models = {
        "Mamba": mamba_cls(VOCAB_SIZE, N_EMBD, N_LAYER).to(local_rank),
        "TGN": TGNModel(VOCAB_SIZE, N_EMBD, N_HEAD, N_LAYER).to(local_rank)
    }
    
    for name in models:
        models[name] = DDP(models[name], device_ids=[local_rank], find_unused_parameters=False)
        
    results = {name: [] for name in models}
    run_order = ["TGN", "Mamba"]
    
    for name in run_order:
        model = models[name]
        if is_main_process():
            print(f"\n>>> Training {name} (Extreme) <<<")
            
        opt = torch.optim.AdamW(model.parameters(), lr=LR)
        # Use num_workers=0 to avoid pickling recursion limit with deep nesting?
        # Let's try workers=0 for safety.
        ds = DeepListOpsDataset(SEQ_LEN, dist.get_rank())
        loader = DataLoader(ds, batch_size=BATCH_SIZE, num_workers=0)
        iter_loader = iter(loader)
        
        t0 = time.time()
        
        for step in range(MAX_ITERS):
            opt.zero_grad()
            try:
                x, y, last_idx = next(iter_loader)
            except RecursionError:
                if is_main_process(): print("Skipping batch (Recursion Limit)")
                continue
                
            x = x.to(local_rank)
            y = y.to(local_rank).squeeze()
            last_idx = last_idx.to(local_rank).squeeze()
            
            logits, gates = model(x, last_idx)
            loss = F.cross_entropy(logits, y)
            
            gate_val = 0.0
            if name == "TGN" and len(gates) > 0:
                g_mean = torch.stack([g.mean() for g in gates]).mean()
                loss = loss + ENERGY_PENALTY * g_mean
                gate_val = g_mean.item()
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            
            # Accuracy
            preds = logits.argmax(dim=-1)
            acc = (preds == y).float().mean()
            dist.all_reduce(acc, op=dist.ReduceOp.AVG)
            
            if step % EVAL_INTERVAL == 0 and is_main_process():
                gate_str = f" | Gate {gate_val:.4f}" if name == "TGN" else ""
                print(f"Iter {step}: Acc {acc.item()*100:.1f}%{gate_str} | t={time.time()-t0:.1f}s")
                results[name].append(acc.item())
                
    if is_main_process():
        plt.figure(figsize=(10, 6))
        x = [i * EVAL_INTERVAL for i in range(len(results["TGN"]))]
        plt.plot(x, results["Mamba"], label="Mamba (SOTA)", linestyle="--")
        plt.plot(x, results["TGN"], label="TGN (Ours)", linewidth=2)
        plt.xlabel("Steps")
        plt.ylabel("Accuracy")
        plt.title(f"Extreme ListOps (L={SEQ_LEN}, D={NEST_DEPTH}): State Bottleneck Test")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig("paper_archive/figures/lra_extreme.png", dpi=300)
        print("Saved plot to paper_archive/figures/lra_extreme.png")
        
    dist.destroy_process_group()

if __name__ == "__main__":
    run_experiment()
