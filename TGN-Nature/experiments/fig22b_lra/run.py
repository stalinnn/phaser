"""
LRA Benchmark: TGN vs Mamba (Paper Section 6.4)
----------------------------------------------
Goal: Challenge SOTA State-Space Models (Mamba) on hierarchical reasoning tasks.

Task: Simplified ListOps (Nested Bracket Matching)
    Input: [ MAX 2 [ MIN 9 3 ] 1 ]
    Target: 2 (Result of operation)
    
Why this is hard for Mamba:
    Mamba compresses history into a fixed-size state. Deep nesting requires a "stack",
    which finite states struggle to approximate perfectly.
    TGN can use Attention to "jump" from closing bracket to opening operator.

Usage:
    torchrun --nproc_per_node=6 code/experiment_lra_benchmark.py
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

# Ensure this file's directory is importable (works for both flat and `code/` layouts)
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.append(str(_THIS_DIR))

# Prefer official CUDA Mamba if available; otherwise fall back to a pure PyTorch baseline.
HAS_OFFICIAL_MAMBA = False
try:
    from mamba_ssm.modules.mamba_simple import Mamba as OfficialMamba  # type: ignore
    HAS_OFFICIAL_MAMBA = True
except Exception:
    HAS_OFFICIAL_MAMBA = False

from mamba_minimal import MambaModel

# =========================
# Configuration
# =========================
SEED = 42
MAX_ITERS = 500        # REDUCED: 3000 -> 2000 (Longer seq len takes more time)
 # The minimal (loop-based) Mamba baseline is slow; keep the fallback lightweight.
BATCH_SIZE = 8          # REDUCED: 32 -> 8 (To fit 2048 len in VRAM)
SEQ_LEN = 2048          # INCREASED: 512 -> 2048 (Challenge Mamba's memory)
VOCAB_SIZE = 16         # 0-9, MAX/MIN/SUM, [, ], PAD
N_LAYER = 4
N_HEAD = 4
N_EMBD = 64             # Small model to highlight architectural bias
LR = 5e-4               # REDUCED: 1e-3 -> 5e-4 for stability
# With DDP we only print from rank0. Too-large intervals look like a hang.
EVAL_INTERVAL = 10 if HAS_OFFICIAL_MAMBA else 25

# TGN Config
ENERGY_PENALTY = 1e-3

# Token IDs
PAD = 15

# =========================
# Data Generation (ListOps Logic)
# =========================
# Operators:
# 10: MAX, 11: MIN, 12: SUM (mod 10), 13: [ , 14: ]
OP_MAX = 10
OP_MIN = 11
OP_SUM = 12
TOK_OPEN = 13
TOK_CLOSE = 14

class ListOpsDataset(IterableDataset):
    def __init__(self, seq_len, rank):
        self.seq_len = seq_len
        self.rank = rank
        
    def generate_sample(self):
        # Recursive generation of nested expression
        # Returns: (tokens_list, value)
        
        def gen_expr(depth):
            if depth > 10 or np.random.rand() < 0.3:
                # Leaf: digit 0-9
                val = np.random.randint(0, 10)
                return [val], val
            
            # Operator
            op = np.random.choice([OP_MAX, OP_MIN, OP_SUM])
            num_args = np.random.randint(2, 5)
            
            tokens = [TOK_OPEN, op]
            values = []
            
            for _ in range(num_args):
                t, v = gen_expr(depth + 1)
                tokens.extend(t)
                values.append(v)
                
            tokens.append(TOK_CLOSE)
            
            # Compute result
            res = 0
            if op == OP_MAX: res = max(values)
            elif op == OP_MIN: res = min(values)
            elif op == OP_SUM: res = sum(values) % 10
            
            return tokens, res

        # Generate fits in seq_len
        while True:
            tokens, val = gen_expr(0)
            if len(tokens) <= self.seq_len:
                # Pad
                last_idx = len(tokens) - 1
                tokens = tokens + [PAD] * (self.seq_len - len(tokens))
                return torch.LongTensor(tokens), torch.LongTensor([val]), torch.LongTensor([last_idx])

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        seed = self.rank * 1000 + (worker_info.id if worker_info else 0)
        np.random.seed(seed)
        
        while True:
            yield self.generate_sample()

# =========================
# TGN Model (Reusable)
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
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
        att = att.masked_fill(mask == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
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
        # Standard learnable pos embedding for TGN
        self.pos_emb = nn.Embedding(SEQ_LEN, n_embd)
        self.layers = nn.ModuleList([TGNBlock(n_embd, n_head) for _ in range(n_layer)])
        self.norm_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, 10) # Output is digit 0-9

    def forward(self, x, last_idx):
        B, T = x.shape
        pos = torch.arange(0, T, device=x.device)
        x = self.embedding(x) + self.pos_emb(pos)
        
        gates = []
        for layer in self.layers:
            x, g = layer(x)
            gates.append(g)
            
        x = self.norm_f(x)
        # Gather representation at the true last (non-pad) token.
        idx = last_idx.view(-1, 1, 1).expand(B, 1, x.size(-1))
        last_h = x.gather(1, idx).squeeze(1)
        logits = self.head(last_h)
        return logits, gates

class MambaMinimalWrapper(MambaModel):
    def __init__(self, vocab_size, n_embd, n_layer):
        super().__init__(vocab_size, n_embd, n_layer)
        self.head = nn.Linear(n_embd, 10)  # output digits

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
    """Uses CUDA-accelerated Mamba from `mamba-ssm` if installed."""
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
        # Single process mode
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12357"
        os.environ["LOCAL_RANK"] = "0" # Fake local rank
        return 0
    
    dist.init_process_group(backend="nccl")
    return int(os.environ["LOCAL_RANK"])

def is_main_process():
    if not dist.is_available() or not dist.is_initialized():
        return True
    return dist.get_rank() == 0

def run_experiment():
    local_rank = setup_ddp()
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        torch.backends.cuda.matmul.allow_tf32 = True
    
    if is_main_process():
        backend = "official CUDA mamba-ssm" if HAS_OFFICIAL_MAMBA else "mamba_minimal (slow, PyTorch loop)"
        print(f"Running ListOps Benchmark: TGN vs Mamba ({backend})")
    
    mamba_cls = MambaOfficialWrapper if HAS_OFFICIAL_MAMBA else MambaMinimalWrapper
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    
    models = {
        "Mamba": mamba_cls(VOCAB_SIZE, N_EMBD, N_LAYER).to(device),
        "TGN": TGNModel(VOCAB_SIZE, N_EMBD, N_HEAD, N_LAYER).to(device)
    }
    
    # Wrap DDP only if distributed
    if dist.is_initialized():
        for name in models:
            models[name] = DDP(models[name], device_ids=[local_rank], find_unused_parameters=False)
            
    results = {name: [] for name in models}
    
    # Reorder to run Mamba first
    run_order = ["Mamba", "TGN"]
    
    for name in run_order:
        model = models[name]
        if is_main_process():
            print(f"\n>>> Training {name} <<<")
            
        opt = torch.optim.AdamW(model.parameters(), lr=LR)
        # Use simple dataset if not distributed
        rank = dist.get_rank() if dist.is_initialized() else 0
        ds = ListOpsDataset(SEQ_LEN, rank)
        
        loader = DataLoader(
            ds,
            batch_size=BATCH_SIZE,
            num_workers=0, # Safer default
        )
        iter_loader = iter(loader)
        
        t0 = time.time()
        
        for step in range(MAX_ITERS):
            opt.zero_grad()
            x, y, last_idx = next(iter_loader)
            x = x.to(device)
            y = y.to(device).squeeze()
            last_idx = last_idx.to(device).squeeze()
            
            logits, gates = model(x, last_idx)
            loss = F.cross_entropy(logits, y)
            
            # TGN Penalty and Gate Logging
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
            
            if dist.is_initialized():
                dist.all_reduce(acc, op=dist.ReduceOp.AVG)
            
            if step % EVAL_INTERVAL == 0 and is_main_process():
                gate_str = f" | Gate {gate_val:.4f}" if name == "TGN" else ""
                print(f"Iter {step}: Acc {acc.item()*100:.1f}%{gate_str} | t={time.time()-t0:.1f}s")
                results[name].append(acc.item())
                
        if is_main_process():
            print(f"Finished {name} in {time.time()-t0:.1f}s")
            
    # Plotting
    if is_main_process():
        plt.figure(figsize=(10, 6))
        x = [i * EVAL_INTERVAL for i in range(len(results["TGN"]))]
        plt.plot(x, results["Mamba"], label="Mamba (SOTA)", linestyle="--", color='#3498db')
        plt.plot(x, results["TGN"], label="TGN (Ours)", linewidth=2, color='#e67e22')
        plt.xlabel("Steps")
        plt.ylabel("Accuracy")
        plt.title("ListOps (Hierarchical Reasoning): TGN vs Mamba")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig("figures/lra_benchmark.png", dpi=300)
        print("Saved plot to figures/lra_benchmark.png")
        
    dist.destroy_process_group()

if __name__ == "__main__":
    run_experiment()
