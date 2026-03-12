"""
MQAR Benchmark: TGN vs Mamba (The "Capacity Test")
-------------------------------------------------
Goal: Test the memory capacity of the models.
Task: Multi-Query Associative Recall (MQAR)
    Sequence: k1 v1 ... k2 v2 ... k3 v3 ... [Query: k2?] -> v2
    
    Crucial: The sequence contains MANY pairs (e.g., 32, 64).
    Mamba (Finite State) must compress all pairs into its fixed state.
    TGN (Attention) can keep them in KV cache and retrieve on demand.

Hypothesis:
    As number of pairs increases, Mamba's accuracy will drop sharply (State Saturation).
    TGN will maintain high accuracy.

Usage:
    torchrun --nproc_per_node=6 code/experiment_mqar.py
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

from mamba_minimal import MambaModel, MambaBlock

# =========================
# Configuration
# =========================
SEED = 2024
MAX_ITERS = 3000        # INCREASED: 2000 -> 3000
BATCH_SIZE = 32         
SEQ_LEN = 64            # TINY: 128 -> 64
NUM_PAIRS = 8           # FEWER: 16 -> 8
VOCAB_SIZE = 64         # SMALLER: 256 -> 64
N_LAYER = 2           
N_HEAD = 4
N_EMBD = 64
LR = 3e-3               # HIGHER LR again to speed up demo
EVAL_INTERVAL = 50
ENERGY_PENALTY = 0.001    # ENABLED: Force TGN to be sparse!

# =========================
# Data Generation (MQAR)
# =========================
class MQARDataset(IterableDataset):
    def __init__(self, seq_len, num_pairs, vocab_size, rank):
        self.seq_len = seq_len
        self.num_pairs = num_pairs
        self.vocab_size = vocab_size
        self.rank = rank
        
    def generate_sample(self):
        # 1. Generate unique keys and values
        # Reserve 0 for padding, 1 for query start marker
        available_tokens = np.arange(2, self.vocab_size)
        
        # Pick 2 * num_pairs unique tokens
        if len(available_tokens) < 2 * self.num_pairs:
            raise ValueError("Vocab too small for pairs")
            
        chosen = np.random.choice(available_tokens, 2 * self.num_pairs, replace=False)
        keys = chosen[:self.num_pairs]
        vals = chosen[self.num_pairs:]
        
        # 2. Distribute pairs in sequence
        # Fix: Ensure pairs do not overlap!
        # We need num_pairs slots of size 2.
        
        # Create a list of available slots (0, 2, 4, ...)
        # context_len is the available region size
        context_len = int(self.seq_len * 0.8)
        max_slots = context_len // 2
        
        if max_slots < self.num_pairs:
             raise ValueError(f"SeqLen {self.seq_len} too short for {self.num_pairs} pairs")
             
        slot_indices = np.random.choice(max_slots, self.num_pairs, replace=False)
        slot_indices.sort() # Optional, but keeps order random
        
        seq = np.zeros(self.seq_len, dtype=int)
        
        for i, slot_idx in enumerate(slot_indices):
            pos = slot_idx * 2
            seq[pos] = keys[i]
            seq[pos+1] = vals[i]
            
        # 3. Generate Query (Last part)
        # Pick a random key to query
        query_idx = np.random.randint(0, self.num_pairs)
        query_key = keys[query_idx]
        target_val = vals[query_idx]
        
        # Place query at the very end
        seq[-2] = 1 # Marker
        seq[-1] = query_key
        
        return torch.LongTensor(seq), torch.LongTensor([target_val])

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        seed = self.rank * 10000 + (worker_info.id if worker_info else 0)
        np.random.seed(seed)
        while True:
            yield self.generate_sample()

# =========================
# Models (Reuse from previous)
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
        
        # FlashAttention
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)

class HybridMambaBlock(nn.Module):
    def __init__(self, n_embd, n_head, mamba_layer):
        super().__init__()
        self.ln = nn.LayerNorm(n_embd)
        self.mamba = mamba_layer # Pre-instantiated mamba layer
        self.attn = CausalSelfAttention(n_embd, n_head)
        self.ffn = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), nn.GELU(),
            nn.Linear(4 * n_embd, n_embd)
        )

    def forward(self, x):
        residual = x
        x_norm = self.ln(x)
        
        # Parallel Hybrid: Mamba + Attention
        # Jamba style usually interleaves, but parallel is a strong baseline too.
        h_mamba = self.mamba(x_norm)
        h_attn = self.attn(x_norm)
        
        # Simple summation (Fixed Hybrid)
        h = residual + h_mamba + h_attn
        h = h + self.ffn(self.ln(h))
        return h

class HybridMambaModel(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(SEQ_LEN, n_embd)
        
        self.layers = nn.ModuleList()
        for _ in range(n_layer):
            if HAS_OFFICIAL_MAMBA:
                mamba_layer = OfficialMamba(d_model=n_embd)
            else:
                mamba_layer = MambaBlock(n_embd) # Use minimal block
                
            self.layers.append(HybridMambaBlock(n_embd, n_head, mamba_layer))
            
        self.norm_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(0, T, device=x.device)
        x = self.embedding(x) + self.pos_emb(pos)
        
        for layer in self.layers:
            x = layer(x)
            
        x = self.norm_f(x)
        return self.head(x[:, -1, :]), []

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
        
        # CRITICAL FIX: Attention must see the original history (x_norm), 
        # not just the compressed RNN state.
        # This allows Geometric Tunneling to bypass RNN bottleneck.
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
        self.head = nn.Linear(n_embd, vocab_size)

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(0, T, device=x.device)
        x = self.embedding(x) + self.pos_emb(pos)
        
        gates = []
        for layer in self.layers:
            x, g = layer(x)
            gates.append(g)
            
        x = self.norm_f(x)
        # Predict based on LAST token (which is the query key)
        logits = self.head(x[:, -1, :]) 
        return logits, gates

class MambaWrapper(MambaModel):
    def __init__(self, vocab_size, n_embd, n_layer):
        super().__init__(vocab_size, n_embd, n_layer)
        self.head = nn.Linear(n_embd, vocab_size) 
        
    def forward(self, x):
        # Mamba forward
        x = super().forward(x)
        return x[:, -1, :], [] 

class MambaOfficialWrapper(nn.Module):
    def __init__(self, vocab_size, n_embd, n_layer):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, n_embd)
        self.layers = nn.ModuleList([OfficialMamba(d_model=n_embd) for _ in range(n_layer)])
        self.norm_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)

    def forward(self, x):
        h = self.embedding(x)
        for layer in self.layers:
            h = h + layer(h)
        h = self.norm_f(h)
        return self.head(h[:, -1, :]), []

# =========================
# DDP Helpers (REMOVED FOR WINDOWS COMPATIBILITY)
# =========================
def run_experiment():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")
    
    backend = "Official CUDA" if HAS_OFFICIAL_MAMBA else "Minimal PyTorch"
    print(f"Running MQAR (L={SEQ_LEN}, Pairs={NUM_PAIRS}): TGN vs Mamba ({backend})")
    
    mamba_cls = MambaOfficialWrapper if HAS_OFFICIAL_MAMBA else MambaWrapper
    
    models = {
        "Mamba (SOTA)": mamba_cls(VOCAB_SIZE, N_EMBD, N_LAYER).to(device),
        "TGN (Ours)": TGNModel(VOCAB_SIZE, N_EMBD, N_HEAD, N_LAYER).to(device)
    }
    
    # NO DDP WRAPPING
        
    results = {name: [] for name in models}
    run_order = ["TGN (Ours)", "Mamba (SOTA)"]
    
    for name in run_order:
        model = models[name]
        print(f"\n>>> Training {name} (MQAR) <<<")
            
        opt = torch.optim.AdamW(model.parameters(), lr=LR)
        # Fix: removed rank arg
        ds = MQARDataset(SEQ_LEN, NUM_PAIRS, VOCAB_SIZE, 0)
        loader = DataLoader(ds, batch_size=BATCH_SIZE, num_workers=0) # Windows: workers=0 is safer
        iter_loader = iter(loader)
        
        t0 = time.time()
        
        for step in range(MAX_ITERS):
            opt.zero_grad()
            try:
                x, y = next(iter_loader)
            except StopIteration:
                iter_loader = iter(loader)
                x, y = next(iter_loader)
                
            x, y = x.to(device), y.to(device).view(-1)
            
            logits, gates = model(x)
            
            # Debug: Check shapes once
            if step == 0 and name == run_order[0]:
                print(f"DEBUG: logits {logits.shape}, y {y.shape}")
            
            loss = F.cross_entropy(logits, y)
            
            gate_val = 0.0
            if name == "TGN (Ours)" and len(gates) > 0:
                g_mean = torch.stack([g.mean() for g in gates]).mean()
                loss = loss + ENERGY_PENALTY * g_mean
                gate_val = g_mean.item()
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            
            preds = logits.argmax(dim=-1)
            acc = (preds == y).float().mean()
            # NO DIST ALL_REDUCE
            
            if step % EVAL_INTERVAL == 0:
                gate_str = f" | Gate {gate_val:.4f}" if name == "TGN (Ours)" else ""
                print(f"Iter {step}: Acc {acc.item()*100:.1f}%{gate_str} | t={time.time()-t0:.1f}s")
                results[name].append(acc.item())
                
    # Plotting
    plt.figure(figsize=(10, 6))
    
    # We only have one TGN/Mamba in results, so extract them directly
    # Adjust x-axis based on length of results
    steps_x = [i * EVAL_INTERVAL for i in range(len(results["TGN (Ours)"]))]
    
    plt.plot(steps_x, results["Mamba (SOTA)"], label="Mamba (SOTA)", linestyle="--", color='#3498db')
    plt.plot(steps_x, results["TGN (Ours)"], label="TGN (Ours)", linewidth=2, color='#e67e22')
    plt.xlabel("Steps")
    plt.ylabel("Accuracy")
    plt.title(f"MQAR Benchmark (Pairs={NUM_PAIRS}, Dim={N_EMBD})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("figures/mqar_benchmark.png", dpi=300)
    print("Saved plot to figures/mqar_benchmark.png")


if __name__ == "__main__":
    run_experiment()
