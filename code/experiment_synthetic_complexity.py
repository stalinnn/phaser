"""
Plan B: Synthetic Complexity Scaling
------------------------------------
Goal: Prove the "S-Curve" by varying Task Difficulty instead of Model Size.
Hypothesis: 
    - Easy Task -> Low Gate Rate (RNN suffices)
    - Hard Task -> High Gate Rate (Attention needed)
    - Impossible Task -> Saturation (Physical Limit)

Task: Multi-hop Associative Recall
    Input: "a=1 b=2 c=3 ... a?" -> Output: "1"
    Difficulty Control: Sequence Length & Num Pairs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
import csv
from pathlib import Path

# ==========================================
# Configuration
# ==========================================
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
TRAIN_ITERS = 5000  # Increased for convergence
EVAL_INTERVAL = 500 # Less frequent logging

# Fixed Model Capacity (Small-ish)
N_EMBD = 128
N_HEAD = 4
N_LAYER = 4
DROPOUT = 0.1

# Task Difficulties to Scan
DIFFICULTIES = [
    {"name": "Easy",   "seq_len": 32,  "vocab": 20},   # RNN can memorize
    {"name": "Medium", "seq_len": 128, "vocab": 50},   # Needs some Attention
    {"name": "Hard",   "seq_len": 512, "vocab": 100},  # Needs lots of Attention
    {"name": "Extreme","seq_len": 1024,"vocab": 200}, # Saturation test
]

# ==========================================
# Synthetic Dataset
# ==========================================
class AssociativeDataset(Dataset):
    def __init__(self, size, seq_len, vocab_size):
        self.size = size
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        # Format: k1 v1 k2 v2 ... k_query ?
        # We use a simple integer encoding
        # 0: padding/sep, 1..vocab: keys/values
        
        # Generate random key-value pairs
        num_pairs = (self.seq_len - 2) // 2
        keys = np.random.randint(1, self.vocab_size, size=num_pairs)
        values = np.random.randint(1, self.vocab_size, size=num_pairs)
        
        # Select a query from keys
        query_idx = np.random.randint(0, num_pairs)
        query_key = keys[query_idx]
        target_val = values[query_idx]
        
        # Construct sequence
        seq = []
        for k, v in zip(keys, values):
            seq.extend([k, v])
        
        # Truncate or Pad
        seq = seq[:self.seq_len-2]
        seq.append(query_key)
        # Target is next token prediction at the end
        
        x = torch.tensor(seq, dtype=torch.long)
        y = torch.tensor(target_val, dtype=torch.long) # Classification target
        
        return x, y

# ==========================================
# TGN Model (Simplified for speed)
# ==========================================
class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=-1)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        
        att = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
        att = att.masked_fill(mask == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.drop(self.proj(y))

class TGNBlock(nn.Module):
    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
        self.ln = nn.LayerNorm(n_embd)
        self.rnn = nn.GRU(n_embd, n_embd, batch_first=True)
        self.gate = nn.Sequential(
            nn.Linear(n_embd, 16),
            nn.Tanh(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        # NEUTRAL INIT: Let it learn to go up or down
        nn.init.constant_(self.gate[2].bias, 0.0) 
        
        self.attn = CausalSelfAttention(n_embd, n_head, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), nn.GELU(),
            nn.Linear(4 * n_embd, n_embd), nn.Dropout(dropout)
        )

    def forward(self, x):
        residual = x
        x_norm = self.ln(x)
        h_rnn, _ = self.rnn(x_norm)
        h_attn = self.attn(x_norm)
        g = self.gate(h_rnn)
        h_mixed = (1 - g) * h_rnn + g * h_attn
        h = residual + h_mixed
        h = h + self.ffn(self.ln(h))
        return h, g

class MiniTGN(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size + 1, N_EMBD) # +1 for padding
        self.pos_emb = nn.Embedding(2048, N_EMBD) # Max len
        self.blocks = nn.ModuleList([TGNBlock(N_EMBD, N_HEAD, DROPOUT) for _ in range(N_LAYER)])
        self.ln_f = nn.LayerNorm(N_EMBD)
        self.head = nn.Linear(N_EMBD, vocab_size + 1)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        
        gates = []
        for block in self.blocks:
            x, g = block(x)
            gates.append(g)
        
        x = self.ln_f(x)
        
        # Only predict the last token (target value)
        logits = self.head(x[:, -1, :]) 
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits, targets)
            
            # Energy Penalty (STRONGER PENALTY)
            # We want to force it to close if not needed.
            # If Loss is around 3.0, penalty should be ~0.3 (10%) to be significant.
            g_mean = torch.stack([g.mean() for g in gates]).mean()
            penalty = 0.1 * g_mean 
            loss = loss + penalty
            
        return logits, loss, gates

# ==========================================
# Main Experiment Loop
# ==========================================
def run_experiment():
    print(f"Plan B: Synthetic Complexity Scaling on {DEVICE}")
    print("------------------------------------------------")
    
    results = []
    
    for diff in DIFFICULTIES:
        print(f"\n>>> Testing Difficulty: {diff['name']} (Len={diff['seq_len']}, Vocab={diff['vocab']})")
        
        # 1. Prepare Data (More data for training)
        train_ds = AssociativeDataset(20000, diff['seq_len'], diff['vocab'])
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        
        # 2. Init Model (Fresh start for each difficulty)
        model = MiniTGN(diff['vocab']).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        
        # 3. Train
        model.train()
        gate_history = []
        
        for i, (x, y) in enumerate(train_loader):
            if i >= TRAIN_ITERS: break
            
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits, loss, gates = model(x, y)
            loss.backward()
            optimizer.step()
            
            g_val = torch.stack([g.mean() for g in gates]).mean().item()
            gate_history.append(g_val)
            
            if i % EVAL_INTERVAL == 0:
                print(f"Iter {i:4d} | Loss {loss.item():.4f} | Gate {g_val:.4f}")
        
        # 4. Record Final Converged Gate Rate
        final_gate = np.mean(gate_history[-50:]) # Average of last 50 steps
        print(f"--> Final Gate Rate: {final_gate:.4f}")
        results.append((diff['name'], diff['seq_len'], final_gate))

    # Summary
    print("\n\n=== Final Results: S-Curve Verification ===")
    print("Difficulty\tSeqLen\tGateRate")
    for name, seq, gate in results:
        print(f"{name}\t\t{seq}\t{gate:.4f}")

if __name__ == "__main__":
    run_experiment()
