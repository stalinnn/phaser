"""
Ablation Study: TGN Components (Paper Section 6.x)
-------------------------------------------------
Goal: Dissect the TGN architecture to prove the necessity of both channels.
We compare three variants under IDENTICAL training conditions (Small config):

1. Full TGN: Inertia (RNN) + Geometry (Attn) + Gate.
2. No-RNN (Pure Attn): Essentially a Gated Transformer (or just Transformer if Gate=1).
3. No-Attn (Pure RNN): A deep Residual GRU network.

Hypothesis:
- No-Attn will saturate early (high loss) due to lack of geometric tunneling.
- No-RNN will be slow/expensive and lack local inductive bias.
- Full TGN will achieve the best Pareto frontier of Loss vs Compute.

Usage:
    python code/experiment_ablation_study.py
"""

import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import requests

# =========================
# Configuration (Small Config for Speed)
# =========================================
SEED = 42
MAX_ITERS = 2000
BLOCK_SIZE = 1024  # Increased from 256 to stress-test RNN and highlight Attention
BATCH_SIZE = 8     # Lowered from 32 to fit in VRAM with larger context
GRAD_ACCUM = 8     # Increased to maintain effective batch size
N_LAYER = 8
N_HEAD = 8
N_EMBD = 384
DROPOUT = 0.1
LR = 3e-4
VAL_FRAC = 0.1
EVAL_INTERVAL = 200

# TGN Specifics
ENERGY_PENALTY = 1e-3
ENERGY_PENALTY_WARMUP = 2000

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)

def get_project_root() -> Path:
    try:
        start = Path(__file__).resolve().parent
    except NameError:
        start = Path.cwd()
    if (start / "paper_archive").exists():
        return start
    for p in [start] + list(start.parents):
        if (p / "paper_archive").exists():
            return p
    return start

# =========================
# Data
# =========================
class CharDataset(Dataset):
    def __init__(self, text, block_size):
        chars = sorted(list(set(text)))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.vocab_size = len(chars)
        self.data = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)
        self.block_size = block_size
    def __len__(self): return self.data.numel() - self.block_size - 1
    def __getitem__(self, idx):
        return self.data[idx:idx+self.block_size], self.data[idx+1:idx+1+self.block_size]

# =========================
# Model Components
# =========================
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
        q, k, v = self.qkv(x).split(C, dim=-1)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
        att = att.masked_fill(mask == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = (self.drop(att) @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.drop(self.proj(y))

class AblationTGNLayer(nn.Module):
    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
        self.ln = nn.LayerNorm(n_embd)
        self.rnn = nn.GRU(n_embd, n_embd, batch_first=True)
        self.attn = CausalSelfAttention(n_embd, n_head, dropout)
        self.gate = nn.Sequential(
            nn.Linear(n_embd, 16), nn.Tanh(), nn.Linear(16, 1), nn.Sigmoid()
        )
        # Bias Init Trick
        nn.init.constant_(self.gate[2].bias, 2.0) 
        self.ffn = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), nn.GELU(),
            nn.Linear(4 * n_embd, n_embd), nn.Dropout(dropout)
        )

    def forward(self, x, mode="full"):
        residual = x
        x = self.ln(x)
        
        # --- Ablation Logic ---
        if mode == "no_rnn":
            # Pure Attention Mode (Gate=1)
            # We skip RNN computation to save time, but logic is h_rnn=0, g=1
            h_attn = self.attn(x)
            h_mixed = h_attn
            g_out = torch.ones(x.shape[:2] + (1,), device=x.device)
            
        elif mode == "no_attn":
            # Pure RNN Mode (Gate=0)
            h_rnn, _ = self.rnn(x)
            h_mixed = h_rnn
            g_out = torch.zeros(x.shape[:2] + (1,), device=x.device)
            
        else: # "full"
            h_rnn, _ = self.rnn(x)
            g = self.gate(h_rnn)
            h_attn = self.attn(h_rnn) # Query from RNN state (Standard TGN)
            h_mixed = (1 - g) * h_rnn + g * h_attn
            g_out = g

        # Residual Fix
        h = residual + h_mixed
        h = h + self.ffn(self.ln(h))
        return h, g_out

class AblationModel(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, dropout):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, n_embd)
        self.pos = nn.Embedding(BLOCK_SIZE, n_embd)
        self.drop = nn.Dropout(dropout)
        self.layers = nn.ModuleList([AblationTGNLayer(n_embd, n_head, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, idx, mode="full", targets=None):
        B, T = idx.shape
        pos = torch.arange(0, T, device=idx.device)
        x = self.drop(self.tok(idx) + self.pos(pos))
        
        gates = []
        for layer in self.layers:
            x, g = layer(x, mode=mode)
            gates.append(g)
            
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss, gates

# =========================
# Experiment Runner
# =========================
def run_ablation():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Ablation on {device}")
    
    # 1. Prepare Data
    root = get_project_root()
    path = root / "data" / "tinyshakespeare_input.txt"
    if not path.exists():
        # Download... (omitted for brevity, assume exists or handled by other scripts)
        pass
    text = path.read_text(encoding="utf-8")
    dataset = CharDataset(text, BLOCK_SIZE)
    train_ds, val_ds = random_split(dataset, [len(dataset)-int(len(dataset)*VAL_FRAC), int(len(dataset)*VAL_FRAC)])
    
    # 2. Define Modes
    modes = ["no_attn", "no_rnn", "full"]
    results = {}
    
    for mode in modes:
        print(f"\n>>> Training Variant: {mode.upper()} <<<")
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
        
        model = AblationModel(dataset.vocab_size, N_EMBD, N_HEAD, N_LAYER, DROPOUT).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=LR)
        
        train_losses = []
        val_losses = []
        
        t0 = time.time()
        iter_train = iter(train_loader)
        
        for step in range(MAX_ITERS):
            opt.zero_grad()
            loss_accum = 0.0
            
            for _ in range(GRAD_ACCUM):
                try: x, y = next(iter_train)
                except StopIteration:
                    iter_train = iter(train_loader)
                    x, y = next(iter_train)
                
                x, y = x.to(device), y.to(device)
                _, loss, gates = model(x, mode=mode, targets=y)
                
                # Penalty only for "full" mode
                if mode == "full":
                    g_mean = torch.stack([g.mean() for g in gates]).mean()
                    ramp = min(1.0, (step+1)/ENERGY_PENALTY_WARMUP)
                    loss = loss + (ENERGY_PENALTY * ramp * g_mean)
                
                loss_accum += loss.item() / GRAD_ACCUM
                (loss / GRAD_ACCUM).backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            
            if step % EVAL_INTERVAL == 0:
                print(f"Iter {step}: Loss {loss_accum:.4f}")
                train_losses.append(loss_accum)
                # Validation
                model.eval()
                v_loss = 0
                with torch.no_grad():
                    for _ in range(5): # Quick val
                        x, y = next(iter(val_loader))
                        _, l, _ = model(x.to(device), mode=mode, targets=y.to(device))
                        v_loss += l.item()
                val_losses.append(v_loss / 5)
                model.train()
        
        results[mode] = val_losses
        print(f"Finished {mode} in {time.time()-t0:.1f}s. Final Val Loss: {val_losses[-1]:.4f}")
        del model, opt
        torch.cuda.empty_cache()

    # 3. Plot
    plt.figure(figsize=(10, 6))
    x = [i * EVAL_INTERVAL for i in range(len(results["full"]))]
    
    plt.plot(x, results["no_attn"], 'r--', label="No-Attn (Pure RNN)")
    plt.plot(x, results["no_rnn"], 'b--', label="No-RNN (Pure Attn)")
    plt.plot(x, results["full"], 'g-', linewidth=2, label="Full TGN")
    
    plt.xlabel("Steps")
    plt.ylabel("Validation Loss")
    plt.title("Ablation Study: Contribution of TGN Components")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_path = root / "paper_archive" / "figures" / "ablation_study.png"
    plt.savefig(save_path, dpi=300)
    print(f"Saved plot to {save_path}")

if __name__ == "__main__":
    run_ablation()
