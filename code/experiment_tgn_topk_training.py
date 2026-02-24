"""
EXP: Thermodynamic Threshold Gating (Dynamic Sparsity)
------------------------------------------------------
Core Idea: Let the model decide HOW MANY tokens to attend to.
Method:
1. Calculate Energy E = Sigmoid(Gate(h))
2. Select tokens where E > Threshold (e.g., 0.5)
3. Load Balancing: Cap at K_MAX to prevent OOM.

This implements "Scheme A (Threshold)" + "Load Balancing".
"""

import math
import time
import csv
from pathlib import Path
import numpy as np
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

# =========================
# Configuration
# =========================
SEED = 42
MAX_ITERS = 1000          
BLOCK_SIZE = 256          
BATCH_SIZE = 32           
N_EMBD = 384              
N_HEAD = 8                
N_LAYER = 8               
DROPOUT = 0.1
LR = 3e-4
VAL_FRAC = 0.1

# Thermodynamic Config
GATE_THRESHOLD = 0.5      # E > 0.5 activates attention
K_CAP_PERCENT = 0.5       # Hard Cap: Max 50% tokens (Load Balancing)

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

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

class CharDataset(Dataset):
    def __init__(self, text: str, block_size: int):
        chars = sorted(list(set(text)))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.vocab_size = len(chars)
        self.data = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)
        self.block_size = block_size

    def __len__(self):
        return max(0, self.data.numel() - self.block_size - 1)

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1 : idx + 1 + self.block_size]
        return x, y

# =========================
# Thermodynamic Components
# =========================

class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, K):
        # x: (B, K, C) - Ragged but padded to K_MAX effectively
        # But wait, in batch processing, K must be same for all batch elements?
        # Yes, we usually take max(K_b) in the batch or fixed K_MAX.
        # Here we assume x is (B, K_actual, C).
        B, _, C = x.shape
        
        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=-1)
        
        q = q.view(B, K, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, K, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, K, self.n_head, self.head_dim).transpose(1, 2)
        
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = torch.tril(torch.ones(K, K, device=x.device)).view(1, 1, K, K)
        att = att.masked_fill(mask == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.drop(att)
        
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, K, C)
        return self.drop(self.proj(y))

class ThermoGateLayer(nn.Module):
    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
        self.rnn = nn.GRU(n_embd, n_embd, batch_first=True)
        self.gate = nn.Sequential(
            nn.Linear(n_embd, 32), nn.Tanh(),
            nn.Linear(32, 1), nn.Sigmoid()
        )
        # Init bias slightly negative to encourage sparsity start
        nn.init.constant_(self.gate[2].bias, -1.0) 
        
        self.attn = CausalSelfAttention(n_embd, n_head, dropout)
        self.ln = nn.LayerNorm(n_embd)
        self.ffn = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), nn.GELU(),
            nn.Linear(4 * n_embd, n_embd), nn.Dropout(dropout)
        )

    def forward(self, x):
        B, T, C = x.shape
        
        # 1. Inertia
        h_rnn, _ = self.rnn(x)
        
        # 2. Gate Energy
        energy = self.gate(h_rnn) # (B, T, 1)
        
        # 3. Dynamic Thresholding + Cap
        # We need to select top K tokens where K is determined by Energy > Threshold
        # BUT for batch efficiency, we must pick a FIXED K for this batch step.
        # Strategy: Let K = count(energy > threshold).avg() or max()?
        # To be safe and fast: Pick Top-K where K = K_CAP (Hard Limit)
        # But mask out the ones that are below threshold.
        
        K_MAX = int(K_CAP_PERCENT * T)
        
        # Sort by energy to find candidates
        # values: (B, T), indices: (B, T)
        sorted_energy, sorted_indices = torch.sort(energy.squeeze(-1), descending=True, dim=1)
        
        # Take Top K_MAX candidates
        top_k_indices = sorted_indices[:, :K_MAX] # (B, K_MAX)
        top_k_energy = sorted_energy[:, :K_MAX]   # (B, K_MAX)
        
        # Restore time order for Causal Attention
        top_k_indices, _ = torch.sort(top_k_indices, dim=1)
        
        # 4. Gather
        gather_indices = top_k_indices.unsqueeze(-1).expand(B, K_MAX, C)
        x_selected = torch.gather(h_rnn, 1, gather_indices) # (B, K_MAX, C)
        
        # 5. Sparse Attention
        x_attn_out = self.attn(x_selected, K_MAX)
        
        # 6. Thermodynamic Masking (The "Real" Gate)
        # Even if we computed attention for K_MAX tokens, we only KEEP the result
        # if energy > threshold.
        # We need to fetch the energy for the sorted time-ordered indices
        # Re-gather energy based on time-sorted indices
        g_selected = torch.gather(energy, 1, top_k_indices.unsqueeze(-1)) # (B, K_MAX, 1)
        
        # Soft Thresholding for Gradient: 
        # mask = sigmoid((energy - threshold) * sharp_factor) ?
        # Or simple multiplication: output = attn * energy
        # Let's use simple multiplication (Soft Gating on top of Top-K routing)
        # This allows gradient to flow to Gate to say "increase energy for this token".
        
        weighted_attn_out = x_attn_out * g_selected
        
        # 7. Scatter
        attn_full = torch.zeros_like(h_rnn)
        attn_full.scatter_(1, gather_indices, weighted_attn_out)
        
        h = h_rnn + attn_full
        h = self.ln(h)
        h = h + self.ffn(h)
        
        # Calculate actual active rate for logging (how many > threshold)
        active_count = (energy > GATE_THRESHOLD).float().sum()
        
        return h, energy, active_count

class ThermoTGN(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, dropout):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.pos_embedding = nn.Embedding(BLOCK_SIZE, n_embd)
        self.drop = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            ThermoGateLayer(n_embd, n_head, dropout) for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)
        x = self.token_embedding(idx) + self.pos_embedding(pos)
        x = self.drop(x)
        
        total_active = 0
        all_energies = []
        
        for layer in self.layers:
            x, energy, act = layer(x)
            total_active += act
            all_energies.append(energy)
            
        x = self.ln_f(x)
        logits = self.head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            
            # Sparsity Penalty: Encourage Energy < Threshold
            # Penalty = mean(ReLU(Energy - Margin))
            # Or simple L1 on Energy
            avg_energy = torch.stack([e.mean() for e in all_energies]).mean()
            loss = loss + 0.05 * avg_energy # Push down
            
        return logits, loss, total_active

# =========================
# Runner
# =========================
def run_experiment():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Thermodynamic Threshold TGN on {device}")
    
    # Data Setup (Standard TinyShakespeare)
    root = get_project_root()
    data_path = root / "data" / "tinyshakespeare_input.txt"
    if not data_path.exists():
        print("Downloading data...")
        data_path.parent.mkdir(parents=True, exist_ok=True)
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        data_path.write_text(requests.get(url).text, encoding="utf-8")
    text = data_path.read_text(encoding="utf-8")
    dataset = CharDataset(text, BLOCK_SIZE)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    
    model = ThermoTGN(dataset.vocab_size, N_EMBD, N_HEAD, N_LAYER, DROPOUT).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    
    start_time = time.time()
    
    for step, (xb, yb) in enumerate(train_loader):
        if step >= MAX_ITERS: break
        xb, yb = xb.to(device), yb.to(device)
        
        optimizer.zero_grad()
        logits, loss, active_count = model(xb, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if step % 50 == 0:
            dt = time.time() - start_time
            # Real Sparsity = (Energy > 0.5) / Total
            # Note: We compute at most K_CAP (50%), but effectively use only High Energy ones
            B, T = xb.shape
            sparsity = active_count / (N_LAYER * B * T)
            print(f"Step {step}: Loss {loss.item():.4f} | DynamicSparsity {sparsity:.2%} | dt={dt:.1f}s")

if __name__ == "__main__":
    run_experiment()