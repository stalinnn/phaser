"""
TGN IQ Test: Gate Rate vs Information Density
---------------------------------------------
Goal: Prove that TGN's Gate Rate is determined by Task Complexity, not just model bias.
Hypothesis:
    - Simple Tokens (Local dependency) -> Gate ≈ 0
    - Complex Tokens (Long-range dependency) -> Gate ≈ 1
    - Overall Gate Rate ≈ Fraction of Complex Tokens (e.g., 20%)

Result: A bimodal distribution of gate values proves "On-Demand Allocation".
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ==========================================
# Configuration
# ==========================================
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
TRAIN_ITERS = 2000
BLOCK_SIZE = 256

# Model Config (Medium-ish capacity to ensure it CAN learn)
N_EMBD = 256
N_HEAD = 4
N_LAYER = 4
DROPOUT = 0.0

# Task Config
SIMPLE_RATIO = 0.8  # 80% simple tasks
VOCAB_SIZE = 100

# ==========================================
# Hybrid Complexity Dataset
# ==========================================
class HybridComplexityDataset(Dataset):
    def __init__(self, size, seq_len):
        self.size = size
        self.seq_len = seq_len
        
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        # We construct a sequence that mixes Simple and Complex dependencies
        # But to make analysis easier, we create samples that are EITHER Simple OR Complex dominant
        # and rely on the batch mixing.
        
        # Actually, let's mix WITHIN sequence to see dynamic gating.
        # Format: [Context] ... [Trigger] -> [Target]
        
        # 1. Simple Pattern (Local Copy): "A B A B" -> next is A
        # 2. Complex Pattern (Long Retrieval): "K ... (noise) ... K" -> next is V associated with K
        
        is_complex_step = np.random.rand() > SIMPLE_RATIO
        
        seq = np.random.randint(1, VOCAB_SIZE, size=self.seq_len).tolist()
        mask_complex = [0] * self.seq_len # 1 if this position requires attention
        
        # Inject patterns
        # We iterate and inject dependencies
        for i in range(10, self.seq_len):
            if np.random.rand() > SIMPLE_RATIO:
                # Inject COMPLEX dependency (Recall token from long ago)
                target_idx = i - np.random.randint(5, i-1) # Long range
                seq[i] = seq[target_idx] # The answer is a copy of previous
                mask_complex[i] = 1 # Mark as complex
            else:
                # Inject SIMPLE dependency (Local repeat)
                # e.g. simply repeat immediate previous or random (RNN handles noise/local well)
                pass 
                
        x = torch.tensor(seq[:-1], dtype=torch.long)
        y = torch.tensor(seq[1:], dtype=torch.long)
        mask = torch.tensor(mask_complex[1:], dtype=torch.float) # Align with targets
        
        return x, y, mask

# ==========================================
# TGN Model (Standard)
# ==========================================
class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.proj = nn.Linear(n_embd, n_embd)

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
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)

class TGNBlock(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.ln = nn.LayerNorm(n_embd)
        self.rnn = nn.GRU(n_embd, n_embd, batch_first=True)
        self.gate = nn.Sequential(
            nn.Linear(n_embd, 16), nn.Tanh(),
            nn.Linear(16, 1), nn.Sigmoid()
        )
        nn.init.constant_(self.gate[2].bias, 0.0) # Neutral init
        self.attn = CausalSelfAttention(n_embd, n_head)
        self.ffn = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd), nn.GELU(),
            nn.Linear(4*n_embd, n_embd)
        )

    def forward(self, x):
        residual = x
        x_norm = self.ln(x)
        h_rnn, _ = self.rnn(x_norm)
        h_attn = self.attn(x_norm)
        g = self.gate(h_rnn)
        h_mixed = (1 - g) * h_rnn + g * h_attn
        return residual + h_mixed + self.ffn(self.ln(residual + h_mixed)), g

class TGN(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(VOCAB_SIZE, N_EMBD)
        self.pos_emb = nn.Embedding(BLOCK_SIZE, N_EMBD)
        self.blocks = nn.ModuleList([TGNBlock(N_EMBD, N_HEAD) for _ in range(N_LAYER)])
        self.ln_f = nn.LayerNorm(N_EMBD)
        self.head = nn.Linear(N_EMBD, VOCAB_SIZE)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=DEVICE))
        
        gates = []
        for block in self.blocks:
            x, g = block(x)
            gates.append(g)
        
        logits = self.head(self.ln_f(x))
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), targets.view(-1))
            # Penalty
            g_mean = torch.stack([g.mean() for g in gates]).mean()
            loss += 0.05 * g_mean # Moderate penalty
            
        return logits, loss, gates

# ==========================================
# Run Experiment
# ==========================================
def main():
    print(">>> Starting TGN 'IQ Test' (Gate Distribution Analysis)...")
    
    # 1. Train
    dataset = HybridComplexityDataset(10000, BLOCK_SIZE)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    model = TGN().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    model.train()
    for i, (x, y, _) in enumerate(loader):
        if i >= TRAIN_ITERS: break
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        _, loss, _ = model(x, y)
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            print(f"Iter {i}: Loss {loss.item():.4f}")

    # 2. Analyze Gate Distribution
    print("\n>>> Analyzing Gate Distribution on Test Set...")
    model.eval()
    all_gates_simple = []
    all_gates_complex = []
    
    test_loader = DataLoader(HybridComplexityDataset(500, BLOCK_SIZE), batch_size=BATCH_SIZE)
    
    with torch.no_grad():
        for x, y, mask in test_loader:
            x, mask = x.to(DEVICE), mask.to(DEVICE)
            _, _, gates = model(x)
            
            # Average gate across layers for each token
            # gates shape: [Layer, B, T, 1] -> [B, T]
            g_avg = torch.stack(gates).mean(dim=0).squeeze(-1)
            
            # Split by complexity mask
            simple_gates = g_avg[mask == 0].cpu().numpy()
            complex_gates = g_avg[mask == 1].cpu().numpy()
            
            all_gates_simple.extend(simple_gates)
            all_gates_complex.extend(complex_gates)
    
    # 3. Stats & Plot
    mean_simple = np.mean(all_gates_simple)
    mean_complex = np.mean(all_gates_complex)
    
    print(f"\nRESULTS:")
    print(f"Simple Token Gate Mean:  {mean_simple:.4f} (Expected: Low)")
    print(f"Complex Token Gate Mean: {mean_complex:.4f} (Expected: High)")
    print(f"Separation Ratio:        {mean_complex / (mean_simple + 1e-6):.2f}x")
    
    if mean_complex > mean_simple * 2:
        print("\n[SUCCESS] TGN successfully distinguishes between simple and complex tokens!")
        print("This proves that gate rate is driven by Information Density.")
    else:
        print("\n[FAIL] TGN failed to separate tasks clearly.")

    # Save distribution plot
    plt.figure(figsize=(10, 6))
    plt.hist(all_gates_simple, bins=50, alpha=0.5, label='Simple Tokens (Local)', density=True)
    plt.hist(all_gates_complex, bins=50, alpha=0.5, label='Complex Tokens (Long-range)', density=True)
    plt.xlabel('Gate Activation Rate')
    plt.ylabel('Density')
    plt.title('TGN Gate Distribution: Simple vs Complex Tasks')
    plt.legend()
    plt.savefig('tgn_iq_test_distribution.png')
    print("Plot saved to tgn_iq_test_distribution.png")

if __name__ == "__main__":
    main()