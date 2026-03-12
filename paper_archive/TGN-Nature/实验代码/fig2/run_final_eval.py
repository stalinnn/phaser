import os
import math
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import tiktoken
from datasets import load_from_disk, load_dataset

# Try to import Mamba
try:
    from mamba_ssm import Mamba
    HAS_MAMBA = True
except ImportError:
    print("WARNING: Mamba not found.")
    HAS_MAMBA = False

# ==========================================
# 1. Config (Must match training)
# ==========================================
class TGNConfig:
    def __init__(self, model_size='medium', vocab_size=50304):
        self.vocab_size = vocab_size
        self.dropout = 0.1
        if model_size == 'medium': 
            self.n_layer = 24; self.n_head = 16; self.n_embd = 1024; self.block_size = 1024
        elif model_size == 'large': 
            self.n_layer = 24; self.n_head = 20; self.n_embd = 1536; self.block_size = 1024

# ==========================================
# 2. Components (Copy-Paste)
# ==========================================
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout_p = config.dropout
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0, is_causal=True) # No dropout in eval
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y

class GeometricGate(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, 64), nn.Tanh(), nn.Linear(64, 1), nn.Sigmoid())
    def forward(self, x): return self.net(x)

class TGNBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        if HAS_MAMBA: self.inertial = Mamba(d_model=config.n_embd, d_state=16, d_conv=4, expand=2)
        else: self.inertial = nn.GRU(config.n_embd, config.n_embd, batch_first=True)
        self.attn = CausalSelfAttention(config)
        self.gate = GeometricGate(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.Sequential(nn.Linear(config.n_embd, 4*config.n_embd), nn.GELU(), nn.Linear(4*config.n_embd, config.n_embd), nn.Dropout(config.dropout))

    def forward(self, x, mode='standard', fixed_mask=None):
        x_norm = self.ln1(x)
        if HAS_MAMBA: h_inertial = self.inertial(x_norm)
        else: h_inertial, _ = self.inertial(x_norm)
        if mode == 'standard': g = self.gate(h_inertial)
        else: g = fixed_mask
        attn_out = self.attn(x_norm)
        mixed = (1 - g) * h_inertial + g * attn_out
        x = x + mixed
        x = x + self.mlp(self.ln2(x))
        return x, g

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([TGNBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.token_embedding.weight = self.head.weight

    def forward(self, idx, targets=None, mode='standard', target_sparsity=None):
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.token_embedding(idx) + self.position_embedding(pos)
        x = self.drop(x)
        total_gate = 0.0
        for block in self.blocks:
            mask = None
            if mode == 'random': 
                mask = torch.bernoulli(torch.full((B, T, 1), target_sparsity, device=idx.device))
            x, g = block(x, mode=mode, fixed_mask=mask)
            total_gate += g.mean()
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        gate_mean = total_gate / len(self.blocks)
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss, gate_mean

# ==========================================
# 3. Main Eval Logic
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--data_path', type=str, default='./wikitext_103_offline')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load Model
    config = TGNConfig(model_size='medium')
    model = GPT(config).to(device)
    
    print(f"Loading checkpoint: {args.ckpt_path}")
    state_dict = torch.load(args.ckpt_path, map_location=device)
    # Handle DDP keys if present
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.eval()

    # Load Validation Data
    print("Loading Validation Set...")
    enc = tiktoken.get_encoding("gpt2")
    if os.path.exists(args.data_path):
        ds = load_from_disk(args.data_path)['validation']
    else:
        ds = load_dataset("wikitext", "wikitext-103-v1", split='validation')
    
    data_list = []
    for item in tqdm(ds):
        if len(item['text']) > 0:
            data_list.extend(enc.encode_ordinary(item['text']) + [enc.eot_token])
    data_tensor = torch.tensor(data_list, dtype=torch.long)
    print(f"Validation tokens: {len(data_tensor)}")

    # Loader
    class ChunkedDataset(torch.utils.data.Dataset):
        def __init__(self, data, block_size):
            self.data = data
            self.block_size = block_size
        def __len__(self): return (len(self.data) - 1) // self.block_size
        def __getitem__(self, idx):
            start_idx = idx * self.block_size
            return self.data[start_idx : start_idx + self.block_size], self.data[start_idx+1 : start_idx + self.block_size + 1]

    val_loader = DataLoader(ChunkedDataset(data_tensor, 1024), batch_size=16, shuffle=False)

    # --- 1. Adaptive TGN ---
    print("\n>>> Evaluating Adaptive TGN...")
    total_loss = 0; total_gate = 0; steps = 0
    with torch.no_grad():
        for X, Y in tqdm(val_loader):
            X, Y = X.to(device), Y.to(device)
            _, loss, gate = model(X, Y, mode='standard')
            total_loss += loss.item()
            total_gate += gate.item()
            steps += 1
    
    ppl_adaptive = math.exp(total_loss / steps)
    avg_sparsity = total_gate / steps
    print(f"Adaptive PPL: {ppl_adaptive:.4f}")
    print(f"Avg Sparsity: {avg_sparsity:.2%}")

    # --- 2. Random Baseline ---
    print(f"\n>>> Evaluating Random Baseline (Sparsity={avg_sparsity:.4f})...")
    total_loss = 0
    with torch.no_grad():
        for X, Y in tqdm(val_loader):
            X, Y = X.to(device), Y.to(device)
            _, loss, _ = model(X, Y, mode='random', target_sparsity=avg_sparsity)
            total_loss += loss.item()
    ppl_random = math.exp(total_loss / steps)
    print(f"Random PPL: {ppl_random:.4f}")

    # --- 3. Full Attention Proxy ---
    print("\n>>> Evaluating Full Attention Proxy (Gate=1.0)...")
    total_loss = 0
    with torch.no_grad():
        for X, Y in tqdm(val_loader):
            X, Y = X.to(device), Y.to(device)
            _, loss, _ = model(X, Y, mode='random', target_sparsity=1.0)
            total_loss += loss.item()
    ppl_full = math.exp(total_loss / steps)
    print(f"Full Attn PPL: {ppl_full:.4f}")

    # Save
    with open('final_eval_results.csv', 'w') as f:
        f.write("model,ppl,sparsity\n")
        f.write(f"adaptive,{ppl_adaptive},{avg_sparsity}\n")
        f.write(f"random,{ppl_random},{avg_sparsity}\n")
        f.write(f"full,{ppl_full},1.0\n")
    print("\nResults saved to final_eval_results.csv")

if __name__ == '__main__':
    main()
