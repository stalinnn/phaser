import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import argparse
from tqdm import tqdm
import tiktoken
from datasets import load_from_disk, load_dataset
import pandas as pd
import numpy as np

# Try to import Mamba
try:
    from mamba_ssm import Mamba
    HAS_MAMBA = True
except ImportError:
    HAS_MAMBA = False

# ==========================================
# 1. Config & Model (Same as before)
# ==========================================
class TGNConfig:
    def __init__(self, model_size='medium', vocab_size=50304):
        self.vocab_size = vocab_size
        self.dropout = 0.0
        if model_size == 'medium': 
            self.n_layer = 24; self.n_head = 16; self.n_embd = 1024; self.block_size = 1024
        elif model_size == 'large': 
            self.n_layer = 24; self.n_head = 20; self.n_embd = 1536; self.block_size = 1024

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
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
        self.mlp = nn.Sequential(nn.Linear(config.n_embd, 4*config.n_embd), nn.GELU(), nn.Linear(4*config.n_embd, config.n_embd))

    def forward(self, x, force_gate=None):
        x_norm = self.ln1(x)
        if HAS_MAMBA: h_inertial = self.inertial(x_norm)
        else: h_inertial, _ = self.inertial(x_norm)
        
        if force_gate is not None:
            g = torch.tensor(force_gate, device=x.device, dtype=x.dtype)
        else:
            g = self.gate(h_inertial)
            
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
        self.blocks = nn.ModuleList([TGNBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.token_embedding.weight = self.head.weight

    def forward(self, idx, targets=None, force_gate=None):
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.token_embedding(idx) + self.position_embedding(pos)
        
        gate_scores = []
        for block in self.blocks:
            x, g = block(x, force_gate=force_gate)
            if isinstance(g, torch.Tensor):
                gate_scores.append(g)
                
        x = self.ln_f(x)
        logits = self.head(x)
        
        # Loss per token (no reduction)
        loss = None
        if targets is not None:
            # Targets are already shifted by caller
            # logits: [B, T, V]
            # targets: [B, T]
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction='none')
            loss = loss.view(B, T)
            
        if gate_scores:
            # [Layers, B, T, 1]
            gate_tensor = torch.stack(gate_scores, dim=0).mean(dim=0).squeeze(-1) # [B, T]
        else:
            gate_tensor = None
            
        return loss, gate_tensor

# ==========================================
# 2. Entity Tracking Logic
# ==========================================
def run_entity_analysis(model, enc, device, data_source):
    print(">>> Analyzing Long-Range Entity Tracking...")
    
    long_range_losses_tgn = []
    long_range_gates_tgn = []
    long_range_losses_mamba = []
    
    short_range_losses_tgn = []
    
    # Analyze a subset of validation data
    max_samples = 500 # Increase samples
    ctx_len = 1024
    
    count = 0
    valid_articles = 0
    
    for item in tqdm(data_source):
        if count >= max_samples: break
        if len(item['text']) < 500: continue # Relax length check
        
        tokens = enc.encode_ordinary(item['text'])
        # if len(tokens) < ctx_len + 1: continue # Don't skip, just truncate/slice
        
        # Take a chunk up to ctx_len
        actual_len = min(len(tokens), ctx_len + 1)
        chunk = tokens[:actual_len]
        if len(chunk) < 200: continue # Skip very short chunks
        
        input_ids = torch.tensor([chunk[:-1]], dtype=torch.long, device=device)
        targets = torch.tensor([chunk[1:]], dtype=torch.long, device=device) # Targets align with input's next token
        
        # Find Entity Recurrence
        token_history = {} 
        long_range_mask = torch.zeros(len(chunk)-1, dtype=torch.bool)
        short_range_mask = torch.zeros(len(chunk)-1, dtype=torch.bool)
        
        chunk_ids = chunk[:-1]
        has_long_range = False
        
        for i, tid in enumerate(chunk_ids):
            if tid in token_history:
                dist = i - token_history[tid]
                if dist > 200: # Relax distance to 200
                    long_range_mask[i] = True
                    has_long_range = True
                elif dist < 50:
                    short_range_mask[i] = True
            token_history[tid] = i
            
        if not has_long_range: continue 
        
        valid_articles += 1
        count += 1
        
        # --- TGN Eval ---
        with torch.no_grad():
            loss_tgn, gate_tgn = model(input_ids, targets)
            # loss_tgn: [1, L]
            # gate_tgn: [1, L]
            
            # Extract Long Range Stats
            lr_loss = loss_tgn[0][long_range_mask].cpu().numpy()
            lr_gate = gate_tgn[0][long_range_mask].cpu().numpy()
            
            long_range_losses_tgn.extend(lr_loss)
            long_range_gates_tgn.extend(lr_gate)
            
            # Extract Short Range Stats (Control Group)
            sr_loss = loss_tgn[0][short_range_mask].cpu().numpy()
            short_range_losses_tgn.extend(sr_loss)

        # --- Mamba Eval ---
        with torch.no_grad():
            loss_mamba, _ = model(input_ids, targets, force_gate=0.0)
            lr_loss_m = loss_mamba[0][long_range_mask].cpu().numpy()
            long_range_losses_mamba.extend(lr_loss_m)

    # Statistics
    avg_loss_tgn_lr = np.mean(long_range_losses_tgn)
    avg_gate_tgn_lr = np.mean(long_range_gates_tgn)
    
    avg_loss_mamba_lr = np.mean(long_range_losses_mamba)
    
    avg_loss_tgn_sr = np.mean(short_range_losses_tgn)
    
    print("\n" + "="*40)
    print("REAL-WORLD ENTITY TRACKING RESULTS")
    print("="*40)
    print(f"Long-Range (>500 tokens) Loss:")
    print(f"  Mamba (Inertial): {avg_loss_mamba_lr:.4f} (PPL {math.exp(avg_loss_mamba_lr):.2f})")
    print(f"  TGN   (Adaptive): {avg_loss_tgn_lr:.4f} (PPL {math.exp(avg_loss_tgn_lr):.2f})")
    print(f"  TGN Gate Rate:    {avg_gate_tgn_lr:.2%}")
    print("-" * 40)
    print(f"Short-Range (<50 tokens) Loss:")
    print(f"  TGN (Control):    {avg_loss_tgn_sr:.4f}")
    print("="*40)
    
    # Save for plotting
    res = {
        "mamba_loss": long_range_losses_mamba,
        "tgn_loss": long_range_losses_tgn,
        "tgn_gate": long_range_gates_tgn
    }
    pd.DataFrame(res).to_csv("entity_tracking_results.csv", index=False)

# ==========================================
# 3. Main
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, required=True)
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    enc = tiktoken.get_encoding("gpt2")
    
    config = TGNConfig(model_size='medium')
    model = GPT(config).to(device)
    
    print(f"Loading {args.ckpt_path}...")
    state_dict = torch.load(args.ckpt_path, map_location=device)
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model.eval()
    
    # Load WikiText Validation
    if os.path.exists("./wikitext_103_offline"):
        ds = load_from_disk("./wikitext_103_offline")['validation']
    else:
        ds = load_dataset("wikitext", "wikitext-103-v1", split='validation')
        
    run_entity_analysis(model, enc, device, ds)

if __name__ == '__main__':
    main()
