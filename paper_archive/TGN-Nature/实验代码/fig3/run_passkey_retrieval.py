import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import argparse

# Try to import Mamba
try:
    from mamba_ssm import Mamba
    HAS_MAMBA = True
except ImportError:
    HAS_MAMBA = False

# ==========================================
# 1. Config & Model Definition (Must match training)
# ==========================================
class TGNConfig:
    def __init__(self, model_size='medium', vocab_size=50304):
        self.vocab_size = vocab_size
        self.dropout = 0.0 # No dropout for eval
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

    def forward(self, idx, force_gate=None):
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
        
        # Stack gates [Layers, B, T, 1]
        if gate_scores:
            gate_tensor = torch.stack(gate_scores, dim=0)
        else:
            gate_tensor = None
            
        return logits, gate_tensor

# ==========================================
# 2. Passkey Retrieval Logic
# ==========================================
def run_passkey_test(model, enc, device, haystack_len=1000, depths=[0.1, 0.5, 0.9]):
    """
    depths: where to hide the needle (0.0 = start, 1.0 = end)
    """
    results = []
    
    # Needle
    passkey = 90210
    needle_text = f" The secret passkey is {passkey}. "
    needle_tokens = enc.encode(needle_text)
    
    # Query
    query_text = " What is the secret passkey? The passkey is"
    query_tokens = enc.encode(query_text)
    
    # Answer
    answer_token = enc.encode(f" {passkey}")[0] # Single token target
    
    # One-shot Demonstration (Teaching the model the task format)
    demo_text = "The secret code is 12345. What is the secret code? The code is 12345.\n"
    demo_tokens = enc.encode(demo_text)
    
    print(f"\n>>> Running Passkey Retrieval (Len={haystack_len})...")
    
    for depth in depths:
        # Construct Context
        # Fill with garbage (WikiText-like)
        garbage_len = haystack_len - len(needle_tokens) - len(query_tokens) - len(demo_tokens)
        garbage = torch.randint(0, 50257, (garbage_len,), dtype=torch.long).tolist()
        
        insert_idx = int(garbage_len * depth)
        
        # Context: Demo + Garbage_Part1 + Needle + Garbage_Part2 + Query
        context = demo_tokens + garbage[:insert_idx] + needle_tokens + garbage[insert_idx:] + query_tokens
        
        input_ids = torch.tensor([context], dtype=torch.long, device=device)
        
        # --- Mode 1: TGN (Adaptive) ---
        with torch.no_grad():
            logits, gates = model(input_ids)
            pred = logits[0, -1, :].argmax().item()
            is_correct = (pred == answer_token)
            
            # Record Gate Activation at Needle position
            # Needle is at [insert_idx : insert_idx+len(needle)]
            needle_gate_avg = gates[:, 0, insert_idx:insert_idx+len(needle_tokens), :].mean().item()
            
            results.append({
                "model": "TGN",
                "depth": depth,
                "correct": is_correct,
                "gate_activation": needle_gate_avg
            })
            print(f"[TGN] Depth {depth:.1f}: {'✅' if is_correct else '❌'} (Gate: {needle_gate_avg:.2%})")

        # --- Mode 2: Mamba (Force Gate=0) ---
        with torch.no_grad():
            logits, _ = model(input_ids, force_gate=0.0)
            pred = logits[0, -1, :].argmax().item()
            is_correct = (pred == answer_token)
            results.append({"model": "Mamba", "depth": depth, "correct": is_correct, "gate_activation": 0.0})
            print(f"[Mamba] Depth {depth:.1f}: {'✅' if is_correct else '❌'}")

    return pd.DataFrame(results)

# ==========================================
# 3. Main
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, required=True)
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    enc = tiktoken.get_encoding("gpt2")
    
    # Load Model
    config = TGNConfig(model_size='medium')
    model = GPT(config).to(device)
    
    print(f"Loading {args.ckpt_path}...")
    state_dict = torch.load(args.ckpt_path, map_location=device)
    # Fix DDP keys
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model.eval()
    
    # Run Test
    # Test at different lengths if possible (start small)
    df = run_passkey_test(model, enc, device, haystack_len=900, depths=[0.1, 0.3, 0.5, 0.7, 0.9])
    
    # Save
    df.to_csv("passkey_results.csv", index=False)
    print("\nResults saved to passkey_results.csv")

if __name__ == '__main__':
    main()
