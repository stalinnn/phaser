import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken
import pandas as pd
from tqdm import tqdm
import argparse
from datasets import load_from_disk, load_dataset

# Try to import Mamba
try:
    from mamba_ssm import Mamba
    HAS_MAMBA = True
except ImportError:
    HAS_MAMBA = False

# ==========================================
# 1. Config & Model (Copy-Paste)
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
        
        if gate_scores:
            gate_tensor = torch.stack(gate_scores, dim=0)
        else:
            gate_tensor = None
            
        return logits, gate_tensor

# ==========================================
# 2. Copy Test Logic
# ==========================================
def run_copy_test(model, data_source, device, enc, noise_lengths=[100, 300, 500, 700]):
    results = []
    
    # 1. Select a random snippet (The "Signal")
    # We want something distinctive, not "the cat is". 
    # Let's take a chunk of 32 tokens.
    signal_len = 32
    # Get a random slice from data
    # Simple logic: assume data_source is a list of items
    import random
    
    # Try to find a good signal
    signal_tokens = []
    attempts = 0
    while len(signal_tokens) < signal_len and attempts < 100:
        item = data_source[random.randint(0, len(data_source)-1)]
        if len(item['text']) > 100:
            tokens = enc.encode_ordinary(item['text'])
            if len(tokens) >= signal_len:
                start = random.randint(0, len(tokens) - signal_len)
                signal_tokens = tokens[start : start + signal_len]
                break
        attempts += 1
        
    if not signal_tokens:
        print("Error: Could not find suitable signal text.")
        return None

    signal_tensor = torch.tensor(signal_tokens, dtype=torch.long, device=device)
    print(f"\nSignal: {enc.decode(signal_tokens)[:50]}... ({len(signal_tokens)} tokens)")

    # Separator
    sep_token = [enc.eot_token] # <|endoftext|>
    
    for noise_len in noise_lengths:
        # Construct Noise
        # Random integers
        noise = torch.randint(0, 50257, (noise_len,), dtype=torch.long, device=device).tolist()
        
        # Prompt: Signal + Noise + Signal_Prefix
        # Task: Predict the NEXT token of the second Signal appearance
        # We prompt with: Signal + Noise + Signal[:-1]
        # Target is: Signal[-1]
        
        # To make it "Copying", we give the first few tokens of Signal as prompt
        prefix_len = 4
        prefix = signal_tokens[:prefix_len]
        
        # Context
        context = signal_tokens + noise + sep_token + prefix
        
        input_ids = torch.tensor([context], dtype=torch.long, device=device)
        
        # Target
        target_token = signal_tokens[prefix_len]
        
        # Check if context fits
        if input_ids.size(1) > 1024:
            print(f"Skipping noise {noise_len} (Too long: {input_ids.size(1)})")
            continue

        print(f"\n--- Noise Length: {noise_len} ---")
        
        # --- TGN ---
        with torch.no_grad():
            logits, gates = model(input_ids)
            pred = logits[0, -1, :].argmax().item()
            is_correct = (pred == target_token)
            
            # Check Gate at the "Recall" moment (last token)
            gate_val = gates[:, 0, -1, :].mean().item() # Avg across layers
            
            print(f"[TGN] Correct: {'✅' if is_correct else '❌'} | Gate: {gate_val:.2%} | Pred: {enc.decode([pred])} vs Target: {enc.decode([target_token])}")
            results.append({"model": "TGN", "noise": noise_len, "correct": is_correct, "gate": gate_val})

        # --- Mamba ---
        with torch.no_grad():
            logits, _ = model(input_ids, force_gate=0.0)
            pred = logits[0, -1, :].argmax().item()
            is_correct = (pred == target_token)
            print(f"[Mamba] Correct: {'✅' if is_correct else '❌'} | Pred: {enc.decode([pred])}")
            results.append({"model": "Mamba", "noise": noise_len, "correct": is_correct, "gate": 0.0})

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
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model.eval()
    
    # Load Data (for signals)
    if os.path.exists("./wikitext_103_offline"):
        print("Loading from local: ./wikitext_103_offline")
        ds = load_from_disk("./wikitext_103_offline")['test']
    else:
        print("Loading from HF Mirror...")
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        ds = load_dataset("wikitext", "wikitext-103-v1", split='test')
    
    # Run
    df = run_copy_test(model, ds, device, enc, noise_lengths=[50, 200, 500, 800])
    print(df)
    df.to_csv("copy_test_results.csv")

if __name__ == '__main__':
    main()
