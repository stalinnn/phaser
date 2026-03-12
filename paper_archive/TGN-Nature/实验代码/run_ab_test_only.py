import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

# --- Config & Model Definitions (Copy-Paste for consistency) ---
# Try to import Mamba
try:
    from mamba_ssm import Mamba
    HAS_MAMBA = True
except ImportError:
    HAS_MAMBA = False

class TGNConfig:
    def __init__(self, model_size='medium', vocab_size=50304):
        self.vocab_size = vocab_size
        self.dropout = 0.1
        # Medium 350M
        self.n_layer = 24
        self.n_head = 16
        self.n_embd = 1024
        self.block_size = 1024

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
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                    .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        # Flash Attn
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=0, is_causal=True
        )
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
            if mode == 'random': mask = torch.bernoulli(torch.full((B, T, 1), target_sparsity, device=idx.device))
            x, g = block(x, mode=mode, fixed_mask=mask)
            total_gate += g.mean()
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        gate_mean = total_gate / len(self.blocks)
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss, gate_mean

def main():
    device = 'cuda'
    config = TGNConfig(model_size='medium')
    model = GPT(config).to(device)
    
    # Load Checkpoint
    ckpt_path = "result_fig2/ckpt_9000_hotfix.pt" # Fallback to 9000
    print(f"Loading checkpoint: {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location=device)
    # Handle DDP prefix "module."
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."): new_state_dict[k[7:]] = v
        else: new_state_dict[k] = v
    model.load_state_dict(new_state_dict)
    model.eval()
    
    # Load Data (WikiText-103)
    print("Loading WikiText-103 from local...")
    enc = tiktoken.get_encoding("gpt2")
    try:
        from datasets import load_from_disk
        # Assuming folder is in current dir
        dataset = load_from_disk("./wikitext_103_offline")['train']
    except Exception as e:
        print(f"Error loading local data: {e}")
        # Try loading from huggingface cache if local folder missing
        dataset = load_dataset("wikitext", "wikitext-103-v1", split='train')

    # Construct a real batch
    print("Constructing batch...")
    batch_size = 4
    X_list, Y_list = [], []
    
    # Grab first 4 long sequences
    count = 0
    # Strategy: Concatenate everything then slice
    full_text = ""
    for i in range(len(dataset)):
        full_text += dataset[i]['text'] + " "
        if len(full_text) > 100000: break # Enough
        
    full_ids = enc.encode_ordinary(full_text)
    
    for i in range(batch_size):
        start = i * 1024
        end = start + 1024
        if end + 1 < len(full_ids):
            X_list.append(torch.tensor(full_ids[start:end], dtype=torch.long))
            Y_list.append(torch.tensor(full_ids[start+1:end+1], dtype=torch.long))
            count += 1
            
    X = torch.stack(X_list).to(device)
    Y = torch.stack(Y_list).to(device)
    print(f"Batch shape: {X.shape}")

    # A/B Test
    print("\n--- Running A/B Test ---")
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
        # A. Adaptive
        _, loss_a, gate_a = model(X, Y, mode='standard')
        ppl_a = math.exp(loss_a.item())
        sparsity = gate_a.item()
        
        # B. Random (Same Sparsity)
        ppl_b_list = []
        for _ in range(5): # Average over 5 runs for stability
            _, loss_b, _ = model(X, Y, mode='random', target_sparsity=sparsity)
            ppl_b_list.append(math.exp(loss_b.item()))
        ppl_b = sum(ppl_b_list) / len(ppl_b_list)
        
    print(f"Adaptive TGN (Sparsity {sparsity:.2%}): PPL = {ppl_a:.2f}")
    print(f"Random Baseline (Sparsity {sparsity:.2%}): PPL = {ppl_b:.2f}")
    print(f"Performance Gap: +{(ppl_b - ppl_a)/ppl_a*100:.1f}%")

if __name__ == '__main__':
    main()
