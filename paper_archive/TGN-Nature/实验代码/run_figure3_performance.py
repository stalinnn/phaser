import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math
import time
import os
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import csv

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# ==========================================
# 1. Config
# ==========================================
class Config:
    def __init__(self):
        self.d_model = 64
        self.n_layer = 2
        self.n_head = 4
        self.vocab_size = 64     # 稍微大一点，避免边缘情况
        self.seq_len = 64        # 长度适中
        self.batch_size = 64
        self.lr = 1e-3
        self.steps = 1000        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.out_dir = "result/figure3_perf"
        os.makedirs(self.out_dir, exist_ok=True)

config = Config()

# ==========================================
# 2. MQAR Dataset
# ==========================================
class MQARDataset(Dataset):
    def __init__(self, size=2000, seq_len=64, num_kv=4):
        self.size = size
        self.seq_len = seq_len
        self.num_kv = num_kv
        self.vocab = config.vocab_size
        
    def __len__(self): return self.size
    
    def __getitem__(self, idx):
        # Ensure data is strictly within [0, vocab-1]
        x = torch.randint(0, self.vocab, (self.seq_len,))
        y = torch.randint(0, self.vocab, (self.seq_len,))
        
        keys = torch.randperm(self.vocab)[:self.num_kv]
        values = torch.randint(0, self.vocab, (self.num_kv,))
        
        # Safe indexing
        valid_range = self.seq_len - 2
        if valid_range < self.num_kv * 2: valid_range = self.num_kv * 2 + 1
        
        avail_pos = torch.randperm(valid_range)[:self.num_kv * 2]
        avail_pos, _ = torch.sort(avail_pos)
        
        for i in range(self.num_kv):
            k_pos = avail_pos[2*i]
            x[k_pos] = keys[i]
            x[k_pos+1] = values[i]
            y[k_pos] = values[i]
            
        q_idx = torch.randint(0, self.num_kv, (1,)).item()
        x[-1] = keys[q_idx]
        y[-1] = values[q_idx]
        
        # Double Check
        assert x.max() < self.vocab, f"X max {x.max()} >= vocab {self.vocab}"
        assert y.max() < self.vocab, f"Y max {y.max()} >= vocab {self.vocab}"
        
        return x, y, torch.tensor(1.0)

# ==========================================
# 3. Models
# ==========================================
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_attn = nn.Linear(config.d_model, 3 * config.d_model)
        self.c_proj = nn.Linear(config.d_model, config.d_model)
        self.n_head = config.n_head
        self.n_embd = config.d_model
        self.register_buffer("bias", torch.tril(torch.ones(config.seq_len, config.seq_len))
                                    .view(1, 1, config.seq_len, config.seq_len))

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)

class MambaProxy(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.emb = nn.Embedding(config.vocab_size, config.d_model)
        self.rnn = nn.GRU(config.d_model, config.d_model, num_layers=config.n_layer, batch_first=True)
        self.head = nn.Linear(config.d_model, config.vocab_size)
    def forward(self, x):
        h = self.emb(x)
        out, _ = self.rnn(h)
        return self.head(out), torch.tensor(0.0)

class TGNModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos = nn.Embedding(config.seq_len, config.d_model)
        
        self.rnn = nn.GRU(config.d_model, config.d_model, num_layers=config.n_layer, batch_first=True)
        self.attn = CausalSelfAttention(config) # Use hand-written attention
        
        self.head = nn.Linear(config.d_model, config.vocab_size)
        self.ln = nn.LayerNorm(config.d_model)
        
    def forward(self, x):
        B, T = x.shape
        # Explicit Positional Encoding
        pos = torch.arange(0, T, dtype=torch.long, device=x.device)
        h = self.emb(x) + self.pos(pos)
        
        r_out, _ = self.rnn(h)
        a_out = self.attn(h) # Pure Transformer path
        
        # Hard Mix: RNN + Attention
        out = r_out + a_out 
        out = self.ln(out)
        return self.head(out), torch.tensor(1.0)

# ==========================================
# 4. Runner
# ==========================================
def run_mqar_battle():
    print("Running MQAR Battle (Baby Mode)...")
    dataset = MQARDataset()
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    models = {
        'Mamba': MambaProxy(config).to(config.device),
        'TGN': TGNModel(config).to(config.device)
    }
    results = {k: [] for k in models.keys()}
    
    for name, model in models.items():
        print(f"Training {name}...")
        optim = torch.optim.AdamW(model.parameters(), lr=config.lr)
        pbar = tqdm(total=config.steps)
        step = 0
        
        while step < config.steps:
            for x, y, _ in loader:
                if step >= config.steps: break
                x, y = x.to(config.device), y.to(config.device)
                
                logits, _ = model(x)
                
                # Check Last Token Prediction
                loss = F.cross_entropy(logits[:, -1, :], y[:, -1])
                
                loss.backward()
                optim.step()
                optim.zero_grad()
                
                pred = logits[:, -1, :].argmax(dim=-1)
                acc = (pred == y[:, -1]).float().mean().item()
                
                results[name].append(acc)
                if step % 10 == 0:
                    pbar.set_description(f"{name} | Acc: {acc:.2%} | Loss: {loss.item():.2f}")
                step += 1
                pbar.update(1)
        pbar.close()
        
    plt.figure(figsize=(8, 6))
    for name, accs in results.items():
        plt.plot(pd.Series(accs).rolling(20).mean(), label=name, linewidth=2)
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.savefig(f"{config.out_dir}/fig3a_mqar.png")

def run_throughput_benchmark():
    pass

if __name__ == "__main__":
    run_mqar_battle()
