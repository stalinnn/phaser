import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from tqdm import tqdm
import csv

# Force Mamba Import (No Fallback)
from mamba_ssm import Mamba
print(">>> Using Real Mamba SSM Kernel.")

# ==========================================
# 1. Config
# ==========================================
class ModelConfig:
    def __init__(self, model_type='tgn', seq_len=1024):
        self.vocab_size = 100 
        self.d_model = 128    
        self.n_layers = 2      
        self.n_heads = 4
        self.d_ff = 512
        self.dropout = 0.0
        self.seq_len = seq_len
        self.model_type = model_type
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.chunk_size = 128 

# ==========================================
# 2. Benchmark Models (Clean)
# ==========================================
class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = nn.MultiheadAttention(config.d_model, config.n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.mlp = nn.Sequential(nn.Linear(config.d_model, config.d_ff), nn.GELU(), nn.Linear(config.d_ff, config.d_model))
        
    def forward(self, x):
        B, L, D = x.shape
        x_norm = self.ln1(x)
        causal_mask = torch.triu(torch.ones(L, L, device=x.device) * float('-inf'), diagonal=1)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, attn_mask=causal_mask, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x, 0.0

class MambaBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.mixer = Mamba(d_model=config.d_model, d_state=16, d_conv=4, expand=2)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.mlp = nn.Sequential(nn.Linear(config.d_model, config.d_ff), nn.GELU(), nn.Linear(config.d_ff, config.d_model))

    def forward(self, x):
        res = x
        x = self.ln1(x)
        x = self.mixer(x)
        x = res + x
        x = x + self.mlp(self.ln2(x))
        return x, 0.0

class ChunkedTGNBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size = config.chunk_size
        self.ln1 = nn.LayerNorm(config.d_model)
        
        # Inertial: Real Mamba
        self.inertial = Mamba(d_model=config.d_model, d_state=16, d_conv=4, expand=2)
            
        self.attn = nn.MultiheadAttention(config.d_model, config.n_heads, batch_first=True)
        self.gate_proj = nn.Linear(config.d_model, 1)
        self.gate_proj.bias.data.fill_(0.0) 
        
        self.ln2 = nn.LayerNorm(config.d_model)
        self.mlp = nn.Sequential(nn.Linear(config.d_model, config.d_ff), nn.GELU(), nn.Linear(config.d_ff, config.d_model))

    def forward(self, x):
        B, L, D = x.shape
        x_norm = self.ln1(x)
        
        # 1. Inertial Path
        h_inertial = self.inertial(x_norm)
            
        # 2. Gating
        gate_score = torch.sigmoid(self.gate_proj(h_inertial)) # [B, L, 1]
        
        # 3. Geometric Path
        causal_mask = torch.triu(torch.ones(L, L, device=x.device) * float('-inf'), diagonal=1)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, attn_mask=causal_mask, need_weights=False)
        
        # 4. Mixing
        out = (1 - gate_score) * h_inertial + gate_score * attn_out
        
        x = x + out
        x = x + self.mlp(self.ln2(x))
        return x, gate_score.mean()

class UniversalModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.seq_len, config.d_model)
        
        self.blocks = nn.ModuleList()
        for _ in range(config.n_layers):
            if config.model_type == 'transformer':
                self.blocks.append(TransformerBlock(config))
            elif config.model_type == 'mamba':
                self.blocks.append(MambaBlock(config))
            elif config.model_type == 'tgn':
                self.blocks.append(ChunkedTGNBlock(config))
        self.head = nn.Linear(config.d_model, config.vocab_size)

    def forward(self, x):
        B, L = x.shape
        pos = torch.arange(0, L, device=x.device).unsqueeze(0)
        h = self.emb(x) + self.pos_emb(pos)
        
        total_g = 0
        count = 0
        for block in self.blocks:
            h, g = block(h)
            if isinstance(g, torch.Tensor):
                total_g += g
                count += 1
        avg_g = total_g / count if count > 0 else 0
        return self.head(h), avg_g

# ==========================================
# 3. MQAR Logic
# ==========================================
def generate_mqar_batch(batch_size, seq_len, vocab_size, num_pairs=8):
    half = vocab_size // 2
    X = torch.randint(0, half, (batch_size, seq_len))
    Y = torch.zeros_like(X)
    
    for i in range(batch_size):
        pairs = []
        keys = np.random.choice(range(half, vocab_size), num_pairs, replace=False)
        for k in keys:
            v = np.random.randint(half, vocab_size)
            pairs.append((k, v))
        
        valid_slots = np.array(range(seq_len // 2 - 1))
        np.random.shuffle(valid_slots)
        slot_idx = 0
        used_mask = np.zeros(seq_len // 2, dtype=bool)
        
        for k, v in pairs:
            while slot_idx < len(valid_slots):
                p = valid_slots[slot_idx]
                slot_idx += 1
                if not used_mask[p] and not used_mask[p+1]:
                    X[i, p] = k
                    X[i, p+1] = v
                    used_mask[p] = True
                    used_mask[p+1] = True
                    break
            
        q_idx = np.random.randint(0, num_pairs)
        q_k, q_v = pairs[q_idx]
        
        X[i, -2] = q_k
        X[i, -1] = 0 
        Y[i, -1] = q_v 
        
    return X, Y

# ==========================================
# 4. Main Run
# ==========================================
def run_mqar_experiment():
    print("\n>>> Running Figure 3a: MQAR Capability Benchmark (Clean)...")
    os.makedirs('result/fig3', exist_ok=True)
    
    configs = [ModelConfig('tgn'), ModelConfig('transformer'), ModelConfig('mamba')]
    
    with open('result/fig3/mqar_training_curves.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['model', 'step', 'loss', 'accuracy', 'gate_rate'])
        
        for conf in configs:
            print(f"\nTraining {conf.model_type.upper()} on MQAR...")
            model = UniversalModel(conf).to(conf.device)
            optim = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)
            
            model.train()
            pbar = tqdm(range(5000)) 
            
            for step in pbar:
                if step < 1000: sl, pairs = 32, 1
                elif step < 3000: sl, pairs = 64, 2
                else: sl, pairs = 128, 4
                
                X, Y = generate_mqar_batch(32, sl, 100, num_pairs=pairs)
                X, Y = X.to(conf.device), Y.to(conf.device)
                
                logits, avg_gate = model(X)
                loss = F.cross_entropy(logits[:, -2, :], Y[:, -1])
                
                loss.backward()
                optim.step()
                optim.zero_grad()
                
                if step % 50 == 0:
                    with torch.no_grad():
                        pred = logits[:, -2, :].argmax(dim=-1)
                        acc = (pred == Y[:, -1]).float().mean().item()
                        gate_val = avg_gate.item() if isinstance(avg_gate, torch.Tensor) else 0
                        writer.writerow([conf.model_type, step, loss.item(), acc, gate_val])
                        pbar.set_postfix({"L": f"{loss.item():.3f}", "A": f"{acc:.1%}", "G": f"{gate_val:.1%}"})
            
            print(f"--> {conf.model_type} Final Accuracy: {acc:.2%}")

if __name__ == '__main__':
    run_mqar_experiment()
