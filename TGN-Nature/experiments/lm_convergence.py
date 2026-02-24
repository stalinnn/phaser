import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
import argparse
import time
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

# --- Configuration & Hyperparameters ---
BLOCK_SIZE = 128   # Spatial extent of the model for its context
BATCH_SIZE = 64
LEARNING_RATE = 3e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EMBED_DIM = 128
NUM_HEADS = 4
NUM_LAYERS = 2     # Number of TGN layers (or Transformer layers)
DROPOUT = 0.2      # Increased dropout
EPOCHS = 30        # Reduced epochs to catch early convergence
EVAL_INTERVAL = 5
SPARSITY_LAMBDA = 0.05 # Penalty for opening the gate

# --- Dataset: Tiny Shakespeare ---
class CharDataset(Dataset):
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size
        
        chars = sorted(list(set(data)))
        self.vocab_size = len(chars)
        self.stoi = { ch:i for i,ch in enumerate(chars) }
        self.itos = { i:ch for i,ch in enumerate(chars) }
    
    def __len__(self):
        return len(self.data) - self.block_size
    
    def __getitem__(self, idx):
        chunk = self.data[idx:idx + self.block_size + 1]
        dix = [self.stoi[c] for c in chunk]
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y

def load_data(path, block_size, split_ratio=0.9):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()
    except FileNotFoundError:
        # Fallback for environment without the file
        print(f"Warning: {path} not found. Using dummy data.")
        text = "To be, or not to be, that is the question. " * 1000
        
    dataset = CharDataset(text, block_size)
    n = len(text)
    train_size = int(n * split_ratio)
    
    train_data = text[:train_size]
    val_data = text[train_size:]
    
    train_dataset = CharDataset(train_data, block_size)
    val_dataset = CharDataset(val_data, block_size)
    
    return train_dataset, val_dataset, dataset.vocab_size

# --- Components ---

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                    .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y

class GeometricGate(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x, None # No gate

class TGNBlock(nn.Module):
    """
    Thermodynamic Gated Block:
    Hybrid of RNN (Inertia) and Attention (Geometry).
    """
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        
        # Inertia Path (Recurrent)
        self.rnn = nn.GRU(config.n_embd, config.n_embd, batch_first=True)
        
        # Geometric Path (Attention)
        self.attn = CausalSelfAttention(config)
        
        # Maxwell's Demon (Gate)
        self.gate = GeometricGate(config.n_embd)
        
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        # x: (B, T, C)
        
        # 1. Inertia (RNN)
        rnn_out, _ = self.rnn(self.ln1(x))
        
        # 2. Compute Gate
        g = self.gate(rnn_out) # (B, T, 1)
        
        # 3. Geometry (Attention)
        attn_out = self.attn(self.ln1(x))
        
        # 4. Gated Combination
        mixed = (1 - g) * rnn_out + g * attn_out
        
        # Residual connection
        x = x + mixed
        
        # MLP
        x = x + self.mlp(self.ln2(x))
        
        return x, g

class LanguageModel(nn.Module):
    def __init__(self, config, model_type='transformer'):
        super().__init__()
        self.config = config
        self.model_type = model_type
        
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        
        self.blocks = nn.ModuleList()
        for _ in range(config.n_layer):
            if model_type == 'transformer':
                self.blocks.append(TransformerBlock(config))
            elif model_type == 'tgn':
                self.blocks.append(TGNBlock(config))
                
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Weight tying
        self.token_embedding.weight = self.head.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        if T > self.config.block_size:
            idx = idx[:, -self.config.block_size:]
            T = idx.size(1)
            
        tok_emb = self.token_embedding(idx) # (B, T, C)
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device)) # (T, C)
        x = self.drop(tok_emb + pos_emb)
        
        total_gate_activation = 0.0
        gate_count = 0
        
        for block in self.blocks:
            x, g = block(x)
            if g is not None:
                total_gate_activation += g.mean()
                gate_count += 1
                
        x = self.ln_f(x)
        logits = self.head(x)
        
        loss = None
        if targets is not None:
            # If we trimmed input, we must trim targets too
            if targets.size(1) > T:
                targets = targets[:, -T:]
                
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            
            # Add sparsity penalty for TGN
            if gate_count > 0:
                avg_gate = total_gate_activation / gate_count
                loss += SPARSITY_LAMBDA * avg_gate
                
        return logits, loss, (total_gate_activation/gate_count if gate_count > 0 else 0)

# --- Training Utils ---

class Config:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.block_size = BLOCK_SIZE
        self.n_embd = EMBED_DIM
        self.n_head = NUM_HEADS
        self.n_layer = NUM_LAYERS
        self.dropout = DROPOUT

@torch.no_grad()
def estimate_loss(model, eval_iters, loader, device):
    out = {}
    model.eval()
    losses = []
    for i, (X, Y) in enumerate(loader):
        if i >= eval_iters: break
        X, Y = X.to(device), Y.to(device)
        _, loss, _ = model(X, Y)
        losses.append(loss.item())
    model.train()
    return np.mean(losses)

def train(model_type='transformer'):
    torch.manual_seed(1337)
    
    # Load Data
    data_path = Path(__file__).parent.parent / "data/tinyshakespeare_input.txt"
    train_dataset, val_dataset, vocab_size = load_data(str(data_path), BLOCK_SIZE)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    config = Config(vocab_size)
    model = LanguageModel(config, model_type=model_type).to(DEVICE)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)
    
    print(f"--- Training {model_type.upper()} ---")
    print(f"Params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    start_time = time.time()
    history = {'train_loss': [], 'val_loss': [], 'gate': []}
    
    for epoch in range(EPOCHS):
        model.train()
        acc_loss = 0
        acc_gate = 0
        
        # Training loop
        max_steps = 100 
        for i, (xb, yb) in enumerate(train_loader):
            if i >= max_steps: break
            
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits, loss, mean_gate = model(xb, yb)
            
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            
            acc_loss += loss.item()
            if isinstance(mean_gate, torch.Tensor):
                acc_gate += mean_gate.item()
            else:
                acc_gate += mean_gate
        
        # Validation
        val_loss = estimate_loss(model, 20, val_loader, DEVICE)
        train_loss = acc_loss / max_steps
        avg_gate = acc_gate / max_steps
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['gate'].append(avg_gate)
        
        if epoch % EVAL_INTERVAL == 0 or epoch == EPOCHS - 1:
            print(f"Epoch {epoch}: Train {train_loss:.4f}, Val {val_loss:.4f}, Gate {avg_gate:.4f}")
            
    dt = time.time() - start_time
    print(f"Finished {model_type} in {dt:.2f}s")
    return history

if __name__ == '__main__':
    print("Starting Comparative Experiment: Transformer vs TGN")
    
    hist_trans = train('transformer')
    hist_tgn = train('tgn')
    
    # Plotting
    out_dir = Path(__file__).parent.parent / "figures"
    out_dir.mkdir(exist_ok=True)
    
    plt.style.use('seaborn-v0_8-paper')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    epochs = range(len(hist_trans['train_loss']))
    
    # Loss Plot (Train)
    ax1.plot(epochs, hist_trans['train_loss'], label='Transformer (Train)', linestyle='--', color='gray')
    ax1.plot(epochs, hist_tgn['train_loss'], label='TGN (Train)', color='#e74c3c', linewidth=2)
    
    ax1.set_title("Language Modeling Convergence")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Cross Entropy Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Gate Plot
    ax2.plot(epochs, hist_tgn['gate'], color='#2ecc71', label='Attention Gate Rate', linewidth=2)
    ax2.set_title("Thermodynamic Sparsity")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Gate Open Prob (0-1)")
    ax2.set_ylim(0, 0.5)
    ax2.axhline(y=0.05, color='gray', linestyle='--', label='Target Sparsity (~5%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_dir / "lm_efficiency.png", dpi=300)
    print(f"Saved plot to {out_dir / 'lm_efficiency.png'}")
