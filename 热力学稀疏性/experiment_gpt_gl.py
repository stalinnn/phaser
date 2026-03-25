"""
Thermodynamic GPT Experiment (Enhanced)
Applying Strong Ginzburg-Landau Regularization to a mini-GPT model on Shakespeare data.
Goal: Observe emergence of structured sparsity in MLP layers and Attention maps.
"""

import math
import inspect
import torch
import torch.nn as nn
from torch.nn import functional as F
import time
import os
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

# Set random seed for reproducibility
torch.manual_seed(42)

# ==========================================
# 1. Causal Self-Attention (With GL Hooks)
# ==========================================

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        
        # Flash Attention mask (not used here for explicit map access)
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                    .view(1, 1, config.block_size, config.block_size))

    def forward(self, x, return_attn_map=False):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att_weights = F.softmax(att, dim=-1)
        att = self.attn_dropout(att_weights)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        y = self.resid_dropout(self.c_proj(y))
        
        if return_attn_map:
            return y, att_weights
        return y

# ==========================================
# 2. MLP (With GL Hooks)
# ==========================================

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.resid_pdrop)
        self.act     = nn.GELU()

    def forward(self, x, return_activations=False):
        h = self.c_fc(x)
        h_act = self.act(h)
        h = self.c_proj(h_act)
        h = self.dropout(h)
        
        if return_activations:
            return h, h_act
        return h

# ==========================================
# 3. GPT Model
# ==========================================

class GPTConfig:
    def __init__(self, vocab_size, block_size, n_layer=4, n_head=4, n_embd=128, 
                 embd_pdrop=0.1, resid_pdrop=0.1, attn_pdrop=0.1):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.embd_pdrop = embd_pdrop
        self.resid_pdrop = resid_pdrop
        self.attn_pdrop = attn_pdrop

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.block_size = config.block_size
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.embd_pdrop),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # init all weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, idx, targets=None, return_internals=False):
        device = idx.device
        b, t = idx.size()
        
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (1, t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        
        internal_acts = []
        internal_attns = []
        
        for block in self.transformer.h:
            if return_internals:
                x, attn, mlp_act = block(x, return_internals=True)
                internal_attns.append(attn)
                internal_acts.append(mlp_act)
            else:
                x = block(x)
                
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        if return_internals:
            return logits, loss, internal_acts, internal_attns
        return logits, loss

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x, return_internals=False):
        if return_internals:
            # Attention Path
            attn_out, attn_map = self.attn(self.ln1(x), return_attn_map=True)
            x = x + attn_out
            
            # MLP Path
            mlp_out, mlp_act = self.mlp(self.ln2(x), return_activations=True)
            x = x + mlp_out
            return x, attn_map, mlp_act
        else:
            x = x + self.attn(self.ln1(x))
            x = x + self.mlp(self.ln2(x))
            return x

# ==========================================
# 4. Thermodynamic Regularization
# ==========================================

def calculate_gl_loss(activations, attns, lambda_mlp=0.01, lambda_attn=0.01):
    """
    1. MLP GL Loss: Encourage 1D domain structure in neuron activations
    2. Attention GL Loss: Encourage block-diagonal structure (Local Attention)
    """
    loss = 0
    
    # 1. MLP Loss
    for act in activations:
        # act: [B, T, 4*n_embd]
        # Calculate spatial gradient along neuron dimension
        # Assuming neurons are topologically 1D
        diff = act[:, :, 1:] - act[:, :, :-1]
        loss += lambda_mlp * torch.mean(diff ** 2)
        
    # 2. Attention Loss (Optional)
    # Encouraging "connected" attention patterns (domains in time)
    # attn: [B, n_head, T, T]
    for attn in attns:
        # Spatial coherence in attention map
        diff_row = attn[:, :, 1:, :] - attn[:, :, :-1, :]
        diff_col = attn[:, :, :, 1:] - attn[:, :, :, :-1]
        loss += lambda_attn * (torch.mean(diff_row**2) + torch.mean(diff_col**2))
        
    return loss

# ==========================================
# 5. Data & Training Loop
# ==========================================

class CharDataset(Dataset):
    def __init__(self, data, block_size):
        chars = sorted(list(set(data)))
        data_size, vocab_size = len(data), len(chars)
        self.stoi = { ch:i for i,ch in enumerate(chars) }
        self.itos = { i:ch for i,ch in enumerate(chars) }
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.data = data
    
    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        chunk = self.data[idx:idx + self.block_size + 1]
        dix = [self.stoi[s] for s in chunk]
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y

def train_model(model_name, use_gl=False):
    # Prepare Data
    if not os.path.exists('input.txt'):
        print("Downloading Shakespeare dataset...")
        os.system('curl https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt -o input.txt')
        
    with open('input.txt', 'r') as f:
        text = f.read()
    
    block_size = 64
    dataset = CharDataset(text, block_size)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0)
    
    # Init Model (DISABLE DROPOUT for clearer structure visualization)
    config = GPTConfig(
        vocab_size=dataset.vocab_size,
        block_size=dataset.block_size,
        n_layer=4, n_head=4, n_embd=128,
        embd_pdrop=0.0, resid_pdrop=0.0, attn_pdrop=0.0 # No Dropout
    )
    model = GPT(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    
    print(f"\nTraining {model_name} (GL={use_gl}, No Dropout)...")
    
    losses = []
    model.train()
    
    # Train for more steps to allow phase transition
    max_steps = 1000 
    for i, (x, y) in enumerate(dataloader):
        if i >= max_steps: break
        
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        
        if use_gl:
            logits, ce_loss, acts, attns = model(x, targets=y, return_internals=True)
            # Increased lambda to force strong domain formation
            gl_loss = calculate_gl_loss(acts, attns, lambda_mlp=1.0, lambda_attn=0.1)
            loss = ce_loss + gl_loss
        else:
            logits, loss = model(x, targets=y)
            
        loss.backward()
        optimizer.step()
        
        if i % 100 == 0:
            print(f"Step {i}: Loss {loss.item():.4f}")
            losses.append(loss.item())
            
    return model, losses

def visualize_activations(model, device, title, filename):
    model.eval()
    # Create a dummy input
    dummy_x = torch.randint(0, 10, (1, 64)).to(device)
    
    with torch.no_grad():
        _, _, acts, attns = model(dummy_x, return_internals=True)
        
    # Visualizing Layer 2 MLP Activation
    layer_idx = 2
    mlp_act = acts[layer_idx][0, 0, :].cpu().numpy() # [First Batch, First Token, All Neurons]
    
    # Sort activations to see if they form clusters (optional check)
    # mlp_act = np.sort(mlp_act) 
    
    plt.figure(figsize=(12, 5))
    plt.bar(range(len(mlp_act)), mlp_act, color='green' if 'Thermo' in title else 'gray', width=1.0)
    plt.title(f'{title} - MLP Activation (Layer {layer_idx})')
    plt.xlabel('Neuron Index')
    plt.ylabel('Activation')
    plt.xlim(0, len(mlp_act))
    plt.savefig(filename)
    print(f"Saved {filename}")

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # 1. Train Baseline
    model_base, loss_base = train_model("Baseline", use_gl=False)
    visualize_activations(model_base, device, "Baseline GPT", "figures/gpt_act_baseline.png")
    
    # 2. Train Thermo
    model_thermo, loss_thermo = train_model("Thermo GPT", use_gl=True)
    visualize_activations(model_thermo, device, "Thermo GPT", "figures/gpt_act_thermo.png")
