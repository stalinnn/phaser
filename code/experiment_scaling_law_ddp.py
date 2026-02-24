"""
Large-Scale DDP Experiment: Baseline Transformer vs TGN (Paper Section 6.3)
-------------------------------------------------------------------------
Goal: Robust head-to-head comparison on Large models (454M) using multi-GPU DDP.
This script is designed to utilize 6x24GB GPUs to overcome optimization difficulties
observed in single-GPU runs (where Large models stuck at loss ~3.3).

Features:
- DistributedDataParallel (DDP) for 6x speedup and global batch size scaling.
- TGN with Residual Fix + Bias Init + Auto-Regulation.
- Baseline Transformer with identical config for fair comparison.

Usage:
    torchrun --nproc_per_node=6 code/experiment_scaling_law_ddp.py
"""

import os
import time
import math
import csv
import functools
from pathlib import Path
from typing import Tuple, List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import requests
import numpy as np

# ==========================================
# Configuration (Optimized for 6x24GB GPUs)
# ==========================================
SEED = 42
BLOCK_SIZE = 256
MAX_ITERS = 2000            # Reduced to 2000 for faster head-to-head comparison
VAL_FRAC = 0.05             # Smaller val split to save memory
EVAL_INTERVAL = 100         # Frequent logging
LR = 1.5e-4                 # Conservative LR for Large models
DROPOUT = 0.1

# Batch Size Strategy
# Per GPU batch size = 12. With 6 GPUs, Global Batch = 72.
# Grad Accum = 4 => Effective Batch = 288.
BATCH_SIZE_PER_GPU = 12     
GRAD_ACCUM_STEPS = 4        

# TGN Specifics
ENERGY_PENALTY = 1e-3
ENERGY_PENALTY_WARMUP_STEPS = 2500  # 50% of training
GATE_THRESHOLD = 0.5

# ==========================================
# Distributed Setup Helpers
# ==========================================
def setup_ddp():
    if "RANK" not in os.environ:
        # Fallback for single GPU debugging
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
    
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup_ddp():
    dist.destroy_process_group()

def is_main_process():
    return dist.get_rank() == 0

def get_project_root() -> Path:
    try:
        start = Path(__file__).resolve().parent
    except NameError:
        start = Path.cwd()
    if (start / "paper_archive").exists():
        return start
    for p in [start] + list(start.parents):
        if (p / "paper_archive").exists():
            return p
    return start

# ==========================================
# Dataset (Shared)
# ==========================================
class CharDataset(Dataset):
    def __init__(self, text: str, block_size: int):
        chars = sorted(list(set(text)))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.vocab_size = len(chars)
        self.data = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)
        self.block_size = block_size

    def __len__(self):
        return max(0, self.data.numel() - self.block_size - 1)

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1 : idx + 1 + self.block_size]
        return x, y

def prepare_data(root: Path):
    data_path = root / "data" / "tinyshakespeare_input.txt"
    if not data_path.exists():
        if is_main_process():
            data_path.parent.mkdir(parents=True, exist_ok=True)
            url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
            print(f"Downloading data to {data_path}...")
            r = requests.get(url)
            data_path.write_text(r.text, encoding="utf-8")
        dist.barrier() # Wait for main process to download
    
    text = data_path.read_text(encoding="utf-8")
    return text

# ==========================================
# Models: Baseline & TGN
# ==========================================

class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=-1)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
        att = att.masked_fill(mask == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.drop(self.proj(y))

# --- Baseline Transformer Block ---
class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x, None # No gate

# --- TGN Block (Fixed) ---
class TGNBlock(nn.Module):
    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
        self.ln = nn.LayerNorm(n_embd)
        
        # 1. Inertia Channel
        self.rnn = nn.GRU(n_embd, n_embd, batch_first=True)
        
        # 2. Maxwell Demon Gate
        self.gate = nn.Sequential(
            nn.Linear(n_embd, 16),
            nn.Tanh(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        # Init Bias Trick: Force open initially
        nn.init.constant_(self.gate[2].bias, 2.0)

        # 3. Geometric Pump Channel
        self.attn = CausalSelfAttention(n_embd, n_head, dropout)
        
        # 4. FFN
        self.ffn = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # x: (B, T, D)
        residual = x
        x_norm = self.ln(x)
        
        # Parallel Architecture (Consistent with other experiments)
        h_rnn, _ = self.rnn(x_norm)
        h_attn = self.attn(x_norm) # Attention sees original history
        
        g = self.gate(h_rnn)
        h_mixed = (1 - g) * h_rnn + g * h_attn
        
        # RESIDUAL CONNECTION
        h = residual + h_mixed
        
        # FFN
        h = h + self.ffn(self.ln(h))
        
        return h, g

class LanguageModel(nn.Module):
    def __init__(self, vocab_size, model_type, n_embd, n_head, n_layer, dropout):
        super().__init__()
        self.model_type = model_type
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(BLOCK_SIZE, n_embd)
        self.drop = nn.Dropout(dropout)
        
        BlockClass = TGNBlock if model_type == "TGN" else TransformerBlock
        self.blocks = nn.ModuleList([
            BlockClass(n_embd, n_head, dropout) for _ in range(n_layer)
        ])
        
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        
        gates = []
        for block in self.blocks:
            x, g = block(x)
            if g is not None:
                gates.append(g)
        
        x = self.ln_f(x)
        logits = self.head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            
        return logits, loss, gates

# ==========================================
# Training Loop
# ==========================================
def train(local_rank, model_type):
    # Setup
    set_seed(SEED + local_rank) # Seed must be different per rank? No, usually same for init, different for sampler
    # Ideally: Same model init (handled by DDP broadcast or loading), different data shuffle.
    # DDP handles model sync automatically on init if random seed is same.
    torch.manual_seed(SEED) 
    
    root = get_project_root()
    text = prepare_data(root)
    dataset = CharDataset(text, BLOCK_SIZE)
    
    # Split
    n_val = int(len(dataset) * VAL_FRAC)
    n_train = len(dataset) - n_val
    # Ensure determinstic split
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(SEED))
    
    # DDP Sampler
    train_sampler = DistributedSampler(train_set, shuffle=True)
    val_sampler = DistributedSampler(val_set, shuffle=False)
    
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE_PER_GPU, sampler=train_sampler, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE_PER_GPU, sampler=val_sampler, drop_last=True)
    
    # Model Config (Large)
    # n_layer=24, n_head=16, n_embd=1024 -> ~454M Params (for Transformer)
    # TGN will have more due to RNN/Gate parameters.
    config = {
        "n_layer": 24,
        "n_head": 16, 
        "n_embd": 1024, 
        "dropout": DROPOUT
    }
    
    model = LanguageModel(dataset.vocab_size, model_type, **config).to(local_rank)
    
    if is_main_process():
        params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"\n[{model_type}] Model Parameters: {params:.2f}M")
    
    # Wrap DDP
    model = DDP(model, device_ids=[local_rank])
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    
    # Training Loop
    model.train()
    iter_loader = iter(train_loader)
    
    start_time = time.time()
    
    for step in range(MAX_ITERS):
        optimizer.zero_grad()
        loss_accum = 0.0
        gate_mean_accum = 0.0
        gate_rate_accum = 0.0
        
        for _ in range(GRAD_ACCUM_STEPS):
            try:
                x, y = next(iter_loader)
            except StopIteration:
                train_sampler.set_epoch(step) # Reshuffle
                iter_loader = iter(train_loader)
                x, y = next(iter_loader)
            
            x, y = x.to(local_rank), y.to(local_rank)
            
            logits, loss, gates = model(x, y)
            
            # TGN Energy Penalty Logic
            loss_total = loss
            g_mean = torch.tensor(0.0, device=local_rank)
            g_rate = torch.tensor(0.0, device=local_rank)
            
            if model_type == "TGN" and len(gates) > 0:
                g_stack = torch.stack([g.mean() for g in gates])
                g_mean = g_stack.mean()
                g_rate = torch.stack([(g > GATE_THRESHOLD).float().mean() for g in gates]).mean()
                
                # Warmup
                ramp = min(1.0, (step + 1) / ENERGY_PENALTY_WARMUP_STEPS)
                
                # Auto-Regulation
                if g_mean < 0.001:
                    penalty = 0.0
                else:
                    penalty = ENERGY_PENALTY * ramp
                    
                loss_total = loss + penalty * g_mean

            loss_total = loss_total / GRAD_ACCUM_STEPS
            loss_total.backward()
            
            loss_accum += loss.item() / GRAD_ACCUM_STEPS
            gate_mean_accum += g_mean.item() / GRAD_ACCUM_STEPS
            gate_rate_accum += g_rate.item() / GRAD_ACCUM_STEPS
        
        # Clip & Step
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Logging (Main process only)
        if step % EVAL_INTERVAL == 0 and is_main_process():
            dt = time.time() - start_time
            print(f"Iter {step:4d} | Loss {loss_accum:.4f} | GateMean {gate_mean_accum:.4f} | GateRate {gate_rate_accum:.4f} | T={dt:.1f}s")
            
            # Append to CSV
            log_path = get_project_root() / "paper_archive" / "results" / f"ddp_large_{model_type.lower()}_log.csv"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_exists = log_path.exists()
            
            with log_path.open("a", newline="") as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(["iter", "loss", "gate_mean", "gate_rate", "time"])
                writer.writerow([step, loss_accum, gate_mean_accum, gate_rate_accum, dt])

    # Final cleanup
    cleanup_ddp()

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)

if __name__ == "__main__":
    local_rank = setup_ddp()
    
    # 1. Run TGN (Parallel Architecture)
    if is_main_process():
        print(">>> STARTING LARGE TGN EXPERIMENT (DDP - Parallel) <<<")
    train(local_rank, "TGN")
    
    # Wait for TGN to finish across all processes
    dist.barrier()
    
    # 2. Run Baseline (Skipping, already have baseline data)
    if is_main_process():
        print("\n\n>>> STARTING LARGE BASELINE EXPERIMENT (DDP) <<<")
    train(local_rank, "Transformer")
