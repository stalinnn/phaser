"""
Multi-Scale Scaling Law Experiment: TGN vs Transformer
------------------------------------------------------
Goal: Verify the "S-Curve Scaling Law" (Emergence & Saturation)
      Includes Gradient Checkpointing for XL/XXL models on 80GB cards.
      Optimized for 8-GPU (A800 80GB) Turbo Training.
      **NOW WITH BF16 SUPPORT FOR XXL**

Configs:
- Small (~21M):  L=8,  D=384
- Medium (~128M): L=12, D=768
- Large (~454M): L=24, D=1024
- XL (~1.5B):    L=48, D=1600 (New!)
- XXL (~3.0B):   L=32, D=2560 (New!)

Usage:
    torchrun --nproc_per_node=6 code/experiment_scaling_law_multiscale.py --size xxl
"""

import os
import time
import math
import csv
import argparse
from pathlib import Path
import functools

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.checkpoint import checkpoint
import requests
import numpy as np

# ==========================================
# Configuration
# ==========================================
SEED = 42
BLOCK_SIZE = 256
MAX_ITERS = 2000            
VAL_FRAC = 0.05             
EVAL_INTERVAL = 50          
DROPOUT = 0.1

# Default settings (will be overridden dynamically)
BATCH_SIZE_PER_GPU = 12     
GRAD_ACCUM_STEPS = 4        

# TGN Specifics (THE CONTROL VARIABLES)
ENERGY_PENALTY = 1e-3       
ENERGY_PENALTY_WARMUP_STEPS = 2500  
GATE_THRESHOLD = 0.5

# ==========================================
# Model Configs
# ==========================================
MODEL_CONFIGS = {
    "small":  {"n_layer": 8,  "n_head": 6,  "n_embd": 384,  "lr": 6e-4}, # ~21M
    "medium": {"n_layer": 12, "n_head": 12, "n_embd": 768,  "lr": 3e-4}, # ~128M
    "large":  {"n_layer": 24, "n_head": 16, "n_embd": 1024, "lr": 1.5e-4}, # ~454M
    # XL Config: ~1.5B Params (12 * 48 * 1600^2 approx)
    "xl":     {"n_layer": 48, "n_head": 32, "n_embd": 1600, "lr": 1.0e-4},
    # XXL Config: ~3.0B Params
    "xxl":    {"n_layer": 32, "n_head": 32, "n_embd": 2560, "lr": 8e-5}
}

# ==========================================
# Distributed Setup
# ==========================================
def setup_ddp():
    if "RANK" not in os.environ:
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12356"
    
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
# Dataset
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
        dist.barrier()
    text = data_path.read_text(encoding="utf-8")
    return text

# ==========================================
# Models
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

class TGNBlock(nn.Module):
    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
        self.ln = nn.LayerNorm(n_embd)
        self.rnn = nn.GRU(n_embd, n_embd, batch_first=True)
        self.gate = nn.Sequential(
            nn.Linear(n_embd, 16),
            nn.Tanh(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        nn.init.constant_(self.gate[2].bias, 2.0)
        self.attn = CausalSelfAttention(n_embd, n_head, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        residual = x
        x_norm = self.ln(x)
        h_rnn, _ = self.rnn(x_norm)
        h_attn = self.attn(x_norm) 
        g = self.gate(h_rnn)
        h_mixed = (1 - g) * h_rnn + g * h_attn
        h = residual + h_mixed
        h = h + self.ffn(self.ln(h))
        return h, g

class LanguageModel(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, dropout, use_checkpointing=False):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(BLOCK_SIZE, n_embd)
        self.drop = nn.Dropout(dropout)
        
        self.blocks = nn.ModuleList([
            TGNBlock(n_embd, n_head, dropout) for _ in range(n_layer)
        ])
        
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        self.use_checkpointing = use_checkpointing

    def run_block(self, block, x):
        # Wrapper for checkpointing
        return block(x)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        
        gates = []
        for block in self.blocks:
            if self.training and self.use_checkpointing:
                x, g = checkpoint(self.run_block, block, x, use_reentrant=False)
            else:
                x, g = block(x)
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
def train(local_rank, size_name):
    cfg = MODEL_CONFIGS[size_name]
    
    # --- GLOBAL DEFAULTS ---
    global BATCH_SIZE_PER_GPU, GRAD_ACCUM_STEPS
    use_checkpoint = False
    
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    # Determine precision: only use bf16 for XXL to save memory
    dtype = torch.float32
    if size_name == "xxl":
        dtype = torch.bfloat16
        if is_main_process():
            print(">>> BF16 Precision Enabled for XXL")

    # =========================================================
    # A800 (6-GPU, 80GB) CONFIGURATION
    # Goal: Maintain Global Batch Size = 288
    # =========================================================
    if world_size == 6:
        if size_name in ["small", "medium", "large"]:
            BATCH_SIZE_PER_GPU = 48
            GRAD_ACCUM_STEPS = 1
            if is_main_process():
                print(f">>> A800-6 Turbo ({size_name}): Batch={BATCH_SIZE_PER_GPU}, Accum={GRAD_ACCUM_STEPS} (Global=288)")

        elif size_name == "xl":
            BATCH_SIZE_PER_GPU = 24
            GRAD_ACCUM_STEPS = 2
            use_checkpoint = False 
            if is_main_process():
                print(f">>> A800-6 Turbo (XL): Batch={BATCH_SIZE_PER_GPU}, Accum={GRAD_ACCUM_STEPS}, Checkpoint=OFF")

        elif size_name == "xxl":
            # 3.0B Params. BF16 ON.
            # With BF16 + Checkpoint, we can probably fit Batch=8 or 12.
            # Let's try Batch=12 (aggressive) or Batch=8 (safe).
            # 6 * 12 * 4 = 288.
            BATCH_SIZE_PER_GPU = 8
            GRAD_ACCUM_STEPS = 6
            use_checkpoint = True 
            if is_main_process():
                print(f">>> A800-6 Turbo (XXL - BF16): Batch={BATCH_SIZE_PER_GPU}, Accum={GRAD_ACCUM_STEPS}, Checkpoint=ON")

    # =========================================================
    # FALLBACK (Single GPU or other setups)
    # =========================================================
    elif size_name == "xl":
        use_checkpoint = True
        BATCH_SIZE_PER_GPU = 2
        total_parallel_batch = BATCH_SIZE_PER_GPU * world_size
        target_global_batch = 48 
        GRAD_ACCUM_STEPS = max(1, target_global_batch // total_parallel_batch)

    set_seed(SEED + local_rank)
    torch.manual_seed(SEED)
    
    root = get_project_root()
    text = prepare_data(root)
    dataset = CharDataset(text, BLOCK_SIZE)
    
    n_val = int(len(dataset) * VAL_FRAC)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(SEED))
    
    train_sampler = DistributedSampler(train_set, shuffle=True)
    val_sampler = DistributedSampler(val_set, shuffle=False)
    
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE_PER_GPU, sampler=train_sampler, drop_last=True)
    
    # Init model in FP32 (weights), but run forward in BF16 if needed
    model = LanguageModel(
        dataset.vocab_size, 
        cfg["n_embd"], 
        cfg["n_head"], 
        cfg["n_layer"], 
        DROPOUT,
        use_checkpointing=use_checkpoint
    ).to(local_rank)
    
    model = DDP(model, device_ids=[local_rank])
    
    if is_main_process():
        params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"\n>>> Starting {size_name.upper()} Experiment <<<")
        print(f"Config: {cfg}")
        print(f"Parameters: {params:.2f}M")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
    
    model.train()
    iter_loader = iter(train_loader)
    start_time = time.time()
    
    log_path = root / "paper_archive" / "results" / f"ddp_{size_name}_tgn_log.csv"
    if is_main_process():
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["iter", "loss", "gate_mean", "gate_rate", "time"])

    for step in range(MAX_ITERS):
        optimizer.zero_grad()
        loss_accum = 0.0
        gate_mean_accum = 0.0
        gate_rate_accum = 0.0
        
        for _ in range(GRAD_ACCUM_STEPS):
            try:
                x, y = next(iter_loader)
            except StopIteration:
                train_sampler.set_epoch(step)
                iter_loader = iter(train_loader)
                x, y = next(iter_loader)
            
            x, y = x.to(local_rank), y.to(local_rank)
            
            # === AMP CONTEXT ===
            with torch.cuda.amp.autocast(dtype=dtype):
                logits, loss, gates = model(x, y)
                
                g_stack = torch.stack([g.mean() for g in gates])
                g_mean = g_stack.mean()
                g_rate = torch.stack([(g > GATE_THRESHOLD).float().mean() for g in gates]).mean()
                
                ramp = min(1.0, (step + 1) / ENERGY_PENALTY_WARMUP_STEPS)
                penalty = 0.0 if g_mean < 0.001 else ENERGY_PENALTY * ramp
                loss_total = (loss + penalty * g_mean) / GRAD_ACCUM_STEPS
            
            # Standard backward (BF16 doesn't strictly need scaler, but fine to run without)
            loss_total.backward()
            
            loss_accum += loss.item() / GRAD_ACCUM_STEPS
            gate_mean_accum += g_mean.item() / GRAD_ACCUM_STEPS
            gate_rate_accum += g_rate.item() / GRAD_ACCUM_STEPS
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if step % EVAL_INTERVAL == 0 and is_main_process():
            dt = time.time() - start_time
            print(f"Iter {step:4d} | Loss {loss_accum:.4f} | Gate {gate_rate_accum:.4f} | Time {dt:.1f}s")
            with log_path.open("a", newline="") as f:
                csv.writer(f).writerow([step, loss_accum, gate_mean_accum, gate_rate_accum, dt])

    cleanup_ddp()

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=str, required=True, choices=["small", "medium", "large", "xl", "xxl"])
    args = parser.parse_args()
    
    local_rank = setup_ddp()
    train(local_rank, args.size)