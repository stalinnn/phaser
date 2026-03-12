import os
import math
import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import tiktoken
from datasets import load_dataset
from tqdm import tqdm
import csv

# --- 强制设置镜像 ---
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0" # 有些环境不开这个更稳

# Try to import Mamba
try:
    from mamba_ssm import Mamba
    HAS_MAMBA = True
except ImportError:
    print("WARNING: Mamba not found, falling back to GRU.")
    HAS_MAMBA = False

# ==========================================
# 1. 350M Medium Config (A800 Optimized)
# ==========================================
class TGNConfig:
    def __init__(self, model_size='medium', vocab_size=50304):
        self.vocab_size = vocab_size
        self.dropout = 0.1
        
        # 350M Parameters
        if model_size == 'medium': 
            self.n_layer = 24
            self.n_head = 16
            self.n_embd = 1024
            self.block_size = 1024
        # 760M Parameters (Optional if you want to push hard)
        elif model_size == 'large': 
            self.n_layer = 24
            self.n_head = 20
            self.n_embd = 1536
            self.block_size = 1024
        else:
            raise ValueError("Use 'medium' or 'large' for A800")

# ==========================================
# 2. Components
# ==========================================
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout_p = config.dropout # Save float value for functional API
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

        # Flash Attention 2.0 Acceleration (Critical for A800)
        y = F.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=None, 
            dropout_p=self.dropout_p if self.training else 0, 
            is_causal=True
        )
        
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y

class GeometricGate(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        self.net[-2].bias.data.fill_(0.0)
    
    def forward(self, x):
        return self.net(x)

class TGNBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        
        # Inertial Channel: Prefer Mamba, fallback to GRU
        if HAS_MAMBA:
            self.inertial = Mamba(
                d_model=config.n_embd,
                d_state=16,
                d_conv=4,
                expand=2
            )
        else:
            self.inertial = nn.GRU(config.n_embd, config.n_embd, batch_first=True)
            
        self.attn = CausalSelfAttention(config)
        self.gate = GeometricGate(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x, mode='standard', fixed_mask=None):
        x_norm = self.ln1(x)
        
        # 1. Inertia (Mamba or GRU)
        if HAS_MAMBA:
            h_inertial = self.inertial(x_norm)
        else:
            h_inertial, _ = self.inertial(x_norm)
            
        # 2. Gate
        if mode == 'standard':
            g = self.gate(h_inertial)
        else: # Random mode
            g = fixed_mask
            
        # 3. Geometry
        attn_out = self.attn(x_norm)
        
        # 4. Mix
        # Convex combination
        mixed = (1 - g) * h_inertial + g * attn_out
        x = x + mixed
        
        # 5. MLP
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
        self.token_embedding.weight = self.head.weight # Weight tying
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, idx, targets=None, sparsity_lambda=0.0, mode='standard', target_sparsity=None):
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.token_embedding(idx) + self.position_embedding(pos)
        x = self.drop(x)
        
        total_gate = 0.0
        
        for block in self.blocks:
            mask = None
            if mode == 'random':
                # Generate random mask with specific sparsity
                mask = torch.bernoulli(torch.full((B, T, 1), target_sparsity, device=idx.device))
            
            x, g = block(x, mode=mode, fixed_mask=mask)
            total_gate += g.mean()
                
        x = self.ln_f(x)
        logits = self.head(x)
        
        loss = None
        gate_mean = total_gate / len(self.blocks)
        
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            if mode == 'standard':
                loss += sparsity_lambda * gate_mean
                
        return logits, loss, gate_mean

# ==========================================
# 3. Data Loading (Robust Map-Style for DDP)
# ==========================================
def get_dataloader(block_size, batch_size, ddp_rank, ddp_world_size):
    enc = tiktoken.get_encoding("gpt2")
    
    # Try loading from local offline cache first
    local_path = "./wikitext_103_offline"
    
    if os.path.exists(local_path):
        if ddp_rank == 0: print(f"Loading from local offline cache: {local_path}")
        from datasets import load_from_disk
        dataset = load_from_disk(local_path)['train']
    else:
        # Only Rank 0 downloads, others wait
        if ddp_rank == 0:
            print(f"Downloading WikiText-103 using mirror: {os.environ['HF_ENDPOINT']} ...")
            try:
                # 尝试下载到本地 cache
                dataset = load_dataset("wikitext", "wikitext-103-v1", split='train')
                print("Download success.")
            except Exception as e:
                print(f"Download failed: {e}")
                print("WARNING: Using Synthetic Data (Network Error)")
                # Fallback to dummy data
                dataset = None
                
        # 同步
        if ddp_world_size > 1:
            torch.distributed.barrier()
            
        # Re-load or create synthetic on all ranks
        if not os.path.exists(local_path):
             try:
                 dataset = load_dataset("wikitext", "wikitext-103-v1", split='train')
             except:
                 dataset = None # Will trigger synthetic below

    if dataset is None:
        # Synthetic Data Fallback
        data_tensor = torch.randint(0, 50257, (1000000,), dtype=torch.long)
        if ddp_rank == 0: print("Generated 1M Synthetic Tokens.")
    else:
        # Pre-tokenize
        if ddp_rank == 0:
            print("Tokenizing WikiText-103...")
            
        data_list = []
        iterator = tqdm(dataset, desc="Loading", disable=(ddp_rank!=0))
        for item in iterator:
            if len(item['text']) > 0:
                data_list.extend(enc.encode_ordinary(item['text']) + [enc.eot_token])
                
        data_tensor = torch.tensor(data_list, dtype=torch.long)
        if ddp_rank == 0:
            print(f"Total tokens: {len(data_tensor)/1e6:.2f}M")

    # Dataset class that slices the long tensor
    class ChunkedDataset(torch.utils.data.Dataset):
        def __init__(self, data, block_size):
            self.data = data
            self.block_size = block_size
            self.length = len(data) - block_size
            
        def __len__(self):
            # Virtual length
            return 1000000 

        def __getitem__(self, idx):
            start_idx = (idx * self.block_size) % (len(self.data) - self.block_size)
            x = self.data[start_idx : start_idx + self.block_size]
            y = self.data[start_idx+1 : start_idx + self.block_size + 1]
            return x, y

    ds = ChunkedDataset(data_tensor, block_size)
    
    # Standard DDP Sampler
    sampler = DistributedSampler(ds, num_replicas=ddp_world_size, rank=ddp_rank, shuffle=True)
    
    return DataLoader(
        ds, 
        batch_size=batch_size, 
        sampler=sampler, 
        num_workers=4, 
        pin_memory=True,
        drop_last=True
    )

# ==========================================
# 4. Main Training Loop
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_size', type=str, default='medium') # 350M
    parser.add_argument('--batch_size', type=int, default=16)       # Per GPU
    parser.add_argument('--grad_accum', type=int, default=2)        # Global Batch ~192
    parser.add_argument('--steps', type=int, default=10000)         
    parser.add_argument('--lr', type=float, default=1.5e-4) # LOWERED LR for Stability
    parser.add_argument('--out_dir', type=str, default='result_fig2')
    args = parser.parse_args()

    # ... (DDP Setup unchanged) ...
    # DDP Setup (Explicit)
    ddp = int(os.environ.get('RANK', -1)) != -1
    if ddp:
        init_process_group(backend='nccl')
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
    else:
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        device = 'cuda'
        master_process = True

    if master_process:
        os.makedirs(args.out_dir, exist_ok=True)
        csv_file = open(os.path.join(args.out_dir, 'training_log.csv'), 'w')
        writer = csv.writer(csv_file)
        writer.writerow(['step', 'loss', 'ppl', 'gate_rate'])

    # Config
    config = TGNConfig(model_size=args.model_size)
    model = GPT(config).to(device)
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    
    raw_model = model.module if ddp else model
    optimizer = torch.optim.AdamW(raw_model.parameters(), lr=args.lr, weight_decay=0.1, betas=(0.9, 0.95))
    
    # Cosine Scheduler with Warmup
    def get_lr(it):
        warmup_iters = 500
        lr_decay_iters = args.steps
        min_lr = args.lr / 10.0
        
        if it < warmup_iters:
            return args.lr * it / warmup_iters
        if it > lr_decay_iters:
            return min_lr
        decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (args.lr - min_lr)

    # Loader
    train_loader = get_dataloader(config.block_size, args.batch_size, ddp_rank, ddp_world_size)
    
    # Training
    if master_process:
        print(f"Starting training 350M TGN on WikiText-103...")
        print(f"Global Batch Size: {args.batch_size * ddp_world_size * args.grad_accum}")
        
    model.train()
    step = 0
    optimizer.zero_grad()
    data_iter = iter(train_loader)
    
    pbar = None
    if master_process:
        pbar = tqdm(total=args.steps, desc="Training")

    while step < args.steps:
        # Update LR
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            
        # Gradient Accumulation
        loss_accum = 0.0
        gate_accum = 0.0
        
        for _ in range(args.grad_accum):
            try:
                X, Y = next(data_iter)
            except StopIteration:
                train_loader.sampler.set_epoch(step) 
                data_iter = iter(train_loader)
                X, Y = next(data_iter)
                
            X, Y = X.to(device), Y.to(device)
            
            # Autocast: Mamba might need fp32 for some parts, handled internally by Mamba impl usually.
            # But we wrap forward in autocast
            with torch.cuda.amp.autocast(dtype=torch.bfloat16): # Use bfloat16 for stability on A800
                logits, loss, gate = model(X, Y, sparsity_lambda=0.002)
                loss = loss / args.grad_accum
            
            loss.backward()
            loss_accum += loss.item()
            gate_accum += gate.item()
        
        # Clip Gradient - Critical for Mamba Stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        optimizer.zero_grad()
        
        step += 1
        
        if master_process:
            ppl = math.exp(min(loss_accum * args.grad_accum, 20)) # Safe exp
            gate_rate = gate_accum / args.grad_accum
            
            writer.writerow([step, loss_accum * args.grad_accum, ppl, gate_rate])
            pbar.set_postfix({'PPL': f"{ppl:.1f}", 'Gate': f"{gate_rate:.2%}", 'LR': f"{lr:.2e}"})
            pbar.update(1)
            
            if step % 1000 == 0:
                torch.save(raw_model.state_dict(), os.path.join(args.out_dir, f'ckpt_{step}.pt'))


    # Phase 2: A/B Testing
    if master_process:
        print("\n>>> Phase 2: A/B Testing (Anti-Forgery Validation)...")
        model.eval()
        with torch.no_grad():
            try:
                val_X, val_Y = next(data_iter)
            except:
                data_iter = iter(train_loader)
                val_X, val_Y = next(data_iter)
                
            val_X, val_Y = val_X.to(device), val_Y.to(device)
            
            # A. Adaptive
            _, loss_a, gate_a = raw_model(val_X, val_Y, mode='standard')
            ppl_a = math.exp(loss_a.item())
            
            # B. Random
            _, loss_b, _ = raw_model(val_X, val_Y, mode='random', target_sparsity=gate_a.item())
            ppl_b = math.exp(loss_b.item())
            
            print(f"Adaptive PPL: {ppl_a:.2f} | Random PPL: {ppl_b:.2f} | Sparsity: {gate_a.item():.2%}")
            
            with open(os.path.join(args.out_dir, 'ab_test.csv'), 'w') as f:
                f.write(f"adaptive,{ppl_a}\nrandom,{ppl_b}\nsparsity,{gate_a.item()}\n")
                
        csv_file.close()
        pbar.close()
    
    destroy_process_group()

if __name__ == '__main__':
    main()
