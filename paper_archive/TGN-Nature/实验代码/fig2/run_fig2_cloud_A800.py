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
from datasets import load_from_disk, load_dataset
from tqdm import tqdm
import csv

# --- 强制设置镜像 ---
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

# Try to import Mamba
try:
    from mamba_ssm import Mamba
    HAS_MAMBA = True
except ImportError:
    print("WARNING: Mamba not found, falling back to GRU.")
    HAS_MAMBA = False

# ==========================================
# 1. Config
# ==========================================
class TGNConfig:
    def __init__(self, model_size='medium', vocab_size=50304):
        self.vocab_size = vocab_size
        self.dropout = 0.1
        
        if model_size == 'medium': 
            self.n_layer = 24
            self.n_head = 16
            self.n_embd = 1024
            self.block_size = 1024
        elif model_size == 'large': 
            self.n_layer = 24
            self.n_head = 20
            self.n_embd = 1536
            self.block_size = 1024
        else:
            raise ValueError("Use 'medium' or 'large'")

# ==========================================
# 2. Components (Same as before)
# ==========================================
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
        # Initialize bias to negative to encourage sparsity at start
        self.net[-2].bias.data.fill_(-1.0)
    
    def forward(self, x):
        return self.net(x)

class TGNBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        
        if HAS_MAMBA:
            self.inertial = Mamba(d_model=config.n_embd, d_state=16, d_conv=4, expand=2)
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
        
        if HAS_MAMBA:
            h_inertial = self.inertial(x_norm)
        else:
            h_inertial, _ = self.inertial(x_norm)
            
        if mode == 'standard':
            g = self.gate(h_inertial)
        else: # Random mode for A/B testing
            g = fixed_mask
            
        attn_out = self.attn(x_norm)
        
        # Soft mixing
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
# 3. Data Loading (Fixed)
# ==========================================
def get_dataloaders(data_path, block_size, batch_size, ddp_rank, ddp_world_size):
    enc = tiktoken.get_encoding("gpt2")
    
    # Auto-detect path
    possible_paths = [
        data_path, 
        "./wikitext_103_offline",
        "../wikitext_103_offline",
        "/TGN/wikitext_103_offline"
    ]
    
    final_path = None
    for p in possible_paths:
        if os.path.exists(p):
            final_path = p
            break
            
    if ddp_rank == 0:
        if final_path:
            print(f"Loading data from: {final_path}")
        else:
            print(f"WARNING: Data path not found, will attempt download...")
    
    # Load dataset
    if final_path:
        dataset_dict = load_from_disk(final_path)
    else:
        dataset_dict = load_dataset("wikitext", "wikitext-103-v1")

    def process_split(split_name):
        if ddp_rank == 0: print(f"Processing {split_name}...")
        data_list = []
        # Only rank 0 needs to show progress bar, but all ranks process
        dataset = dataset_dict[split_name]
        for item in dataset:
            if len(item['text']) > 0:
                data_list.extend(enc.encode_ordinary(item['text']) + [enc.eot_token])
        return torch.tensor(data_list, dtype=torch.long)

    train_data = process_split('train')
    # Load Validation Set!
    val_data = process_split('validation') 
    
    if ddp_rank == 0:
        print(f"Train Tokens: {len(train_data)/1e6:.2f}M")
        print(f"Val Tokens:   {len(val_data)/1e6:.2f}M")

    class ChunkedDataset(torch.utils.data.Dataset):
        def __init__(self, data, block_size):
            self.data = data
            self.block_size = block_size
            
        def __len__(self):
            return (len(self.data) - 1) // self.block_size

        def __getitem__(self, idx):
            start_idx = idx * self.block_size
            x = self.data[start_idx : start_idx + self.block_size]
            y = self.data[start_idx+1 : start_idx + self.block_size + 1]
            return x, y

    train_ds = ChunkedDataset(train_data, block_size)
    val_ds = ChunkedDataset(val_data, block_size)
    
    train_sampler = DistributedSampler(train_ds, num_replicas=ddp_world_size, rank=ddp_rank, shuffle=True)
    val_sampler = DistributedSampler(val_ds, num_replicas=ddp_world_size, rank=ddp_rank, shuffle=False)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=train_sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, sampler=val_sampler, num_workers=4, pin_memory=True)
    
    return train_loader, val_loader

# ==========================================
# 4. Main
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_size', type=str, default='medium')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--grad_accum', type=int, default=2)
    parser.add_argument('--steps', type=int, default=10000)
    parser.add_argument('--lr', type=float, default=1.5e-4)
    parser.add_argument('--out_dir', type=str, default='result_fig2')
    parser.add_argument('--data_path', type=str, default='./wikitext_103_offline')
    parser.add_argument('--sparsity_penalty', type=float, default=0.002) # Add this line
    args = parser.parse_args()

    # DDP Setup
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

    config = TGNConfig(model_size=args.model_size)
    model = GPT(config).to(device)
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    
    raw_model = model.module if ddp else model
    optimizer = torch.optim.AdamW(raw_model.parameters(), lr=args.lr, weight_decay=0.1)
    
    train_loader, val_loader = get_dataloaders(args.data_path, config.block_size, args.batch_size, ddp_rank, ddp_world_size)
    
    if master_process:
        print(f"Starting training TGN...")

    # Training Loop
    model.train()
    step = 0
    optimizer.zero_grad()
    data_iter = iter(train_loader)
    
    pbar = None
    if master_process:
        pbar = tqdm(total=args.steps, desc="Training")

    while step < args.steps:
        loss_accum = 0.0
        gate_accum = 0.0
        
        for _ in range(args.grad_accum):
            try:
                X, Y = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                X, Y = next(data_iter)
                
            X, Y = X.to(device), Y.to(device)
            
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                logits, loss, gate = model(X, Y, sparsity_lambda=args.sparsity_penalty)
                loss = loss / args.grad_accum
            
            loss.backward()
            loss_accum += loss.item()
            gate_accum += gate.item()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        step += 1
        
        if master_process:
            ppl = math.exp(min(loss_accum * args.grad_accum, 20))
            gate_rate = gate_accum / args.grad_accum
            writer.writerow([step, loss_accum * args.grad_accum, ppl, gate_rate])
            csv_file.flush() # Force write to disk immediately
            pbar.set_postfix({'PPL': f"{ppl:.1f}", 'Gate': f"{gate_rate:.2%}"})
            pbar.update(1)
            
            if step % 2000 == 0 or step == args.steps:
                torch.save(raw_model.state_dict(), os.path.join(args.out_dir, f'ckpt_{step}.pt'))

    if master_process:
        csv_file.close()
        pbar.close()

    # ==========================================
    # Phase 2: Rigorous A/B Testing on Validation Set
    # ==========================================
    if master_process:
        print("\n>>> Phase 2: Rigorous A/B Testing on Validation Set...")
    
    # Sync weights before testing
    if ddp: torch.distributed.barrier()
    
    model.eval()
    
    # 1. First pass: Measure avg sparsity on Validation
    total_val_gate = 0.0
    val_steps = 0
    val_loss_adaptive = 0.0
    
    if master_process: print("Evaluating Adaptive TGN on Validation Set...")
    
    with torch.no_grad():
        for val_X, val_Y in tqdm(val_loader, disable=(not master_process)):
            val_X, val_Y = val_X.to(device), val_Y.to(device)
            # Standard TGN
            _, loss, gate = model(val_X, val_Y, mode='standard')
            
            # Gather loss from all GPUs
            if ddp:
                torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.AVG)
                torch.distributed.all_reduce(gate, op=torch.distributed.ReduceOp.AVG)
                
            val_loss_adaptive += loss.item()
            total_val_gate += gate.item()
            val_steps += 1
            
    avg_sparsity = total_val_gate / val_steps
    avg_ppl_adaptive = math.exp(val_loss_adaptive / val_steps)
    
    if master_process:
        print(f"Adaptive Validation PPL: {avg_ppl_adaptive:.4f}")
        print(f"Measured Avg Sparsity: {avg_sparsity:.2%}")
        print(f"Starting Random Baseline Evaluation (Target Sparsity: {avg_sparsity:.4f})...")

    # 2. Second pass: Random Baseline with EXACT same sparsity
    val_loss_random = 0.0
    
    with torch.no_grad():
        for val_X, val_Y in tqdm(val_loader, disable=(not master_process)):
            val_X, val_Y = val_X.to(device), val_Y.to(device)
            # Random TGN
            _, loss, _ = model(val_X, val_Y, mode='random', target_sparsity=avg_sparsity)
            
            if ddp:
                torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.AVG)
                
            val_loss_random += loss.item()

    avg_ppl_random = math.exp(val_loss_random / val_steps)

    # 3. Third pass: Full Attention Baseline (Upper Bound Proxy)
    # Force gate = 1.0 everywhere to simulate "what if we had infinite compute"
    val_loss_full = 0.0
    if master_process: print(f"Evaluating Full Attention Upper Bound (Gate=1.0)...")

    with torch.no_grad():
        for val_X, val_Y in tqdm(val_loader, disable=(not master_process)):
            val_X, val_Y = val_X.to(device), val_Y.to(device)
            # Full mode: force gate to 1
            # We reuse 'random' mode logic but pass target_sparsity=1.0 which makes bernoulli output all 1s
            _, loss, _ = model(val_X, val_Y, mode='random', target_sparsity=1.0)
            
            if ddp:
                torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.AVG)
                
            val_loss_full += loss.item()

    avg_ppl_full = math.exp(val_loss_full / val_steps)

    if master_process:
        print("="*40)
        print(f"FINAL RESULTS (N={val_steps} batches)")
        print("="*40)
        print(f"Full Attention (Proxy): {avg_ppl_full:.4f} (100% Compute)")
        print(f"Adaptive TGN:         {avg_ppl_adaptive:.4f} ({avg_sparsity:.2%} Compute)")
        print(f"Random Baseline:      {avg_ppl_random:.4f} ({avg_sparsity:.2%} Compute)")
        print("="*40)
        
        # Save results
        with open(os.path.join(args.out_dir, 'final_ab_test_rigorous.csv'), 'w') as f:
            f.write("model,ppl,sparsity\n")
            f.write(f"full,{avg_ppl_full},1.0\n")
            f.write(f"adaptive,{avg_ppl_adaptive},{avg_sparsity}\n")
            f.write(f"random,{avg_ppl_random},{avg_sparsity}\n")
            
    if ddp:
        destroy_process_group()

if __name__ == '__main__':
    main()
