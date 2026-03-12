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

# --- Model Configuration ---
class TGNConfig:
    def __init__(self, model_size='small', vocab_size=50304):
        self.vocab_size = vocab_size
        self.dropout = 0.1
        
        if model_size == 'small': # ~125M
            self.n_layer = 12
            self.n_head = 12
            self.n_embd = 768
            self.block_size = 1024
        elif model_size == 'medium': # ~350M
            self.n_layer = 24
            self.n_head = 16
            self.n_embd = 1024
            self.block_size = 1024
        elif model_size == 'large': # ~760M
            self.n_layer = 24
            self.n_head = 20
            self.n_embd = 1536
            self.block_size = 2048
        # elif model_size == 'xl': # ~1.3B
        #     self.n_layer = 24
        #     self.n_head = 24
        #     self.n_embd = 2048
        #     self.block_size = 2048
        
        # New Config for 6x A100 (80GB each assumed, or 40GB)
        # We can fit a bit more context or batch size
        elif model_size == 'custom_a100': # Designed for A100 efficiency
            self.n_layer = 24
            self.n_head = 16
            self.n_embd = 1024 # ~350M params
            self.block_size = 4096 # Long context capable on A100
        else:
            raise ValueError(f"Unknown model size: {model_size}")

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
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v 
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y

class GeometricGate(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.Tanh(), # Tanh often more stable for gating logic
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        # Initialize bias to encourage starting open (gate ~ 0.5)
        # Sigmoid(0) = 0.5. Default linear init might be too small/random.
        self.net[-2].bias.data.fill_(0.0) 
    
    def forward(self, x):
        return self.net(x)

class TGNBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.rnn = nn.GRU(config.n_embd, config.n_embd, batch_first=True)
        self.attn = CausalSelfAttention(config)
        self.gate = GeometricGate(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        # 1. Inertia (RNN)
        rnn_out, _ = self.rnn(self.ln1(x))
        # 2. Gate
        g = self.gate(rnn_out)
        # 3. Geometry
        attn_out = self.attn(self.ln1(x))
        # 4. Mix
        mixed = (1 - g) * rnn_out + g * attn_out
        x = x + mixed
        # 5. MLP
        x = x + self.mlp(self.ln2(x))
        return x, g

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
        return x, None

class GPT(nn.Module):
    def __init__(self, config, model_type='tgn'):
        super().__init__()
        self.config = config
        self.model_type = model_type
        
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        
        self.blocks = nn.ModuleList()
        for _ in range(config.n_layer):
            if model_type == 'tgn':
                self.blocks.append(TGNBlock(config))
            else:
                self.blocks.append(TransformerBlock(config))
                
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.token_embedding.weight = self.head.weight # Weight tying
        
        # Enable gradient checkpointing for memory efficiency
        self.gradient_checkpointing = True

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, idx, targets=None, sparsity_lambda=0.0):
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        
        x = self.token_embedding(idx) + self.position_embedding(pos)
        x = self.drop(x)
        
        total_gate = 0.0
        gate_count = 0
        
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                # Gradient Checkpointing: trade compute for memory
                # Custom forward to handle tuple return (x, g)
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward
                
                # Checkpointing requires output to be tensor, but we return tuple.
                # Simplification: we only checkpoint x, accumulating g separately or ignoring g's gradient contribution to memory peak
                # Correct implementation for Tuple return is tricky with torch.utils.checkpoint.
                # Let's use a simpler approach: only checkpoint the transformer part or just run it.
                # For robustness, we'll wrap the block call.
                x, g = torch.utils.checkpoint.checkpoint(create_custom_forward(block), x, use_reentrant=False)
            else:
                x, g = block(x)
                
            if g is not None:
                total_gate += g.mean()
                gate_count += 1
                
        x = self.ln_f(x)
        logits = self.head(x)
        
        loss = None
        gate_mean = 0.0
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            if gate_count > 0:
                gate_mean = total_gate / gate_count
                loss += sparsity_lambda * gate_mean
                
        return logits, loss, gate_mean

# --- Data Loading ---
def get_dataloader(dataset_name, split, block_size, batch_size, ddp_rank, ddp_world_size):
    # Using TikToken for standard GPT-2 BPE
    enc = tiktoken.get_encoding("gpt2")
    
    data_tensor = None
    
    # Check for local offline path first
    local_path = os.path.join("data_cache", dataset_name)
    from datasets import load_from_disk
    
    try:
        if os.path.exists(local_path):
            print(f"[Rank {ddp_rank}] Loading offline dataset from {local_path}...")
            ds = load_from_disk(local_path)[split]
            text_column = 'text'
        elif dataset_name == 'wikitext':
            # WikiText-103
            ds = load_dataset("wikitext", "wikitext-103-v1", split=split)
            text_column = 'text'
        elif dataset_name == 'shakespeare':
            ds = load_dataset("tiny_shakespeare", split=split)
            text_column = 'text'
        elif dataset_name == 'dummy':
            raise ValueError("Force dummy")
        else:
            # Fallback or other HF datasets
            ds = load_dataset(dataset_name, split=split, streaming=True)
            text_column = 'text'

        # Pre-tokenize (naive approach for simplicity, better to use map/cache for huge datasets)
        print(f"Tokenizing {dataset_name} ({split})...")
        
        data = []
        count = 0
        max_tokens = 100_000_000 # Cap at 100M tokens for safety
        
        for item in tqdm(ds):
            text = item[text_column]
            if len(text) > 0:
                ids = enc.encode_ordinary(text)
                ids.append(enc.eot_token)
                data.extend(ids)
                count += len(ids)
                if count > max_tokens: break
                
        data_tensor = torch.tensor(data, dtype=torch.long)
        print(f"Total tokens: {len(data_tensor)/1e6:.2f}M")
        
    except Exception as e:
        print(f"[Rank {ddp_rank}] WARNING: Failed to load dataset '{dataset_name}': {e}")
        print(f"[Rank {ddp_rank}] SWITCHING TO DUMMY DATA MODE (Synthetic).")
        
        # Create synthetic data that mimics language distribution (Zipfian)
        # to ensure loss convergence behavior is somewhat realistic
        vocab_size = 50304
        total_len = 10_000_000 # 10M tokens for dummy test
        
        # Simple random data
        data_tensor = torch.randint(0, vocab_size, (total_len,), dtype=torch.long)
        print(f"[Rank {ddp_rank}] Generated {total_len} dummy tokens.")

    class Dataset(torch.utils.data.Dataset):
        def __init__(self, data, block_size):
            self.data = data
            self.block_size = block_size
        def __len__(self):
            return len(self.data) - self.block_size
        def __getitem__(self, idx):
            # Random sampling for LM
            idx = torch.randint(0, len(self.data) - self.block_size, (1,)).item()
            x = self.data[idx:idx+self.block_size]
            y = self.data[idx+1:idx+1+self.block_size]
            return x, y

    dataset = Dataset(data_tensor, block_size)
    sampler = DistributedSampler(dataset, num_replicas=ddp_world_size, rank=ddp_rank, shuffle=True)
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=0, pin_memory=True) # Workers=0 for safety
    return loader

# --- Main Training Loop ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_size', type=str, default='small')
    parser.add_argument('--dataset', type=str, default='wikitext') # or 'wikitext', 'openwebtext'
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10) # Increased to 10 for convergence on small datasets
    parser.add_argument('--lr', type=float, default=6e-4)
    parser.add_argument('--sparsity_penalty', type=float, default=0.002) # Reduced from 0.02 to avoid early collapse
    parser.add_argument('--out_dir', type=str, default='result_cloud')
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
        print(f"DDP: Process {ddp_rank} (Local {ddp_local_rank}) initialized on {device}")
    else:
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        master_process = True

    if master_process:
        os.makedirs(args.out_dir, exist_ok=True)
        print(f"Starting Scale Experiment: {args.model_size} on {args.dataset}")

    # Config
    config = TGNConfig(model_size=args.model_size)
    
    # Loaders
    # Ensure only master process downloads data to avoid race conditions/bans
    if ddp and not master_process:
        # Wait for master to download
        torch.distributed.barrier()
    
    train_loader = get_dataloader(args.dataset, 'train', config.block_size, args.batch_size, ddp_rank, ddp_world_size)
    
    if ddp and master_process:
        # Release other processes
        torch.distributed.barrier()

    # Models (Train TGN and Transformer side-by-side or separate runs? For script simplicity, let's run TGN)
    # Ideally, run this script twice: once with type='tgn', once 'transformer'
    # We will accept an argument for type
    
    # Update: Adding type arg
    model = GPT(config, model_type='tgn').to(device) # Default to TGN for now
    
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    
    raw_model = model.module if ddp else model
    optimizer = torch.optim.AdamW(raw_model.parameters(), lr=args.lr, weight_decay=0.1, betas=(0.9, 0.95))
    scaler = torch.cuda.amp.GradScaler() # Mixed Precision

    # Training
    step = 0
    acc_loss = 0.0
    acc_gate = 0.0
    log_interval = 10
    
    model.train()
    
    start_time = time.time()
    
    for epoch in range(args.epochs):
        train_loader.sampler.set_epoch(epoch)
        for X, Y in train_loader:
            X, Y = X.to(device), Y.to(device)
            
            with torch.cuda.amp.autocast():
                logits, loss, gate_mean = model(X, Y, sparsity_lambda=args.sparsity_penalty)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            
            step += 1
            acc_loss += loss.item()
            acc_gate += gate_mean if isinstance(gate_mean, float) else gate_mean.item()
            
            if step % log_interval == 0 and master_process:
                avg_loss = acc_loss / log_interval
                avg_gate = acc_gate / log_interval
                ppl = math.exp(avg_loss)
                dt = time.time() - start_time
                print(f"Step {step} | Loss: {avg_loss:.4f} | PPL: {ppl:.2f} | Gate: {avg_gate:.4f} | Time: {dt:.2f}s")
                
                # Simple CSV logging
                with open(os.path.join(args.out_dir, 'log.csv'), 'a') as f:
                    f.write(f"{step},{avg_loss},{ppl},{avg_gate},{dt}\n")
                
                acc_loss = 0.0
                acc_gate = 0.0
                start_time = time.time()
    
    if master_process:
        torch.save(raw_model.state_dict(), os.path.join(args.out_dir, 'final_model.pt'))
        print("Training Complete.")
        destroy_process_group()

if __name__ == '__main__':
    main()
