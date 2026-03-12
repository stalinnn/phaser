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
# STRICT REQUIREMENT: mamba_ssm MUST be installed and compiled with CUDA
try:
    from mamba_ssm import Mamba as FastMamba
    print(">>> ✅ Mamba-SSM (CUDA Accelerated) loaded successfully.")
except ImportError:
    raise ImportError("❌ CRITICAL: 'mamba_ssm' not found. Please install it with CUDA support to run SOTA comparison.")

from datasets import load_dataset
from tqdm import tqdm

# --- Transformer Components ---
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
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
        y = att @ v 
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)

# --- TGN Gate ---
class GeometricGate(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 64), nn.Tanh(),
            nn.Linear(64, 1), nn.Sigmoid()
        )
        self.net[-2].bias.data.fill_(0.0) 
    def forward(self, x): return self.net(x)

# --- Model Wrapper ---
class UniversalModel(nn.Module):
    def __init__(self, config, model_type='tgn'):
        super().__init__()
        self.config = config
        self.model_type = model_type
        
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        
        self.layers = nn.ModuleList()
        for _ in range(config.n_layer):
            if model_type == 'transformer':
                self.layers.append(self._build_transformer_block(config))
            elif model_type == 'mamba':
                self.layers.append(self._build_mamba_block(config))
            elif model_type == 'tgn':
                self.layers.append(self._build_tgn_block(config))
            elif model_type == 'tgn_plus': # TGN + Mamba Inertia (The Ultimate Weapon)
                self.layers.append(self._build_tgn_plus_block(config))
                
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.token_embedding.weight = self.head.weight
        
        # Enable Gradient Checkpointing for ALL models to survive OOM
        self.gradient_checkpointing = True

    # ... (transformer and mamba blocks remain same) ...

    def _build_tgn_block(self, config):
        # TGN Block
        class Block(nn.Module):
            def __init__(self, config):
                super().__init__()
                self.ln1 = nn.LayerNorm(config.n_embd)
                self.rnn = nn.GRU(config.n_embd, config.n_embd, batch_first=True)
                self.attn = CausalSelfAttention(config)
                self.gate = GeometricGate(config.n_embd)
                self.ln2 = nn.LayerNorm(config.n_embd)
                self.mlp = nn.Sequential(
                    nn.Linear(config.n_embd, 4 * config.n_embd), nn.GELU(),
                    nn.Linear(4 * config.n_embd, config.n_embd), nn.Dropout(config.dropout)
                )
            def forward(self, x):
                rnn_out, _ = self.rnn(self.ln1(x))
                g = self.gate(rnn_out)
                attn_out = self.attn(self.ln1(x))
                mixed = (1 - g) * rnn_out + g * attn_out
                x = x + mixed
                x = x + self.mlp(self.ln2(x))
                return x, g
        return Block(config)

    def forward(self, idx, targets=None, sparsity_lambda=0.0):
        B, T = idx.size()
        if self.model_type != 'mamba': # Mamba handles pos embedding differently usually, but let's keep simple
            pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
            x = self.token_embedding(idx) + self.position_embedding(pos)
        else:
            x = self.token_embedding(idx) # Mamba typically no pos emb or learned
            
        x = self.drop(x)
        
        total_gate = 0.0
        gate_count = 0
        
        for layer in self.layers:
            if self.training and self.gradient_checkpointing:
                # Gradient Checkpointing Wrapper
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward
                
                # Checkpointing for TGN (tuple return) vs Others (tensor return)
                if self.model_type == 'tgn':
                    # TGN returns (x, g), checkpointing handles tuples but needs care.
                    # Simplification: Only checkpoint the heavy computation, ignore gate gradient flow through checkpoint boundary if tricky
                    # But torch.utils.checkpoint supports tuple return since PyTorch 1.10+
                    x, g = torch.utils.checkpoint.checkpoint(create_custom_forward(layer), x, use_reentrant=False)
                    total_gate += g.mean()
                    gate_count += 1
                else:
                    x = torch.utils.checkpoint.checkpoint(create_custom_forward(layer), x, use_reentrant=False)
            else:
                # No Checkpointing (Validation or Inference)
                if self.model_type == 'tgn':
                    x, g = layer(x)
                    total_gate += g.mean()
                    gate_count += 1
                else:
                    x = layer(x)
                
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

# --- Config & Main ---
class Config:
    def __init__(self, model_size='small'):
        self.vocab_size = 50304
        self.dropout = 0.1
        if model_size == 'medium': # ~350M
            self.n_layer = 24
            self.n_head = 16
            self.n_embd = 1024
            self.block_size = 1024
        elif model_size == 'large': # ~760M
            self.n_layer = 24
            self.n_head = 20
            self.n_embd = 1536
            self.block_size = 1024

def get_dataloader(dataset_name, split, block_size, batch_size, ddp_rank, ddp_world_size):
    enc = tiktoken.get_encoding("gpt2")
    # ... (Reuse offline loading logic from previous script) ...
    # Simplified for brevity in this tool call
    # Assume offline data_cache/wikitext exists
    from datasets import load_from_disk
    local_path = os.path.join("data_cache", dataset_name)
    
    if os.path.exists(local_path):
        ds = load_from_disk(local_path)[split]
    else:
        # Fallback to load_dataset if internet allows
        ds = load_dataset("wikitext", "wikitext-103-v1", split=split)

    data = []
    # Tokenize (Simplified)
    # Ideally use map, but for script self-containment:
    count = 0
    for item in ds:
        text = item['text']
        if len(text)>0:
            data.extend(enc.encode_ordinary(text))
            count += len(text)
            if count > 10_000_000: break # Limit for demo speed
    
    data_tensor = torch.tensor(data, dtype=torch.long)
    
    class Dataset(torch.utils.data.Dataset):
        def __init__(self, data, block_size):
            self.data = data
            self.block_size = block_size
        def __len__(self): return len(self.data) - self.block_size
        def __getitem__(self, idx):
            idx = torch.randint(0, len(self.data) - self.block_size, (1,)).item()
            return self.data[idx:idx+self.block_size], self.data[idx+1:idx+1+self.block_size]

    dataset = Dataset(data_tensor, block_size)
    sampler = DistributedSampler(dataset, num_replicas=ddp_world_size, rank=ddp_rank)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, required=True, choices=['transformer', 'mamba', 'tgn'])
    parser.add_argument('--model_size', type=str, default='medium')
    parser.add_argument('--out_dir', type=str, default='result_sota')
    args = parser.parse_args()
    
    # DDP Setup
    ddp = int(os.environ.get('RANK', -1)) != -1
    if ddp:
        init_process_group(backend='nccl')
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
    else:
        ddp_rank = 0; ddp_local_rank = 0; device = 'cuda'; master_process = True

    # --- SOTA Optimization: TF32 ---
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if master_process:
        os.makedirs(args.out_dir, exist_ok=True)

    config = Config(model_size=args.model_size)
    model = UniversalModel(config, model_type=args.model_type).to(device)
    
    # --- Battle Stats ---
    if master_process:
        param_count = sum(p.numel() for p in model.parameters())
        param_str = f"{param_count/1e6:.2f}M"
        print(f"\n{'='*40}")
        print(f"🥊 MODEL: {args.model_type.upper()}")
        print(f"📏 SIZE:  {args.model_size} ({param_str} Params)")
        print(f"🚀 MODE:  CUDA (FastMamba)")
        print(f"{'='*40}\n")
    
    if ddp: 
        # Enable find_unused_parameters for Mamba to handle complex graph
        model = DDP(model, device_ids=[ddp_local_rank], find_unused_parameters=(args.model_type == 'mamba'))
    
    # Loader
    # Scale batch size by world size? Ideally yes, but here we fix per-gpu batch size
    # A800 usually has 80GB, we can increase batch size if needed
    batch_size = 32 # Default for A800
    loader = get_dataloader('wikitext', 'train', config.block_size, batch_size, ddp_rank, int(os.environ.get('WORLD_SIZE', 1)))
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scaler = torch.cuda.amp.GradScaler() # AMP
    
    model.train()
    for step, (X, Y) in enumerate(loader):
        if step > 5000: break # Run 5000 steps for serious convergence
        X, Y = X.to(device), Y.to(device)
        
        with torch.cuda.amp.autocast(): # Mixed Precision
            logits, loss, gate = model(X, Y, sparsity_lambda=0.002)
        
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        if ddp_rank == 0 and step % 10 == 0:
            print(f"{args.model_type} | Step {step} | Loss {loss.item():.4f} | Gate {gate:.4f}")
            with open(f"{args.out_dir}/log_{args.model_type}.csv", "a") as f:
                f.write(f"{step},{loss.item()},{gate}\n")

    if ddp: destroy_process_group()

if __name__ == '__main__':
    main()
