import os
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import tiktoken
from datasets import load_dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
import csv

# ==========================================
# 1. 论文级配置 (Paper-level Config)
# ==========================================
class PaperConfig:
    def __init__(self):
        self.vocab_size = 50257  # GPT-2
        self.block_size = 1024   # 论文级上下文长度
        self.n_layer = 12        # GPT-2 Small 规模
        self.n_head = 12
        self.n_embd = 768
        self.dropout = 0.1
        self.batch_size = 4      # 显存优化：从 8 降到 4
        self.lr = 6e-4
        self.max_steps = 5000    # 训练步数
        self.sparsity_lambda = 0.005 
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.out_dir = "result/paper_repro"
        self.log_path = os.path.join(self.out_dir, "training_log.csv")
        os.makedirs(self.out_dir, exist_ok=True)

config = PaperConfig()

# ==========================================
# 2. 核心架构 (TGN Architecture)
# ==========================================
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                    .view(1, 1, config.block_size, config.block_size))

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

class TGNBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.rnn = nn.GRU(config.n_embd, config.n_embd, batch_first=True)
        self.attn = CausalSelfAttention(config)
        self.gate_net = nn.Sequential(
            nn.Linear(config.n_embd, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd)
        )

    def forward(self, x, mode='standard', fixed_mask=None):
        x_norm = self.ln1(x)
        # 惯性通道
        h_rnn, _ = self.rnn(x_norm)
        # 门控决策
        if mode == 'standard':
            g = self.gate_net(h_rnn)
        else: # random mode
            g = fixed_mask
        # 几何通道
        h_attn = self.attn(x_norm)
        # 并行混合
        x = x + (1 - g) * h_rnn + g * h_attn
        x = x + self.mlp(self.ln2(x))
        return x, g

class TGNModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.ModuleList([TGNBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.token_emb.weight = self.head.weight # Weight Tying

    def forward(self, idx, targets=None, mode='standard', target_sparsity=None):
        B, T = idx.size()
        x = self.token_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device))
        
        total_g = 0
        for block in self.blocks:
            mask = None
            if mode == 'random':
                mask = torch.bernoulli(torch.full((B, T, 1), target_sparsity, device=idx.device))
            x, g = block(x, mode=mode, fixed_mask=mask)
            total_g += g.mean()
            
        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            if mode == 'standard':
                # 注意力惩罚
                loss += config.sparsity_lambda * (total_g / len(self.blocks))
                
        return logits, loss, total_g / len(self.blocks)

# ==========================================
# 3. 数据加载 (WikiText-103)
# ==========================================
def get_data_loader():
    print("Loading WikiText-103...")
    ds = load_dataset("wikitext", "wikitext-103-v1", split='train')
    enc = tiktoken.get_encoding("gpt2")
    
    def tokenize(example):
        ids = enc.encode_ordinary(example['text'])
        ids.append(enc.eot_token)
        return {'ids': ids}
    
    tokenized = ds.map(tokenize, remove_columns=['text'], num_proc=4)
    # 展平所有 tokens
    all_ids = []
    for item in tqdm(tokenized['ids'], desc="Processing tokens"):
        all_ids.extend(item)
    data_tensor = torch.tensor(all_ids, dtype=torch.long)
    
    class IterDataset(torch.utils.data.Dataset):
        def __init__(self, data, block_size):
            self.data = data
            self.block_size = block_size
        def __len__(self): return len(self.data) - self.block_size
        def __getitem__(self, i):
            return self.data[i:i+self.block_size], self.data[i+1:i+1+self.block_size]

    return DataLoader(IterDataset(data_tensor, config.block_size), batch_size=config.batch_size, shuffle=True)

# ==========================================
# 4. 主程序
# ==========================================
def train():
    loader = get_data_loader()
    model = TGNModel(config).to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=0.1)
    
    # 初始化 CSV 日志
    with open(config.log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['step', 'loss', 'ppl', 'gate_rate'])

    print(f"\n>>> Phase 1: Training on WikiText-103 (Device: {config.device})")
    model.train()
    scaler = torch.cuda.amp.GradScaler() # 混合精度训练
    
    step = 0
    start_time = time.time()
    
    pbar = tqdm(total=config.max_steps, desc="Training")
    for X, Y in loader:
        X, Y = X.to(config.device), Y.to(config.device)
        
        optimizer.zero_grad()
        
        # 使用 AMP 自动混合精度
        with torch.cuda.amp.autocast():
            logits, loss, gate = model(X, Y)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        if step % 10 == 0:
            loss_val = loss.item()
            gate_val = gate.item()
            ppl = math.exp(min(loss_val, 20)) # 防止溢出
            
            # 保存到 CSV
            with open(config.log_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([step, f"{loss_val:.4f}", f"{ppl:.2f}", f"{gate_val:.4f}"])
            
            pbar.set_postfix({"Loss": f"{loss_val:.3f}", "Gate": f"{gate_val:.2%}", "PPL": f"{ppl:.1f}"})
        
        step += 1
        pbar.update(1)
        if step >= config.max_steps: break
    
    pbar.close()

    # Phase 2: A/B 测试逻辑
    print("\n>>> Phase 2: A/B Testing...")
    model.eval()
    with torch.no_grad():
        # 获取一个 Batch 进行公平对比
        test_X, test_Y = next(iter(loader))
        test_X, test_Y = test_X.to(config.device), test_Y.to(config.device)
        
        with torch.cuda.amp.autocast():
            _, loss_a, gate_a = model(test_X, test_Y, mode='standard')
            ppl_a = math.exp(loss_a.item())
            
            _, loss_b, _ = model(test_X, test_Y, mode='random', target_sparsity=gate_a.item())
            ppl_b = math.exp(loss_b.item())
        
        print(f"\nFinal Comparison:")
        print(f"Adaptive TGN PPL: {ppl_a:.2f}")
        print(f"Random Mask PPL:  {ppl_b:.2f}")
        print(f"Sparsity Rate:    {gate_a.item():.2%}")
        
        # 结果存入 CSV
        with open(os.path.join(config.out_dir, "ab_test_result.csv"), 'w') as f:
            f.write(f"metric,value\nadaptive_ppl,{ppl_a}\nrandom_ppl,{ppl_b}\nsparsity,{gate_a.item()}\n")

if __name__ == "__main__":
    train()
