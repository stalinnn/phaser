import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import math
import time
import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ==========================================
# 1. 配置与超参数 (Configuration)
# ==========================================
class Config:
    def __init__(self):
        self.vocab_size = 1000   # 快速验证不需要太大的词表
        self.d_model = 256       # 减小维度以加快速度
        self.n_layers = 4        
        self.n_heads = 4
        self.d_ff = 1024
        self.dropout = 0.1
        self.max_seq_len = 128   # 较短的序列，但足以体现长程依赖
        self.batch_size = 32
        self.lr = 1e-3           # 提高学习率以快速收敛
        self.epochs = 50         # 增加轮数
        self.gate_lambda = 0.01  # 降低惩罚，鼓励“觉醒”
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.output_dir = "result/ablation_experiment"
        os.makedirs(self.output_dir, exist_ok=True)

config = Config()

# ==========================================
# 2. TGN 模型定义 (TGN Architecture)
# ==========================================
class TGNLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        # 惯性通道 (Inertial Channel)
        self.rnn = nn.Linear(d_model, d_model) # 简化表示
        
        # 几何通道 (Geometric Channel)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        
        # 门控网络 (Maxwell's Demon)
        self.gate_net = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x, mode='standard', fixed_mask=None):
        # 1. 惯性流
        h_inertial = self.rnn(x)
        
        # 2. 门控决策
        if mode == 'standard':
            gate_score = self.gate_net(h_inertial) 
        elif mode == 'random':
            gate_score = fixed_mask
        
        # 3. 几何流 (Attention)
        batch, seq_len, _ = x.shape
        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(x.device)
        attn_out, _ = self.attn(x, x, x, attn_mask=causal_mask, need_weights=False)
        
        # 4. 物理融合 (Convex Combination)
        # 核心：(1-g)*Inertia + g*Geometry
        mixed = (1 - gate_score) * h_inertial + gate_score * attn_out
        
        out = self.ln1(x + mixed)
        out = out + self.ffn(self.ln2(out))
        
        return out, gate_score

class TGNModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.max_seq_len, config.d_model))
        self.layers = nn.ModuleList([
            TGNLayer(config.d_model, config.n_heads, config.d_ff, config.dropout)
            for _ in range(config.n_layers)
        ])
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
    def forward(self, x, mode='standard', target_sparsity=None):
        batch, seq_len = x.shape
        h = self.embedding(x) + self.pos_emb[:, :seq_len, :]
        
        total_gate_activation = 0
        for layer in self.layers:
            fixed_mask = None
            if mode == 'random':
                fixed_mask = torch.bernoulli(torch.full((batch, seq_len, 1), target_sparsity, device=x.device))
            
            h, gate = layer(h, mode=mode, fixed_mask=fixed_mask)
            total_gate_activation += gate.mean()
            
        logits = self.head(h)
        return logits, total_gate_activation / len(self.layers)

# ==========================================
# 3. 结构化数据：长程复制任务 (Copy Task)
# ==========================================
class StructuredCopyDataset(Dataset):
    def __init__(self, seq_len, vocab_size, size=2000):
        self.size = size
        self.seq_len = seq_len
        self.half = seq_len // 2
        # 生成 [Pattern_A, Pattern_A] 形式的数据
        # 前半部分随机，后半部分重复前半部分。模型必须通过 Attention 越过中间位置查找前面的 Token
        pattern = torch.randint(1, vocab_size, (size, self.half))
        self.data = torch.cat([pattern, pattern], dim=1)
    
    def __len__(self): return self.size
    def __getitem__(self, idx): return self.data[idx], self.data[idx]

# ==========================================
# 4. 实验流程
# ==========================================
def run_experiment():
    print(f"Device: {config.device} | Task: Long-range Copy (Structured)")
    model = TGNModel(config).to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    criterion = nn.CrossEntropyLoss()
    
    train_loader = DataLoader(StructuredCopyDataset(config.max_seq_len, config.vocab_size, 5000), batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(StructuredCopyDataset(config.max_seq_len, config.vocab_size, 500), batch_size=config.batch_size)
    
    print("\n>>> Phase 1: Training (Watching for 'Hysteretic Awakening')...")
    model.train()
    for epoch in range(config.epochs):
        total_loss, total_gate = 0, 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(config.device), target.to(config.device)
            optimizer.zero_grad()
            logits, avg_gate = model(data, mode='standard')
            
            # Autoregressive loss
            loss = criterion(logits[:, :-1, :].reshape(-1, config.vocab_size), target[:, 1:].reshape(-1))
            (loss + config.gate_lambda * avg_gate).backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_gate += avg_gate.item()
            
        print(f"Epoch {epoch} | Loss: {total_loss/len(train_loader):.4f} | Gate: {total_gate/len(train_loader):.2%}")

    print("\n>>> Phase 2: A/B Testing...")
    model.eval()
    
    def evaluate(mode, target_sparsity=None):
        nll, tokens, gates = 0, 0, 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(config.device), target.to(config.device)
                logits, gate = model(data, mode=mode, target_sparsity=target_sparsity)
                loss = criterion(logits[:, :-1, :].reshape(-1, config.vocab_size), target[:, 1:].reshape(-1))
                nll += loss.item() * (config.max_seq_len - 1) * data.size(0)
                tokens += (config.max_seq_len - 1) * data.size(0)
                gates += gate.item()
        return math.exp(nll / tokens), gates / len(val_loader)

    ppl_a, sparsity_a = evaluate('standard')
    print(f"Group A (Adaptive) | PPL: {ppl_a:.2f} | Gate: {sparsity_a:.2%}")
    
    ppl_b, _ = evaluate('random', target_sparsity=sparsity_a)
    print(f"Group B (Random)   | PPL: {ppl_b:.2f} | Gate: {sparsity_a:.2%} (Fixed)")

    # Plotting
    labels, ppls = ['TGN (Adaptive)', 'Random (Baseline)'], [ppl_a, ppl_b]
    plt.figure(figsize=(6, 5), dpi=150)
    bars = plt.bar(labels, ppls, color=['#2ca02c', '#7f7f7f'], alpha=0.8, width=0.6)
    for bar in bars:
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height(), f'{bar.get_height():.2f}', ha='center', va='bottom', fontweight='bold')
    plt.ylabel('Perplexity'), plt.title(f'Mechanism Verification (Sparsity ~{sparsity_a*100:.1f}%)')
    plt.savefig(Path(config.output_dir) / 'figure_2b_ablation.png')
    print(f"Result saved to {config.output_dir}/figure_2b_ablation.png")

if __name__ == '__main__':
    run_experiment()
