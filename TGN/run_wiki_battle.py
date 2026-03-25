import os
# 设置 HuggingFace 镜像源以解决国内网络问题
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import tiktoken
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
import csv

# Import the centralized TGN architecture
# Make sure to import mamba first to check if it's available
import sys
try:
    import mamba_ssm
    print(f"Mamba-SSM Version: {mamba_ssm.__version__}")
except ImportError:
    pass

from tgn import UniversalModel

# ==========================================
# 1. Config
# ==========================================
class Config:
    def __init__(self, model_type='tgn', size='500M'):
        self.vocab_size = 50257  # GPT-2
        self.max_seq_len = 1024  
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_type = model_type
        
        # Scaling configurations
        if size == '500M':
            self.n_layers = 24
            self.d_model = 1024
            self.n_heads = 16
        elif size == '2B':
            self.n_layers = 24
            self.d_model = 2048
            self.n_heads = 16
        else: # debug
            self.n_layers = 8
            self.d_model = 512
            self.n_heads = 8
            
        self.batch_size = 32      
        self.lr = 3e-4 # Lower LR for larger models
        self.max_steps = 100    
        self.sparsity_lambda = 0.001
        
        self.out_dir = f"TGN/result/wiki_battle_{size}"
        os.makedirs(self.out_dir, exist_ok=True)
        self.log_path = os.path.join(self.out_dir, f"{model_type}_log.csv")

# ==========================================
# 2. Data Loading (WikiText-103)
# ==========================================
def get_data_loaders(config):
    print("Loading WikiText-103 dataset from HuggingFace...")
    
    # 加载真正的 WikiText-103
    dataset = load_dataset("wikitext", "wikitext-103-v1")
    enc = tiktoken.get_encoding("gpt2")
    
    def encode_split(split_name, max_tokens=None):
        print(f"Tokenizing {split_name} split...")
        # 过滤掉空行，合并文本
        texts = [text for text in dataset[split_name]['text'] if len(text.strip()) > 0]
        
        # 如果只想快速跑，可以截断数据；如果是正式实验，建议全部使用
        if max_tokens is not None:
            # 简单粗暴：只取前几百篇文章来快速达到 max_tokens
            text_str = "\n".join(texts[:5000])
        else:
            text_str = "\n".join(texts)
            
        tokens = enc.encode(text_str)
        if max_tokens is not None:
            tokens = tokens[:max_tokens]
            
        return torch.tensor(tokens, dtype=torch.long)

    # 训练集：1亿Token有点大，为了单卡测试效率，我们取前 10,000,000 个Token (约10%数据) 
    # 如果你要发论文跑全量，把 max_tokens=10000000 删掉即可
    train_data = encode_split('train') 
    # 验证集：用完整的 validation split (大约 200,000 tokens)
    val_data = encode_split('validation')

    class IterDataset(torch.utils.data.Dataset):
        def __init__(self, data, block_size):
            self.data = data
            self.block_size = block_size
        def __len__(self): 
            # 改进：避免滑动窗口重叠，标准的 Chunk 划分
            return (len(self.data) - 1) // self.block_size
        def __getitem__(self, i):
            idx = i * self.block_size
            return self.data[idx:idx+self.block_size], self.data[idx+1:idx+1+self.block_size]

    train_loader = DataLoader(IterDataset(train_data, config.max_seq_len), batch_size=config.batch_size, shuffle=True)
    # 验证集也开启 shuffle，确保 100 步能够随机抽样到不同文章段落
    val_loader = DataLoader(IterDataset(val_data, config.max_seq_len), batch_size=config.batch_size, shuffle=True)
    
    return train_loader, val_loader

# ==========================================
# 3. Main Run
# ==========================================
def train_model(model_type, train_loader, val_loader, size='500M'):
    config = Config(model_type, size=size)
    model = UniversalModel(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        max_seq_len=config.max_seq_len,
        model_type=config.model_type
    )
    
    # Enable DataParallel if multiple GPUs are available
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
        
    model = model.to(config.device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=0.1)
    
    with open(config.log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['step', 'loss', 'ppl', 'gate_rate'])

    print(f"\n>>> Training {model_type.upper()} on WikiText-103 (Device: {config.device})")
    model.train()
    scaler = torch.cuda.amp.GradScaler() 
    
    step = 0
    pbar = tqdm(total=config.max_steps, desc=f"Training {model_type}")
    
    loader_iter = iter(train_loader)
    
    while step < config.max_steps:
        try:
            X, Y = next(loader_iter)
        except StopIteration:
            loader_iter = iter(train_loader)
            X, Y = next(loader_iter)
            
        X, Y = X.to(config.device), Y.to(config.device)
        optimizer.zero_grad()
        
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            # nn.DataParallel 封装后，原始参数会移动到 model.module
            if isinstance(model, nn.DataParallel):
                logits, loss, gate = model(X, targets=Y, mode='standard')
            else:
                logits, loss, gate = model(X, Y, mode='standard')
                
            # 当使用 DataParallel 时，loss 和 gate 返回的是各个 GPU 上结果的 list/tensor，需要求均值
            if isinstance(model, nn.DataParallel):
                loss = loss.mean()
                gate = gate.mean()
                
            if model_type == 'tgn':
                loss += config.sparsity_lambda * gate
                
        # Handle nan gracefully during training loop
        if torch.isnan(loss):
            print(f"NaN Loss at step {step}")
            break
        
        scaler.scale(loss).backward()
        
        # gradient clipping to prevent NaN
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # 放宽一点梯度裁剪，有助于大batch收敛
        
        scaler.step(optimizer)
        scaler.update()
        
        if step % 20 == 0:
            loss_val = loss.item()
            gate_val = gate.item()
            if loss_val > 100 or math.isnan(loss_val): 
                ppl = float('inf')
            else:
                ppl = math.exp(loss_val) 
            
            with open(config.log_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([step, f"{loss_val:.4f}", f"{ppl:.2f}", f"{gate_val:.4f}"])
            
            pbar.set_postfix({
                "Loss": f"{loss_val:.3f}", 
                "PPL": f"{ppl:.1f}", 
                "Gate": f"{gate_val:.2%}"
            })
        
        step += 1
        pbar.update(1)
        if step >= config.max_steps: break
    
    pbar.close()
    # Phase 2: A/B 测试逻辑 (Evaluation)
    print(f"\n>>> Running Evaluation for {model_type.upper()} on Validation Set...")
    model.eval()
    eval_losses = []
    
    with torch.no_grad():
        eval_steps = 100
        loader_iter = iter(val_loader)
        
        eval_gates = []
        for _ in range(eval_steps):
            try:
                X, Y = next(loader_iter)
            except StopIteration:
                loader_iter = iter(val_loader)
                X, Y = next(loader_iter)
                
            X, Y = X.to(config.device), Y.to(config.device)
            
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                # 注意这里我们依然传入 mode='standard'，它会自动触发 tgn.py 里的硬门控逻辑 (if not self.training:)
                if isinstance(model, nn.DataParallel):
                    logits, loss, gate = model(X, targets=Y, mode='standard')
                    loss = loss.mean()
                    gate = gate.mean()
                else:
                    logits, loss, gate = model(X, Y, mode='standard')
                
                # 过滤掉 NaN 的 loss，防止 PPL 爆炸
                if not math.isnan(loss.item()):
                    eval_losses.append(loss.item())
                if gate is not None:
                    eval_gates.append(gate.item())
                
    if len(eval_losses) > 0:
        final_eval_ppl = math.exp(np.mean(eval_losses))
    else:
        final_eval_ppl = float('inf')
    avg_eval_gate = np.mean(eval_gates) if eval_gates else 0.0
    print(f"--> {model_type.upper()} Final Eval PPL: {final_eval_ppl:.2f} (Gate Rate: {avg_eval_gate:.2%})\n")
    return final_eval_ppl, avg_eval_gate, model

if __name__ == "__main__":
    import sys
    
    # 允许通过命令行传参，默认跑 50M
    size = '50M'
    if len(sys.argv) > 1:
        size = sys.argv[1]
        
    print(f"==========================================")
    print(f"Starting Benchmark for size: {size}")
    print(f"==========================================")
    
    train_loader, val_loader = get_data_loaders(Config(size=size))
    
    results = {}
    gate_results = {}
    tgn_model = None
    tgn_hard_gate = 0.0
    for mt in [
        'tgn',
        # 'jamba', 
        # 'mamba'
        ]:
        final_ppl, final_gate, model = train_model(mt, train_loader, val_loader, size=size)
        results[mt] = final_ppl
        gate_results[mt] = final_gate
        if mt == 'tgn':
            tgn_model = model
            tgn_hard_gate = final_gate
        
    print("\n" + "="*40)
    print(f"FINAL WIKITEXT-103 PPL COMPARISON ({size})")
    print("="*40)
    for mt, ppl in results.items():
        print(f"{mt.upper():<10} | PPL: {ppl:.2f} | Gate Rate: {gate_results[mt]:.2%}")
        
    # 2. 第二步：在 TGN 自身上的消融
    print("\n" + "="*40)
    print(f"Phase 2: Ablation on TGN")
    print("="*40)
    
    eval_steps = 100
    loader_iter = iter(val_loader)
    config = Config(size=size)
    
    # Random Gating Baseline (Target same sparsity)
    print(f"\n>>> Running Evaluation for Random Gating (Sparsity: {tgn_hard_gate:.2%})...")
    random_losses = []
    with torch.no_grad():
        for _ in range(eval_steps):
            try:
                X, Y = next(loader_iter)
            except StopIteration:
                loader_iter = iter(val_loader)
                X, Y = next(loader_iter)
            X, Y = X.to(config.device), Y.to(config.device)
            
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                if isinstance(tgn_model, nn.DataParallel):
                    logits, loss, _ = tgn_model(X, targets=Y, mode='random', target_sparsity=tgn_hard_gate)
                    loss = loss.mean()
                else:
                    logits, loss, _ = tgn_model(X, Y, mode='random', target_sparsity=tgn_hard_gate)
                    
                if not math.isnan(loss.item()):
                    random_losses.append(loss.item())
                    
    random_ppl = math.exp(np.mean(random_losses)) if random_losses else float('inf')
    
    
    print("\n" + "="*40)
    print(f"FINAL ABLATION RESULTS ({size})")
    print("="*40)
    print(f"TGN (Hard, Sparsity={tgn_hard_gate:.2%}) | PPL: {results['tgn']:.2f}")
    print(f"Random (Sparsity={tgn_hard_gate:.2%})      | PPL: {random_ppl:.2f}")
    
    # Save final comparison results to CSV
    final_csv_path = os.path.join(config.out_dir, "final_results.csv")
    with open(final_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['model_type', 'ppl', 'gate_rate'])
        for mt in results:
            writer.writerow([mt, f"{results[mt]:.4f}", f"{gate_results[mt]:.4f}"])
            
    # Save ablation results to CSV
    ablation_csv_path = os.path.join(config.out_dir, "ablation_results.csv")
    with open(ablation_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['ablation_type', 'ppl', 'gate_rate'])
        writer.writerow(['tgn_hard', f"{results['tgn']:.4f}", f"{tgn_hard_gate:.4f}"])
        writer.writerow(['random', f"{random_ppl:.4f}", f"{tgn_hard_gate:.4f}"])
    
    print(f"\nExperiment complete. Check '{config.out_dir}' for detailed logs.")