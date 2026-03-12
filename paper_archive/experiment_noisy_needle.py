import torch
import torch.nn as nn
import torch.nn.functional as F
from train_complex_tgn import ComplexTransformer, ComplexTGNLayer
from complex_layers import ComplexLinear
import math
import time
import random

# ==========================================
# 1. 高级数据生成器：带噪声的大海捞针
# ==========================================
class NoisyNeedleDataset(torch.utils.data.Dataset):
    def __init__(self, size=1000, seq_len=4096, noise_ratio=0.5, curriculum_step=0, total_steps=1000):
        self.size = size
        self.seq_len = seq_len
        self.noise_ratio = noise_ratio
        self.vocab_size = 1000 
        self.needle_trigger = [10, 20, 30] 
        self.needle_len = 4
        
        # Curriculum: progress from 0.0 (easiest, at end) to 1.0 (hardest, random)
        self.difficulty = min(1.0, curriculum_step / (total_steps * 0.5))

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        text = torch.randint(100, self.vocab_size, (self.seq_len,))
        
        noise_mask = torch.rand(self.seq_len) < self.noise_ratio
        text[noise_mask] = torch.randint(0, self.vocab_size, (noise_mask.sum(),))
        
        # Curriculum: 
        # Easy mode: insert near the end
        # Hard mode: insert anywhere
        max_idx = self.seq_len - self.needle_len - 1
        min_idx = int(max_idx * (1 - self.difficulty)) # If diff=0, min=max (end only)
        
        insert_idx = random.randint(min_idx, max_idx)
        
        target_code = random.randint(100, self.vocab_size - 1)
        needle = torch.tensor(self.needle_trigger + [target_code])
        
        text[insert_idx : insert_idx + self.needle_len] = needle
        
        input_ids = text[:-1]
        labels = torch.full((self.seq_len-1,), -100, dtype=torch.long)
        
        pred_pos = insert_idx + len(self.needle_trigger) - 1
        labels[pred_pos] = target_code
        
        return input_ids, labels

# ==========================================
# 2. 基线模型：Real Transformer (Same scale)
# ==========================================
class RealTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, n_layer, n_head, d_ff, max_len):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, dim_feedforward=d_ff, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layer)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device)
        x = self.emb(x) + self.pos(pos)
        # Causal Mask
        mask = nn.Transformer.generate_square_subsequent_mask(T).to(x.device)
        
        x = self.encoder(x, mask=mask, is_causal=True)
        return self.head(x)

# ==========================================
# 3. 实验主程序
# ==========================================
def run_noisy_needle_experiment():
    # --- Config ---
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running Noisy Needle Experiment on {DEVICE}")
    
    # Curriculum: Start Easy
    SEQ_LEN = 128      # Reduced from 1024
    VOCAB_SIZE = 1000
    NOISE_RATIO = 0.1  # Reduced from 0.6
    
    BATCH_SIZE = 32    # Increased for stability
    STEPS = 1000       # Increased steps
    LR = 1e-3          # Increased LR
    
    # Model Config 
    D_MODEL = 64       # Smaller dim for faster convergence on simple task
    N_LAYER = 2
    N_HEAD = 2
    D_FF = 256
    
    # --- Data ---
    train_ds = NoisyNeedleDataset(size=STEPS*BATCH_SIZE, seq_len=SEQ_LEN, noise_ratio=NOISE_RATIO)
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE)
    
    # --- Init Models ---
    print("Initializing Models...")
    # Real Baseline
    real_model = RealTransformer(VOCAB_SIZE, D_MODEL, N_LAYER, N_HEAD, D_FF, SEQ_LEN).to(DEVICE)
    # Complex TGN (Ours)
    comp_model = ComplexTransformer(VOCAB_SIZE, D_MODEL, N_LAYER, N_HEAD, D_FF, SEQ_LEN).to(DEVICE)
    
    opt_real = torch.optim.AdamW(real_model.parameters(), lr=LR)
    opt_comp = torch.optim.AdamW(comp_model.parameters(), lr=LR)
    
    criterion = nn.CrossEntropyLoss(ignore_index=-100) # 只计算 Needle 处的 Loss
    
    print(f"Real Params: {sum(p.numel() for p in real_model.parameters())/1e6:.2f}M")
    print(f"Comp Params: {sum(p.numel() for p in comp_model.parameters())/1e6:.2f}M")
    
    print("\nStarting Training (Goal: Retrieve 'Needle' from Heavy Noise)...")
    print(f"{'Step':<6} | {'Real Loss':<10} | {'Comp Loss':<10} | {'Real Acc':<10} | {'Comp Acc':<10} | Diff")
    print("-" * 75)
    
    start_time = time.time()
    
    # --- PHASED CURRICULUM: Overfit -> Slow Drift -> Generalize ---
    print("DEBUG: Starting Phased Curriculum...")
    
    # Initial fixed batch
    train_ds.difficulty = 0.0
    batch_inputs = []
    batch_labels = []
    for _ in range(BATCH_SIZE):
        bx, by = train_ds[0]
        batch_inputs.append(bx)
        batch_labels.append(by)
    x = torch.stack(batch_inputs).to(DEVICE)
    y = torch.stack(batch_labels).to(DEVICE)
    
    for step in range(STEPS):
        # Strategy:
        # Steps 0-200: Fix batch (Warmup)
        # Steps 200-600: Change batch every 50 steps (Slow Drift)
        # Steps 600+: Change batch every step (Full Online)
        
        if step < 200:
            pass # Keep x, y fixed
        elif step < 600:
            if step % 50 == 0:
                # Resample batch
                # Increase difficulty slightly
                train_ds.difficulty = min(1.0, (step - 200) / 400.0)
                batch_inputs, batch_labels = [], []
                for _ in range(BATCH_SIZE):
                    bx, by = train_ds[0]
                    batch_inputs.append(bx)
                    batch_labels.append(by)
                x = torch.stack(batch_inputs).to(DEVICE)
                y = torch.stack(batch_labels).to(DEVICE)
        else:
            # Full random per step
            train_ds.difficulty = 1.0
            batch_inputs, batch_labels = [], []
            for _ in range(BATCH_SIZE):
                bx, by = train_ds[0]
                batch_inputs.append(bx)
                batch_labels.append(by)
            x = torch.stack(batch_inputs).to(DEVICE)
            y = torch.stack(batch_labels).to(DEVICE)
        
        # 1. Train Real
        logits_real = real_model(x)
        # Reshape: (B*T, V) vs (B*T)
        loss_real = criterion(logits_real.view(-1, VOCAB_SIZE), y.view(-1))
        
        opt_real.zero_grad()
        loss_real.backward()
        # torch.nn.utils.clip_grad_norm_(real_model.parameters(), 1.0)
        opt_real.step()
        
        # 2. Train Complex
        logits_comp = comp_model(x)
        loss_comp = criterion(logits_comp.view(-1, VOCAB_SIZE), y.view(-1))
        
        opt_comp.zero_grad()
        loss_comp.backward()
        # torch.nn.utils.clip_grad_norm_(comp_model.parameters(), 1.0)
        opt_comp.step()
        
        if step % 20 == 0:
            # Calc Accuracy on Needle (Ignore -100)
            mask = y != -100
            
            # Real Acc
            pred_real = logits_real.argmax(dim=-1)
            acc_real = (pred_real[mask] == y[mask]).float().mean().item()
            
            # Comp Acc
            pred_comp = logits_comp.argmax(dim=-1)
            acc_comp = (pred_comp[mask] == y[mask]).float().mean().item()
            
            print(f"{step:<6} | {loss_real.item():<10.4f} | {loss_comp.item():<10.4f} | {acc_real:<10.2%} | {acc_comp:<10.2%} | Diff: {train_ds.difficulty:.2f}")

    print(f"\nExperiment Finished in {time.time()-start_time:.1f}s")

if __name__ == "__main__":
    run_noisy_needle_experiment()
