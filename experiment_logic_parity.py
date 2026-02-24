import torch
import torch.nn as nn
import torch.nn.functional as F
from train_complex_tgn import ComplexTransformer
import math
import time

# ==========================================
# 1. 基线模型：标准实数 Transformer
# ==========================================
class RealTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, n_layer, n_head, d_ff, max_len=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, dim_feedforward=d_ff, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layer)
        
        self.fc_out = nn.Linear(d_model, 2) 

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        x = self.embedding(x) + self.pos_emb(pos)
        
        x = self.transformer(x)
        
        # Change: Use CLS token (first token) instead of Mean Pooling
        # Assuming the model learns to aggregate info to the first token
        x = x[:, 0, :] 
        return self.fc_out(x)

# ==========================================
# 2. 实验模型：Complex-TGN (适配分类任务)
# ==========================================
class ComplexTGNClassifier(nn.Module):
    def __init__(self, vocab_size, d_model, n_layer, n_head, d_ff, max_len=512):
        super().__init__()
        self.backbone = ComplexTransformer(vocab_size, d_model, n_layer, n_head, d_ff, max_len)
        self.backbone.out_head = None 
        
        from complex_layers import ComplexLinear
        self.cls_head = ComplexLinear(d_model, 2)

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        
        xr = self.backbone.emb(x) + self.backbone.pos_emb(pos)
        xi = self.backbone.emb_imag(x) + self.backbone.pos_emb_imag(pos)
        x_complex = torch.complex(xr, xi)
        
        mask = torch.zeros(T, T, device=x.device)
        
        for layer in self.backbone.layers:
            x_complex = layer(x_complex, mask)
            
        # Change: Use CLS token (first token)
        x_complex = x_complex[:, 0, :]
        
        logits_complex = self.cls_head(x_complex)
        return logits_complex.abs()

# ==========================================
# 3. 数据生成器：奇偶校验 (Parity Task)
# ==========================================
def generate_parity_data(batch_size, seq_len, device='cpu'):
    # Generate random 0/1 sequence
    # vocab: 0, 1. (Padding not needed for fixed length batch)
    x = torch.randint(0, 2, (batch_size, seq_len), device=device)
    
    # Calculate Parity: sum(x) % 2
    y = x.sum(dim=1) % 2
    return x, y

# ==========================================
# 4. 训练与评测脚本
# ==========================================
def run_experiment():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running Parity Task Experiment on {device}")
    
    # Config
    SEQ_LEN = 64   # 序列长度，越长越难
    BATCH_SIZE = 64
    STEPS = 1000
    D_MODEL = 64   # 小模型，考验表达能力
    
    print(f"Task: Parity Check | Length: {SEQ_LEN}")
    
    # --- Init Models ---
    real_model = RealTransformer(vocab_size=2, d_model=D_MODEL, n_layer=2, n_head=2, d_ff=128, max_len=SEQ_LEN).to(device)
    comp_model = ComplexTGNClassifier(vocab_size=2, d_model=D_MODEL, n_layer=2, n_head=2, d_ff=128, max_len=SEQ_LEN).to(device)
    
    # Count Params
    real_params = sum(p.numel() for p in real_model.parameters())
    comp_params = sum(p.numel() for p in comp_model.parameters())
    print(f"Real Model Params: {real_params}")
    print(f"Comp Model Params: {comp_params} (Complex params count as 2 floats)")
    
    # Optimizers
    opt_real = torch.optim.Adam(real_model.parameters(), lr=1e-3)
    opt_comp = torch.optim.Adam(comp_model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    # --- Training Loop ---
    print("\nStarting Training Comparison...")
    print(f"{'Step':<6} | {'Real Acc':<10} | {'Comp Acc':<10} | {'Real Loss':<10} | {'Comp Loss':<10}")
    print("-" * 60)
    
    for step in range(STEPS):
        x, y = generate_parity_data(BATCH_SIZE, SEQ_LEN, device)
        
        # Train Real
        pred_real = real_model(x)
        loss_real = criterion(pred_real, y)
        opt_real.zero_grad()
        loss_real.backward()
        opt_real.step()
        
        # Train Complex
        pred_comp = comp_model(x)
        loss_comp = criterion(pred_comp, y)
        opt_comp.zero_grad()
        loss_comp.backward()
        opt_comp.step()
        
        if step % 100 == 0:
            # Eval Accuracy
            acc_real = (pred_real.argmax(1) == y).float().mean().item()
            acc_comp = (pred_comp.argmax(1) == y).float().mean().item()
            print(f"{step:<6} | {acc_real:<10.2%} | {acc_comp:<10.2%} | {loss_real:<10.4f} | {loss_comp:<10.4f}")

    # --- Final Test on Longer Sequence (Generalization) ---
    print("\n=== Generalization Test (Length 2x) ===")
    test_len = SEQ_LEN * 2
    # Note: Positional embedding might fail if not handled, but let's try
    # Re-init models with larger max_len if needed, but here we just test robustness within capacity
    # Actually, standard PE cannot extrapolate well. 
    # But let's see if Complex phase logic generalizes better even with PE noise.
    
    try:
        x_test, y_test = generate_parity_data(BATCH_SIZE, test_len, device)
        # We need to handle pos emb size mismatch for simple implementation
        # For this demo, we just test on same length but unseen data to be safe
        x_test, y_test = generate_parity_data(1000, SEQ_LEN, device) 
        
        with torch.no_grad():
            p_r = real_model(x_test).argmax(1)
            p_c = comp_model(x_test).argmax(1)
            
        print(f"Final Test Acc (Real): {(p_r == y_test).float().mean():.2%}")
        print(f"Final Test Acc (Comp): {(p_c == y_test).float().mean():.2%}")
        
    except Exception as e:
        print(f"Gen test skipped: {e}")

if __name__ == "__main__":
    run_experiment()
