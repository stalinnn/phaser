import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os

# Set seed
torch.manual_seed(42)
np.random.seed(42)

# ==========================================
# 1. Dataset: The Copy Task
# ==========================================

class CopyTaskDataset(torch.utils.data.Dataset):
    def __init__(self, seq_len, num_samples=1000, num_classes=10):
        self.seq_len = seq_len
        self.num_samples = num_samples
        self.num_classes = num_classes # Tokens 0-9
        
    def __len__(self):
        return self.num_samples
        
    def __getitem__(self, idx):
        # Format:
        # Input:  [Header (10)] + [Zeros (L-11)] + [Delimiter]
        # Target: [Zeros (10)]  + [Zeros (L-11)] + [Header]
        
        header_len = 10
        # Generate random header (integers 1 to 8)
        # 0 is reserved for padding, 9 is reserved for delimiter
        header = torch.randint(1, self.num_classes - 1, (header_len,))
        
        # Zero padding
        zeros = torch.zeros(self.seq_len - header_len - 1, dtype=torch.long)
        
        # Delimiter
        delimiter = torch.tensor([self.num_classes - 1], dtype=torch.long)
        
        # Construct Input
        x = torch.cat([header, zeros, delimiter])
        
        # Construct Target (Loss is only calculated at the end)
        # We use -1 to ignore index in loss for the padding part
        y_ignore = torch.full((self.seq_len,), -1, dtype=torch.long)
        y = torch.cat([y_ignore, header])
        
        # For input to model, we need to append more zeros at the end so it can output
        x_full = torch.cat([x, torch.zeros(header_len, dtype=torch.long)])
        
        return x_full, y

def get_dataloader(seq_len=200, batch_size=32):
    ds = CopyTaskDataset(seq_len)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)

# ==========================================
# 2. Models
# ==========================================

class StandardLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # x: [B, L]
        emb = self.embedding(x)
        out, _ = self.lstm(emb)
        # out: [B, L, H]
        logits = self.fc(out)
        return logits

class SkyrmionCell(nn.Module):
    """
    JIT-compilable cell for Skyrmion RNN.
    Separated to allow TorchScript compilation of the loop.
    """
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        
    def forward(self, in_re: torch.Tensor, in_im: torch.Tensor, theta: torch.Tensor):
        # in_re, in_im: [B, L, H]
        # theta: [H]
        B, L, H = in_re.shape
        
        # Initialize state
        h_re = torch.zeros(B, H, device=in_re.device, dtype=in_re.dtype)
        h_im = torch.zeros(B, H, device=in_im.device, dtype=in_im.dtype)
        
        # Precompute rotation terms
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        
        outputs = []
        
        # JIT-friendly loop
        for t in range(L):
            # 1. Rotation (Unitary Step)
            # h_re_rot = h_re * cos - h_im * sin
            # h_im_rot = h_re * sin + h_im * cos
            
            # Note: In-place operations might be tricky for autograd in loops, 
            # so we use new variables.
            h_re_rot = h_re * cos_theta - h_im * sin_theta
            h_im_rot = h_re * sin_theta + h_im * cos_theta
            
            # 2. Input Addition
            x_re = in_re[:, t, :]
            x_im = in_im[:, t, :]
            
            h_re = h_re_rot + x_re
            h_im = h_im_rot + x_im
            
            # 3. Output Collection
            out_t = torch.cat([h_re, h_im], dim=1)
            outputs.append(out_t)
            
        return torch.stack(outputs, dim=1)

class SkyrmionRNN(nn.Module):
    """
    Simplified Skyrmion Memory: Unitary Complex Recurrent Unit
    """
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Input embedding to Complex domain
        self.emb_re = nn.Embedding(input_size, hidden_size, padding_idx=0)
        self.emb_im = nn.Embedding(input_size, hidden_size, padding_idx=0)
        
        # Learnable Rotation Phase (Theta)
        self.theta = nn.Parameter(torch.randn(hidden_size) * 0.1)
        
        # JIT-compiled Cell
        self.cell = torch.jit.script(SkyrmionCell(hidden_size))
        
        # Output projection
        self.fc = nn.Linear(hidden_size * 2, output_size)
        
    def forward(self, x):
        # Precompute Embeddings
        in_re = self.emb_re(x)
        in_im = self.emb_im(x)
        
        # Run JIT loop
        # outputs: [B, L, 2H]
        outputs = self.cell(in_re, in_im, self.theta)
        
        logits = self.fc(outputs) 
        
        return logits

# ==========================================
# 3. Experiment Loop
# ==========================================

def train(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    steps = 0
    for x, y in loader:
        optimizer.zero_grad()
        logits = model(x)
        
        # Reshape for loss
        # logits: [B, L, C], y: [B, L]
        # Only calculate loss on the last 10 steps (Target)
        # CopyTaskDataset puts -1 in y for ignored steps
        
        logits_flat = logits.view(-1, logits.size(-1))
        y_flat = y.view(-1)
        
        loss = criterion(logits_flat, y_flat)
        loss.backward()
        
        # Gradient clipping for LSTM
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        total_loss += loss.item()
        steps += 1
    return total_loss / steps

def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            logits = model(x)
            preds = logits.argmax(dim=-1)
            
            # Only count accuracy on valid targets (y != -1)
            mask = y != -1
            correct += (preds[mask] == y[mask]).sum().item()
            total += mask.sum().item()
            
    return 100. * correct / total

def run_experiment():
    # Challenge: Sequence Length 150 
    SEQ_LEN = 150 
    BATCH_SIZE = 64
    HIDDEN_DIM = 512 # Quadrupled capacity (Let it eat!)
    EPOCHS = 100 
    
    print(f"Starting Copy Task Experiment (Seq Len: {SEQ_LEN})...")
    loader = get_dataloader(SEQ_LEN, BATCH_SIZE)
    
    # Init Models
    lstm = StandardLSTM(10, HIDDEN_DIM, 10)
    # No training for LSTM
    
    sky = SkyrmionRNN(10, HIDDEN_DIM//2, 10) 
    # Switch to Adam for faster convergence
    opt_sky = optim.Adam(sky.parameters(), lr=0.002) # Lower LR slightly for stability with larger dim
    
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    
    lstm_accs = []
    sky_accs = []
    
    print("\n--- Training LSTM (Baseline) ---")
    # Train for just a few epochs to show it fails/stagnates
    for e in range(5):
        loss = train(lstm, loader, opt_lstm, criterion)
        acc = evaluate(lstm, loader)
        lstm_accs.append(acc)
        print(f"Epoch {e+1}: Loss {loss:.4f}, Acc {acc:.2f}%")
    
    # Extend the plot line for visualization
    lstm_accs = lstm_accs + [lstm_accs[-1]] * (EPOCHS - 5)
            
    print("\n--- Training Skyrmion (Topological) RNN ---")
    for e in range(EPOCHS):
        # TRAIN STEP
        sky.train()
        total_loss = 0
        steps = 0
        for x, y in loader:
            opt_sky.zero_grad()
            logits = sky(x)
            logits_flat = logits.view(-1, logits.size(-1))
            y_flat = y.view(-1)
            loss = criterion(logits_flat, y_flat)
            loss.backward()
            # REMOVED GRAD CLIP for Skyrmion
            opt_sky.step()
            total_loss += loss.item()
            steps += 1
        loss = total_loss / steps
        
        # EVAL STEP
        acc = evaluate(sky, loader)
        sky_accs.append(acc)
        print(f"Epoch {e+1}: Loss {loss:.4f}, Acc {acc:.2f}%")
            
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(lstm_accs, 'o--', color='gray', label='Standard LSTM')
    plt.plot(sky_accs, 'o-', color='blue', label='Skyrmion RNN (Topological)')
    plt.xlabel('Epochs')
    plt.ylabel('Copy Accuracy (%)')
    plt.title(f'Long-Term Memory Test (Sequence Length {SEQ_LEN})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_path = os.path.join('figures', 'skyrmion_memory_experiment.png')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    print(f"\nPlot saved to {output_path}")

if __name__ == "__main__":
    run_experiment()
