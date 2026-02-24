import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from pathlib import Path
import time
import argparse

def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class AssociativeRecallDataset(Dataset):
    """
    Task: Key-Value association with long delay.
    Input:  A:1 ... B:2 ... C:3 ... ? A
    Output: 1
    """
    def __init__(self, vocab_size=30, seq_len=50, num_samples=2000):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples
        
    def __len__(self):
        return self.num_samples
        
    def __getitem__(self, idx):
        # Generate random keys and values
        # We use simple integers as tokens
        # 0: pad, 1: query_marker, 2-vocab: symbols
        
        # Structure: Key Val ...noise... Marker Key -> Val
        
        key = np.random.randint(2, self.vocab_size, size=(1,))
        val = np.random.randint(2, self.vocab_size, size=(1,))
        
        # Noise
        noise_len = self.seq_len - 4
        noise = np.random.randint(2, self.vocab_size, size=(noise_len,))
        
        # Construct sequence
        # [Key, Val, ...noise..., Marker, Key]
        input_seq = np.concatenate([key, val, noise, np.array([1]), key])
        target = val[0]
        
        return torch.LongTensor(input_seq), torch.LongTensor([target])

class CausalSelfAttentionLast(nn.Module):
    """
    Causal self-attention, but computed ONLY for the last position.
    This is sufficient for the associative-recall setup where we only predict at the last step.
    Complexity is O(T·D) for projections + O(T) for attention weights (vs O(T^2) for full self-attn).
    """
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.key = nn.Linear(d_model, d_model)
        self.query = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, C = x.size()
        k = self.key(x)  # (B, T, C)
        v = self.value(x)  # (B, T, C)
        q_last = self.query(x[:, -1:, :])  # (B, 1, C)

        # Last-step causal mask is trivial (it can attend to all previous incl itself)
        wei = (q_last @ k.transpose(-2, -1)) * (C ** -0.5)  # (B, 1, T)
        wei = F.softmax(wei, dim=-1)

        out_last = wei @ v  # (B, 1, C)
        out_last = self.proj(out_last)  # (B, 1, C)
        return out_last, wei

class GeometricGate(nn.Module):
    """
    Learnable Thermodynamic Sensor.
    Instead of hardcoding rank, we let a small MLP look at the local state 
    and decide if it's 'confused' (High Entropy).
    """
    def __init__(self, d_model):
        super().__init__()
        # Input: Local hidden state
        # Output: Probability of activating Attention (Gate)
        self.net = nn.Sequential(
            nn.Linear(d_model, 16),
            nn.Tanh(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # x: [B, T, D]
        # We want a gate per token per batch
        return self.net(x) # [B, T, 1]

class ThermodynamicGatedNetwork(nn.Module):
    def __init__(self, vocab_size, d_model, model_type='tgn'):
        super().__init__()
        self.model_type = model_type
        self.d_model = d_model
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        # Positional Embedding (for Transformer baseline)
        self.pos_embedding = nn.Embedding(500, d_model)  # max seq len 500 for safety
        
        # Components
        self.rnn = nn.GRU(d_model, d_model, batch_first=True)  # Inertia Engine
        self.attn = CausalSelfAttentionLast(d_model)  # Plasma Engine (last-step only)
        self.gate_net = GeometricGate(d_model)  # Maxwell's Demon
        
        self.head = nn.Linear(d_model, vocab_size)
        
    def forward(self, x, hard_gate_inference: bool = False, gate_threshold: float = 0.5):
        # x: [B, T]
        x_emb = self.embedding(x)
        
        # 1. Inertia Pass (Always Run - Low Cost)
        # RNN captures local dependencies and history efficiently
        rnn_out, _ = self.rnn(x_emb) # [B, T, D]
        
        gate_values = None
        attn_weights = None
        
        if self.model_type == 'lstm':
            final_last = rnn_out[:, -1, :]
            
        elif self.model_type == 'transformer':
            # Add Positional Embeddings
            B, T = x.shape
            positions = torch.arange(0, T, device=x.device).unsqueeze(0)
            x_with_pos = x_emb + self.pos_embedding(positions)
            
            attn_last, attn_weights = self.attn(x_with_pos)  # (B,1,D)
            final_last = attn_last[:, 0, :]
            
        elif self.model_type == 'tgn':
            # Thermodynamic Gated Network (Ours)
            
            # 2. Demon Senses State
            # Gate determines: "Do I need to look back?"
            # gate -> 1 (Use Attention), gate -> 0 (Use RNN/Inertia)
            gate = self.gate_net(rnn_out) # [B, T, 1]
            gate_values = gate
            gate_last = gate[:, -1, :]  # [B, 1]
            
            # 3. Conditional Attention
            # Training: soft mixture. Inference: optional hard routing to realize compute savings.
            if hard_gate_inference and (not self.training):
                final_last = rnn_out[:, -1, :].clone()
                mask = (gate_last[:, 0] > gate_threshold)
                if mask.any():
                    idx = mask.nonzero(as_tuple=False).squeeze(-1)
                    attn_last_sub, attn_weights = self.attn((rnn_out[idx] + x_emb[idx]))  # (B',1,D)
                    attn_last_sub = attn_last_sub[:, 0, :]
                    # Residual + layer norm as in the demo
                    rescued = rnn_out[idx, -1, :] + attn_last_sub
                    rescued = F.layer_norm(rescued, (self.d_model,))
                    final_last[idx] = rescued
            else:
                attn_last, attn_weights = self.attn(rnn_out + x_emb)  # (B,1,D)
                attn_last = attn_last[:, 0, :]
                rnn_last = rnn_out[:, -1, :]
                final_last = (1 - gate_last) * rnn_last + gate_last * attn_last
            
        logits = self.head(final_last)  # Predict only at last step
        
        return logits, gate_values, attn_weights

@torch.no_grad()
def evaluate_accuracy(model, dataloader, device, hard_gate_inference: bool = False):
    model.eval()
    correct = 0
    total = 0
    gate_sum = 0.0
    gate_count = 0
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        y = y.squeeze()
        logits, gate, _ = model(X, hard_gate_inference=hard_gate_inference)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
        if gate is not None:
            gate_sum += gate.mean().item()
            gate_count += 1
    acc = correct / max(total, 1)
    avg_gate = gate_sum / max(gate_count, 1)
    return acc, avg_gate

def train_one_seed(seed: int, seq_len: int = 200, epochs: int = 60):
    set_seed(seed)
    vocab_size = 30
    # Long-range setting to stress RNN
    d_model = 32
    batch_size = 64
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    dataset = AssociativeRecallDataset(vocab_size, seq_len, num_samples=4000)
    # Train/test split
    n_train = 3000
    n_test = len(dataset) - n_train
    train_set, test_set = torch.utils.data.random_split(dataset, [n_train, n_test], generator=torch.Generator().manual_seed(seed))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    
    models = {
        'GRU (Inertia)': ThermodynamicGatedNetwork(vocab_size, d_model, 'lstm').to(device),
        'Transformer (Full Attention)': ThermodynamicGatedNetwork(vocab_size, d_model, 'transformer').to(device),
        'TGN (Ours)': ThermodynamicGatedNetwork(vocab_size, d_model, 'tgn').to(device)
    }
    
    history = {name: {'train_loss': [], 'test_acc': [], 'gate_usage': []} for name in models}
    
    # Loss penalty for using Attention (Energy Cost)
    # Lowered from 0.05 to 0.005 to allow TGN to learn long-range dependencies first
    energy_penalty = 0.005 
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            total_loss = 0
            total_gate = 0.0
            
            model.train()
            for X, y in train_loader:
                X, y = X.to(device), y.to(device)
                y = y.squeeze()
                
                optimizer.zero_grad()
                logits, gate, _ = model(X)
                
                # Main Task Loss
                task_loss = criterion(logits, y)
                
                # Energy Loss (Only for TGN)
                reg_loss = 0
                if name == 'TGN (Ours)' and gate is not None:
                    # We want gate to be 0 mostly, but 1 when needed.
                    # L1 regularization on gate activation
                    reg_loss = energy_penalty * torch.mean(gate)
                    total_gate += gate.mean().item()
                
                loss = task_loss + reg_loss
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / max(len(train_loader), 1)
            avg_gate = total_gate / max(len(train_loader), 1) if name == 'TGN (Ours)' else 0.0
            test_acc, _ = evaluate_accuracy(model, test_loader, device, hard_gate_inference=False)
            
            history[name]['train_loss'].append(avg_loss)
            history[name]['test_acc'].append(test_acc)
            history[name]['gate_usage'].append(avg_gate)
            
            if epoch % 10 == 0 or epoch == epochs - 1:
                print(f"Seed {seed} | {name} | Epoch {epoch}: TrainLoss={avg_loss:.4f}, TestAcc={test_acc:.4f}, Gate={avg_gate:.4f}")

    # Timing: compare inference-time hard gating vs others on a fixed batch
    # (This is a coarse wall-clock measure; still useful to sanity-check savings.)
    batch_X, _ = next(iter(test_loader))
    batch_X = batch_X.to(device)
    timing = {}
    for name, model in models.items():
        # Warmup
        for _ in range(5):
            _ = model(batch_X)
        torch.cuda.synchronize() if device.type == 'cuda' else None
        t0 = time.perf_counter()
        iters = 30
        for _ in range(iters):
            if name == 'TGN (Ours)':
                _ = model(batch_X, hard_gate_inference=True, gate_threshold=0.5)
            else:
                _ = model(batch_X)
        torch.cuda.synchronize() if device.type == 'cuda' else None
        dt = (time.perf_counter() - t0) / iters
        timing[name] = dt

    # Return model weights needed for visualization
    tgn_state = models['TGN (Ours)'].state_dict()
    return history, timing, tgn_state

def train_and_evaluate():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5, help="Number of random seeds (default: 5)")
    parser.add_argument("--epochs", type=int, default=60, help="Training epochs per seed (default: 60)")
    parser.add_argument("--seq-len", type=int, default=200, help="Sequence length (default: 200)")
    args = parser.parse_args()

    # Multi-seed evaluation for robustness
    seeds = list(range(args.seeds))
    seq_len = args.seq_len
    epochs = args.epochs

    all_histories = []
    all_timings = []
    last_tgn_state = None
    for s in seeds:
        print(f"\n=== Running seed {s} ===")
        hist, timing, tgn_state = train_one_seed(seed=s, seq_len=seq_len, epochs=epochs)
        all_histories.append(hist)
        all_timings.append(timing)
        last_tgn_state = tgn_state

    # Aggregate metrics
    model_names = list(all_histories[0].keys())
    summary = {}
    for name in model_names:
        final_accs = np.array([h[name]['test_acc'][-1] for h in all_histories], dtype=float)
        final_gates = np.array([h[name]['gate_usage'][-1] for h in all_histories], dtype=float)
        summary[name] = {
            'acc_mean': float(final_accs.mean()),
            'acc_std': float(final_accs.std(ddof=1) if len(final_accs) > 1 else 0.0),
            'gate_mean': float(final_gates.mean()),
            'gate_std': float(final_gates.std(ddof=1) if len(final_gates) > 1 else 0.0),
        }

    # --- Plotting Results (mean±std across seeds) ---
    out_dir = (Path(__file__).resolve().parents[1] / "figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / "architecture_validation.png"

    plt.style.use('seaborn-v0_8-paper')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    x = np.arange(epochs)
    for name in model_names:
        acc_curves = np.array([h[name]['test_acc'] for h in all_histories], dtype=float)  # [S,E]
        acc_mean = acc_curves.mean(axis=0)
        acc_std = acc_curves.std(axis=0, ddof=1) if len(seeds) > 1 else np.zeros_like(acc_mean)
        if name == 'TGN (Ours)':
            ax1.plot(x, acc_mean, label=f"{name} (mean)", linewidth=3, color='#e74c3c')
            ax1.fill_between(x, np.clip(acc_mean - acc_std, 0, 1), np.clip(acc_mean + acc_std, 0, 1), color='#e74c3c', alpha=0.15)
        else:
            ax1.plot(x, acc_mean, label=f"{name} (mean)", alpha=0.7, linestyle='--')
            ax1.fill_between(x, np.clip(acc_mean - acc_std, 0, 1), np.clip(acc_mean + acc_std, 0, 1), alpha=0.12)

    ax1.set_title(f"Associative Recall (SeqLen={seq_len}) — Test Accuracy", fontweight='bold')
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Accuracy")
    ax1.set_ylim(-0.05, 1.05)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Gate activation profile (single illustrative sample from the last seed)
    # This panel is illustrative; the table/summary gives mean±std gate usage.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vocab_size = 30
    d_model = 32
    demo_dataset = AssociativeRecallDataset(vocab_size, seq_len, num_samples=1)
    demo_x, _ = demo_dataset[0]
    demo_x = demo_x.unsqueeze(0).to(device)
    model = ThermodynamicGatedNetwork(vocab_size, d_model, 'tgn').to(device)
    if last_tgn_state is not None:
        model.load_state_dict(last_tgn_state)
    model.eval()
    with torch.no_grad():
        _, gate, _ = model(demo_x)
    gate_np = gate.squeeze().detach().cpu().numpy()

    ax2.plot(gate_np, color='#e74c3c', label='Gate Probability')
    ax2.set_title("TGN Gate Activation (Illustrative)", fontweight='bold')
    ax2.set_xlabel("Sequence Position")
    ax2.set_ylabel("Gate Probability (Open=1)")
    ax2.fill_between(range(len(gate_np)), gate_np, color='#e74c3c', alpha=0.2)
    ax2.axvline(x=0, color='gray', linestyle=':', label='Key/Value start')
    ax2.axvline(x=len(gate_np)-1, color='gray', linestyle=':', label='Query (last)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"\nSaved figure to {str(save_path)}")

    print("\n=== Final Summary (mean±std across seeds) ===")
    for name, stats in summary.items():
        print(f"{name}: TestAcc={stats['acc_mean']:.4f}±{stats['acc_std']:.4f}, Gate={stats['gate_mean']:.4f}±{stats['gate_std']:.4f}")

    print("\n=== Inference Timing (avg seconds/iter on one batch; hard-gated for TGN) ===")
    # aggregate timings
    for name in model_names:
        vals = np.array([t[name] for t in all_timings], dtype=float)
        print(f"{name}: {vals.mean():.6f}±{(vals.std(ddof=1) if len(vals)>1 else 0.0):.6f}")

    # --- Plotting Results ---
    return summary

if __name__ == "__main__":
    train_and_evaluate()
