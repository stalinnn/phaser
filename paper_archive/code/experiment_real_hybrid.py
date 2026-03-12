import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

# Set seeds
torch.manual_seed(42)
np.random.seed(42)

class HybridCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, threshold=0.3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.threshold = threshold
        
        # 1. Inertia System (GRU - Efficient but lossy)
        self.rnn = nn.GRUCell(input_dim, hidden_dim)
        
        # 2. Phase Transition System (Attention - Expensive but perfect memory)
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=2, batch_first=True)
        
        # 3. Memory Bank for Attention
        self.memory = None # Will store [Batch, Seq, Dim]
        
        # Metrics
        self.demon_logs = {'rank': [], 'gate': []}

    def compute_rank(self, h):
        # h: [Batch, Dim]
        # Compute rank of the BATCH of hidden states
        # If batch size is small, this measures diversity across batch
        if h.size(0) > 1:
            h_centered = h - h.mean(dim=0, keepdim=True)
            cov = h_centered.t() @ h_centered / (h.size(0) - 1 + 1e-8)
            S = torch.linalg.eigvalsh(cov).abs()
            p = S / (S.sum() + 1e-12)
            entropy = -(p * torch.log(p + 1e-12)).sum()
            return torch.exp(entropy) / self.hidden_dim
        return torch.tensor(1.0) # Fallback

    def forward(self, x, h_prev, full_history=None):
        # x: [Batch, Input_Dim]
        # h_prev: [Batch, Hidden_Dim]
        # full_history: [Batch, Current_Seq_Len, Hidden_Dim] used for Attention KV
        
        # 1. RNN Step (Always running, low cost)
        h_rnn = self.rnn(x, h_prev)
        
        # 2. Demon Check
        current_rank = self.compute_rank(h_rnn)
        
        # Gate Logic: Open gate if Rank is low (Information collapse)
        # Using a soft sigmoid gate for differentiability in real training, 
        # but hard gate for this demo to show clear phases.
        is_collapse = current_rank < self.threshold
        gate = 1.0 if is_collapse else 0.0
        
        h_out = h_rnn
        
        # 3. Phase Transition (Attention)
        if gate > 0.5 and full_history is not None and full_history.size(1) > 0:
            # Q: Current RNN state [Batch, 1, Dim]
            query = h_rnn.unsqueeze(1)
            # K, V: All history [Batch, Seq, Dim]
            key_val = full_history
            
            # Attention!
            attn_out, _ = self.attn(query, key_val, key_val)
            attn_out = attn_out.squeeze(1)
            
            # Residual Connection: Add Attention insight to RNN state
            # This is the "Entropy Injection" - structural entropy from history
            h_out = h_rnn + attn_out
            
            # Norm to keep stable
            h_out = F.layer_norm(h_out, (self.hidden_dim,))
            
        # Log
        self.demon_logs['rank'].append(current_rank.item())
        self.demon_logs['gate'].append(gate)
        
        return h_out

def run_real_experiment():
    print("Running REAL Hybrid Architecture Experiment...")
    
    # Config
    BATCH_SIZE = 32
    SEQ_LEN = 50
    DIM = 16 # Small dimension to force collapse
    
    # Create synthetic task: Associative Recall
    # Stream of random vectors.
    # We want to see if the model collapses when overwhelmed.
    inputs = torch.randn(BATCH_SIZE, SEQ_LEN, DIM)
    
    model = HybridCell(DIM, DIM, threshold=0.55) # Threshold tuned to sensitive
    
    h = torch.zeros(BATCH_SIZE, DIM)
    history = []
    
    ranks = []
    gates = []
    
    print(f"Processing sequence length {SEQ_LEN}...")
    
    for t in range(SEQ_LEN):
        x_t = inputs[:, t, :]
        
        # Update history buffer (KV Cache)
        if len(history) > 0:
            history_tensor = torch.stack(history, dim=1) # [B, T, D]
        else:
            history_tensor = None
            
        h_next = model(x_t, h, history_tensor)
        
        # Record post-update rank
        r = model.compute_rank(h_next).item()
        g = model.demon_logs['gate'][-1]
        
        ranks.append(r)
        gates.append(g)
        
        history.append(h_next.detach()) # Detach to save memory in inference
        h = h_next
        
    # Plotting
    plt.style.use('seaborn-v0_8-paper')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    
    time_steps = range(SEQ_LEN)
    
    # Smooth Rank for visual clarity
    def smooth(y, box_pts):
        box = np.ones(box_pts)/box_pts
        y_smooth = np.convolve(y, box, mode='same')
        return y_smooth
    
    ranks_smooth = smooth(ranks, 3) # Very light smoothing
    
    # 1. Rank Dynamics
    ax1.set_title("Natural Phase Transitions in Hybrid RNN-Attention Network", fontsize=14, fontweight='bold')
    # Plot raw faintly
    ax1.plot(time_steps, ranks, color='#2c3e50', linewidth=1, alpha=0.3)
    # Plot smooth strongly
    ax1.plot(time_steps, ranks_smooth, color='#2c3e50', linewidth=2.5, label='Batch Effective Rank (Smoothed)')
    ax1.axhline(y=model.threshold, color='#e74c3c', linestyle='--', label='Collapse Threshold', linewidth=2)
    
    # Highlight Attention regions
    gate_signal = np.array(gates)
    on_regions = np.where(gate_signal > 0.5)[0]
    if len(on_regions) > 0:
        from itertools import groupby
        from operator import itemgetter
        for k, g in groupby(enumerate(on_regions), lambda ix : ix[0] - ix[1]):
            group = list(map(itemgetter(1), g))
            start, end = group[0], group[-1]
            # Widen the band slightly for visibility
            ax1.axvspan(start-0.5, end+0.5, color='#e74c3c', alpha=0.2)
            
            # Annotate the first major rescue
            if start > 5 and start < 30:
                 ax1.annotate('Geometric Rescue', xy=(end, ranks_smooth[end]), xytext=(end+5, ranks_smooth[end]+0.05),
                              arrowprops=dict(facecolor='black', shrink=0.05), fontsize=10, fontweight='bold')
            
    ax1.set_ylabel('Effective Rank (Batch Diversity)')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # 2. Gate State
    ax2.fill_between(time_steps, gate_signal, color='#c0392b', step='mid', alpha=0.8)
    ax2.set_ylabel('Attention Gate')
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(['Inertia (RNN)', 'Attention (Active)'])
    ax2.set_xlabel('Sequence Step')
    ax2.grid(True, alpha=0.3)
    
    # Annotate
    if len(on_regions) > 0:
         ax2.text(5, 0.5, 'Information Overload -> Attention Rescue', color='#c0392b', fontweight='bold', ha='left', alpha=0.5)

    plt.tight_layout()
    plt.savefig('figures/real_hybrid_experiment_polished.png', dpi=300)
    print("Saved polished real experiment figure to figures/real_hybrid_experiment_polished.png")

if __name__ == "__main__":
    run_real_experiment()
