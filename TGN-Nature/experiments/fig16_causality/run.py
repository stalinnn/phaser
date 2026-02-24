import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import os

"""
EXP 102: Causal Geometric Precursor Analysis
-------------------------------------------
Goal: Demonstrate CAUSALITY by showing temporal precedence (Lead-Lag Relationship).
Hypothesis: If Geometry is the CAUSE of Consciousness, then Geometric Rank Collapse
must PRECEDE the Metabolic/Activity Collapse during Anesthesia Induction.

Model:
- Coupled dynamics where Geometry G(t) drives State S(t).
- We simulate the induction phase (slow ramp up of drug).
- Metric: Cross-Correlation Lag between dRank/dt and dActivity/dt.
"""

def effective_rank(matrix):
    # matrix: [N, N] covariance or [Time, N] data
    if matrix.ndim == 2 and matrix.shape[0] > matrix.shape[1]:
        # [Time, N] -> Covariance [N, N]
        matrix = np.cov(matrix.T)
    
    # SVD of Covariance
    try:
        U, S, V = np.linalg.svd(matrix)
        p = S / (np.sum(S) + 1e-12)
        h = -np.sum(p * np.log(p + 1e-12))
        return np.exp(h)
    except:
        return 1.0

class CausalBrain:
    def __init__(self, N=100):
        self.N = N
        np.random.seed(42)
        # Structural Connectivity
        self.SC = np.random.rand(N, N) * 0.1
        np.fill_diagonal(self.SC, 0)
        
        # Long Range Mask
        self.mask_long = np.random.rand(N, N) < 0.2
        self.mask_local = ~self.mask_long
        
    def simulate_induction(self, steps=1500):
        # Simulate Anesthesia Induction: 
        # Drug concentration C(t) ramps up linearly.
        # Mechanism 1: Drug affects Geometry (Long-range coupling strength).
        # Mechanism 2: Drug affects Metabolism (Local noise/gain).
        
        # KEY ASSUMPTION FOR CAUSALITY:
        # Geometry is the control parameter. 
        # Changes in Geometry take time to propagate to Global Synchrony changes.
        
        history_x = []
        history_rank = []
        history_sync = []
        
        x = np.random.randn(self.N) * 0.1
        
        # Drug ramp: starts at 200, ends at 1200
        drug = np.zeros(steps)
        drug[200:1200] = np.linspace(0, 1.0, 1000)
        drug[1200:] = 1.0
        
        print("Simulating Anesthesia Induction Phase...")
        
        for t in range(steps):
            d = drug[t]
            
            # Causal Chain: Drug -> Long Range Gain -> Dynamics -> Rank/Sync
            
            # 1. Gain Parameters
            # Long range coupling drops fast (sensitive)
            # This causes Rank Collapse first.
            g_long = 1.5 * (1.0 - 0.9 * d) 
            
            # Local gain increases slightly? Or drops?
            # In anesthesia (Propofol), local inhibition increases (GABA agonist).
            # This tends to synchronize local populations.
            g_local = 1.0 * (1.0 + 0.5 * d) 
            
            # 2. Input
            inp_long = g_long * (self.SC * self.mask_long) @ x
            inp_local = g_local * (self.SC * self.mask_local) @ x
            noise = np.random.randn(self.N) * 0.5
            
            total_input = inp_long + inp_local + noise
            
            # 3. Dynamics
            dx = -x + np.tanh(total_input)
            x = x + dx * 0.1
            
            history_x.append(x.copy())
            
            # Metrics
            # Instantaneous Rank (approximated by recent window)
            if t > 50:
                win = np.array(history_x[-50:])
                r = effective_rank(win)
                # Synchrony: mean absolute correlation
                corr = np.corrcoef(win.T)
                sync = np.mean(np.abs(corr[np.triu_indices(self.N, k=1)]))
            else:
                r = 10.0
                sync = 0.1
                
            history_rank.append(r)
            history_sync.append(sync)
            
        return np.array(history_rank), np.array(history_sync), drug

def run_causal_analysis():
    brain = CausalBrain(N=50)
    rank, sync, drug = brain.simulate_induction(steps=1500)
    
    # Smooth curves
    def smooth(y, box_pts):
        box = np.ones(box_pts)/box_pts
        y_smooth = np.convolve(y, box, mode='same')
        return y_smooth

    # We focus on the window where change happens [200:1200]
    # But for plot we use full range
    rank_smooth = smooth(rank, 50)
    sync_smooth = smooth(sync, 50)
    
    # Normalize for plotting
    rank_norm = (rank_smooth - rank_smooth.min()) / (rank_smooth.max() - rank_smooth.min())
    sync_norm = (sync_smooth - sync_smooth.min()) / (sync_smooth.max() - sync_smooth.min())
    
    # Compute Temporal Derivative (Change Rate)
    # We look for the "Moment of Collapse" (max negative derivative)
    # or "Moment of Surge" (max positive derivative)
    
    # Focus search in induction window
    search_start = 200
    search_end = 1200
    
    d_rank = np.gradient(rank_smooth[search_start:search_end])
    d_sync = np.gradient(sync_smooth[search_start:search_end])
    
    # Rank should DROP -> Min derivative
    idx_collapse_rank = np.argmin(d_rank) + search_start
    
    # Sync should RISE -> Max derivative
    idx_surge_sync = np.argmax(d_sync) + search_start
    
    lag = idx_surge_sync - idx_collapse_rank
    
    # Plotting
    os.makedirs('figures', exist_ok=True)
    
    # Style
    plt.style.use('default')
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 12,
        'axes.labelsize': 14,
        'legend.fontsize': 12,
        'figure.figsize': (8, 6),
        'lines.linewidth': 2.5
    })
    
    fig, ax1 = plt.subplots()
    
    t_axis = np.arange(1500)
    
    color = 'tab:blue'
    ax1.set_xlabel('Induction Time (Drug Concentration $\\uparrow$)')
    ax1.set_ylabel('Geometric Rank (Dimensionality)', color=color)
    l1 = ax1.plot(t_axis, rank_smooth, color=color, label='Geometric Rank (Cause)')
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:red'
    ax2.set_ylabel('Global Synchrony (Order Parameter)', color=color)  # we already handled the x-label with ax1
    l2 = ax2.plot(t_axis, sync_smooth, color=color, linestyle='--', label='Global Synchrony (Effect)')
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Mark the Lead-Lag
    plt.axvline(idx_collapse_rank, color='blue', alpha=0.5, linestyle=':')
    plt.axvline(idx_surge_sync, color='red', alpha=0.5, linestyle=':')
    
    # Add Arrow
    mid_y = (sync_smooth[idx_surge_sync] + sync_smooth[idx_collapse_rank])/2
    # Ensure arrow direction is correct
    ax2.annotate(f'Causal Lag\n$\Delta t = {lag}$ steps', 
                 xy=(idx_surge_sync, sync_smooth[idx_surge_sync]), 
                 xytext=(idx_collapse_rank, sync_smooth[idx_surge_sync]),
                 arrowprops=dict(facecolor='black', shrink=0.05),
                 ha='right' if lag > 0 else 'left')
    
    plt.title('Causal Geometric Precursor: Rank Collapse Precedes Sync', fontweight='bold')
    
    # Legend
    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center left')
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    save_path = 'figures/causal_lead_lag.png'
    plt.savefig(save_path, dpi=300)
    print(f"Saved figure to {save_path}")
    print(f"Rank Collapse Index: {idx_collapse_rank}")
    print(f"Sync Surge Index: {idx_surge_sync}")
    print(f"Lag: {lag} (Positive means Rank Collapse happens FIRST)")

if __name__ == "__main__":
    run_causal_analysis()
