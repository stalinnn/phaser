import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os

"""
EXP 101: 5-HT2A Psychedelic Simulation (Entropic Expansion)
-------------------------------------
Goal: Test the falsifiable prediction that 5-HT2A-induced entropic states
lead to Geometric Rank Hyper-expansion, distinguishing it from normal waking state.

Hypothesis:
- Anesthesia: Rank Collapse (Hypersynchrony/Silence).
- Awake: Critical Rank (Balanced).
- Psychedelic: Rank Expansion (Hyper-associative/Entropic).

Timeline:
0-500: Baseline (Normal Awake)
500-1500: Drug Onset (Entropic Forcing + Long Range Disinhibition)
1500-2000: Recovery
"""

def effective_rank(matrix):
    # matrix: [Time, Regions]
    if matrix.shape[0] < 2: return 1.0
    # Center
    matrix = matrix - matrix.mean(axis=0)
    # Covariance
    C = np.cov(matrix.T)
    # SVD
    try:
        U, S, V = np.linalg.svd(C)
        # Normalize
        p = S / (np.sum(S) + 1e-12)
        # Entropy
        h = -np.sum(p * np.log(p + 1e-12))
        return np.exp(h)
    except:
        return 1.0

class PsychedelicBrain:
    def __init__(self, N=100):
        self.N = N
        np.random.seed(42)
        
        # 1. Structural Connectome (Fixed SC)
        # Modular Small World
        self.SC = np.random.rand(N, N) * 0.05 # Weak background
        # Modules
        n_mod = 5
        size = N // n_mod
        for i in range(n_mod):
            # Intra-module strong
            self.SC[i*size:(i+1)*size, i*size:(i+1)*size] += 0.5
            
        np.fill_diagonal(self.SC, 0)
        
        # Long-Range Mask
        self.mask_long = np.ones((N, N))
        for i in range(n_mod):
            self.mask_long[i*size:(i+1)*size, i*size:(i+1)*size] = 0
            
    def simulate(self, steps=2000, dt=0.1):
        history = []
        x = np.random.randn(self.N) * 0.1
        
        # Metrics tracking
        ranks = []
        window = 100
        
        print("Running Psychedelic Simulation (Entropic Expansion)...")
        
        for t in tqdm(range(steps)):
            # Condition
            if 500 <= t < 1500:
                # Psychedelic State: 
                # 1. Increased excitability/entropy (REBUS model)
                # 2. Increased Global Functional Connectivity (Long Range Disinhibition)
                g_local = 0.5
                g_long = 0.8 # Stronger global integration
                noise_level = 1.2 # High entropy injection
            else:
                # Baseline State (Awake)
                g_local = 0.8 # Stronger local constraints
                g_long = 0.2 # Weak global
                noise_level = 0.4 # Moderate noise
                
            # Input computation
            # Local Drive 
            input_local = (self.SC * (1 - self.mask_long)) @ x
            
            # Global Drive
            input_long = (self.SC * self.mask_long) @ x
            
            # Dynamics
            # dx = -x + S(inputs)
            total_input = g_local * input_local + g_long * input_long + np.random.randn(self.N) * noise_level
            dx = -x + np.tanh(total_input)
            x = x + dx * dt
            
            history.append(x.copy())
            
            # Compute Rank periodically
            if len(history) > window:
                recent_data = np.array(history[-window:])
                r = effective_rank(recent_data)
                ranks.append(r)
            else:
                ranks.append(1.0) # Padding
                
        return np.array(history), np.array(ranks)

def run_experiment():
    brain = PsychedelicBrain(N=100)
    data, ranks = brain.simulate(steps=2000)
    
    # Plotting
    # Use Nature-like style
    plt.style.use('default')
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.figsize': (8, 10),
        'lines.linewidth': 2,
        'grid.alpha': 0.3
    })

    os.makedirs('figures', exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10), sharex=True)
    
    # 1. Raster Plot
    sns.heatmap(data.T, ax=ax1, cmap='magma', cbar=False, vmin=-2, vmax=2)
    ax1.set_title('Neural Activity Raster (N=100) - Entropic Expansion', fontweight='bold')
    ax1.set_ylabel('Brain Regions')
    ax1.axvline(500, color='cyan', linestyle='--')
    ax1.axvline(1500, color='cyan', linestyle='--')
    ax1.text(250, 50, 'Baseline', color='white', ha='center', fontweight='bold')
    ax1.text(1000, 50, 'Psychedelic State', color='white', ha='center', fontweight='bold')
    
    # 2. Geometric Rank
    t_axis = np.arange(len(ranks))
    ax2.plot(t_axis, ranks, 'm-', linewidth=2.5, label='Effective Geometric Rank')
    
    # Highlight
    ax2.axvspan(500, 1500, color='magenta', alpha=0.1, label='Drug Window')
    
    # Baseline mean
    base_mean = ranks[200:500].mean()
    ax2.axhline(base_mean, color='gray', linestyle='--', alpha=0.8, label=f'Baseline ({base_mean:.1f})')
    
    ax2.set_title('Geometric Manifold Hyper-Expansion', fontweight='bold')
    ax2.set_ylabel('Effective Rank ($R_{eff}$)')
    ax2.set_xlabel('Time (Simulation Steps)')
    ax2.legend(loc='lower right', frameon=False)
    ax2.grid(True, alpha=0.3)
    
    # Annotate peak
    peak_val = ranks[500:1500].max()
    peak_idx = 500 + ranks[500:1500].argmax()
    ax2.annotate(
        f'Hyper-Rank Peak\n({peak_val:.1f})',
        xy=(peak_idx, peak_val),
        xytext=(peak_idx+100, peak_val-1),
        arrowprops=dict(facecolor='black', shrink=0.05),
        fontsize=11
    )
    
    plt.tight_layout()
    save_path = 'figures/psychedelic_rank_expansion.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved figure to {save_path}")
    
    # Stats
    drug_mean = ranks[600:1400].mean()
    print(f"Baseline Rank: {base_mean:.2f}")
    print(f"Psychedelic Rank: {drug_mean:.2f}")
    print(f"Expansion Factor: {drug_mean/base_mean:.2f}x")

if __name__ == "__main__":
    run_experiment()
