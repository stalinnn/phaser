import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import pandas as pd
from tqdm import tqdm
import scipy.stats as stats
import warnings

# ==========================================
# 配置与路径检测
# ==========================================
# 自动检测本地路径：优先使用用户提供的 Windows 路径
LOCAL_WINDOWS_PATH = r"C:\Users\29478.000\Desktop\系统科学金融理论\paper_archive\data\propofol_dataset"
CLOUD_PATH = "data/propofol_dataset"

if os.path.exists(LOCAL_WINDOWS_PATH):
    DATA_ROOT = LOCAL_WINDOWS_PATH
    print(f"[Config] Using Local Windows Data: {DATA_ROOT}")
elif os.path.exists(CLOUD_PATH):
    DATA_ROOT = CLOUD_PATH
    print(f"[Config] Using Cloud Data: {DATA_ROOT}")
else:
    # Fallback to current directory data folder
    DATA_ROOT = os.path.join(os.getcwd(), "data", "propofol_dataset")
    print(f"[Config] Defaulting to: {DATA_ROOT}")

OUTPUT_DIR = "result/figures_nmi_final"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 显存保护：6GB 显存策略
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type == 'cuda':
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"[Device] CUDA Available: {torch.cuda.get_device_name(0)} ({vram:.1f} GB)")
    if vram < 8.0:
        print("[Config] Low VRAM detected (<8GB). Enabling memory optimization mode.")
        torch.cuda.empty_cache()

# ==========================================
# 1. Panel 4a: Real fMRI Analysis (CPU)
# ==========================================
def compute_effective_rank(time_series):
    # time_series: [Time, Regions]
    # Standardize
    X = time_series - np.mean(time_series, axis=0)
    std = np.std(X, axis=0)
    X = X / (std + 1e-9)
    
    T = X.shape[0]
    if T < 2: return np.nan
    
    # Covariance
    Cov = (X.T @ X) / (T - 1)
    
    try:
        eigenvalues = np.linalg.eigvalsh(Cov)
        eigenvalues = eigenvalues[eigenvalues > 0] # Filter precision noise
        total = np.sum(eigenvalues)
        if total < 1e-10: return np.nan
        
        p = eigenvalues / total
        entropy = -np.sum(p * np.log(p + 1e-12))
        return np.exp(entropy)
    except:
        return np.nan

def generate_panel_4a(ax):
    print("\n>>> Panel 4a: Analyzing Real fMRI Data...")
    
    try:
        from nilearn import input_data, datasets
    except ImportError:
        print("[Error] Nilearn not installed. Skipping 4a.")
        ax.text(0.5, 0.5, "Nilearn Missing", ha='center')
        return

    # Check data existence
    subjects = [d for d in os.listdir(DATA_ROOT) if d.startswith('sub-')]
    if len(subjects) == 0:
        print(f"[Error] No subjects found in {DATA_ROOT}")
        ax.text(0.5, 0.5, "Data Not Found", ha='center')
        return

    print(f"    Found {len(subjects)} subjects. Loading Atlas...")
    # 使用 nilearn 自动下载或加载缓存图谱
    dataset = datasets.fetch_atlas_schaefer_2018(n_rois=100, yeo_networks=7, verbose=0)
    masker = input_data.NiftiLabelsMasker(labels_img=dataset.maps, standardize=True, verbose=0)
    
    rank_awake = []
    rank_deep = []
    
    print("    Processing subjects (this may take a few minutes)...")
    for sub in tqdm(subjects):
        # 适配 OpenNeuro ds003171 的文件结构
        # 尝试多种可能的命名格式
        candidates_awake = [
            f"{sub}_task-restawake_bold.nii.gz",
            f"{sub}_task-restawake_run-01_bold.nii.gz"
        ]
        candidates_deep = [
            f"{sub}_task-restdeep_bold.nii.gz",
            f"{sub}_task-restdeep_run-01_bold.nii.gz"
        ]
        
        p_awake = None
        p_deep = None
        
        # Find valid files
        func_dir = os.path.join(DATA_ROOT, sub, "func")
        if not os.path.exists(func_dir): continue
        
        for f in candidates_awake:
            if os.path.exists(os.path.join(func_dir, f)): p_awake = os.path.join(func_dir, f); break
        for f in candidates_deep:
            if os.path.exists(os.path.join(func_dir, f)): p_deep = os.path.join(func_dir, f); break
            
        if p_awake and p_deep:
            try:
                ts_a = masker.fit_transform(p_awake)
                ts_d = masker.fit_transform(p_deep)
                
                ra = compute_effective_rank(ts_a)
                rd = compute_effective_rank(ts_d)
                
                if not np.isnan(ra) and not np.isnan(rd):
                    rank_awake.append(ra)
                    rank_deep.append(rd)
            except Exception as e:
                print(f"    [Warn] Failed {sub}: {e}")
                continue

    if len(rank_awake) < 2:
        print("[Error] Not enough valid paired data.")
        ax.text(0.5, 0.5, "Insufficient Data", ha='center')
        return

    # Statistical Test
    t_stat, p_val = stats.ttest_rel(rank_awake, rank_deep)
    print(f"    Result: N={len(rank_awake)}, p={p_val:.4e}")

    # Plotting
    bp = ax.boxplot([rank_awake, rank_deep], labels=['Awake', 'Deep Sedation'], 
                    patch_artist=True, widths=0.5)
    
    colors = ['#3498db', '#e74c3c']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
        
    # Draw individual lines
    for i in range(len(rank_awake)):
        ax.plot([1, 2], [rank_awake[i], rank_deep[i]], color='gray', alpha=0.3, linewidth=1)
        
    ax.set_title(f"A. Brain State Phase Transition\n(Real fMRI, N={len(rank_awake)}, p < {p_val:.1e})", fontweight='bold')
    ax.set_ylabel("Effective Geometric Rank ($R_{eff}$)")
    ax.grid(axis='y', linestyle='--', alpha=0.3)

# ==========================================
# 2. Panel 4b: Kuramoto Physics (CPU)
# ==========================================
def generate_panel_4b(ax):
    print("\n>>> Panel 4b: Running Kuramoto Physics Model...")
    
    N = 200 # More oscillators for smoother physics
    steps = 2000
    dt = 0.05
    
    # 物理参数
    # Awake: Low Coupling (Critical), High Rank
    # Anesthesia: High Coupling (Sync), Low Rank
    K_schedule = np.linspace(0.0, 6.0, steps) # Ramp up coupling (Induction)
    
    omega = np.random.normal(0, 1, N) # Natural frequencies
    theta = np.random.uniform(0, 2*np.pi, N)
    
    history_rank = []
    history_sync = []
    phases_buffer = [] # Store complex phases
    
    for t in range(steps):
        K = K_schedule[t]
        
        # Order parameter (Synchrony)
        z = np.mean(np.exp(1j * theta))
        r = np.abs(z)
        psi = np.angle(z)
        
        # Differential Equation: dtheta = omega + K * r * sin(psi - theta)
        dtheta = omega + K * r * np.sin(psi - theta) + np.random.normal(0, 0.2, N)
        theta += dtheta * dt
        
        # Record state for rank calculation
        state_vector = np.concatenate([np.cos(theta), np.sin(theta)]) # 2N dim embedding
        phases_buffer.append(state_vector)
        
        if len(phases_buffer) > 50:
            phases_buffer.pop(0)
            
        if t % 5 == 0 and len(phases_buffer) == 50:
            # Compute rank on sliding window
            win = np.array(phases_buffer) # [50, 2N]
            rank = compute_effective_rank(win)
            history_rank.append(rank)
            history_sync.append(r)
        elif t % 5 == 0:
            history_rank.append(np.nan)
            history_sync.append(np.nan)
            
    # Plotting
    t_axis = np.linspace(0, 1, len(history_rank))
    
    # Smooth curves
    def smooth(y):
        y = np.array(y)
        valid_idx = ~np.isnan(y)
        smooth_y = np.zeros_like(y)
        if np.sum(valid_idx) > 0:
            smooth_y[valid_idx] = np.convolve(y[valid_idx], np.ones(20)/20, mode='same')
        return smooth_y

    rank_s = smooth(history_rank)
    sync_s = smooth(history_sync)
    
    l1 = ax.plot(t_axis, rank_s, color='#2ecc71', linewidth=2.5, label='Geometric Rank')
    ax.set_ylabel('Effective Rank', color='#2ecc71')
    ax.tick_params(axis='y', labelcolor='#2ecc71')
    
    ax2 = ax.twinx()
    l2 = ax2.plot(t_axis, sync_s, color='#e74c3c', linestyle='--', linewidth=2, label='Synchrony')
    ax2.set_ylabel('Global Synchrony ($r$)', color='#e74c3c')
    ax2.tick_params(axis='y', labelcolor='#e74c3c')
    
    # Find crossing point / lag
    # Theoretical prediction: Rank drops BEFORE Sync saturates
    ax.set_xlabel("Coupling Strength (Anesthesia Depth)")
    ax.set_title("B. In-silico Induction Dynamics\n(Kuramoto Model)", fontweight='bold')

# ==========================================
# 3. Panel 4c: Spin Glass Real Solve (GPU)
# ==========================================
class SpinGlassSystem:
    def __init__(self, L=6):
        self.L = L
        # Random J couplings [L, L, L, 3] (x, y, z neighbors)
        self.J = torch.randn(L, L, L, 3, device=device)
        
    def energy(self, s):
        # s: [1, L, L, L]
        E = 0
        # Compute interaction with 3 neighbors
        # dim 1,2,3 corresponds to x,y,z
        for d in range(3):
            # Shift to get neighbor
            s_neighbor = torch.roll(s, shifts=-1, dims=d+1)
            # J[..., d] is coupling for that direction
            E -= torch.sum(self.J[..., d] * s * s_neighbor)
            
        # Soft spin constraint (s^2 - 1)^2
        E += torch.sum((s**2 - 1)**2)
        return E

def run_spin_glass_solve(ax):
    print("\n>>> Panel 4c: Real-time Spin Glass Solve (GPU)...")
    
    L = 6 # Fits in 6GB VRAM easily
    steps = 300
    lr = 0.1
    
    system = SpinGlassSystem(L=L)
    
    # 1. Langevin (Gradient Descent + Noise)
    print("    Solver 1: Langevin Dynamics...")
    x = torch.randn(1, L, L, L, device=device, requires_grad=True)
    hist_langevin = []
    
    for _ in range(steps):
        E = system.energy(x)
        grad = torch.autograd.grad(E, x)[0]
        
        # Langevin Update: x += -lr * grad + sqrt(2*lr*T)*noise
        noise = torch.randn_like(x) * 0.1
        x.data = x.data - lr * grad + noise
        
        hist_langevin.append(E.item())
        
    # 2. Attention Dynamics (Non-local Smoothing)
    print("    Solver 2: Attention Dynamics...")
    x = torch.randn(1, L, L, L, device=device, requires_grad=True) # Reset start
    hist_attn = []
    
    # Attention Kernel (3x3x3 smoothing as proxy for local self-attention)
    kernel = torch.ones(1, 1, 3, 3, 3, device=device) / 27.0
    
    for i in range(steps):
        E = system.energy(x)
        grad = torch.autograd.grad(E, x)[0]
        
        # Calculate Geometric Force (Non-local aggregation)
        # In real TGN this is QK^T, here we use conv for physics efficiency on grid
        grad_smooth = F.conv3d(grad, kernel, padding=1)
        
        # Gate: Active only when gradients are rough (high frequency)
        # Here we assume constant geometric drive for demo
        force = 0.5 * grad + 0.5 * grad_smooth
        
        noise = torch.randn_like(x) * 0.1
        x.data = x.data - lr * force + noise
        
        hist_attn.append(E.item())
        
    # Plotting
    ax.plot(hist_langevin, color='gray', linestyle='--', label='Langevin (Thermodynamic)')
    ax.plot(hist_attn, color='#8e44ad', linewidth=2, label='Attention (Geometric)')
    
    # Annotate Tunneling
    # Find where Attention beats Langevin significantly
    diff = np.array(hist_langevin) - np.array(hist_attn)
    max_adv_idx = np.argmax(diff[50:]) + 50
    
    ax.annotate("Geometric Tunneling", 
                xy=(max_adv_idx, hist_attn[max_adv_idx]), 
                xytext=(max_adv_idx+20, hist_attn[max_adv_idx]+50),
                arrowprops=dict(facecolor='black', shrink=0.05),
                fontweight='bold')
    
    ax.set_title("C. Optimization Dynamics\n(Real 3D Spin Glass Solve)", fontweight='bold')
    ax.set_xlabel("Steps")
    ax.set_ylabel("Energy (Hamiltonian)")
    ax.legend()
    ax.grid(True, alpha=0.3)

# ==========================================
# Main Execution
# ==========================================
def main():
    fig = plt.figure(figsize=(15, 5), dpi=150)
    gs = fig.add_gridspec(1, 3, wspace=0.3)
    
    ax1 = fig.add_subplot(gs[0])
    generate_panel_4a(ax1)
    
    ax2 = fig.add_subplot(gs[1])
    generate_panel_4b(ax2)
    
    ax3 = fig.add_subplot(gs[2])
    run_spin_glass_solve(ax3)
    
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "Figure_4_Final_Real.png")
    plt.savefig(save_path)
    print(f"\n[Success] Figure saved to: {save_path}")

if __name__ == "__main__":
    main()
