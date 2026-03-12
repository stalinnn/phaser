import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import os
import seaborn as sns

# ==========================================
# 统一风格设置 (NMI Standard)
# ==========================================
plt.style.use('default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'sans-serif']
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['xtick.major.width'] = 1.0
plt.rcParams['ytick.major.width'] = 1.0
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12

def plot_fig2_combined():
    # 1. Load Data
    path1 = "TGN-Nature/实验代码/fig2/training_log.csv"
    path2 = "TGN-Nature/实验代码/fig2/training_log_hotfix.csv"
    
    if not os.path.exists(path1): path1 = "result_fig2/training_log.csv"
    if not os.path.exists(path2): path2 = "result_fig2/training_log_hotfix.csv"
    
    df1 = pd.read_csv(path1)
    df2 = pd.read_csv(path2)
    
    # PPL Correction
    df1['ppl_corrected'] = np.exp(df1['loss'] / 2)
    df2['ppl_corrected'] = np.exp(df2['loss'] / 2)
    
    # Concatenate
    df = pd.concat([df1[df1['step'] <= 6000], df2]).sort_values('step')
    
    # Smooth data
    window = 50
    df['ppl_smooth'] = df['ppl_corrected'].rolling(window=window).mean()
    df['gate_smooth'] = df['gate_rate'].rolling(window=window).mean()
    
    # Create Figure (Unified Size: 15x5)
    fig = plt.figure(figsize=(15, 5), dpi=300)
    
    # --- Panel A: Dynamics (Left) ---
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = ax1.twinx()
    
    # Plot PPL (Left Axis)
    l1, = ax1.plot(df['step'], df['ppl_smooth'], color='#d62728', alpha=0.9, linewidth=2.5, label='Perplexity (PPL)')
    ax1.set_ylabel('Perplexity (Lower is Better)', color='#d62728', fontweight='bold')
    ax1.set_yscale('log')
    ax1.tick_params(axis='y', labelcolor='#d62728')
    
    # Plot Gate (Right Axis)
    l2, = ax2.plot(df['step'], df['gate_smooth'], color='#2ca02c', alpha=0.9, linewidth=2.5, label='Gate Rate (Sparsity)')
    ax2.set_ylabel('Gate Open Rate', color='#2ca02c', fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='#2ca02c')
    ax2.set_ylim(0, 0.6) # Focus on 0-60%
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    
    # Annotations
    ax2.annotate('Inertial Collapse\n(Low Cost Preference)', xy=(1000, 0.45), xytext=(1500, 0.55),
                 arrowprops=dict(facecolor='gray', shrink=0.05), fontsize=10)
    
    ax2.axvline(x=6000, color='gray', linestyle='--', alpha=0.5)
    ax2.text(6200, 0.55, 'Annealing Phase\n($\lambda \\uparrow$)', fontsize=11, color='gray')
    
    ax2.annotate('Forced Sparsification', xy=(7000, 0.35), xytext=(8000, 0.45),
                 arrowprops=dict(facecolor='#2ca02c', shrink=0.05), fontsize=10)

    ax1.set_xlabel('Training Steps', fontweight='bold')
    ax1.set_title('(a) Hysteretic Dynamics & Annealing', fontweight='bold', loc='left')
    ax1.grid(True, linestyle=':', alpha=0.4)
    
    # --- Panel B: A/B Test (Right) ---
    ax3 = fig.add_subplot(1, 2, 2)
    
    # Data
    methods = ['Adaptive TGN\n(Ours)', 'Random Baseline\n(Control)']
    ppls = [14.75, 974.99]
    colors = ['#2ca02c', '#7f7f7f']
    
    bars = ax3.bar(methods, ppls, color=colors, alpha=0.8, width=0.5)
    
    # Add values
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 10,
                 f'{height:.1f}',
                 ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Add Gap Arrow
    ax3.annotate('', xy=(0, 14.75), xytext=(0, 900),
                 arrowprops=dict(arrowstyle='<->', linewidth=2, color='red'))
    ax3.text(0.1, 450, 'Gap: +6500%', fontsize=12, color='red', fontweight='bold')
    
    ax3.set_ylabel('Perplexity (Lower is Better)', fontweight='bold')
    ax3.set_title(f'(b) Mechanism Verification\n(Sparsity = 33.4%)', fontweight='bold', loc='left')
    ax3.grid(axis='y', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('TGN-Nature/实验代码/fig2/fig2_final_nmi.png', dpi=300, bbox_inches='tight')
    print("Saved to TGN-Nature/实验代码/fig2/fig2_final_nmi.png")

if __name__ == "__main__":
    plot_fig2_combined()
