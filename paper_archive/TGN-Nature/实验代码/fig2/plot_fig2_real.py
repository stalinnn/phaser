import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

# NMI Style
plt.style.use('default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['xtick.major.width'] = 1.5
plt.rcParams['ytick.major.width'] = 1.5
plt.rcParams['font.size'] = 12

def plot_fig2_real():
    fig = plt.figure(figsize=(14, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1])
    
    # ==========================================
    # Panel A: Evolution of Intelligence (Gate vs PPL)
    # ==========================================
    ax1 = fig.add_subplot(gs[0])
    
    checkpoints = ['Phase 1\n(Ckpt 2000)', 'Phase 2\n(Hotfix 0.02)', 'Phase 3\n(Hotfix 0.05)']
    gates = [45.79, 34.40, 31.40]
    ppls_lr = [15.64, 14.55, 14.57]
    
    # Plot TGN Curve
    ax1.plot(gates, ppls_lr, 'o-', color='#e74c3c', linewidth=3, markersize=10, label='TGN (Adaptive)')
    
    # Plot Mamba Line (as a Line Plot, not axhline)
    # Create a line spanning the X-axis range
    x_mamba_line = np.linspace(28, 48, 100)
    y_mamba_line = [28.62] * 100 
    ax1.plot(x_mamba_line, y_mamba_line, '--', color='#9b59b6', linewidth=2, label='Mamba Baseline')
    
    # Add Random Baseline (from run_final_eval, PPL ~310, off-chart but we show direction)
    # Just to show it's bad
    # ax1.arrow(35, 25, 0, 5, head_width=1, head_length=1, fc='gray', ec='gray')
    # ax1.text(35, 26, 'Random Mix\n(PPL > 300)', color='gray', ha='center')
    
    # Smart Annotation (Manual Offset)
    # Phase 1: Top-Left
    ax1.annotate(checkpoints[0], (gates[0], ppls_lr[0]), xytext=(-10, 15), textcoords='offset points', 
                 ha='center', fontsize=10, fontweight='bold')
                 
    # Phase 2: Bottom
    ax1.annotate(checkpoints[1], (gates[1], ppls_lr[1]), xytext=(0, -35), textcoords='offset points', 
                 ha='center', fontsize=10, fontweight='bold')
                 
    # Phase 3: Left
    ax1.annotate(checkpoints[2], (gates[2], ppls_lr[2]), xytext=(-40, -10), textcoords='offset points', 
                 ha='right', fontsize=10, fontweight='bold')
        
    ax1.invert_xaxis()
    
    # Adjust Limits
    ax1.set_xlim(48, 28) 
    ax1.set_ylim(10, 35) # Adjusted to fit Mamba (28.6) comfortably at top
    
    ax1.set_xlabel('Gate Activation Rate (%)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Long-Range PPL (Lower is Better)', fontsize=14, fontweight='bold')
    ax1.set_title('A. Evolution of Geometric Intelligence', fontsize=16, pad=15)
    ax1.legend(loc='lower right', frameon=True, framealpha=0.9) 
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    # Optimized Arrow
    ax1.annotate('Sparsity Annealing', xy=(33, 15.0), xytext=(42, 17.0),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8), 
                 fontsize=12, fontweight='bold')

    # ==========================================
    # Panel B: The Pareto Breakthrough
    # ==========================================
    ax2 = fig.add_subplot(gs[1])
    
    # Data
    x_mamba, y_mamba = 0, 1/3.3542
    x_tgn, y_tgn = 31.40, 1/2.6793
    x_tf, y_tf = 100, 1/2.65
    
    # Scatter
    ax2.scatter([x_mamba], [y_mamba], s=200, c='#9b59b6', marker='s', label='Mamba')
    ax2.scatter([x_tgn], [y_tgn], s=400, c='#e74c3c', marker='*', label='TGN (Ours)', zorder=10)
    ax2.scatter([x_tf], [y_tf], s=200, c='#3498db', marker='D', label='Transformer (Full)')
    ax2.scatter([31.40], [1/5.0], s=100, c='gray', marker='X', label='Random Mix')
    
    # Frontier
    ax2.plot([x_mamba, x_tgn, x_tf], [y_mamba, y_tgn, y_tf], '--', color='gray', alpha=0.3)
    ax2.plot([x_mamba, x_tf], [y_mamba, y_tf], ':', color='red', alpha=0.5, label='Linear Trade-off')
    
    ax2.set_xlabel('Computational Cost (Gate %)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Recall Efficiency (1/Loss)', fontsize=14, fontweight='bold')
    ax2.set_title('B. Breaking the Pareto Frontier', fontsize=16, pad=15)
    ax2.legend(loc='lower right')
    ax2.grid(True, linestyle=':', alpha=0.6)
    
    # Shaded Region
    ax2.fill_between([0, 31.4, 100], [y_mamba, y_mamba + (y_tf-y_mamba)*0.314, y_tf], [y_mamba, y_tgn, y_tf], 
                     color='#e74c3c', alpha=0.1)
    ax2.text(50, 0.34, 'Geometric\nAdvantage', color='#c0392b', ha='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig('fig2_final_real.png', dpi=300, bbox_inches='tight')
    print("Saved fig2_final_real.png")

if __name__ == '__main__':
    plot_fig2_real()
