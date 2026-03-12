import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from pathlib import Path

def plot_scaling_results(csv_path):
    # Load Data
    # CSV format: Step, Loss, PPL, Gate, Time
    # Header is missing in the raw log, so we define it manually based on script
    # Columns: step, avg_loss, ppl, avg_gate, dt
    
    try:
        df = pd.read_csv(csv_path, header=None, names=['Step', 'Loss', 'PPL', 'Gate', 'Time'])
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # Create figure with dual y-axis
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Configure plot style
    plt.style.use('seaborn-v0_8-paper')
    color_ppl = '#e74c3c'  # Red
    color_gate = '#2ecc71' # Green
    
    # Plot PPL (Left Axis)
    ax1.set_xlabel('Training Steps')
    ax1.set_ylabel('Perplexity (PPL)', color=color_ppl, fontweight='bold')
    # Use log scale for PPL because it drops from 8000 to 25
    ax1.set_yscale('log')
    
    line1, = ax1.plot(df['Step'], df['PPL'], color=color_ppl, linewidth=2, label='TGN Perplexity')
    ax1.tick_params(axis='y', labelcolor=color_ppl)
    
    # Customize Log Ticks
    ax1.yaxis.set_major_formatter(ticker.ScalarFormatter())
    ax1.set_yticks([25, 50, 100, 500, 1000, 5000])
    
    # Plot Gate Rate (Right Axis)
    ax2 = ax1.twinx()
    ax2.set_ylabel('Gate Open Rate (Sparsity)', color=color_gate, fontweight='bold')
    line2, = ax2.plot(df['Step'], df['Gate'], color=color_gate, linewidth=2, label='Attention Sparsity', alpha=0.8)
    ax2.tick_params(axis='y', labelcolor=color_gate)
    ax2.set_ylim(0, 0.5) # Focus on 0-50% range
    
    # Add Reference Line for Theoretical Ideal (~10%)
    ax2.axhline(y=0.10, color='gray', linestyle='--', alpha=0.5, label='Theoretical Equilibrium (~10%)')
    
    # Annotations
    # 1. Early collapse (Inertia Phase)
    min_gate_idx = df['Gate'].idxmin()
    min_gate_step = df.loc[min_gate_idx, 'Step']
    min_gate_val = df.loc[min_gate_idx, 'Gate']
    
    ax2.annotate('Inertial Collapse\n(Gate < 1%)', xy=(min_gate_step, min_gate_val), xytext=(min_gate_step+500, min_gate_val+0.05),
                 arrowprops=dict(facecolor='black', shrink=0.05, alpha=0.5), fontsize=9)

    # 2. Hysteresis Awakening (Geometric Phase)
    # Find where Gate crosses 5% again
    recovery_df = df[df['Step'] > min_gate_step]
    awakening = recovery_df[recovery_df['Gate'] > 0.05].head(1)
    if not awakening.empty:
        aw_step = awakening['Step'].values[0]
        aw_val = awakening['Gate'].values[0]
        ax2.annotate('Hysteresis Awakening\n(Geometry Re-emerges)', xy=(aw_step, aw_val), xytext=(aw_step-1000, aw_val+0.08),
                     arrowprops=dict(facecolor=color_gate, shrink=0.05), fontsize=9, color='darkgreen')

    # 3. Final Convergence
    final_step = df.iloc[-1]['Step']
    final_ppl = df.iloc[-1]['PPL']
    final_gate = df.iloc[-1]['Gate']
    
    ax1.annotate(f'Final PPL: {final_ppl:.1f}', xy=(final_step, final_ppl), xytext=(final_step-1000, final_ppl*3),
                 arrowprops=dict(facecolor=color_ppl, shrink=0.05), fontsize=10, fontweight='bold', color=color_ppl)
                 
    ax2.annotate(f'Stable Sparsity: {final_gate:.1%}', xy=(final_step, final_gate), xytext=(final_step-1500, final_gate-0.03),
                 arrowprops=dict(facecolor=color_gate, shrink=0.05), fontsize=10, fontweight='bold', color='darkgreen')

    # Title and Layout
    plt.title('Emergence of Sparsity on Natural Language Manifold (WikiText-103)', fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3)
    
    # Legend
    lines = [line1, line2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center right')
    
    # Save
    out_path = Path(csv_path).parent / 'figures' / 'lm_efficiency_cloud.png'
    out_path.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {out_path}")

if __name__ == "__main__":
    plot_scaling_results('log.csv')
