import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

# NMI Style Settings
plt.style.use('default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['xtick.major.width'] = 1.5
plt.rcParams['ytick.major.width'] = 1.5
plt.rcParams['font.size'] = 14

def plot_merged_training_dynamics():
    # 1. Load Data
    df1 = pd.read_csv('TGN-Nature/实验代码/fig2/training_log.csv')
    df2 = pd.read_csv('TGN-Nature/实验代码/fig2/training_log_hotfix_phase2.csv')
    df3 = pd.read_csv('TGN-Nature/实验代码/fig2/training_log_hotfix_lambda_0.05.csv')

    # 2. Offset Steps for Continuity
    offset1 = df1['step'].max()
    df2['step'] = df2['step'] + offset1
    
    offset2 = df2['step'].max()
    df3['step'] = df3['step'] + offset2

    # 3. Concatenate
    df_all = pd.concat([df1, df2, df3], ignore_index=True)
    
    # Smooth Curves (Moving Average)
    window = 50
    df_all['ppl_smooth'] = df_all['ppl'].rolling(window, min_periods=1).mean()
    df_all['gate_smooth'] = df_all['gate_rate'].rolling(window, min_periods=1).mean()

    # 4. Plotting
    fig, ax1 = plt.subplots(figsize=(12, 7))

    # --- PPL Curve (Left Axis) ---
    color_ppl = '#c0392b' # Deep Red
    ax1.semilogy(df_all['step'], df_all['ppl_smooth'], color=color_ppl, linewidth=2.5, label='Adaptive TGN (Ours)')
    
    # --- Random Baseline Reference ---
    # Random PPL is ~310. Draw it as a dashed line.
    ax1.axhline(y=310, color='gray', linestyle='--', linewidth=2, alpha=0.6, label='Random Baseline (Control)')
    ax1.text(df_all['step'].iloc[-1], 330, 'Random Baseline\n(PPL ~310)', color='gray', ha='right', va='bottom', fontsize=11, fontweight='bold')

    ax1.set_xlabel('Training Steps', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Perplexity (Lower is Better)', color=color_ppl, fontsize=16, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor=color_ppl)
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    # Set Y-limit for PPL
    ax1.set_ylim(10, 1000) 

    # --- Gate Rate Curve (Right Axis) ---
    ax2 = ax1.twinx()
    color_gate = '#27ae60' # Green
    ax2.plot(df_all['step'], df_all['gate_smooth'], color=color_gate, linewidth=3, linestyle='-', label='Gate Activation Rate')
    ax2.set_ylabel('Gate Open Rate', color=color_gate, fontsize=16, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor=color_gate)
    
    # Set Y-limit for Gate (0% - 60%)
    ax2.set_ylim(0, 0.6)
    ax2.set_yticklabels(['{:.0%}'.format(x) for x in ax2.get_yticks()])

    # --- Vertical Lines for Phases ---
    ax1.axvline(x=offset1, color='gray', linestyle='--', linewidth=1.5)
    ax1.axvline(x=offset2, color='gray', linestyle='--', linewidth=1.5)
    
    # Phase Labels
    ax1.text(offset1/2, 500, 'Phase 1:\nNatural Awakening', ha='center', fontsize=12, fontweight='bold', color='#2c3e50')
    ax1.text(offset1 + (offset2-offset1)/2, 500, 'Phase 2:\nSparsity Annealing\n(λ=0.02)', ha='center', fontsize=12, fontweight='bold', color='#2c3e50')
    ax1.text(offset2 + 500, 500, 'Phase 3:\nExtreme\n(λ=0.05)', ha='center', fontsize=12, fontweight='bold', color='#2c3e50')

    # --- Key Mechanism Annotations ---
    
    # 1. Inertial Collapse
    early_phase = df_all.iloc[:500]
    min_gate_idx = early_phase['gate_smooth'].idxmin()
    min_gate_step = df_all.loc[min_gate_idx, 'step']
    min_gate_val = df_all.loc[min_gate_idx, 'gate_smooth']
    
    ax2.annotate('Inertial Collapse', 
                 xy=(min_gate_step, min_gate_val), xytext=(min_gate_step+800, min_gate_val+0.08),
                 arrowprops=dict(facecolor='black', shrink=0.05), fontsize=11, fontweight='bold')

    # 2. Hysteretic Awakening
    phase1_range = df_all[df_all['step'] < offset1]
    max_gate_idx = phase1_range['gate_smooth'].idxmax()
    max_gate_step = df_all.loc[max_gate_idx, 'step']
    max_gate_val = df_all.loc[max_gate_idx, 'gate_smooth']
    
    ax2.annotate('Hysteretic Awakening', 
                 xy=(max_gate_step, max_gate_val), xytext=(max_gate_step+1000, max_gate_val+0.05),
                 arrowprops=dict(facecolor='black', shrink=0.05), fontsize=11, fontweight='bold')

    # 3. Forced Sparsification
    final_gate_val = df_all['gate_smooth'].iloc[-1]
    final_step = df_all['step'].iloc[-1]
    
    ax2.annotate('Adaptive Sparsification\n(Gate: {:.1%})'.format(final_gate_val), 
                 xy=(final_step, final_gate_val), xytext=(final_step-1500, final_gate_val+0.1),
                 arrowprops=dict(facecolor='green', shrink=0.05), fontsize=11, fontweight='bold', color='green')

    # Add Legend for lines
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', frameon=True, framealpha=0.9)

    plt.title('Figure 2a: Hysteretic Dynamics & Sparsity Annealing in TGN', fontsize=18, pad=20)
    plt.tight_layout()
    
    # Save
    out_path = 'TGN-Nature/实验代码/fig2/fig2_final_merged_nmi.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved merged figure to {out_path}")

if __name__ == '__main__':
    plot_merged_training_dynamics()
