import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from pathlib import Path

def plot_sota_comparison(result_dir='result_sota_a800'):
    # Load Data
    try:
        df_mamba = pd.read_csv(f"{result_dir}/log_mamba.csv", header=None, names=['Step', 'Loss', 'Gate'])
        df_trans = pd.read_csv(f"{result_dir}/log_transformer.csv", header=None, names=['Step', 'Loss', 'Gate'])
        df_tgn = pd.read_csv(f"{result_dir}/log_tgn.csv", header=None, names=['Step', 'Loss', 'Gate'])
    except Exception as e:
        print(f"Error reading CSVs: {e}")
        return

    # Create figure with dual y-axis
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Configure plot style
    plt.style.use('seaborn-v0_8-paper')
    
    # Colors
    c_mamba = '#9b59b6' # Purple
    c_trans = '#3498db' # Blue
    c_tgn = '#e74c3c'   # Red
    c_gate = '#2ecc71'  # Green
    
    # Plot Losses (Left Axis)
    ax1.set_xlabel('Training Steps')
    ax1.set_ylabel('Training Loss (Log Scale)', fontweight='bold')
    ax1.set_yscale('log')
    
    l1, = ax1.plot(df_mamba['Step'], df_mamba['Loss'], color=c_mamba, linestyle='--', alpha=0.8, label='Mamba (SOTA)')
    l2, = ax1.plot(df_trans['Step'], df_trans['Loss'], color=c_trans, linestyle='-', linewidth=2, label='Transformer (Baseline)')
    l3, = ax1.plot(df_tgn['Step'], df_tgn['Loss'], color=c_tgn, linewidth=3, label='TGN (Ours)')
    
    ax1.yaxis.set_major_formatter(ticker.ScalarFormatter())
    ax1.set_yticks([0.5, 1, 2, 5, 10])
    
    # Plot Gate Rate (Right Axis) for TGN
    ax2 = ax1.twinx()
    ax2.set_ylabel('TGN Gate Rate (Sparsity)', color=c_gate, fontweight='bold')
    l4, = ax2.plot(df_tgn['Step'], df_tgn['Gate'], color=c_gate, linestyle=':', linewidth=2, label='TGN Sparsity')
    ax2.tick_params(axis='y', labelcolor=c_gate)
    ax2.set_ylim(0, 0.6)
    
    # Annotations
    final_step = df_tgn.iloc[-1]['Step']
    tgn_loss = df_tgn.iloc[-1]['Loss']
    trans_loss = df_trans.iloc[-1]['Loss']
    mamba_loss = df_mamba.iloc[-1]['Loss']
    tgn_gate = df_tgn.iloc[-1]['Gate']
    
    # Mamba
    ax1.annotate(f'Mamba: {mamba_loss:.2f}', xy=(final_step, mamba_loss), xytext=(final_step-100, mamba_loss*0.6),
                 arrowprops=dict(facecolor=c_mamba, shrink=0.05), color=c_mamba, fontweight='bold')
    
    # Transformer
    ax1.annotate(f'Transformer: {trans_loss:.2f}', xy=(final_step, trans_loss), xytext=(final_step+10, trans_loss),
                 color=c_trans, fontweight='bold', ha='left', va='center')

    # TGN
    ax1.annotate(f'TGN: {tgn_loss:.2f}', xy=(final_step, tgn_loss), xytext=(final_step+10, tgn_loss-0.5),
                 color=c_tgn, fontweight='bold', ha='left', va='center')
                 
    # Gate
    ax2.annotate(f'Gate: {tgn_gate:.1%}', xy=(final_step, tgn_gate), xytext=(final_step-150, tgn_gate+0.1),
                 arrowprops=dict(facecolor=c_gate, shrink=0.05), color=c_gate, fontweight='bold')

    # Title
    plt.title('Early Training Dynamics: TGN vs SOTA (WikiText-103)', fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3)
    
    # Legend
    lines = [l1, l2, l3, l4]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center left')
    
    # Save
    out_path = Path('figures/sota_comparison_battle.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {out_path}")

if __name__ == "__main__":
    plot_sota_comparison()
