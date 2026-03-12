import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Fix for matplotlib incompatibility with numpy 2.0
if not hasattr(np, 'Inf'):
    np.Inf = np.inf

def plot_scaling_law_comparison():
    root = Path(__file__).resolve().parent.parent
    tgn_path = root / "result" / "ddp_large_tgn_log.csv"
    trans_path = root / "result" / "ddp_large_transformer_log.csv"
    
    if not tgn_path.exists() or not trans_path.exists():
        print(f"Error: Logs not found in {root}/result")
        return

    df_tgn = pd.read_csv(tgn_path)
    df_trans = pd.read_csv(trans_path)
    
    # Setup plot
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # 1. Plot Loss (Left Axis)
    # Transformer
    l1, = ax1.plot(df_trans['iter'], df_trans['loss'], 
             label="Transformer (Baseline)", color='#3498db', linestyle='--', linewidth=2, marker='o', markersize=4, alpha=0.7)
    
    # TGN
    l2, = ax1.plot(df_tgn['iter'], df_tgn['loss'], 
             label="TGN (Ours)", color='#2ecc71', linewidth=2.5, marker='s', markersize=4)
    
    ax1.set_xlabel("Training Steps", fontsize=12)
    ax1.set_ylabel("Cross Entropy Loss", color='#2c3e50', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='#2c3e50')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 4.5) # Based on log data
    
    # 2. Plot Gate Rate (Right Axis) - Only for TGN
    ax2 = ax1.twinx()
    l3, = ax2.plot(df_tgn['iter'], df_tgn['gate_rate'], 
             label="TGN Gate Rate", color='#e74c3c', linestyle=':', linewidth=2)
    
    ax2.set_ylabel("Gate Activation Rate (Sparsity)", color='#e74c3c', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='#e74c3c')
    ax2.set_ylim(0, 1.05)
    
    # Annotations
    final_loss_tgn = df_tgn['loss'].iloc[-1]
    final_loss_trans = df_trans['loss'].iloc[-1]
    final_gate = df_tgn['gate_rate'].iloc[-1]
    
    # Text box for final stats
    stats_text = (
        f"Iter {df_tgn['iter'].iloc[-1]}\n"
        f"Transformer Loss: {final_loss_trans:.4f}\n"
        f"TGN Loss: {final_loss_tgn:.4f} (-{(1 - final_loss_tgn/final_loss_trans)*100:.1f}%)\n"
        f"TGN Sparsity: {final_gate*100:.1f}% Active"
    )
    
    props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray')
    ax1.text(0.05, 0.2, stats_text, transform=ax1.transAxes, fontsize=11,
            verticalalignment='bottom', bbox=props)

    # Combined Legend
    lines = [l1, l2, l3]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right', frameon=True, fontsize=10)
    
    plt.title("Large-Scale Language Modeling (454M Params): TGN vs Transformer", fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    
    save_path = root / "figures" / "llm_scaling_law_empirical2.png"
    plt.savefig(save_path, dpi=300)
    print(f"Saved plot to {save_path}")

if __name__ == "__main__":
    plot_scaling_law_comparison()
