import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

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

def plot_fig3_combined():
    # Load data
    df_mqar = pd.read_csv('TGN-Nature/实验代码/fig3/mqar_training_curves.csv')
    df_throughput = pd.read_csv('TGN-Nature/实验代码/fig3/throughput_benchmark.csv')

    # Create figure (Unified Size: 15x5)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), dpi=300)

    # --- Plot 3a: MQAR Accuracy ---
    ax1 = axes[0]

    # Filter data
    tgn_data = df_mqar[df_mqar['model'] == 'tgn']
    mamba_data = df_mqar[df_mqar['model'] == 'mamba']
    transformer_data = df_mqar[df_mqar['model'] == 'transformer']

    # Plot lines
    # Transformer as baseline (dashed)
    ax1.plot(transformer_data['step'], transformer_data['accuracy'], 
             label='Transformer (Full Attn)', color='gray', linestyle='--', linewidth=2, alpha=0.7)

    # Mamba (Blue)
    ax1.plot(mamba_data['step'], mamba_data['accuracy'], 
             label='Mamba (SSM)', color='#1f77b4', linewidth=3)

    # TGN (Red)
    ax1.plot(tgn_data['step'], tgn_data['accuracy'], 
             label='TGN (Adaptive)', color='#d62728', linewidth=3)

    ax1.set_xlabel('Training Steps', fontweight='bold')
    ax1.set_ylabel('Accuracy', fontweight='bold')
    ax1.set_title('(a) Multi-Query Associative Recall (L=1024)', fontweight='bold', loc='left')
    
    # Move legend to center right
    ax1.legend(loc='center right', frameon=True, fontsize=10)
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Add annotation for Mamba collapse
    ax1.annotate('State Compression\nCollapse', xy=(2500, 0.02), xytext=(2500, 0.25),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
                 fontsize=10, color='#1f77b4', ha='center')

    # --- Plot 3b: Throughput ---
    ax2 = axes[1]

    x = df_throughput['seq_len']

    # Handle 0.0 values for Transformer (OOM)
    transf_vals = df_throughput['Transformer'].copy()
    transf_vals[transf_vals == 0] = np.nan

    # Transformer (Gray)
    ax2.plot(x, transf_vals, 
             label='Transformer (O(N²))', color='gray', marker='o', linestyle='--', linewidth=2.5)

    # Mamba (Blue)
    ax2.plot(x, df_throughput['Mamba'], 
             label='Mamba (O(N))', color='#1f77b4', marker='s', linewidth=3)

    # TGN (Red)
    ax2.plot(x, df_throughput['TGN (Chunked 20%)'], 
             label='TGN (Chunked, 20% Active)', color='#d62728', marker='^', linewidth=3)

    ax2.set_xlabel('Sequence Length', fontweight='bold')
    ax2.set_ylabel('Throughput (tokens/sec)', fontweight='bold')
    ax2.set_title('(b) Inference Throughput Benchmark', fontweight='bold', loc='left')

    ax2.set_xscale('log', base=2)
    ax2.set_yscale('log')

    # Format x-axis ticks
    ax2.set_xticks(x)
    ax2.set_xticklabels(x)
    
    # Annotate OOM
    oom_x = x[df_throughput['Transformer'] == 0]
    for ox in oom_x:
        ax2.text(ox, 1e4, 'OOM', color='gray', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax2.legend(loc='lower left', frameon=True, fontsize=10)
    ax2.grid(True, which="both", ls="-", alpha=0.2)
    ax2.grid(True, which="major", ls="--", alpha=0.6)

    # Add text for speedup
    idx_8k = df_throughput[df_throughput['seq_len'] == 8192].index[0]
    val_tgn_8k = df_throughput['TGN (Chunked 20%)'].iloc[idx_8k]
    val_trf_8k = df_throughput['Transformer'].iloc[idx_8k]
    speedup = val_tgn_8k / val_trf_8k

    ax2.annotate(f'{speedup:.1f}x Speedup\n(vs Transformer)', 
                 xy=(8192, val_tgn_8k), 
                 xytext=(4096, val_tgn_8k * 2),
                 arrowprops=dict(facecolor='#d62728', shrink=0.05),
                 fontsize=11, color='#d62728', fontweight='bold', ha='center')

    # Highlight 32k TGN performance
    val_tgn_32k = df_throughput['TGN (Chunked 20%)'].iloc[-1]
    ax2.annotate('Scales to 32k', 
                 xy=(32768, val_tgn_32k), 
                 xytext=(16384, val_tgn_32k * 0.5),
                 arrowprops=dict(facecolor='#d62728', shrink=0.05),
                 fontsize=10, color='#d62728', fontweight='bold')

    plt.tight_layout()
    plt.savefig('TGN-Nature/实验代码/fig3/fig3_combined_nmi.png', dpi=300, bbox_inches='tight')
    print("Figure saved to TGN-Nature/实验代码/fig3/fig3_combined_nmi.png")

if __name__ == "__main__":
    plot_fig3_combined()
