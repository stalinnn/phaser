import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

"""
EXP 103: Empirical Geometric Scaling Laws (0.1B -> 7B)
-----------------------------------------------------
Goal: Visualize the real rank evolution extracted from cloud experiments.
Data Sources:
1. GPT-2 (0.1B)
2. TinyLlama (1.1B)
3. Qwen2.5-7B (7B) - Proxy for Llama-3-8B class

Key Finding: 
Larger models exhibit "Deeper Compression" (Semantic Crystallization) 
and "Stronger Recovery" (Geometric Tunneling).
"""

def run_empirical_plot():
    # 1. Real Data (Manually transcribed from cloud logs)
    
    # GPT-2 Small (12 Layers)
    gpt2_ranks = [
        110.00798, 114.40558, 112.57105, 81.011986, 100.82561, 116.141685, 
        142.31944, 158.72725, 187.8509, 196.79851, 205.60776, 207.2163, 42.938873
    ]
    # Remove last layer (usually head collapse)
    gpt2_ranks = gpt2_ranks[:-1]
    
    # TinyLlama 1.1B (22 Layers)
    tiny_ranks = [
        100.9, 124.1, 142.9, 33.5, 47.8, 63.5, 79.6, 98.2, 85.8, 105.9, 
        127.3, 145.9, 169.5, 184.6, 198.5, 209.2, 236.2, 247.6, 259.5, 
        269.3, 277.5, 282.8, 298.5
    ]
    
    # Qwen2.5-7B (28 Layers)
    qwen_ranks = [
        91.6, 140.9, 143.9, 152.4, # Embedding/Early
        8.9, 9.0, 10.7, 13.3, 16.6, 20.8, 20.2, 21.8, # Deep Compression zone
        23.9, 25.7, 28.4, 31.6, 36.2, 40.2, 45.4, 48.5, 54.9, 66.9, 
        81.5, 106.7, 126.9, 144.3, 165.4, 274.7, 253.5
    ]
    # Smooth Qwen slightly for visualization (moving average)
    qwen_smooth = np.convolve(qwen_ranks, np.ones(3)/3, mode='valid')
    # Pad to match original length for x-axis
    qwen_plot = qwen_ranks # Use raw for authenticity
    
    # 2. Normalize Depth for Comparison (0.0 to 1.0)
    def get_norm_x(data):
        return np.linspace(0, 1, len(data))
    
    # 3. Plotting
    # Use Nature-like style
    plt.style.use('default')
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 14,
        'axes.labelsize': 16,
        'axes.titlesize': 16,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
        'figure.figsize': (10, 7),
        'lines.linewidth': 3,
        'grid.alpha': 0.3
    })
    
    fig, ax = plt.subplots()
    
    # Plot curves
    ax.plot(get_norm_x(gpt2_ranks), gpt2_ranks, 'o--', color='#1f77b4', alpha=0.6, label='GPT-2 (0.1B)')
    ax.plot(get_norm_x(tiny_ranks), tiny_ranks, 's--', color='#ff7f0e', alpha=0.8, label='TinyLlama (1.1B)')
    ax.plot(get_norm_x(qwen_plot), qwen_plot, 'D-', color='#d62728', alpha=1.0, label='Qwen2.5 (7B) - SOTA')
    
    # Annotations
    # 1. Semantic Crystallization (The drop in Qwen)
    min_idx = 4 # Index of 8.9
    min_val = qwen_ranks[min_idx]
    ax.annotate('Semantic Crystallization\n(Rank $\\approx$ 9)', 
                xy=(get_norm_x(qwen_plot)[min_idx], min_val), 
                xytext=(0.25, 100),
                arrowprops=dict(facecolor='black', shrink=0.05),
                fontsize=12, fontweight='bold')
    
    # 2. Geometric Tunneling (The rise)
    max_idx = -2 # Near end
    max_val = qwen_ranks[max_idx]
    ax.annotate('Geometric Hyper-Expansion\n(Rank $\\times 30$)', 
                xy=(get_norm_x(qwen_plot)[max_idx], max_val), 
                xytext=(0.6, 200),
                arrowprops=dict(facecolor='red', shrink=0.05),
                fontsize=12, fontweight='bold', color='red')
    
    ax.set_title('Empirical Geometric Scaling Laws (0.1B $\\rightarrow$ 7B)', fontweight='bold')
    ax.set_xlabel('Normalized Network Depth (Input $\\rightarrow$ Output)')
    ax.set_ylabel('Effective Geometric Rank ($R_{eff}$)')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper center', frameon=False)
    
    # Insight Text
    text = "Insight: Larger models compress deeper\nand expand wider (Stronger 'Breathing')"
    ax.text(0.02, 250, text, fontsize=12, bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

    plt.tight_layout()
    save_path = 'figures/llm_scaling_law_empirical2.png'
    plt.savefig(save_path, dpi=300)
    print(f"Saved to {save_path}")

if __name__ == "__main__":
    run_empirical_plot()
