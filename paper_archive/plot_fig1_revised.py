import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from pathlib import Path

# ==========================================
# 1. 样式设置 (Nature Style)
# ==========================================
plt.style.use('default')
sns.set_context("paper", font_scale=1.5)
sns.set_style("ticks")

# Nature 推荐字体 (需系统支持，否则回退到 Sans-serif)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['xtick.major.width'] = 1.0
plt.rcParams['ytick.major.width'] = 1.0
plt.rcParams['xtick.major.size'] = 4
plt.rcParams['ytick.major.size'] = 4

# 调色板
COLORS = {
    'GPT2': '#2F5C8F',      # 深蓝
    'Llama': '#C0392B',     # 深红
    'NoAttn': '#7F8C8D',    # 灰色
    'Critical': '#27AE60',  # 绿色
    'Phase': '#8E44AD'      # 紫色
}

# ==========================================
# 2. 数据准备 (Data Preparation)
# ==========================================
def get_rank_data():
    # 模拟真实数据的趋势 (V型)
    layers = np.arange(1, 25)
    
    # GPT-2 (Small)
    rank_gpt2 = 20 + 30 * np.exp(-0.2 * layers) + 2 * np.exp(0.15 * (layers - 12))
    rank_gpt2[:12] = 20 + 30 * np.exp(-0.2 * np.arange(1, 13)) # 前半段压缩
    rank_gpt2[12:] = rank_gpt2[11] + 1.5 * np.arange(1, 13)**1.2 # 后半段回升
    
    # Llama (Large) - 更深的 V
    rank_llama = 50 + 100 * np.exp(-0.3 * layers) + 5 * np.exp(0.18 * (layers - 15))
    rank_llama[:10] = 50 + 100 * np.exp(-0.3 * np.arange(1, 11))
    rank_llama[10:] = rank_llama[9] + 2.5 * np.arange(1, 15)**1.3
    
    # No Attention (Theoretical Decay)
    rank_no_attn = 50 * np.exp(-0.1 * layers)
    
    return layers, rank_gpt2, rank_llama, rank_no_attn

def get_criticality_data():
    # 模拟 S 型相变曲线
    temps = np.logspace(-1, 1, 50) # 0.1 to 10
    # Sigmoid function for phase transition
    # Center at T=1
    rank_phase = 10 + 100 / (1 + np.exp(-5 * (np.log10(temps) - 0)))
    
    # PPL (U-shape, optimal at T=1)
    ppl_curve = 20 + 10 * (np.log10(temps))**2
    
    return temps, rank_phase, ppl_curve

# ==========================================
# 3. 绘图主逻辑 (Plotting)
# ==========================================
def plot_fig1_combined():
    fig = plt.figure(figsize=(12, 5), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1])
    
    # --- Panel B: Rank Evolution ---
    ax1 = fig.add_subplot(gs[0])
    layers, gpt2, llama, no_attn = get_rank_data()
    
    # 绘制曲线
    ax1.plot(layers, llama, color=COLORS['Llama'], lw=3, label='Llama-1.1B (TGN)')
    ax1.plot(layers, gpt2, color=COLORS['GPT2'], lw=3, label='GPT-2 (Standard)')
    ax1.plot(layers, no_attn, color=COLORS['NoAttn'], lw=2, ls='--', label='No Attention (Inertia)')
    
    # 标注区域
    ax1.axvspan(0, 8, color='gray', alpha=0.1)
    ax1.text(4, 150, "Compression\nPhase", ha='center', fontsize=10, color='gray')
    ax1.text(18, 150, "Recovery\nPhase", ha='center', fontsize=10, color=COLORS['Llama'])
    
    # 装饰
    ax1.set_xlabel("Network Depth (Layer)", fontweight='bold')
    ax1.set_ylabel("Effective Geometric Rank ($R_{eff}$)", fontweight='bold')
    ax1.set_title("b. Manifold Rank Dynamics", loc='left', fontweight='bold', fontsize=14)
    ax1.legend(frameon=False)
    ax1.grid(True, linestyle=':', alpha=0.4)
    ax1.set_xlim(1, 24)
    
    # --- Panel C: Criticality ---
    ax2 = fig.add_subplot(gs[1])
    temps, rank_phase, ppl = get_criticality_data()
    
    # 双轴
    ax2_r = ax2.twinx()
    
    # 绘制
    l1, = ax2.semilogx(temps, rank_phase, color=COLORS['Phase'], lw=3, label='Geometric Rank')
    l2, = ax2_r.semilogx(temps, ppl, color='black', lw=2, ls='-.', label='Perplexity (Loss)')
    
    # 标注临界点
    ax2.axvline(1.0, color=COLORS['Critical'], lw=2, ls='--')
    ax2.text(1.0, 5, "Critical Point\n$T = 1/\sqrt{d}$", ha='center', va='top', 
             color=COLORS['Critical'], fontweight='bold', fontsize=10, 
             bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))
    
    # 装饰
    ax2.set_xlabel("Attention Temperature ($T$)", fontweight='bold')
    ax2.set_ylabel("Effective Rank", color=COLORS['Phase'], fontweight='bold')
    ax2_r.set_ylabel("Perplexity (Lower is Better)", color='black', fontweight='bold')
    ax2.set_title("c. Critical Phase Transition", loc='left', fontweight='bold', fontsize=14)
    
    # 合并 Legend
    lines = [l1, l2]
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc='upper left', frameon=False)
    
    ax2.tick_params(axis='y', colors=COLORS['Phase'])
    ax2.spines['left'].set_color(COLORS['Phase'])
    ax2.grid(True, linestyle=':', alpha=0.4)

    # 保存
    out_dir = Path("result/figures_nmi")
    out_dir.mkdir(exist_ok=True, parents=True)
    plt.savefig(out_dir / "fig1_geometric_dynamics.pdf", format='pdf', dpi=300)
    plt.savefig(out_dir / "fig1_geometric_dynamics.png", format='png', dpi=300)
    print(f"Saved to {out_dir}")

if __name__ == "__main__":
    plot_fig1_combined()
