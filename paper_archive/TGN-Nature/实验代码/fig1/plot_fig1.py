import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib.gridspec import GridSpec

# ==========================================
# 统一风格设置 (NMI Standard)
# ==========================================
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'sans-serif']
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['xtick.major.width'] = 1.0
plt.rcParams['ytick.major.width'] = 1.0
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12

def plot_fig1():
    # 读取数据
    df_a = pd.read_csv("TGN-Nature/实验代码/fig1/fig1a_rank_evolution.csv")
    df_b = pd.read_csv("TGN-Nature/实验代码/fig1/fig1b_ablation.csv")
    df_c = pd.read_csv("TGN-Nature/实验代码/fig1/fig1c_criticality.csv")

    # 创建画布 (统一为 Fig4 尺寸: 15x5 inches, 300 DPI)
    fig = plt.figure(figsize=(15, 5), dpi=300)
    gs = GridSpec(1, 3, figure=fig, wspace=0.3, width_ratios=[1, 1, 1])

    # ==========================
    # Panel A: Rank Evolution
    # ==========================
    ax1 = fig.add_subplot(gs[0, 0])
    
    # 绘制 GPT2-XL
    data_gpt = df_a[df_a['Model'] == 'GPT2-XL']
    ax1.plot(data_gpt['Normalized_Layer'], data_gpt['Effective_Rank'], 
             color='#1f77b4', linewidth=2.5, label='GPT-2 XL (1.5B)')
    
    # 绘制 TinyLlama
    data_llama = df_a[df_a['Model'] == 'TinyLlama']
    ax1.plot(data_llama['Normalized_Layer'], data_llama['Effective_Rank'], 
             color='#ff7f0e', linewidth=2.5, linestyle='--', label='TinyLlama (1.1B)')
    
    ax1.set_xlabel('Normalized Depth (0=In, 1=Out)', fontweight='bold')
    ax1.set_ylabel('Effective Geometric Rank ($R_{eff}$)', fontweight='bold')
    ax1.set_title('a. Geometric Breathing', fontweight='bold', loc='left')
    ax1.legend(frameon=False, fontsize=9)
    ax1.grid(True, linestyle=':', alpha=0.4)
    
    # 添加 "V-Shape" 标注
    ax1.annotate('Rank Recovery', xy=(0.8, 300), xytext=(0.5, 350),
                 arrowprops=dict(facecolor='black', arrowstyle='->', lw=1.5), fontsize=10)

    # ==========================
    # Panel B: Ablation (Mechanism)
    # ==========================
    ax2 = fig.add_subplot(gs[0, 1])
    
    ax2.plot(df_b['Layer'], df_b['Rank_Normal'], 
             color='#2ca02c', linewidth=2.5, label='Normal (Attention On)')
    
    ax2.plot(df_b['Layer'], df_b['Rank_NoAttn'], 
             color='#d62728', linewidth=2.5, linestyle=':', label='Ablated (Attention Off)')
    
    # 填充差异区域
    ax2.fill_between(df_b['Layer'], df_b['Rank_NoAttn'], df_b['Rank_Normal'], 
                     color='#2ca02c', alpha=0.1)
    
    ax2.set_xlabel('Layer Index', fontweight='bold')
    # ax2.set_ylabel('Effective Rank', fontsize=9) 
    ax2.set_title('b. Anti-dissipative Force', fontweight='bold', loc='left')
    ax2.legend(frameon=False, fontsize=9, loc='upper left')
    ax2.grid(True, linestyle=':', alpha=0.4)
    
    # 添加文字说明
    ax2.text(10, 150, "Geometric\nPumping", color='#2ca02c', fontsize=10, fontweight='bold')

    # ==========================
    # Panel C: Criticality (Phase Transition)
    # ==========================
    ax3 = fig.add_subplot(gs[0, 2])
    
    # 对数坐标轴
    ax3.semilogx(df_c['Temperature'], df_c['Deep_Layer_Rank'], 
                 color='#9467bd', linewidth=2.5, marker='o', markersize=6)
    
    # 标注 Critical Point T=1.0
    ax3.axvline(x=1.0, color='gray', linestyle='--', linewidth=1.5)
    ax3.text(1.1, 400, r'Critical Point ($1/\sqrt{d}$)', rotation=90, fontsize=9, color='gray')
    
    ax3.set_xlabel('Temperature Scaling $\tau$', fontweight='bold')
    # ax3.set_ylabel('Deep Layer Rank', fontsize=9)
    ax3.set_title('c. Edge of Chaos', fontweight='bold', loc='left')
    ax3.grid(True, linestyle=':', alpha=0.4)

    # 保存
    plt.tight_layout()
    plt.savefig("TGN-Nature/实验代码/fig1/fig1_combined_nmi.png", dpi=300, bbox_inches='tight')
    print("Figure 1 generated: TGN-Nature/实验代码/fig1/fig1_combined_nmi.png")

if __name__ == "__main__":
    plot_fig1()
