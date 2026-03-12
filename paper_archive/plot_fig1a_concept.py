import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
from matplotlib.collections import LineCollection

def plot_concept_manifold():
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    
    # 1. 生成流形线条 (Inertial Flow)
    t = np.linspace(0, 10, 500)
    n_lines = 20
    
    # 收缩阶段 (Compression)
    for i in range(n_lines):
        offset = (i - n_lines/2) * 0.1
        # 初始发散 -> 逐渐收缩
        spread = 1.0 / (1 + 0.5 * t)  
        y = np.sin(t + i*0.2) * 0.2 + offset * spread
        
        # 颜色：从蓝变红 (Temperature Increase)
        color = plt.cm.coolwarm(t/10)
        
        # 使用 LineCollection 实现渐变色
        points = np.array([t, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap='viridis', norm=plt.Normalize(0, 10), alpha=0.6, linewidth=1.5)
        lc.set_array(t)
        ax.add_collection(lc)

    # 2. 绘制 Attention 虫洞 (Geometric Shortcuts)
    # 在 t=4 和 t=8 之间建立连接
    x_start, x_end = 2.0, 8.0
    y_start, y_end = 0.0, 0.0
    
    # 绘制弧线代表非局部连接
    arc = patches.Arc(((x_start + x_end)/2, 0.5), width=x_end-x_start, height=1.5, 
                      theta1=0, theta2=180, color='#E74C3C', linewidth=2, linestyle='--')
    ax.add_patch(arc)
    
    # 3. 几何膨胀 (Expansion)
    # 在虫洞落地后 (t > 8)，流形突然膨胀
    for i in range(n_lines):
        t_exp = np.linspace(8, 10, 100)
        offset = (i - n_lines/2) * 0.1
        # 突然发散
        spread = 0.3 + 2.0 * (t_exp - 8)**2 
        y = np.sin(t_exp) * 0.2 + offset * spread
        ax.plot(t_exp, y, color='#E74C3C', alpha=0.6, linewidth=1.5)

    # 4. 标注与装饰
    ax.text(1.0, 0.5, "Inertial Flow\n(Low Rank)", ha='center', fontsize=12, fontweight='bold', color='#2980B9')
    ax.text(9.0, 1.5, "Geometric\nExpansion\n(High Rank)", ha='center', fontsize=12, fontweight='bold', color='#C0392B')
    
    ax.text(5.0, 1.4, "Non-local Attention\n(Topological Shortcut)", ha='center', fontsize=10, color='#E74C3C', backgroundcolor='white')
    
    # 箭头标注
    ax.arrow(2.0, 0.1, 0, 0.2, head_width=0.1, head_length=0.1, fc='#E74C3C', ec='#E74C3C')
    ax.arrow(8.0, 1.2, 0, -0.2, head_width=0.1, head_length=0.1, fc='#E74C3C', ec='#E74C3C')

    # 去除坐标轴
    ax.set_xlim(0, 10)
    ax.set_ylim(-2, 2)
    ax.axis('off')
    
    # 标题
    ax.set_title("a. Thermodynamic Gating Mechanism", loc='left', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig("result/figures_nmi/fig1a_concept_generated.png")
    print("Generated fig1a_concept_generated.png")

if __name__ == "__main__":
    plot_concept_manifold()
