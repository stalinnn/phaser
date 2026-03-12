import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
from matplotlib.collections import LineCollection

def plot_concept_manifold_v2():
    # 使用宽屏布局
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    
    # ------------------------------------------------
    # 1. 惯性流 (Inertial Manifold) - 冷峻的蓝灰色
    # ------------------------------------------------
    t = np.linspace(0, 10, 800)
    n_lines = 25
    
    for i in range(n_lines):
        # 归一化索引 (-1 到 1)
        norm_i = (i - n_lines/2) / (n_lines/2)
        
        # 物理模型：阻尼振荡收缩
        # y = A * exp(-lambda * t) * sin(omega * t + phase)
        decay = np.exp(-0.15 * t)
        
        # 初始扰动大，逐渐收敛到中心流形
        y = norm_i * 1.5 * decay + 0.05 * np.sin(3*t + i) * decay
        
        # 颜色：从浅蓝到深蓝，带一点透明度
        # 越靠近中心越深，越外围越浅
        alpha = 0.3 + 0.4 * np.exp(-2 * norm_i**2)
        color = '#2C3E50' # 专业的深蓝灰
        
        ax.plot(t, y, color=color, alpha=alpha, linewidth=1.5)

    # ------------------------------------------------
    # 2. 几何流 (Geometric Expansion) - 爆发的红金色
    # ------------------------------------------------
    # 触发点
    t_trigger = 7.5
    
    for i in range(n_lines):
        norm_i = (i - n_lines/2) / (n_lines/2)
        
        t_exp = np.linspace(t_trigger, 10, 200)
        
        # 衔接点 continuity
        decay_at_trigger = np.exp(-0.15 * t_trigger)
        y_start = norm_i * 1.5 * decay_at_trigger + 0.05 * np.sin(3*t_trigger + i) * decay_at_trigger
        
        # 膨胀模型：指数发散
        # y = y_start * exp(lambda * (t - t_trigger))
        # 引入一点混沌扰动
        y_exp = y_start * np.exp(1.2 * (t_exp - t_trigger)) + 0.1 * norm_i * (t_exp - t_trigger)**2
        
        # 颜色：红色渐变到橙色
        color = '#E74C3C' # 亮红
        ax.plot(t_exp, y_exp, color=color, alpha=0.6, linewidth=1.8)

    # ------------------------------------------------
    # 3. 虫洞连接 (The Shortcut)
    # ------------------------------------------------
    # 从 t=2.0 (Historical State) 到 t=7.5 (Current State)
    x_hist, x_curr = 2.0, t_trigger
    
    # 画一条贝塞尔曲线或者大弧线
    arc = patches.FancyArrowPatch(
        (x_hist, 0.8), (x_curr, 0.8),
        connectionstyle="arc3,rad=-0.4",
        color='#D35400',
        arrowstyle='Simple,tail_width=0.5,head_width=4,head_length=4',
        linewidth=2,
        linestyle='--'
    )
    ax.add_patch(arc)
    
    # ------------------------------------------------
    # 4. 专业标注
    # ------------------------------------------------
    # 区域背景
    ax.axvspan(0, t_trigger, color='#ECF0F1', alpha=0.3, lw=0) # 灰色背景 (Inertia)
    ax.axvspan(t_trigger, 10, color='#FDEDEC', alpha=0.3, lw=0) # 红色背景 (Geometry)
    
    # 文字
    ax.text(1.0, -1.5, "Local Inertial Flow\n(Entropy Increase / Dissipation)", 
            ha='left', fontsize=11, fontweight='bold', color='#2C3E50', family='sans-serif')
    
    ax.text(9.0, -1.5, "Non-local Geometric Expansion\n(Entropy Decrease / Work)", 
            ha='right', fontsize=11, fontweight='bold', color='#C0392B', family='sans-serif')
    
    ax.text((x_hist+x_curr)/2, 2.2, "Attention Mechanism\n(Topological Shortcut)", 
            ha='center', fontsize=10, fontweight='bold', color='#D35400', 
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#D35400', alpha=0.9))

    # 关键点标注
    ax.scatter([x_hist], [0.5], color='#2C3E50', s=50, zorder=10)
    ax.text(x_hist, 0.3, "$h_{t-\\tau}$", ha='center', color='#2C3E50')
    
    ax.scatter([x_curr], [0.1], color='#C0392B', s=50, zorder=10)
    ax.text(x_curr, -0.2, "$h_t$", ha='center', color='#C0392B')

    # ------------------------------------------------
    # 5. 收尾
    # ------------------------------------------------
    ax.set_xlim(0, 10)
    ax.set_ylim(-2.5, 3.0)
    ax.axis('off') # 隐藏坐标轴
    
    # 主标题
    ax.text(0, 2.8, "a. Thermodynamic Gating Mechanism", fontsize=16, fontweight='bold', color='black')

    plt.tight_layout()
    plt.savefig("result/figures_nmi/fig1a_concept_v2.png", bbox_inches='tight')
    plt.savefig("result/figures_nmi/fig1a_concept_v2.pdf", bbox_inches='tight')
    print("Generated fig1a_concept_v2.png")

if __name__ == "__main__":
    plot_concept_manifold_v2()
