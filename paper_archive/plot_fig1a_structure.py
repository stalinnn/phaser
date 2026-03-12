import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def plot_structure_concept():
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    
    # ---------------------------------------------------
    # 1. 基础设置
    # ---------------------------------------------------
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis('off')
    
    # 颜色定义
    c_inertial = '#34495E' # 深蓝灰 (惯性)
    c_geo = '#E74C3C'      # 亮红 (几何)
    c_gate = '#27AE60'     # 绿色 (门控)
    c_token = '#BDC3C7'    # 浅灰 (Token)

    # ---------------------------------------------------
    # 2. 绘制时间轴与 Token (The Timeline)
    # ---------------------------------------------------
    tokens = [1, 3, 5, 7, 9, 11]
    labels = ["$x_{t-k}$", "...", "$x_{t-2}$", "$x_{t-1}$", "$x_t$\n(Current)", "$x_{t+1}$"]
    
    for i, x in enumerate(tokens):
        # Token Node
        circle = patches.Circle((x, 1), radius=0.4, fc='white', ec=c_token, lw=2)
        ax.add_patch(circle)
        ax.text(x, 0.3, labels[i], ha='center', va='top', fontsize=11, fontweight='bold', color='gray')

    # ---------------------------------------------------
    # 3. 绘制惯性通道 (Inertial Channel - Mamba/RNN)
    # ---------------------------------------------------
    # 用一条粗壮的、渐变的流管连接底部
    # 模拟“绝热过程”
    path_y = 1.0
    
    # 绘制连接箭头
    for i in range(len(tokens)-1):
        start = tokens[i]
        end = tokens[i+1]
        
        # 惯性流 (蓝色箭头)
        arrow = patches.FancyArrowPatch(
            (start+0.4, path_y), (end-0.4, path_y),
            arrowstyle='Simple,tail_width=2,head_width=8,head_length=8',
            color=c_inertial, alpha=0.8, zorder=1
        )
        ax.add_patch(arrow)
    
    ax.text(6, 0.8, "Inertial Channel (Mamba)\nLow Energy / Local Context", 
            ha='center', va='top', fontsize=10, color=c_inertial, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8, ec='none'))

    # ---------------------------------------------------
    # 4. 绘制几何通道 (Geometric Channel - Attention)
    # ---------------------------------------------------
    # 从 t-k (x=1) 跳跃到 t (x=9)
    # 这是一个巨大的红色弧线
    
    # 只有当 Gate 开启时才存在
    arc = patches.FancyArrowPatch(
        (1, 1.5), (9, 1.5),
        connectionstyle="arc3,rad=-0.5", # 向上弯曲
        arrowstyle='Simple,tail_width=1.5,head_width=10,head_length=10',
        color=c_geo, alpha=0.9, linestyle='--', zorder=5
    )
    ax.add_patch(arc)
    
    ax.text(5, 4.5, "Geometric Channel (Attention)\nHigh Energy / Non-local Shortcut", 
            ha='center', va='bottom', fontsize=10, color=c_geo, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8, ec='none'))

    # ---------------------------------------------------
    # 5. 绘制麦克斯韦妖 (The Gate / Maxwell's Demon)
    # ---------------------------------------------------
    # 在 x_t (9, 1) 处画一个控制阀
    
    # 传感器线 (从惯性流引出)
    ax.plot([9, 9], [1.4, 2.2], color=c_gate, lw=2, linestyle=':')
    
    # 门控开关 (菱形)
    gate_box = patches.RegularPolygon((9, 2.5), numVertices=4, radius=0.6, fc='white', ec=c_gate, lw=2.5, zorder=10)
    ax.add_patch(gate_box)
    
    ax.text(9, 2.5, "Gate\n$g_t$", ha='center', va='center', fontweight='bold', color=c_gate, fontsize=12)
    
    # 门控逻辑标注
    ax.text(10.5, 2.5, r"$\Delta \mathcal{L} > \lambda$" + "\n(High Surprisal)", 
            ha='left', va='center', fontsize=11, color=c_gate)
    
    # 阻断/允许符号
    # 画一个“开关”打开的动作
    ax.text(9, 3.3, "OPEN", ha='center', va='bottom', fontsize=10, fontweight='bold', color=c_geo)

    # ---------------------------------------------------
    # 6. 物理隐喻标注
    # ---------------------------------------------------
    # 左侧：低熵区
    # 右侧：高熵区 -> 降熵
    
    ax.text(1, 5.5, "a. Thermodynamic Gating Architecture", fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig("result/figures_nmi/fig1a_structure.png", bbox_inches='tight')
    plt.savefig("result/figures_nmi/fig1a_structure.pdf", bbox_inches='tight')
    print("Generated fig1a_structure.png")

if __name__ == "__main__":
    plot_structure_concept()
