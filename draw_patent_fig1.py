import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.font_manager as fm

# 设置中文字体（尝试自动寻找 Windows 常见中文字体）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False # 解决负号显示问题

def draw_tgn_architecture():
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Styles
    box_props = dict(boxstyle='round,pad=0.5', facecolor='#e1f5fe', edgecolor='#0277bd', linewidth=2)
    gate_props = dict(boxstyle='round,pad=0.5', facecolor='#fff9c4', edgecolor='#fbc02d', linewidth=2)
    attn_props = dict(boxstyle='round,pad=0.5', facecolor='#ffcdd2', edgecolor='#c62828', linewidth=2)
    io_props = dict(boxstyle='round,pad=0.3', facecolor='#f5f5f5', edgecolor='black', linewidth=1)

    # 1. Input
    ax.text(6, 9, "输入序列 (Xt)", ha='center', va='center', size=12, bbox=io_props)
    ax.arrow(6, 8.6, 0, -0.6, head_width=0.15, head_length=0.2, fc='black', ec='black')

    # 2. Inertial Channel (RNN) - Always Active
    ax.text(2, 6, "惯性处理单元\n(RNN / SSM)\n低成本 O(N)", ha='center', va='center', size=12, bbox=box_props)
    ax.arrow(6, 8, -4, -1.2, head_width=0.15, head_length=0.2, fc='black', ec='black') # Input -> RNN
    ax.text(3.5, 7.5, "路径 A (常驻)", size=10)

    # 3. Gating Module (Maxwell's Demon)
    ax.text(6, 6, "门控判决单元\n(熵检测)\n预测 gt", ha='center', va='center', size=12, bbox=gate_props)
    ax.arrow(2, 5.2, 3, 0, head_width=0.15, head_length=0.2, fc='black', ec='black') # RNN -> Gate
    ax.text(3.5, 5.4, "隐状态", size=10)

    # 4. Geometric Channel (Attention) - Conditionally Active
    ax.text(10, 6, "几何修正单元\n(Attention)\n高成本 O(N^2)", ha='center', va='center', size=12, bbox=attn_props)
    
    # Conditional Arrow
    ax.annotate("", xy=(10, 5.2), xytext=(7, 5.2), arrowprops=dict(arrowstyle="->", linestyle="dashed", linewidth=2))
    ax.text(8.5, 5.4, "gt > 阈值?", size=10, ha='center', color='#c62828')

    # Input to Attention (Skip Connection)
    ax.arrow(6, 8, 4, -1.2, head_width=0.15, head_length=0.2, fc='gray', ec='gray', linestyle='dotted')
    ax.text(8.5, 7.5, "路径 B (按需)", size=10, color='gray')

    # 5. Fusion
    ax.text(6, 2, "自适应融合\n(1-gt)*RNN + gt*Attn", ha='center', va='center', size=12, bbox=box_props)
    
    # Arrows to Fusion
    ax.arrow(2, 5.2, 3, -2.4, head_width=0.15, head_length=0.2, fc='black', ec='black') # RNN -> Fusion
    ax.arrow(10, 5.2, -3, -2.4, head_width=0.15, head_length=0.2, fc='black', ec='black') # Attn -> Fusion

    # 6. Output
    ax.arrow(6, 1.2, 0, -0.6, head_width=0.15, head_length=0.2, fc='black', ec='black')
    ax.text(6, 0.2, "输出状态 (Ht)", ha='center', va='center', size=12, bbox=io_props)

    # Title
    ax.text(6, 9.8, "图 1: 热力学门控网络 (TGN) 系统架构", ha='center', size=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig('patent_fig1_architecture.png', dpi=300)
    print("Generated patent_fig1_architecture.png (Chinese Version)")

if __name__ == "__main__":
    draw_tgn_architecture()
