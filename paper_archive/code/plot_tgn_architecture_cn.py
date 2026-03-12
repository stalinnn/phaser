
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS'] 
plt.rcParams['axes.unicode_minus'] = False

def draw_tgn_architecture():
    # Setup canvas
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')  # Turn off axis

    # Function to draw box with text
    def draw_box(x, y, w, h, text, color='#E0E0E0', edge='#333333', fontsize=12, alpha=1.0):
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=1", 
                                      linewidth=2, edgecolor=edge, facecolor=color, alpha=alpha)
        ax.add_patch(rect)
        ax.text(x + w/2 + 1, y + h/2 + 1, text, ha='center', va='center', 
                fontsize=fontsize, fontweight='bold', color='#333333')
        return rect

    # Function to draw arrow
    def draw_arrow(x1, y1, x2, y2, color='#333333', style='->', lw=2):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle=style, color=color, lw=lw, mutation_scale=20))

    # --- 1. Inputs ---
    draw_box(40, 5, 20, 5, "输入序列\n(Tokens)", color='#F5F5F5')
    
    # Arrow up
    draw_arrow(51, 12, 51, 20)

    # --- 2. Inertia Channel (Left/Main) ---
    # RNN Box
    draw_box(30, 20, 42, 10, "惯性通道\n(RNN / GRU / SSM)", color='#D6EAF8', edge='#2E86C1')
    
    ax.text(15, 25, "低熵态\n成本 ≈ O(N)", ha='right', va='center', fontsize=10, color='#2E86C1', style='italic')

    # Arrow from RNN to Fork
    draw_arrow(51, 32, 51, 40)

    # --- 3. Maxwell Demon (Gate) ---
    # The decision point
    draw_box(42, 40, 18, 10, "麦克斯韦妖\n(熵门控)", color='#D7BDE2', edge='#884EA0')
    
    # Gate output signal
    ax.text(65, 45, "秩坍缩?\n熵 > τ ?", ha='left', va='center', fontsize=10, color='#884EA0', style='italic')

    # Arrow to Switch
    draw_arrow(51, 52, 51, 58)

    # --- 4. The Switch (Router) ---
    # Circle for switch
    switch_circle = patches.Circle((51, 60), 2, facecolor='white', edgecolor='black', linewidth=2)
    ax.add_patch(switch_circle)
    
    # Two paths
    # Path A: Inertia Only (Straight)
    draw_arrow(49, 60, 30, 75, color='#2E86C1', style='->', lw=2)
    ax.text(35, 68, "低能耗\n(维持惯性)", ha='right', fontsize=10, color='#2E86C1')
    
    # Path B: Geometric Rescue (Right) - The Wormhole
    draw_arrow(53, 60, 75, 75, color='#C0392B', style='->', lw=3) 
    ax.text(70, 68, "高能耗\n(几何救援)", ha='left', fontsize=10, color='#C0392B')

    # --- 5. Geometric Channel (Right/Sparse) ---
    # Attention Box
    draw_box(65, 75, 25, 10, "几何通道\n(稀疏注意力)", color='#FADBD8', edge='#C0392B')
    ax.text(95, 80, "成本 ≈ O(K²)\n全局上下文", ha='left', va='center', fontsize=10, color='#C0392B', style='italic')

    # Inertia Bypass (Left)
    draw_box(15, 75, 25, 10, "恒等 / 残差", color='#EAEDED', edge='#999999')

    # --- 6. Fusion ---
    # Arrow from Left
    draw_arrow(27, 87, 45, 93)
    # Arrow from Right
    draw_arrow(77, 87, 57, 93)
    
    # Fusion Box
    draw_box(40, 93, 22, 5, "自适应融合\n(LayerNorm)", color='#F5F5F5')

    # --- Annotations ---
    # Title
    ax.text(51, 102, "热力学门控网络 (TGN) 核心架构", 
            ha='center', fontsize=16, fontweight='bold')
    
    # Save
    out_path = get_project_root() / "paper_archive" / "figures" / "tgn_architecture_diagram_cn.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Architecture diagram (Chinese) saved to {out_path}")
    plt.close()

def get_project_root() -> Path:
    try:
        start = Path(__file__).resolve().parent
    except NameError:
        start = Path.cwd()
    if (start / "paper_archive").exists():
        return start
    for p in [start] + list(start.parents):
        if (p / "paper_archive").exists():
            return p
    return start

if __name__ == "__main__":
    draw_tgn_architecture()
