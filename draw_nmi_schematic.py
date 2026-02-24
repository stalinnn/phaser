import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# ==========================================
# 1. NMI 风格配色 (Nature Palette)
# ==========================================
COLORS = {
    'background': '#FFFFFF',
    'rnn_fill': '#4E84C4',      # Steel Blue (Inertial)
    'rnn_edge': '#295F99',
    'attn_line': '#D64045',     # Firebrick (Geometric)
    'attn_glow': '#FFD700',     # Gold (Glow)
    'gate_fill': '#2A2A2A',     # Dark Grey (Gate)
    'text': '#333333'
}

def draw_3d_cube(ax, center, size, color, alpha=0.9, label=None):
    """绘制一个伪 3D 立方体"""
    x, y = center
    w = size
    h = size
    d = size * 0.4  # depth
    
    # Front face
    rect = patches.Rectangle((x - w/2, y - h/2), w, h, linewidth=1, edgecolor=COLORS['rnn_edge'], facecolor=color, alpha=alpha, zorder=10)
    ax.add_patch(rect)
    
    # Top face
    poly_top = patches.Polygon([
        (x - w/2, y + h/2), (x - w/2 + d, y + h/2 + d),
        (x + w/2 + d, y + h/2 + d), (x + w/2, y + h/2)
    ], closed=True, linewidth=1, edgecolor=COLORS['rnn_edge'], facecolor=color, alpha=alpha*0.8, zorder=9)
    ax.add_patch(poly_top)
    
    # Side face
    poly_side = patches.Polygon([
        (x + w/2, y - h/2), (x + w/2 + d, y - h/2 + d),
        (x + w/2 + d, y + h/2 + d), (x + w/2, y + h/2)
    ], closed=True, linewidth=1, edgecolor=COLORS['rnn_edge'], facecolor=color, alpha=alpha*0.6, zorder=8)
    ax.add_patch(poly_side)
    
    if label:
        ax.text(x, y, label, ha='center', va='center', fontsize=10, color='white', fontweight='bold', zorder=11)

def draw_glowing_curve(ax, p1, p2, height, color, glow_color):
    """绘制带有发光效果的贝塞尔曲线 (Attention)"""
    # 贝塞尔控制点
    mid_x = (p1[0] + p2[0]) / 2
    mid_y = p1[1] + height
    
    # Path patch for Bezier
    verts = [
        (p1[0], p1[1]),  # Start
        (p1[0], mid_y),  # Control 1 (vertical up)
        (p2[0], mid_y),  # Control 2 (horizontal over)
        (p2[0], p2[1])   # End
    ]
    
    # 模拟贝塞尔曲线 (简化为二次)
    path_codes = [patches.Path.MOVETO, patches.Path.CURVE4, patches.Path.CURVE4, patches.Path.CURVE4]
    path = patches.Path(verts, path_codes)
    
    # Glow effect (multiple transparent lines)
    for w, alpha in [(6, 0.1), (4, 0.2), (2, 0.4)]:
        patch = patches.PathPatch(path, facecolor='none', edgecolor=glow_color, linewidth=w, alpha=alpha, zorder=5)
        ax.add_patch(patch)
        
    # Main line
    patch = patches.PathPatch(path, facecolor='none', edgecolor=color, linewidth=1.5, zorder=6)
    ax.add_patch(patch)

def draw_gate(ax, center, radius):
    """绘制门控阀门图标 (麦克斯韦妖)"""
    x, y = center
    # Diamond shape
    gate = patches.RegularPolygon((x, y), numVertices=4, radius=radius, orientation=0, 
                                  facecolor=COLORS['gate_fill'], edgecolor='black', zorder=20)
    ax.add_patch(gate)
    ax.text(x, y, "G", color='white', ha='center', va='center', fontweight='bold', fontsize=9, zorder=21)

def create_schematic():
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.set_aspect('equal')
    ax.axis('off') # Hide axis
    
    # 1. 绘制时间轴 (Arrow)
    ax.arrow(0.5, 1, 11, 0, head_width=0.2, head_length=0.3, fc='gray', ec='gray', width=0.02, zorder=1)
    ax.text(11.5, 0.7, "Time (t)", fontsize=12, style='italic', color='gray')

    # 2. 绘制 RNN 节点 (Inertial Channel)
    nodes_x = [1.5, 3.5, 5.5, 7.5, 9.5]
    node_y = 1.5
    size = 0.8
    
    for i, x in enumerate(nodes_x):
        label = f"$h_{{{i+1}}}$"
        draw_3d_cube(ax, (x, node_y), size, COLORS['rnn_fill'], label=label)
        
        # RNN 连接线
        if i < len(nodes_x) - 1:
            ax.arrow(x + size/2 + 0.2, node_y, 1.0, 0, head_width=0.15, fc=COLORS['rnn_fill'], ec=COLORS['rnn_fill'], zorder=2)

    # 3. 绘制 Attention 连接 (Geometric Channel) - 仅在关键点触发
    # 从 h1 -> h3 (Short skip)
    draw_glowing_curve(ax, (nodes_x[0], node_y + size/2), (nodes_x[2], node_y + size/2), 1.5, COLORS['attn_line'], COLORS['attn_glow'])
    
    # 从 h1 -> h5 (Long skip)
    draw_glowing_curve(ax, (nodes_x[0], node_y + size/2), (nodes_x[4], node_y + size/2), 3.0, COLORS['attn_line'], COLORS['attn_glow'])
    
    # 从 h2 -> h5 (Another skip)
    draw_glowing_curve(ax, (nodes_x[1], node_y + size/2), (nodes_x[4], node_y + size/2), 2.2, COLORS['attn_line'], COLORS['attn_glow'])

    # 4. 绘制门控 (Maxwell's Demon)
    # 在 h3 和 h5 上方放置门控
    gate_y = node_y + 1.2
    draw_gate(ax, (nodes_x[2], gate_y), 0.25)
    draw_gate(ax, (nodes_x[4], gate_y), 0.25)
    
    # Gate connection
    ax.plot([nodes_x[2], nodes_x[2]], [node_y + size/2, gate_y - 0.25], 'k:', lw=1, zorder=5)
    ax.plot([nodes_x[4], nodes_x[4]], [node_y + size/2, gate_y - 0.25], 'k:', lw=1, zorder=5)

    # 5. 添加标注
    ax.text(0.5, 3.5, "Geometric Channel\n(High Entropy / Attention)", color=COLORS['attn_line'], fontsize=11, fontweight='bold', va='center')
    ax.text(0.5, 1.5, "Inertial Channel\n(Low Entropy / RNN)", color=COLORS['rnn_fill'], fontsize=11, fontweight='bold', va='center')
    
    # Title
    ax.text(6, 5.5, "Thermodynamic Gated Network (TGN) Architecture", ha='center', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig('nmi_schematic_fig1.png', dpi=300, bbox_inches='tight')
    print("Generated nmi_schematic_fig1.png")

if __name__ == "__main__":
    create_schematic()
