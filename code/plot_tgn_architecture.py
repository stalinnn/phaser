
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

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
    draw_box(40, 5, 20, 5, "Input Sequence\n(Tokens)", color='#F5F5F5')
    
    # Arrow up
    draw_arrow(51, 12, 51, 20)

    # --- 2. Inertia Channel (Left/Main) ---
    # RNN Box
    draw_box(30, 20, 42, 10, "Inertia Channel\n(RNN / GRU / SSM)", color='#D6EAF8', edge='#2E86C1')
    
    ax.text(15, 25, "Low Entropy\nCost ≈ O(N)", ha='right', va='center', fontsize=10, color='#2E86C1', style='italic')

    # Arrow from RNN to Fork
    draw_arrow(51, 32, 51, 40)

    # --- 3. Maxwell Demon (Gate) ---
    # The decision point
    draw_box(42, 40, 18, 10, "Maxwell Demon\n(Entropy Gate)", color='#D7BDE2', edge='#884EA0')
    
    # Gate output signal
    ax.text(65, 45, "Rank Collapse?\nEntropy > τ ?", ha='left', va='center', fontsize=10, color='#884EA0', style='italic')

    # Arrow to Switch
    draw_arrow(51, 52, 51, 58)

    # --- 4. The Switch (Router) ---
    # Circle for switch
    switch_circle = patches.Circle((51, 60), 2, facecolor='white', edgecolor='black', linewidth=2)
    ax.add_patch(switch_circle)
    
    # Two paths
    # Path A: Inertia Only (Straight)
    draw_arrow(49, 60, 30, 75, color='#2E86C1', style='->', lw=2)
    ax.text(35, 68, "Low Energy\n(Keep Inertia)", ha='right', fontsize=10, color='#2E86C1')
    
    # Path B: Geometric Rescue (Right) - The Wormhole
    draw_arrow(53, 60, 75, 75, color='#C0392B', style='->', lw=3) 
    ax.text(70, 68, "High Energy\n(Attention)", ha='left', fontsize=10, color='#C0392B')

    # --- 5. Geometric Channel (Right/Sparse) ---
    # Attention Box
    draw_box(65, 75, 25, 10, "Geometric Channel\n(Sparse Attention)", color='#FADBD8', edge='#C0392B')
    ax.text(95, 80, "Cost ≈ O(K²)\nGlobal Context", ha='left', va='center', fontsize=10, color='#C0392B', style='italic')

    # Inertia Bypass (Left)
    draw_box(15, 75, 25, 10, "Identity / Residual", color='#EAEDED', edge='#999999')

    # --- 6. Fusion ---
    # Arrow from Left
    draw_arrow(27, 87, 45, 93)
    # Arrow from Right
    draw_arrow(77, 87, 57, 93)
    
    # Fusion Box
    draw_box(40, 93, 22, 5, "Adaptive Fusion\n(LayerNorm)", color='#F5F5F5')

    # --- Annotations ---
    # Title
    ax.text(51, 102, "Thermodynamic Gated Network (TGN) Architecture", 
            ha='center', fontsize=16, fontweight='bold')
    
    # Background Physics
    # Draw a faint background box to group components
    # rect_bg = patches.Rectangle((5, 15), 95, 80, linewidth=1, edgecolor='#DDDDDD', facecolor='none', linestyle='--')
    # ax.add_patch(rect_bg)
    # ax.text(98, 17, "Thermodynamic System Boundary", ha='right', fontsize=8, color='gray')

    # Save
    out_path = get_project_root() / "paper_archive" / "figures" / "tgn_architecture_diagram.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Architecture diagram saved to {out_path}")
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
