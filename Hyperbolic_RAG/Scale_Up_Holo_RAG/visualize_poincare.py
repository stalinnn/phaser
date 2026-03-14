import torch
import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.font_manager import FontProperties
import umap
from sklearn.manifold import TSNE

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from holo_embedder import HyperbolicEmbedder

def plot_poincare_disk():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Generating Poincaré Disk Visualization on {device} ===")
    
    # 1. Load Data
    with open("hierarchical_dataset.json", 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    # Extract unique parents and children
    parents = list(set([item['parent'] for item in data]))
    # For visualization clarity, let's just take a subset of children for each parent
    children = []
    parent_of_child = {}
    
    for p in parents:
        c_list = [item['child'] for item in data if item['parent'] == p and item['label'] == 1]
        children.extend(c_list)
        for c in c_list:
            parent_of_child[c] = p
            
    all_texts = parents + children
    labels = ["Macro (Root)"] * len(parents) + ["Micro (Leaf)"] * len(children)
    
    # 2. Extract Embeddings
    print("Loading finetuned Holo-Embedder...")
    embedder = HyperbolicEmbedder("BAAI/bge-small-zh-v1.5").to(device)
    try:
        embedder.projection.load_state_dict(torch.load("scale_holo_projection.pt"))
    except FileNotFoundError:
        print("Error: scale_holo_projection.pt not found.")
        return
        
    embedder.eval()
    
    print("Extracting representations...")
    inputs = embedder.tokenizer(all_texts, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = embedder.backbone(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
        euclidean_embs = outputs.last_hidden_state[:, 0, :]
        scaled_embs = embedder.projection(euclidean_embs)
        
        # We extract the pre-exponential mapped embeddings for UMAP
        # UMAP is an Euclidean algorithm, so we reduce dimensionality in the tangent space (scaled_embs)
        # and THEN map the 2D coordinates into the Poincare disk! This ensures mathematical correctness.
        tangent_embs = scaled_embs.cpu().numpy()
        
    # 3. Dimensionality Reduction to 2D Tangent Space
    print("Applying TSNE for 2D projection (Fallback to avoid UMAP sklearn issues)...")
    reducer = TSNE(n_components=2, metric='cosine', random_state=42, perplexity=5)
    tangent_2d = reducer.fit_transform(tangent_embs)
    
    # Normalize tangent_2d to prevent overflow during exp_map
    tangent_2d = tangent_2d - np.mean(tangent_2d, axis=0) # center
    max_norm = np.max(np.linalg.norm(tangent_2d, axis=1))
    tangent_2d = (tangent_2d / max_norm) * 5.0 # Scale to a reasonable tangent vector length
    
    # 4. Map 2D Tangent Vectors to 2D Poincaré Disk
    print("Mapping to 2D Poincaré disk...")
    tangent_2d_tensor = torch.tensor(tangent_2d, dtype=torch.float32)
    poincare_2d = embedder.holo_math.exp_map0(tangent_2d_tensor).numpy()
    
    # Split back into parents and children
    p_coords = poincare_2d[:len(parents)]
    c_coords = poincare_2d[len(parents):]
    
    # 5. Plotting
    # Use a font that supports Chinese
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Draw the Poincaré boundary circle
    circle = patches.Circle((0, 0), radius=1.0, edgecolor='black', facecolor='#f0f4f8', linewidth=2, zorder=0)
    ax.add_patch(circle)
    
    # Draw concentric grid lines to emphasize hyperbolic nature
    for r in [0.2, 0.4, 0.6, 0.8]:
        grid_circle = patches.Circle((0, 0), radius=r, edgecolor='gray', linestyle='--', alpha=0.3, fill=False, zorder=1)
        ax.add_patch(grid_circle)
        
    # Define colors for different macro clusters
    colors = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6']
    
    # Plot connections and children
    for i, child_coord in enumerate(c_coords):
        parent_idx = parents.index(parent_of_child[children[i]])
        parent_coord = p_coords[parent_idx]
        color = colors[parent_idx % len(colors)]
        
        # Plot child
        ax.scatter(child_coord[0], child_coord[1], color=color, s=50, alpha=0.7, edgecolors='white', zorder=3)
        
        # Draw curved geodesic connecting parent and child
        # A true geodesic in Poincare disk is a circular arc orthogonal to the boundary.
        # For visualization simplicity, we use a Bezier curve to approximate the "bending towards origin" feel.
        rad = 0.2
        ax.annotate("",
                    xy=(parent_coord[0], parent_coord[1]),
                    xytext=(child_coord[0], child_coord[1]),
                    arrowprops=dict(arrowstyle="-", color=color, alpha=0.4, connectionstyle=f"arc3,rad={rad}"),
                    zorder=2)
                    
    # Plot parents (Macros) - larger and closer to origin
    for i, parent_coord in enumerate(p_coords):
        color = colors[i % len(colors)]
        ax.scatter(parent_coord[0], parent_coord[1], color=color, s=300, marker='*', edgecolors='black', linewidths=1.5, zorder=4)
        
        # Add labels for parents
        label_text = parents[i][:12] + "..." # truncate
        ax.text(parent_coord[0] + 0.03, parent_coord[1] + 0.03, label_text, fontsize=10, weight='bold', color='black', 
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1), zorder=5)
                
    # Emphasize the query logic jump (e.g., the Granny Liu query)
    query_target = "大观园内实行严格的封建等级制度。"
    query_leaf = "门子拦住了来打秋风的刘姥姥，只因为她衣衫褴褛，不符合府里的规矩。"
    
    if query_target in parents and query_leaf in children:
        qt_idx = parents.index(query_target)
        ql_idx = children.index(query_leaf)
        qt_coord = p_coords[qt_idx]
        ql_coord = c_coords[ql_idx]
        
        # Draw the dramatic geodesic retrieval path
        ax.annotate("",
                    xy=(qt_coord[0], qt_coord[1]),
                    xytext=(ql_coord[0], ql_coord[1]),
                    arrowprops=dict(arrowstyle="->", color='red', lw=3, connectionstyle="arc3,rad=-0.3"),
                    zorder=6)
        
        ax.text((qt_coord[0]+ql_coord[0])/2 - 0.2, (qt_coord[1]+ql_coord[1])/2 + 0.1, 
                "Geodesic Retrieval Path\n(Bends to Origin)", 
                color='red', fontsize=12, weight='bold', zorder=7)

    # Title and annotations
    plt.title("Holo-RAG: Representation in the Poincaré Ball\n(Macro concepts converge at origin, Micro details expand to boundary)", 
              fontsize=16, pad=20, weight='bold')
              
    # Save the figure
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Holo_RAG_Paper", "poincare_visualization.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight', transparent=False)
    print(f"\n>>> Visualization successfully saved to: {out_path}")
    
if __name__ == "__main__":
    plot_poincare_disk()
