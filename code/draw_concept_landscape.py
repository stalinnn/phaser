import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

def rastrigin(X, Y, A=10):
    return A * 2 + (X**2 - A * np.cos(2 * np.pi * X)) + (Y**2 - A * np.cos(2 * np.pi * Y))

def generate_concept_figure(out_path='figures/concept_energy_landscape.png'):
    # Setup grid
    x = np.linspace(-4, 4, 100)
    y = np.linspace(-4, 4, 100)
    X, Y = np.meshgrid(x, y)
    Z = rastrigin(X, Y)

    # Setup Plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot Surface (Glassy Landscape)
    # Make surface more transparent to see lines behind peaks
    surf = ax.plot_surface(X, Y, Z, cmap=cm.viridis, alpha=0.6, linewidth=0, antialiased=True)
    
    # --- Trajectory 1: Local Diffusion (Trapped) ---
    # Start at (-2, -2) valley, try to go to (0,0) but get trapped in local ring
    path1_x = np.linspace(-2.5, -1.5, 30)
    path1_y = np.linspace(-2.5, -1.5, 30)
    # Add spiral noise to simulate trapped diffusion
    theta = np.linspace(0, 4*np.pi, 30)
    path1_x += 0.3 * np.cos(theta)
    path1_y += 0.3 * np.sin(theta)
    
    # Lift path significantly above surface to ensure visibility
    path1_z = rastrigin(path1_x, path1_y) + 15 
    
    ax.plot(path1_x, path1_y, path1_z, color='#e74c3c', linewidth=3, linestyle='-', label='Local Diffusion (Trapped)', zorder=10)
    # Mark start and end of trap
    ax.scatter(path1_x[-1], path1_y[-1], path1_z[-1], color='#e74c3c', s=80, marker='x', zorder=10)

    # --- Trajectory 2: Geometric Tunneling (Attention) ---
    # Start at (-3, 3) (another trap), tunnel to (0,0) (Global Min)
    start_point = np.array([-3, 3])
    end_point = np.array([0, 0])
    
    # Draw an arc (Wormhole)
    t = np.linspace(0, 1, 100)
    path2_x = (1-t)*start_point[0] + t*end_point[0]
    path2_y = (1-t)*start_point[1] + t*end_point[1]
    # Arc height: Parabola (much higher to be visible)
    base_z = np.maximum(rastrigin(path2_x, path2_y), 0)
    path2_z = base_z + 80 * np.sin(np.pi * t) + 10 # High arch bridge
    
    ax.plot(path2_x, path2_y, path2_z, color='#2ecc71', linewidth=4, label='Geometric Tunneling (Attention)', zorder=10)
    
    # Mark Start and End
    start_z = rastrigin(start_point[0], start_point[1]) + 10
    end_z = rastrigin(end_point[0], end_point[1]) + 10
    ax.scatter(start_point[0], start_point[1], start_z, color='black', s=100, zorder=10)
    ax.scatter(end_point[0], end_point[1], end_z, color='#f1c40f', s=200, marker='*', label='Global Optima', zorder=10)

    # View angle - Higher elevation to look down into valleys
    ax.view_init(elev=60, azim=-30)
    
    # Remove axis clutter for cleaner look
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    
    # Styling
    ax.set_title('Thermodynamic Gated Network: Escaping Glassy Landscapes', fontweight='bold', pad=20)
    
    # Legend
    ax.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9)
    
    # Save
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    print(f"Saved concept figure to {out_path}")

if __name__ == "__main__":
    generate_concept_figure()
