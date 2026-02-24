import numpy as np
import matplotlib.pyplot as plt
from simulation import EconomySimulationBase

# --- 复杂的非凸 2D 势能面 (Rosenbrock-like but with multiple minima) ---
def complex_potential(x, y):
    """
    一个经典的复杂地形：Rosenbrock function (香蕉函数) 的变体
    有一个弯曲的长峡谷。Global Min 在 (1, 1)。
    对于盲目搜索者，这非常难，因为梯度方向往往不指向最小值。
    """
    # V(x,y) = (1-x)^2 + 100 * (y - x^2)^2
    # 这是一个弯曲的山谷。
    return (1 - x)**2 + 100 * (y - x**2)**2

def potential_gradient(x, y):
    """
    返回 [dV/dx, dV/dy]
    """
    # dV/dx = -2(1-x) + 100 * 2(y-x^2) * (-2x)
    #       = -2 + 2x - 400x(y-x^2)
    #       = -2 + 2x - 400xy + 400x^3
    dx = -2 * (1 - x) - 400 * x * (y - x**2)
    
    # dV/dy = 100 * 2(y-x^2) * 1
    #       = 200(y - x^2)
    dy = 200 * (y - x**2)
    return np.array([dx, dy])

def metric_tensor_inverse(x, y):
    """
    构造黎曼度量张量的逆 G^-1 (Fisher Information Matrix 的逆)
    对于 Rosenbrock，Hessian 是一个很好的近似。
    Natural Gradient Descent 利用 Hessian 的逆来校正方向。
    """
    # 这里我们简化，构造一个能够感知"峡谷"方向的张量
    # 在峡谷里，沿着峡谷底部的曲率小，垂直峡谷壁的曲率大。
    # G 应该反映这种曲率。
    
    # 为了演示，我们直接使用真实的 Hessian 的近似
    # H = [[ dxx, dxy], [dyx, dyy]]
    # dxx = 2 - 400y + 1200x^2
    # dxy = -400x
    # dyy = 200
    
    hxx = 2 - 400*y + 1200*x**2
    hxy = -400*x
    hyy = 200
    
    H = np.array([[hxx, hxy], [hxy, hyy]])
    
    # 为了保证正定性和数值稳定性，做正则化
    eigenvalues, eigenvectors = np.linalg.eigh(H)
    eigenvalues = np.abs(eigenvalues) + 0.1 # 避免 0 或负特征值
    
    # 重构 Hessian
    H_spd = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    
    # G^-1 近似为 H^-1 (牛顿法/自然梯度)
    G_inv = np.linalg.inv(H_spd)
    
    return G_inv

class LangevinComparison:
    def __init__(self, T=1.0):
        self.T = T
        # Start at a difficult point (-1, 1)
        # Target is (1, 1)
        # The agents must travel along the curved banana valley.
        self.start_pos = np.array([-1.0, 1.0])
        
        self.path_scalar = [self.start_pos.copy()]
        self.path_tensor = [self.start_pos.copy()]
        
    def run(self, steps=1000, dt=0.001):
        print(f"Running Langevin Comparison (T={self.T})...")
        
        curr_scalar = self.start_pos.copy()
        curr_tensor = self.start_pos.copy()
        
        np.random.seed(42)
        
        for t in range(steps):
            # --- 1. Euclidean Langevin Dynamics (Baseline) ---
            # dot_x = - grad V + sqrt(2T) * noise
            grad_s = potential_gradient(curr_scalar[0], curr_scalar[1])
            noise_s = np.random.normal(0, 1, 2)
            
            dx_s = -grad_s * dt + np.sqrt(2 * self.T * dt) * noise_s
            curr_scalar += dx_s
            self.path_scalar.append(curr_scalar.copy())
            
            # --- 2. Riemannian Langevin Dynamics (Natural Gradient) ---
            # dot_x = - G^-1 grad V + sqrt(2T) * G^-1/2 * noise
            # 注意：这是自然梯度 (Natural Gradient) 的随机版本
            
            grad_t = potential_gradient(curr_tensor[0], curr_tensor[1])
            G_inv = metric_tensor_inverse(curr_tensor[0], curr_tensor[1])
            
            # Deterministic drift (Newton-like step)
            drift = - G_inv @ grad_t
            
            # Stochastic diffusion (Anisotropic noise)
            # We need sqrt(G^-1). Since G^-1 is SPD, we can use Cholesky or Eig.
            # L @ L.T = G_inv. So L corresponds to G^-1/2 acting on noise
            # Noise term = L @ standard_normal
            try:
                L = np.linalg.cholesky(G_inv)
            except:
                # Fallback if numerical issues
                L = np.eye(2) * 0.01
                
            noise_t_raw = np.random.normal(0, 1, 2)
            diffusion = L @ noise_t_raw
            
            dx_t = drift * dt + np.sqrt(2 * self.T * dt) * diffusion
            curr_tensor += dx_t
            self.path_tensor.append(curr_tensor.copy())
            
    def plot(self):
        path_s = np.array(self.path_scalar)
        path_t = np.array(self.path_tensor)
        
        # Setup Contour Plot
        x_grid = np.linspace(-2, 2, 100)
        y_grid = np.linspace(-1, 3, 100)
        X, Y = np.meshgrid(x_grid, y_grid)
        Z = complex_potential(X, Y)
        
        plt.figure(figsize=(10, 8))
        
        # Log scale contour for better visibility of the valley
        plt.contourf(X, Y, np.log(Z+1), levels=30, cmap='viridis_r', alpha=0.6)
        plt.colorbar(label='Log Potential Energy')
        
        # Plot Paths
        plt.plot(path_s[:,0], path_s[:,1], 'r-', linewidth=1, alpha=0.7, label='Euclidean Dynamics (Isotropic Noise)')
        plt.plot(path_t[:,0], path_t[:,1], 'w-', linewidth=2, label='Riemannian Dynamics (Anisotropic Noise)')
        
        # Markers
        plt.plot(self.start_pos[0], self.start_pos[1], 'ko', label='Start')
        plt.plot(1, 1, 'r*', markersize=15, label='Global Opt')
        
        plt.title('Riemannian vs Euclidean Langevin Dynamics\n(The "Banana Valley" Problem)', fontsize=14)
        plt.xlabel('X (Resource A)')
        plt.ylabel('Y (Resource B)')
        plt.legend()
        
        import os
        if not os.path.exists('figures'): os.makedirs('figures')
        plt.savefig('figures/riemannian_langevin.png', dpi=300)
        print("Saved to figures/riemannian_langevin.png")

if __name__ == "__main__":
    # T needs to be small enough to settle, but large enough to show fluctuation
    sim = LangevinComparison(T=5.0)
    sim.run(steps=2000, dt=0.0005) 
    sim.plot()

