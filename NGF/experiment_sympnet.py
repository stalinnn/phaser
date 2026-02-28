import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ==========================================
# 1. 物理模拟：简单的谐振子 (Harmonic Oscillator)
# ==========================================
def get_oscillator_data(n_samples=50, t_span=[0, 10]):
    """
    生成谐振子数据 H = 0.5 * (p^2 + q^2)
    解是圆周运动 q = A cos(t + phi), p = -A sin(t + phi)
    """
    def dynamics_fn(t, y):
        q, p = y
        return [p, -q] # dq/dt = p, dp/dt = -q

    t_eval = np.linspace(t_span[0], t_span[1], 100)
    data = []
    for _ in range(n_samples):
        y0 = np.random.randn(2)
        sol = solve_ivp(dynamics_fn, t_span, y0, t_eval=t_eval)
        
        # 构造训练对: (x_t) -> (x_{t+dt})
        # 辛网络通常学习离散映射 (Map)，而不是连续流 (Flow)
        # 我们学习步长 dt 的演化
        X_t = sol.y[:, :-1].T
        X_next = sol.y[:, 1:].T
        data.append((X_t, X_next))
        
    return data

# ==========================================
# 2. 模型定义：Symplectic Network (SympNet)
# ==========================================

class LinearSympNet(nn.Module):
    """
    最简单的辛网络：由一系列辛矩阵乘法组成。
    每一层都是一个 Upper/Lower Triangular 变换，这种变换天然保体积。
    """
    def __init__(self, dim=2, layers=3):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(layers):
            # LA-SympNet (Linear-Activation) 的变体
            # Up变换: q' = q + f(p), p' = p
            # Low变换: p' = p + g(q), q' = q
            self.layers.append(nn.Linear(1, 1)) # 用于拟合力场
            self.layers.append(nn.Linear(1, 1))

    def forward(self, x):
        # x: [batch, 2] -> [q, p]
        # 【修复】增加容错，确保切片后维度正确
        q, p = x[:, 0:1], x[:, 1:2]
        
        dt = 0.1 # 假设的时间步长
        
        for i in range(0, len(self.layers), 2):
            # 1. p-update (Kick)
            # p_{k+1} = p_k - dt * H_q(q_k)
            # 这里用神经网络拟合力场 F(q)
            force = torch.tanh(self.layers[i](q)) 
            p = p + dt * force
            
            # 2. q-update (Drift)
            # q_{k+1} = q_k + dt * H_p(p_{k+1})
            velocity = p # 简单谐振子 H_p = p
            q = q + dt * velocity
            
        return torch.cat([q, p], dim=1)

# ==========================================
# 3. 对比实验
# ==========================================
class StandardMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 32), nn.Tanh(),
            nn.Linear(32, 32), nn.Tanh(),
            nn.Linear(32, 2)
        )
    def forward(self, x):
        # 也是预测 x_{t+1} = x_t + f(x_t)
        return x + 0.1 * self.net(x)

def train_and_vis():
    device = torch.device("cpu")
    print("Training Symplectic Network...")
    
    # 获取数据 [q, p] -> [q_next, p_next]
    data = get_oscillator_data()
    X = torch.tensor(np.concatenate([d[0] for d in data]), dtype=torch.float32)
    Y = torch.tensor(np.concatenate([d[1] for d in data]), dtype=torch.float32)
    
    # 【修复】X 的形状目前是 [N*99, 2]，我们需要切分成 q 和 p
    # 在 LinearSympNet.forward 中，我们用了 q = x[:, 0:1]
    # 如果 X 的形状不对，就会导致切片异常
    
    print(f"Data shape: {X.shape}")
    mlp = StandardMLP().to(device)
    
    # 2. 初始化模型
    symp = LinearSympNet().to(device)
    mlp = StandardMLP().to(device)
    
    opt_s = torch.optim.Adam(symp.parameters(), lr=0.01)
    opt_m = torch.optim.Adam(mlp.parameters(), lr=0.01)
    
    for ep in range(1000):
        # Train
        loss_s = ((symp(X) - Y)**2).mean()
        opt_s.zero_grad(); loss_s.backward(); opt_s.step()
        
        loss_m = ((mlp(X) - Y)**2).mean()
        opt_m.zero_grad(); loss_m.backward(); opt_m.step()
        
    # Long-term prediction
    print("Evaluating long-term stability...")
    traj_s = [torch.tensor([[1.0, 0.0]])]
    traj_m = [torch.tensor([[1.0, 0.0]])]
    
    for _ in range(500):
        traj_s.append(symp(traj_s[-1]))
        traj_m.append(mlp(traj_m[-1]))
        
    traj_s = torch.cat(traj_s).detach().numpy()
    traj_m = torch.cat(traj_m).detach().numpy()
    
    plt.figure(figsize=(6, 6))
    plt.plot(traj_m[:,0], traj_m[:,1], 'b--', alpha=0.6, label='Standard MLP')
    plt.plot(traj_s[:,0], traj_s[:,1], 'r-', label='Symplectic Net')
    plt.title("Symplectic Network vs MLP (Phase Space)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.savefig('sympnet_result.png', dpi=300)
    print("Saved to sympnet_result.png")

if __name__ == "__main__":
    train_and_vis()