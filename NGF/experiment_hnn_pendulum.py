import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import os

# ==========================================
# 1. 物理模拟：生成理想单摆数据 (Ground Truth)
# ==========================================
def get_pendulum_data(n_samples=50, t_span=[0, 10]):
    """
    生成单摆的相空间轨迹 (q, p)
    H = p^2/2 + (1-cos(q))  (假设 m=1, g=1, l=1)
    """
    def dynamics_fn(t, coords):
        q, p = coords
        dq_dt = p             # Hamilton Eq 1
        dp_dt = -np.sin(q)    # Hamilton Eq 2
        return [dq_dt, dp_dt]

    t_eval = np.linspace(t_span[0], t_span[1], 100)
    data = []
    
    # 生成多条不同初始能量的轨迹
    for _ in range(n_samples):
        q0 = np.random.uniform(-np.pi/2, np.pi/2) # 初始角度
        p0 = np.random.uniform(-1, 1)             # 初始动量
        
        # 使用 scipy 的高精度积分器生成真值
        sol = solve_ivp(dynamics_fn, t_span, [q0, p0], t_eval=t_eval, rtol=1e-10)
        
        # 整理数据: 输入(q,p) -> 输出(dq/dt, dp/dt)
        # 我们要训练网络预测导数
        q, p = sol.y
        dq_dt = p
        dp_dt = -np.sin(q)
        
        # [N, 2] state, [N, 2] derivative
        X = np.stack([q, p], axis=1)
        dX = np.stack([dq_dt, dp_dt], axis=1)
        data.append((X, dX))
        
    return data

# ==========================================
# 2. 模型定义：Baseline MLP vs HNN (哈密顿网络)
# ==========================================

class BaselineMLP(nn.Module):
    """普通神经网络：直接拟合动力学方程 F = ma"""
    def __init__(self, input_dim=2, hidden_dim=200):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, input_dim) # 直接输出 [dq, dp]
        )
        
    def forward(self, x):
        return self.net(x)

class HamiltonianNN(nn.Module):
    """
    哈密顿神经网络 (HNN)：学习哈密顿量 H(q,p)，而非直接学习导数
    这保证了能量守恒和辛几何结构。
    """
    def __init__(self, input_dim=2, hidden_dim=200):
        super().__init__()
        # 网络只输出一个标量：能量 H
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1) # 输出标量 H
        )
        
    def forward(self, x):
        # x = [q, p]
        # 我们需要对输入求导，所以必须开启 grad
        with torch.set_grad_enabled(True):
            x = x.requires_grad_(True)
            H = self.net(x)
            
            # 自动微分求梯度: dH/dx = [dH/dq, dH/dp]
            grads = torch.autograd.grad(H.sum(), x, create_graph=True)[0]
            
            dH_dq = grads[:, 0].unsqueeze(1)
            dH_dp = grads[:, 1].unsqueeze(1)
            
            # 哈密顿方程 (Hamilton's Equations)
            # dq/dt = dH/dp
            # dp/dt = -dH/dq
            dq_dt = dH_dp
            dp_dt = -dH_dq
            
            return torch.cat([dq_dt, dp_dt], dim=1)

# ==========================================
# 3. 实验运行：长程预测对比
# ==========================================
def train_and_evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running HNN Experiment on {device}...")
    
    # 1. 准备数据
    data = get_pendulum_data(n_samples=50)
    X_train = torch.tensor(np.concatenate([d[0] for d in data]), dtype=torch.float32).to(device)
    Y_train = torch.tensor(np.concatenate([d[1] for d in data]), dtype=torch.float32).to(device)
    
    # 2. 初始化模型
    mlp = BaselineMLP().to(device)
    hnn = HamiltonianNN().to(device)
    
    opt_mlp = torch.optim.Adam(mlp.parameters(), lr=1e-3)
    opt_hnn = torch.optim.Adam(hnn.parameters(), lr=1e-3)
    
    # 3. 训练
    print("Training models...")
    for epoch in range(2000):
        # Train MLP
        opt_mlp.zero_grad()
        pred_mlp = mlp(X_train)
        loss_mlp = ((pred_mlp - Y_train)**2).mean()
        loss_mlp.backward()
        opt_mlp.step()
        
        # Train HNN
        opt_hnn.zero_grad()
        pred_hnn = hnn(X_train)
        loss_hnn = ((pred_hnn - Y_train)**2).mean()
        loss_hnn.backward()
        opt_hnn.step()
        
        if epoch % 500 == 0:
            print(f"Epoch {epoch}: MLP Loss {loss_mlp.item():.6f}, HNN Loss {loss_hnn.item():.6f}")

    # 4. 长程预测评估 (Long-term Integration)
    print("\nEvaluating long-term energy conservation...")
    # 时间必须极长，才能暴露出 MLP 的数值耗散 (Energy Dissipation)
    t_span_test = [0, 500] 
    
    # 初始能量必须很大，处于非线性极强的区域 (接近倒立点)
    # q0 = 2.0 (rad) 接近 115度，非线性极强
    q0, p0 = 2.0, 0.0     
    
    # 使用训练好的神经网络作为 ODE 的导数函数
    def mlp_field(t, y):
        with torch.no_grad():
            y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(0).to(device)
            return mlp(y_tensor).cpu().numpy().flatten()
            
    def hnn_field(t, y):
        with torch.no_grad():
            y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(0).to(device)
            return hnn(y_tensor).cpu().numpy().flatten()

    # 积分预测
    t_eval = np.linspace(0, 500, 5000)
    sol_mlp = solve_ivp(mlp_field, t_span_test, [q0, p0], t_eval=t_eval)
    sol_hnn = solve_ivp(hnn_field, t_span_test, [q0, p0], t_eval=t_eval)
    
    # 计算能量 H = p^2/2 + (1-cos(q))
    def get_energy(q, p):
        return 0.5 * p**2 + (1 - np.cos(q))
        
    E_mlp = get_energy(sol_mlp.y[0], sol_mlp.y[1])
    E_hnn = get_energy(sol_hnn.y[0], sol_hnn.y[1])
    
    # 【修复绘图】：将绝对能量转换为相对误差，并使用对数坐标
    # 这样更能体现 HNN 在保持能量守恒上的指数级优势
    
    plt.figure(figsize=(12, 5))
    
    # 相空间轨迹
    plt.subplot(1, 2, 1)
    # 绘制全部轨迹，让 MLP 的螺旋收缩更加显眼
    plt.plot(sol_mlp.y[0], sol_mlp.y[1], 'b--', alpha=0.4, label='Baseline MLP', linewidth=1)
    plt.plot(sol_hnn.y[0], sol_hnn.y[1], 'r-', label='Hamiltonian NN (Ours)', linewidth=1.5)
    plt.title("Phase Space Trajectory (Spiraling vs Conserved)")
    plt.xlabel("Position (q)")
    plt.ylabel("Momentum (p)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 能量守恒曲线 (归一化并取对数)
    plt.subplot(1, 2, 2)
    # 计算能量漂移率: |E(t) - E(0)| / E(0)
    drift_mlp = np.abs((E_mlp - E_mlp[0]) / E_mlp[0]) + 1e-8
    drift_hnn = np.abs((E_hnn - E_hnn[0]) / E_hnn[0]) + 1e-8
    
    plt.semilogy(t_eval, drift_mlp, 'b--', label='Baseline MLP (Drift)', linewidth=2)
    plt.semilogy(t_eval, drift_hnn, 'r-', label='Hamiltonian NN (Ours)', linewidth=2)
    plt.title("Energy Drift (Log Scale)")
    plt.xlabel("Time")
    plt.ylabel("|(E_t - E_0) / E_0|")
    plt.legend()
    plt.grid(True, alpha=0.3, which="both")
    
    plt.tight_layout()
    save_path = "hnn_pendulum_result.png"
    plt.savefig(save_path, dpi=300)
    print(f"Experiment Complete. Visualization saved to {save_path}")

if __name__ == "__main__":
    train_and_evaluate()