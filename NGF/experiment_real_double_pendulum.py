import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import os

# ==========================================
# 1. 数据加载器 (处理真实实验数据)
# ==========================================
def load_real_double_pendulum_data(csv_path):
    """
    加载 IBM Double Pendulum 数据集 (CSV 格式)
    CSV 包含: Time, Theta1, Theta2 (角度)
    我们需要数值差分计算角速度 Omega1, Omega2 作为状态变量
    """
    try:
        # 假设数据格式: time, theta1, theta2
        # 注意：真实数据可能有噪声，我们需要简单的平滑处理
        df = pd.read_csv(csv_path)
        t = df.iloc[:, 0].values
        theta1 = df.iloc[:, 1].values
        theta2 = df.iloc[:, 2].values
        
        # 计算角速度 (Finite Difference)
        dt = t[1] - t[0]
        omega1 = np.gradient(theta1, dt)
        omega2 = np.gradient(theta2, dt)
        
        # 构造状态 X = [theta1, theta2, omega1, omega2] (近似 q, p)
        X = np.stack([theta1, theta2, omega1, omega2], axis=1)
        
        # 构造输入输出对: X_t -> dX_dt
        # HNN 需要学习的是导数场 dX/dt
        dX = np.gradient(X, dt, axis=0)
        
        return torch.tensor(X, dtype=torch.float32), torch.tensor(dX, dtype=torch.float32)
        
    except Exception as e:
        print(f"Error loading real data: {e}")
        print("Falling back to synthetic data generation for demo.")
        return generate_synthetic_double_pendulum()

def generate_synthetic_double_pendulum(n_steps=1000):
    """(Fallback) 生成模拟双摆数据，以防没有真实 csv"""
    from scipy.integrate import solve_ivp
    def deriv(t, y):
        theta1, theta2, p1, p2 = y
        # 简化双摆方程 (哈密顿形式极其复杂，这里用近似)
        # 仅作为占位符，真实实验请务必使用 load_real_double_pendulum_data
        return [p1, p2, -2*np.sin(theta1), -np.sin(theta2)]
    
    sol = solve_ivp(deriv, [0, 10], [1.0, 1.0, 0.0, 0.0], t_eval=np.linspace(0, 10, n_steps))
    X = sol.y.T
    dt = 10/n_steps
    dX = np.gradient(X, dt, axis=0)
    return torch.tensor(X, dtype=torch.float32), torch.tensor(dX, dtype=torch.float32)

# ==========================================
# 2. HNN 模型 (4D 相空间)
# ==========================================
class DoublePendulumHNN(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=200):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1) # Output Scalar Energy H
        )
    
    def forward(self, x):
        # x = [q1, q2, p1, p2]
        with torch.set_grad_enabled(True):
            x = x.requires_grad_(True)
            H = self.net(x)
            grads = torch.autograd.grad(H.sum(), x, create_graph=True)[0]
            
            # Hamilton Equations for 2 Degrees of Freedom
            # dq/dt = dH/dp
            # dp/dt = -dH/dq
            dH_dq = grads[:, :2] # [dH/dq1, dH/dq2]
            dH_dp = grads[:, 2:] # [dH/dp1, dH/dp2]
            
            return torch.cat([dH_dp, -dH_dq], dim=1)

# ==========================================
# 3. 训练主循环
# ==========================================
def train_real_world_hnn():
    # 假设你已经把 csv 上传到了这个路径
    # 如果没有，它会自动使用模拟数据降级运行
    csv_path = "/gz-data/real_physics/double_pendulum_track.csv"
    
    print(f"Loading Real-World Double Pendulum Data from {csv_path}...")
    X_train, Y_train = load_real_double_pendulum_data(csv_path)
    
    dataset = TensorDataset(X_train, Y_train)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DoublePendulumHNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    print("Start Training HNN on Real Physics Data...")
    for epoch in range(500):
        total_loss = 0
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            
            optimizer.zero_grad()
            pred = model(bx)
            # L2 Loss: 拟合相空间导数
            loss = ((pred - by)**2).mean()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss {total_loss/len(loader):.6f}")
            
    print("Training Complete. Generating Chaos Prediction Plot...")
    
    # 4. 预测混沌轨迹
    # 从数据的最后一个点开始往后预测
    t_span = 200
    x0 = X_train[0].unsqueeze(0).to(device) # 从起点开始重构
    trajectory = [x0.cpu().detach().numpy()]
    
    curr_x = x0
    dt = 0.01
    for _ in range(t_span):
        dx = model(curr_x)
        curr_x = curr_x + dx * dt # Euler integration
        trajectory.append(curr_x.cpu().detach().numpy())
        
    traj = np.concatenate(trajectory, axis=0)
    real_traj = X_train[:t_span+1].cpu().numpy()
    
    plt.figure(figsize=(10, 5))
    plt.plot(real_traj[:, 0], real_traj[:, 1], 'k-', label='Real Experiment (Ground Truth)', alpha=0.6)
    plt.plot(traj[:, 0], traj[:, 1], 'r--', label='HNN Prediction (Learned Physics)')
    plt.title("Double Pendulum: Real vs HNN (Configuration Space q1-q2)")
    plt.xlabel("Theta 1")
    plt.ylabel("Theta 2")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('real_double_pendulum_result.png')
    print("Saved to real_double_pendulum_result.png")

if __name__ == "__main__":
    train_real_world_hnn()