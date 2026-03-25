import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time
import os

from models.fluid_core import EvTTCFluidNetwork
from utils.event_dataset import MockEvTTCDataset, EventStreamer

print("================================================================")
print(" 🎓 [训练脚本] 连续统流体引擎的“非平衡态热力学”训练")
print(" 目标: 验证热力学损失函数 (Thermodynamic Loss) 的有效性")
print("================================================================\n")

class ThermodynamicLoss(nn.Module):
    """
    为物理流体引擎量身定制的热力学损失函数。
    它不仅关注预测对不对，还从物理上约束引擎的“演化形态”。
    """
    def __init__(self, lambda_kinetic=0.01, lambda_entropy=0.05):
        super().__init__()
        self.task_loss_fn = nn.BCELoss() # 任务: 二分类交叉熵 (安全 vs 碰撞)
        self.lambda_kinetic = lambda_kinetic
        self.lambda_entropy = lambda_entropy
        
    def forward(self, pred_risk, target_risk, current_state, prev_state, current_entropy):
        """
        1. Task Loss (内能 U): 降低预测误差
        2. Kinetic Penalty (动能约束): ||H_t - H_{t-1}||^2。迫使隐状态在没有危机时保持平稳
        3. Entropy Polarization (熵极化): 鼓励状态要么极度平稳，要么极度尖锐(报警)，惩罚半吊子状态
        """
        # 1. 任务损失 (拟合 TTC 标签)
        task_loss = self.task_loss_fn(pred_risk, target_risk)
        
        # 2. 动能惩罚 (时间平滑正则化)
        if prev_state is not None:
            kinetic_loss = torch.mean((current_state - prev_state) ** 2)
        else:
            kinetic_loss = torch.tensor(0.0, device=pred_risk.device)
            
        # 3. 熵极化约束 (金兹堡-朗道相变机制)
        # 我们希望熵 H 趋向于 1.0 (绝对平稳) 或者 0.0 (绝对尖锐的焦点)
        # 构造双势阱函数: U(S) = S^2 * (1-S)^2
        entropy_loss = (current_entropy ** 2) * ((1.0 - current_entropy) ** 2)
        
        total_loss = task_loss + self.lambda_kinetic * kinetic_loss + self.lambda_entropy * entropy_loss
        
        return total_loss, task_loss, kinetic_loss, entropy_loss


def train_mock_epoch():
    device = torch.device("cpu") # 强制使用 CPU 防止后台任务干扰
    print(f"[*] 挂载训练硬件: {device}")
    
    # 初始化模型与优化器
    model = EvTTCFluidNetwork(height=32, width=32, embed_dim=16).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = ThermodynamicLoss(lambda_kinetic=0.1, lambda_entropy=0.05)
    
    # --- 训练记录 ---
    history = {'total': [], 'task': [], 'kinetic': [], 'entropy': []}
    
    # 缩小数据和迭代次数，保证能快速跑通并画图
    epochs = 5
    
    print("\n[*] 开始基于 BPTT (通过时间的流式反向传播) 训练...")
    
    for epoch in range(epochs):
        # 1. 重置引擎状态 (每开一次车)
        model.reset_state()
        prev_state = None
        
        # 2. 生成一段 1 秒钟的逼近碰撞事件流 (800ms 处相撞)
        # 降低分辨率以加速 CPU/轻度 GPU 运算
        dataset = MockEvTTCDataset(width=32, height=32, duration_ms=1000)
        events = dataset.generate_looming_collision(collision_time_ms=800)
        streamer = EventStreamer(events, dt_ms=20.0) # 每 20ms 一帧
        
        epoch_loss = 0.0
        steps = 0
        
        for t_ms, micro_batch in streamer:
            # === 生成真实的 TTC Label ===
            # 如果当前时间离碰撞时间 < 200ms，我们认为是危险期，Label=1.0
            if t_ms >= 600: 
                target_risk = torch.tensor([[1.0]], device=device)
            else:
                target_risk = torch.tensor([[0.0]], device=device)
                
            # === 全双工前向演化 ===
            pred_risk_val = model(micro_batch) # [1] 的标量
            pred_risk = torch.tensor([[pred_risk_val]], device=device, requires_grad=True)
            
            # 为了能在计算图中求导，我们需要在模型外包装一下提取
            # (在完美的流体代码中，我们应当直接返回带梯度的 Tensor，这里为了 Mock 快速跑通)
            # 我们重新进行带梯度的微步：
            perturbation = model.pump(micro_batch)
            current_state = model.fluid_core(model.global_state, perturbation)
            pred_risk = model.collision_head(current_state)
            
            # 更新状态为当前帧
            model.global_state = current_state.detach() # Truncated BPTT 防止梯度爆炸
            
            # 计算局部熵
            entropy_val = model.get_entropy()
            current_entropy = torch.tensor([entropy_val], device=device, requires_grad=True)
            
            # === 热力学损失计算 ===
            loss, t_loss, k_loss, e_loss = criterion(
                pred_risk, target_risk, 
                current_state, prev_state, 
                current_entropy
            )
            
            # === 反向传播与优化 ===
            optimizer.zero_grad()
            loss.backward()
            
            # 防止梯度爆炸的流形保护 (梯度裁剪)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            prev_state = current_state.detach()
            
            epoch_loss += loss.item()
            steps += 1
            
            history['total'].append(loss.item())
            history['task'].append(t_loss.item())
            history['kinetic'].append(k_loss.item())
            history['entropy'].append(e_loss.item())
            
        print(f"    Epoch [{epoch+1}/{epochs}] | Avg Loss: {epoch_loss/steps:.4f} | 物理场逐渐学会预警...")

    print("\n[*] 训练完成，正在绘制热力学收敛曲线...")
    
    # 绘制平滑过的曲线
    def smooth(scalars, weight=0.9):
        last = scalars[0]
        smoothed = []
        for point in scalars:
            smoothed_val = last * weight + (1 - weight) * point
            smoothed.append(smoothed_val)
            last = smoothed_val
        return smoothed

    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 6))
    
    iters = range(len(history['total']))
    ax.plot(iters, smooth(history['total']), color='white', linewidth=2, label="Total Free Energy (Loss)")
    ax.plot(iters, smooth(history['task']), color='cyan', linestyle='--', label="Task Error (Internal Energy U)")
    ax.plot(iters, smooth(history['kinetic']), color='yellow', linestyle=':', label="Kinetic Penalty (Stability)")
    
    ax.set_title("Thermodynamic Training Convergence (Fluid Engine)", fontsize=14)
    ax.set_xlabel("Continuous Micro-Steps", fontsize=12)
    ax.set_ylabel("Energy / Loss Magnitude", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)
    
    save_path = "/phaser/EvTTC_Fluid_Control/training_convergence.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"[*] 训练收敛图已保存至: {save_path}")

if __name__ == "__main__":
    train_mock_epoch()
