import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class AccumulatingGate(nn.Module):
    def __init__(self, threshold=1.0):
        super().__init__()
        self.threshold = threshold

    def forward(self, surprise_scores):
        """
        surprise_scores: [B, L] 局部惊奇度 (0~1之间)
        返回门控激活序列: [B, L]，值为 0 或 1
        """
        B, L = surprise_scores.shape
        gates = torch.zeros_like(surprise_scores)
        accumulated_surprise = torch.zeros(B, device=surprise_scores.device)
        
        for t in range(L):
            # 累加当前的惊奇度
            accumulated_surprise += surprise_scores[:, t]
            
            # 判断是否超过阈值
            trigger = (accumulated_surprise >= self.threshold).float()
            
            # 如果触发了，当前时间步门控为 1
            gates[:, t] = trigger
            
            # 触发后，将累加器清零 (通过减去 trigger * 累加器的值)
            accumulated_surprise = accumulated_surprise * (1 - trigger)
            
        return gates

# 测试一下
torch.manual_seed(42)
B, L = 1, 50
# 模拟一些平缓的惊奇度，偶尔有几个尖峰
surprise = torch.rand(B, L) * 0.2
surprise[0, 10] = 0.8
surprise[0, 25] = 0.9

gate_module = AccumulatingGate(threshold=1.0)
gates = gate_module(surprise)

print("Surprise:\n", surprise[0].numpy())
print("Gates:\n", gates[0].numpy())
