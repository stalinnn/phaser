import torch
import numpy as np

class HessianTracker:
    def __init__(self, model, max_iters=5, tol=1e-3, device="cuda"):
        """
        基于 Power Iteration 的轻量级 Hessian 最大特征值估算器。
        专为 TELM (Training Energy Landscape Monitor) 探针 A 设计。
        """
        self.model = model
        self.max_iters = max_iters
        self.tol = tol
        self.device = device
        
        # 提取需要追踪的参数 (只取 requires_grad=True 的)
        self.params = [p for p in self.model.parameters() if p.requires_grad]
        
        # 预分配随机向量 v，并进行归一化
        self.v = [torch.randn_like(p).to(self.device) for p in self.params]
        self._normalize(self.v)

    def _normalize(self, v_list):
        norm = torch.sqrt(sum(torch.sum(v ** 2) for v in v_list))
        if norm > 0:
            for v in v_list:
                v.div_(norm)
        return norm

    def compute_max_eigenvalue(self, loss):
        """
        使用 HVP (Hessian-Vector Product) 和幂迭代法估算最大特征值。
        注意：这必须在 loss.backward() 之前调用。
        """
        # 计算一次梯度，必须 create_graph=True 才能算二阶导
        # allow_unused=True 是因为当模型稀疏时，某些门控可能完全关闭分支，导致部分参数未被使用
        grads = torch.autograd.grad(loss, self.params, create_graph=True, retain_graph=True, allow_unused=True)
        grads = tuple(g if g is not None else torch.zeros_like(p) for g, p in zip(grads, self.params))
        
        lambda_max = 0.0
        
        # 幂迭代
        for i in range(self.max_iters):
            # 将梯度与当前的随机向量 v 做点积
            grad_v_dot = sum(torch.sum(g * v_i) for g, v_i in zip(grads, self.v))
            
            # 再对参数求导，得到 Hessian 与 v 的乘积 (HVP)
            # 为了保证外部 loss.backward() 可以正常执行，我们这里必须全程 retain_graph=True
            hvp = torch.autograd.grad(grad_v_dot, self.params, retain_graph=True, allow_unused=True)
            hvp = tuple(h if h is not None else torch.zeros_like(p) for h, p in zip(hvp, self.params))
            
            # 计算 Rayleigh 商: v^T * H * v
            # 因为 v 已经被归一化，分母为 1
            lambda_max_prev = lambda_max
            lambda_max = sum(torch.sum(h_i * v_i) for h_i, v_i in zip(hvp, self.v)).item()
            
            # 更新向量 v 以备下一次迭代
            self.v = [h_i.detach() for h_i in hvp]
            self._normalize(self.v)
            
            if abs(lambda_max - lambda_max_prev) < self.tol:
                break
                
        # 清理由于计算二阶导产生的临时图缓存（不影响一阶 loss 的图）
        # 让 Python 垃圾回收即可
        return lambda_max
