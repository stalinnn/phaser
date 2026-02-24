import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import scipy.sparse as sp
import time
import os
import pickle

class EconomySimulationBase:
    def __init__(self, n_agents=500, seed=42, graph_type='geometric'):
        np.random.seed(seed)
        self.N = n_agents
        self.time_steps = []
        self.is_real_data = False
        
        # --- 拓扑初始化 (Topology) ---
        if graph_type == 'geometric':
            self.G_graph = nx.random_geometric_graph(n_agents, radius=np.sqrt(6/n_agents))
            self.adj_matrix = nx.to_numpy_array(self.G_graph)
        elif graph_type == 'scale_free':
            self.G_graph = nx.barabasi_albert_graph(n_agents, m=2)
            self.adj_matrix = nx.to_numpy_array(self.G_graph)
        elif graph_type == 'watts_strogatz':
            self.G_graph = nx.watts_strogatz_graph(n_agents, k=6, p=0.1)
            self.adj_matrix = nx.to_numpy_array(self.G_graph)
        elif graph_type.endswith('.gpickle'):
            print(f"Loading real-world topology from {graph_type}...")
            with open(graph_type, 'rb') as f:
                self.G_graph = pickle.load(f)
            self.N = self.G_graph.number_of_nodes()
            self.adj_matrix = nx.to_numpy_array(self.G_graph, weight='weight')
            self.is_real_data = True
        
        # 缓存边数，用于计算带宽成本
        self.n_edges = self.G_graph.number_of_edges()
        self.total_information_flow = 0.0 # 单位：Float-Hops (传输一个浮点数经过一条边算1)

        # --- 状态空间 (Manifold State) ---
        self.x = np.random.uniform(0.0, 2.0, self.N)       
        self.x_target = self.x + np.random.normal(0, 0.5, self.N)
        
        # --- 黎曼度量张量场 (Riemannian Metric Tensor Field) ---
        # g_ij 代表“交换阻力”或“局部耦合强度”
        # 理论对应：g_{ij} = - \partial^2 \Phi / \partial x^i \partial x^j
        
        # 1. 对角线 g_ii (Self-Metric): 拥有资源的边际效用衰减率
        diag_values = np.random.uniform(1.0, 5.0, self.N)
        
        # 2. 非对角线 g_ij (Interaction): 
        # 在网络拓扑中，耦合权重越高(Weight大)，信息传递阻力越小(g_ij越低)
        # 我们构建 Laplacian-like 结构来保证正定性
        if self.is_real_data:
            # 真实信任网络：W_ij 是信任度
            W = self.adj_matrix
            # 我们定义 g_ij = -0.5 * W_ij (信任降低“势垒”)
            off_diag = -0.5 * W 
        else:
            # 随机网络：随机耦合
            off_diag = self.adj_matrix * np.random.uniform(-0.5, 0.5, (self.N, self.N))
            off_diag = (off_diag + off_diag.T) / 2
            
        self.G_dense = np.diag(diag_values) + off_diag
        
        # 3. 强制正定性 (Ensuring Convexity of Potential)
        # 物理含义：确保系统处于稳态势阱中，而非鞍点
        min_eig = np.min(np.linalg.eigvals(self.G_dense))
        if min_eig < 0.1:
            self.G_dense += (0.1 - min_eig) * np.eye(self.N)
            
        self.G_sparse = sp.csr_matrix(self.G_dense)
        
        # 预计算 Jacobi 迭代所需的矩阵分量 (For Belief Propagation)
        self.D_inv = 1.0 / np.diag(self.G_dense) # 对角矩阵逆
        self.R_matrix = self.G_dense - np.diag(np.diag(self.G_dense)) # 残差矩阵 (off-diagonal)
        self.R_sparse = sp.csr_matrix(self.R_matrix)
        
        self.history = {'error': [], 'entropy_production': [], 'volume_form': [], 'spectral_gap': []}
        
    def calculate_geometric_observables(self):
        """
        计算黎曼几何观测量 (Riemannian Geometric Observables)
        """
        try:
            sign, logdet = np.linalg.slogdet(self.G_dense)
            volume = 0.5 * logdet
        except:
            volume = 0.0
            
        try:
            # Laplacian L = D - W. 这里 G 类似于 Laplacian.
            # 找第二小的特征值 (Fiedler value)
            vals = sp.linalg.eigsh(self.G_sparse, k=2, which='SM', return_eigenvectors=False)
            gap = vals[1] if len(vals) > 1 else vals[0]
        except:
            gap = 0.0
            
        return volume, gap

    def potential_energy(self, x_curr):
        # Phi(x) = 0.5 * (x-x*)^T G (x-x*)
        error = x_curr - self.x_target
        return 0.5 * error @ (self.G_dense @ error)

class TensorSynergySimulation(EconomySimulationBase):
    """
    Model B: 二阶张量耦合网络 (Second-Order Tensor Coupling Network)
    理论依据：利用高斯信念传播 (Gaussian Belief Propagation, GaBP) 求解牛顿方向
    近似分布式牛顿法 (Distributed Newton Method)
    """
    def run(self, steps=200, dt=0.05, shock_at=None, shock_mag=0.0, bp_steps=5, volatility=0.0):
        print(f"Running Second-Order Tensor Coupling (BP Steps={bp_steps})...")
        x_curr = self.x.copy()
        velocity = np.zeros(self.N) # 初始速度场 (梯度流)
        
        for t in range(steps):
            # 1. 外部冲击 (External Shock) & 持续波动 (Volatility)
            if t == shock_at:
                shock = np.random.normal(0, shock_mag, self.N)
                x_curr += shock
            
            if volatility > 0:
                drift = np.random.normal(0, volatility, self.N)
                self.x_target += drift
                
            # 2. 计算一阶梯度 (First-Order Gradient / Force)
            error = x_curr - self.x_target
            gradient_force = -error 
            
            # 3. 求解二阶修正方向 (Second-Order Correction)
            # 理论修正：使用 Belief Propagation (Jacobi Iteration) 模拟 Hessian 逆的作用
            # 通信成本核算：每次内部迭代，所有边都交换一次信息
            
            v_bp = velocity.copy() # Warm Start
            
            # === 带宽成本计算 ===
            # 二阶网络为了获得高精度的牛顿方向，需要在每一步物理时间内
            # 进行 bp_steps 次的信息交换。
            # Cost = 边数 * 2 (双向) * 迭代次数
            self.total_information_flow += self.n_edges * 2 * bp_steps
            
            for _ in range(bp_steps):
                interaction_msg = self.R_sparse.dot(v_bp) 
                v_bp = (gradient_force - interaction_msg) * self.D_inv
            
            velocity = v_bp
            
            # 4. 状态更新 (Natural Gradient Step)
            x_curr += velocity * dt
            
            # 5. 记录指标
            current_error_norm = np.linalg.norm(error)
            
            # --- 动态曲率更新 (Adaptive Curvature/Hessian) ---
            # Levenberg-Marquardt style damping
            congestion = 0.1 * (velocity ** 2)
            self.G_dense += np.diag(congestion * dt) 
            self.D_inv = 1.0 / np.diag(self.G_dense) 
            
            # 计算瞬时收敛成本 (Convergence Cost / Loss Drop)
            # 对应于二次型损失函数的下降量
            dissipation = velocity @ (self.G_dense @ velocity)
            
            self.history['error'].append(current_error_norm)
            self.history['entropy_production'].append(dissipation)

class MarketScalarSimulation(EconomySimulationBase):
    """
    Model A: 一阶标量耦合模型 (First-Order Scalar Coupling Model)
    理论依据：分布式梯度下降 (Distributed Gradient Descent) + 扩散 (Diffusion)
    """
    def run(self, steps=200, dt=0.02, shock_at=None, shock_mag=0.0, volatility=0.0, diffusion_coeff=0.5):
        print(f"Running Discrete First-Order Model (Diffusion Coeff={diffusion_coeff})...")
        x_curr = self.x.copy()
        
        G_diag = np.diag(self.G_dense) 
        
        for t in range(steps):
            if t == shock_at:
                shock = np.random.normal(0, shock_mag, self.N)
                x_curr += shock

            if volatility > 0:
                drift = np.random.normal(0, volatility, self.N)
                self.x_target += drift
                
            error = x_curr - self.x_target
            
            # --- 解决 "只看对角线" 问题 ---
            if diffusion_coeff > 0:
                # 邻居的影响。
                # === 带宽成本计算 ===
                # 标量市场开启扩散模式，每一步仅交换一次标量信息
                # Cost = 边数 * 2 (双向) * 1 (一次扩散)
                self.total_information_flow += self.n_edges * 2
                
                neighbor_influence = self.R_sparse.dot(error)
                effective_signal = error + diffusion_coeff * neighbor_influence
            else:
                # 零通信成本模式 (Autarky)
                effective_signal = error

            # 梯度下降动力学 (Gradient Descent Dynamics)
            velocity = - (1.0 / G_diag) * effective_signal 
            
            x_curr += velocity * dt
            
            current_error_norm = np.linalg.norm(error)
            
            # 标量系统的收敛成本
            # 1. 误差收敛慢 (积分大)
            # 2. 轨迹震荡 (路径长)
            dissipation = velocity @ (np.diag(G_diag) @ velocity)
            
            self.history['error'].append(current_error_norm)
            self.history['entropy_production'].append(dissipation)

class SmartMarketSimulation(EconomySimulationBase):
    """
    Model A+ (Augmented): 增强型标量模型 (Momentum-based Scalar Model)
    
    Upgrade: 引入 Adam 优化器逻辑模拟 "自适应动量" (Adaptive Momentum)
    这代表了标量系统在时间维度上挖掘信息的极限：
    不仅利用了一阶趋势(First Moment)，还利用了二阶波动(Second Moment)来调整步长。
    """
    def run(self, steps=200, dt=0.02, shock_at=None, shock_mag=0.0, volatility=0.0, 
            beta1=0.9, beta2=0.999, epsilon=1e-8, diffusion_coeff=0.5):
        
        print(f"Running Adaptive Momentum Model (Adam-style: Trend + Volatility)...")
        x_curr = self.x.copy()
        
        # Adam Internal States
        # m: 市场对价格趋势的共识 (First Moment)
        # v: 市场对价格波动的风险评估 (Second Moment)
        m = np.zeros(self.N) 
        v = np.zeros(self.N)
        
        G_diag = np.diag(self.G_dense)
        
        for t in range(1, steps + 1):
            if t == shock_at:
                shock = np.random.normal(0, shock_mag, self.N)
                x_curr += shock
                
            if volatility > 0:
                self.x_target += np.random.normal(0, volatility, self.N)
            
            # 1. 原始梯度信号 (供需缺口)
            grad = x_curr - self.x_target
            
            # 2. 信号扩散 (Signal Diffusion)
            # 即使是标量市场，也可以通过观察邻居的价格压力来修正自己的判断
            # Cost = Edges * 2
            self.total_information_flow += self.n_edges * 2 
            neighbor_influence = self.R_sparse.dot(grad)
            
            # 有效梯度：自身需求 + 邻居压力
            effective_grad = grad + diffusion_coeff * neighbor_influence
            
            # 3. Adam Update Logic (自适应市场核心)
            # 更新趋势认知 (Momentum)
            m = beta1 * m + (1 - beta1) * effective_grad
            
            # 更新风险认知 (Volatility)
            v = beta2 * v + (1 - beta2) * (effective_grad ** 2)
            
            # 偏差修正 (Bias Correction) - 模拟市场启动初期的学习过程
            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)
            
            # 4. 价格调整速度
            # 核心机制：当波动率 v_hat 很大时，分母变大，调整速度变慢 (风险厌恶)
            # 1/G_diag 仍然是个体对价格的基础敏感度
            
            # Adam 的标准更新公式: step = - alpha * m / (sqrt(v) + eps)
            velocity = - (1.0 / G_diag) * m_hat / (np.sqrt(v_hat) + epsilon)
            
            x_curr += velocity * dt
            
            # 记录
            current_error_norm = np.linalg.norm(grad)
            dissipation = velocity @ (np.diag(G_diag) @ velocity)
            
            self.history['error'].append(current_error_norm)
            self.history['entropy_production'].append(dissipation)

# --- 保持向下兼容的辅助函数 ---
def run_thermodynamic_comparison():
    pass 

if __name__ == "__main__":
    run_thermodynamic_comparison()
