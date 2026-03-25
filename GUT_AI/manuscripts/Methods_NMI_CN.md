# 方法 (Methods)

## 理论推导 (Theoretical Derivations)

### Attention 与热扩散方程的同构性 (Isomorphism between Attention and Heat Diffusion)

为了建立 Attention 机制的物理基础，我们基于 Coifman & Lafon 的扩散映射 (Diffusion Maps) 理论，证明 Softmax Attention 在特定条件下严格等价于黎曼流形上的热扩散算子。

**1. 几何约束与距离映射**
在 Transformer 架构中，LayerNorm 操作将特征向量 $\mathbf{x}$ 约束在 $d$ 维超球面的一个切层附近。对于严格归一化的向量 $\|\mathbf{q}\| = \|\mathbf{k}\| = R$（其中 $R=\sqrt{d}$），欧氏距离与点积存在一一映射：
$$
\|\mathbf{q} - \mathbf{k}\|^2 = \|\mathbf{q}\|^2 + \|\mathbf{k}\|^2 - 2\mathbf{q}^T \mathbf{k} = 2R^2 - 2\mathbf{q}^T \mathbf{k}
$$
这意味着点积（相似度）可以直接转化为距离（差异度）：$\mathbf{q}^T \mathbf{k} = R^2 - \frac{1}{2} \|\mathbf{q} - \mathbf{k}\|^2$。

将其代入标准的 Softmax Attention 公式：
$$
A_{ij} = \frac{\exp(\mathbf{q}_i^T \mathbf{k}_j / \tau)}{\sum_l \exp(\mathbf{q}_i^T \mathbf{k}_l / \tau)} = \frac{\exp(R^2/\tau) \cdot \exp\left(-\frac{\|\mathbf{q}_i - \mathbf{k}_j\|^2}{2\tau}\right)}{\sum_l \exp(R^2/\tau) \cdot \exp\left(-\frac{\|\mathbf{q}_i - \mathbf{k}_l\|^2}{2\tau}\right)}
$$
这表明 $A_{ij}$ 在数学上严格等价于高斯热核 (Gaussian Heat Kernel) $K_\tau(x, y) = \exp(-\|x-y\|^2/2\tau)$ 的归一化形式。

**2. 动力学演化方程**
考虑残差连接下的单层 Attention 更新 $\mathbf{x}_i^{t+1} = \mathbf{x}_i^t + \sum_j A_{ij} (\mathbf{x}_j^t - \mathbf{x}_i^t)$。这对应于热扩散方程 $\partial_t \mathbf{u} = \Delta \mathbf{u}$ 的显式欧拉离散化：
$$
\frac{\mathbf{x}^{t+1} - \mathbf{x}^t}{\Delta t} \approx \Delta_{\mathcal{M}} \mathbf{x}^t
$$
其中 $\Delta_{\mathcal{M}}$ 为流形上的 Laplace-Beltrami 算子。因此，Transformer 的一层更新物理上对应于流形上的一步热流演化，其作用是最小化系统的狄利克雷能量 (Dirichlet Energy) $E(\mathbf{X}) = \frac{1}{2} \sum_{i,j} A_{ij} \|\mathbf{x}_i - \mathbf{x}_j\|^2$。

### 缩放因子与随机矩阵理论 (Scaling Factor and Random Matrix Theory)

我们通过随机矩阵理论 (RMT) 推导缩放因子 $\lambda = 1/\sqrt{d}$ 的必然性。假设 $\mathbf{Q}, \mathbf{K} \in \mathbb{R}^{N \times d}$ 的元素独立同分布于 $\mathcal{N}(0, 1)$。根据中心极限定理，点积 $z_{ij} = \mathbf{q}_i^T \mathbf{k}_j$ 的方差为 $d$。

若不进行缩放，极值理论表明 $\max_j(z_{ij})$ 将随 $d$ 线性增长，导致 Softmax 进入饱和区（梯度消失）或退化为 One-hot（谱秩坍缩）。引入缩放因子 $\lambda$ 后，输入分布变为 $\mathcal{N}(0, \lambda^2 d)$。为了保持系统处于“临界态”（既不饱和也不退化为均匀分布），需控制输入的方差为常数：
$$
\text{Var}(\lambda z_{ij}) = \lambda^2 d = 1 \implies \lambda = \frac{1}{\sqrt{d}}
$$
我们的 RMT 数值模拟证实，在此临界点，Attention 矩阵的奇异值分布呈现幂律长尾特征，且有效秩达到最大，对应于系统处于“混沌边缘” (Edge of Chaos)。

---

## 物理仿真 (Physical Simulations)

### 自旋玻璃模型 (Spin Glass Model)
我们采用 3D Edwards-Anderson (EA) 自旋玻璃模型作为非凸优化基准。其哈密顿量定义为：
$$
H(\mathbf{s}) = -\sum_{\langle i,j \rangle} J_{ij} s_i s_j + \sum_i (s_i^2 - 1)^2
$$
其中 $s_i$ 为连续自旋变量，相互作用强度 $J_{ij}$ 服从高斯分布 $\mathcal{N}(0, 1)$。

**对比算法设置：**
1.  **Langevin Dynamics**: 遵循过阻尼朗之万方程 $d\mathbf{s} = -\nabla H(\mathbf{s})dt + \sqrt{2T}dW$。
2.  **Parallel Tempering**: 设置 10 个温度副本 $T \in [0.1, 5.0]$，每 $100$ 步尝试交换相邻副本，交换概率遵循 Metropolis 准则。
3.  **Attention Dynamics**: 在 Langevin 梯度更新的基础上，引入基于 Transformer 的非局部平滑项。Transformer 模型仅在简单的几何去噪任务上进行预训练（零样本迁移）。

实验在 $L \in \{4, 6, 8, 10, 12\}$ 的立方晶格上进行，总自旋数 $N=L^3$。对于每个尺寸，我们运行 50 个独立的无序样本，并记录能量密度的收敛轨迹。

---

## 神经影像分析 (Neuroimaging Analysis)

### fMRI 数据预处理
我们使用 OpenNeuro 数据集 (ds003171)，包含 17 名健康受试者在清醒、轻度镇静（右美托咪定/异丙酚）和深度镇静状态下的 fMRI 扫描。
数据经过标准 fMRIPrep 流程处理，包括运动校正、切片时间校正和 MNI 空间标准化。利用 Schaefer 2018 图谱将大脑划分为 100 个功能感兴趣区 (ROI)。

### 几何秩计算 (Geometric Rank Calculation)
对于每个时间窗（窗口大小 $W=100$ TR），我们计算区域间的 Pearson 相关矩阵 $\mathbf{C} \in \mathbb{R}^{100 \times 100}$。将负相关值置零（保留兴奋性连接）并归一化得到概率转移矩阵 $\mathbf{P}$。
有效几何秩 $R_{eff}$ 定义为矩阵奇异值分布 $\sigma_i$ 的香农熵的指数形式：
$$
R_{eff}(\mathbf{P}) = \exp\left( -\sum_{i=1}^d \tilde{\sigma}_i \log \tilde{\sigma}_i \right), \quad \text{其中 } \tilde{\sigma}_i = \frac{\sigma_i}{\sum \sigma_j}
$$
该指标衡量了功能连接流形的有效维度。$R_{eff} \to 1$ 表示系统高度同步（低维），$R_{eff} \to N$ 表示系统完全去相关（高维混沌）。

---

## 神经网络实验 (Neural Network Experiments)

### 热力学门控网络 (Thermodynamic Gated Networks, TGN)

**架构设计：**
TGN 包含一个惯性通道（GRU/SSM）和一个几何通道（Causal Attention）。门控机制 $g_t$ 基于惯性通道的隐状态 $h_t^{rnn}$ 生成：
$$
g_t = \sigma(W_g h_t^{rnn} + b_g)
$$
最终状态更新为 $\mathbf{h}_t = (1-g_t) \odot \mathbf{h}_t^{rnn} + g_t \odot \mathbf{h}_t^{attn}$。

**损失函数：**
为了诱导稀疏性，我们在交叉熵损失之外引入 $L_1$ 正则化项：
$$
\mathcal{L} = \mathcal{L}_{CE} + \lambda \cdot \frac{1}{T} \sum_{t=1}^T |g_t|
$$
其中 $\lambda$ 为热力学惩罚系数。在自适应实验中，我们采用动态惩罚策略 $\lambda(\alpha) = \lambda_{max} \cdot \alpha^\gamma$，其中 $\alpha$ 为当前任务准确率。

### 训练细节 (Training Details)
**WikiText-103 语言建模：**
*   **模型规模**：125M 参数（对标 GPT-Neo Small）。
*   **层数**：12层，隐层维度 768。
*   **优化器**：AdamW ($\beta_1=0.9, \beta_2=0.95$)。
*   **学习率**：峰值 $6 \times 10^{-4}$，采用 Cosine Decay 调度。
*   **Batch Size**：全局 Batch Size 512。
*   **硬件**：6x NVIDIA A100 (80GB)。
*   **稀疏性目标**：TGN 模型在无显式稀疏率约束（仅通过 $\lambda$ 惩罚）下训练，最终自发收敛至 ~9% 的门控率。

**基线模型 (Baselines)：**
*   **Transformer**: 标准 GPT-2 架构，使用 RoPE 位置编码。
*   **Mamba (SSM)**: Mamba-1 架构，使用官方 CUDA 实现。

### 扩展数据图表说明 (Extended Data)
所有补充图表（如零样本拓扑推断、MQAR 容量测试细节、不同架构的秩演化对比）均在附录文件 Extended Data 中提供。
