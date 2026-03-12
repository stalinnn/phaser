[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18340659.svg)](https://doi.org/10.5281/zenodo.18340659)

# Thermodynamic Gated Networks: Attention as a Geometric Anti-Dissipative Force
## 论文代码与数据复现指南 (Code & Data Reproduction Guide)

本仓库包含了论文《大规模协同网络中热力学耗散的几何抑制机制》中所有理论仿真、金融实证与 AI 实验的源代码。

### 环境依赖 (Dependencies)
请确保安装以下 Python 库：
```bash
pip install numpy scipy matplotlib seaborn pandas yfinance torch transformers networkx statsmodels tqdm
```

---

### 实验与图表对应表 (Figure Mapping)

#### 第一部分：理论与仿真 (Theory & Simulation)

*   **图 1: 热力学稳定性对比 (Thermodynamic Stability)**
    *   **描述**：对比标量扩散与张量协同在不同噪声温度下的误差收敛。
    *   **脚本**：`code/simulation_distributed_mechanism.py`
    *   **输出**：`figures/first_principles_network_dynamics_fixed.png`

*   **图 2: 有限尺寸标度 (Finite Size Scaling)**
    *   **描述**：验证协同误差随系统规模 $N$ 的幂律增长（复杂性陷阱）及几何协同的压制效果。
    *   **脚本**：`code/simulation_finite_size_scaling.py`
    *   **输出**：`figures/finite_size_scaling_ring.png`

*   **图 3: 自适应相变验证 (Adaptive Phase Transition)**
    *   **描述**：在 GRU+Attention 混合架构中观测到的“认知呼吸”现象。展示了 Attention 如何作为“几何泵”在关键时刻逆转信息耗散。此图也验证了麦克斯韦妖机制。
    *   **脚本**：`code/experiment_real_hybrid.py`
    *   **输出**：`figures/real_hybrid_experiment_polished.png`

*   **图 4: 非单调相变 (Non-monotonic Transition)**
    *   **描述**：揭示 $K=1$ 时的“盲目协同”灾难与 $K \ge 2$ 的几何有序相变。
    *   **脚本**：`code/simulation_criticality_search.py`
    *   **输出**：`figures/criticality_search.png`

*   **图 4: 热力学注意力的涌现 (Emergence of Thermodynamic Attention)**
    *   **描述**：在非线性流形（Kuramoto）上，展示 Attention 机制如何通过门控（Softmax）解决长程退相干问题。
    *   **脚本**：`code/simulation_attention.py`
    *   **输出**：`figures/attention_fix.png`

*   **图 5: 架构创新验证 (TGN Architecture Validation)**
    *   **描述**：工程基准测试。在 SeqLen=200 的极限条件下，对比 LSTM、Transformer 和 TGN。证明 TGN 能以 1% 的稀疏能耗实现 Transformer 级的长程推理。
    *   **脚本**：`code/experiment_architecture_innovation.py`
    *   **输出**：`figures/architecture_validation.png`

#### 第二部分：金融实证 (Financial Evidence)

*   **图 5: 金融市场的几何相变与因果检验 (Geometric Phase Transition)**
    *   **描述**：S&P 500 (2000-2023) 全周期几何熵演化，以及几何熵对 VIX 的格兰杰因果检验 (Lag=14 days)。
    *   **脚本**：`code/financial_causality.py`
    *   **输出**：`figures/empirical_finance_entropy.png` (时间序列), `figures/granger_causality.png` (P值分析)

*   **图 6: 市场的微观拓扑重构 (Microscopic Topological Reconfiguration)**
    *   **描述**：对比 2006 年平稳期（模块化结构）与 2008 年危机期（全连接坍缩）的市场相关性矩阵。
    *   **脚本**：`code/financial_topology_viz.py`
    *   **输出**：`figures/market_topology_collapse.png`

#### 第三部分：AI 作为模式生物 (AI as Model Organism)

*   **图 7: 神经-物理同构 (Neural-Physical Isomorphism)**
    *   **描述**：对比 Transformer 自发学习到的 Attention 权重与理论推导的物理格林函数 $\mathbf{G}^{-1}$。
    *   **脚本**：`code/ai_transformer_verification.py`
    *   **输出**：`figures/ai_transformer_geometry.png`

*   **图 8: 有效秩演化与消融实验 (Rank Renormalization & Ablation)**
    *   **描述**：展示 GPT-2 深层网络的有效秩回升现象，并通过消融实验证明 Attention 是唯一的“流形泵”。
    *   **脚本**：`code/llm_rank_ablation.py`
    *   **输出**：`figures/llm_rank_ablation.png`

*   **图 9: 长程有序性与近临界态 (Long-Range Order)**
    *   **描述**：测量 GPT-2 生成文本的两点关联函数，展示其超长的特征关联长度 ($\xi \approx 600$)。
    *   **脚本**：`code/llm_criticality_test.py`
    *   **输出**：`figures/llm_criticality.png`

---

### 如何运行 (How to Run)
所有脚本均为独立运行，无需额外配置。在项目根目录下执行：

```bash
# 运行所有仿真实验
python code/simulation_distributed_mechanism.py
python code/simulation_attention.py
python code/experiment_architecture_innovation.py # TGN 架构验证

# 运行金融实证 (需网络连接下载数据)
python code/financial_causality.py

# 运行 AI 实验 (需安装 PyTorch/Transformers)
python code/llm_rank_ablation.py
```

### 数据说明 (Data Availability)
*   **金融数据**：通过 `yfinance` API 实时从 Yahoo Finance 获取。
*   **AI 模型**：使用 HuggingFace `gpt2` 预训练权重。
*   **仿真数据**：由脚本中的物理方程实时生成。

