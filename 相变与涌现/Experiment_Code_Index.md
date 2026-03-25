# 物理人工智能大统一理论 - 实验与代码索引

本文档汇总了 `Phase_Transition_Emergence` 目录下的核心理论验证代码、生成的相变实验图表以及关键的日志文件说明。所有临时调试和废弃的脚本均已清理，保留的是最终可复现顶会级理论的完整代码栈。

---

## 1. 核心理论文档

*   **[`Theory_and_Experiment_Report.md`](./Theory_and_Experiment_Report.md)**: 最新的大统一理论实验报告，包含“引入内生温度代理与残差防坍塌机制”以及“跨架构物理相图”的深度分析。
*   **[`Manuscript_Math_Derivations.md`](./Manuscript_Math_Derivations.md)**: 严格的数学推导手稿，从 Langevin 动力学、Fokker-Planck 方程一直推导到完整的无量纲参数 $\Lambda(t)$。

---

## 2. 核心实验脚本

这些脚本直接驱动了理论的验证与相变坍缩图的生成。

### 基础验证：直接双曲坐标优化
*   **[`emerge_holographic_bulk.py`](./emerge_holographic_bulk.py)**: 基础物理演化脚本，通过 `RiemannianAdam` 直接在庞加莱球中驱动节点移动（无神经网络），验证纯粹的语义纠缠向逻辑结构的重整化流。
*   **[`run_universal_collapse.py`](./run_universal_collapse.py)**: 运行不同 Batch Size 和 Learning Rate 的组合，证明大批量等效于低温，并生成 **`universality_collapse.png`**。
*   **[`run_model_size_collapse.py`](./run_model_size_collapse.py)**: 运行不同的双曲流形维度 ($d_{hyp}$)，证明几何自由度作为相变分母项的正确性，并生成 **`model_size_collapse.png`**。

### 进阶突破：跨网络架构的真实动力学
*   **[`emerge_arch_holographic.py`](./emerge_arch_holographic.py)**: 核心突破脚本。测试不同的神经网络拓扑（MLP, Transformer(带残差), Mamba-Proxy）在映射欧氏语义到双曲空间时的相变动力学。提取真实的梯度经验方差作为内部热力学温度代理。
*   **[`run_arch_collapse.py`](./run_arch_collapse.py)**: 统筹运行不同的架构，利用 $D_{param}$ 作为宏观容量因子，证明同类架构内的完美坍缩，并揭示不同拓扑产生的“物理相图分离”（归纳偏置），生成 **`architecture_collapse_v4.png`**。

### 终极大一统：计算图拓扑谱间隙
*   **[`graph_topology_analyzer.py`](./graph_topology_analyzer.py)**: 计算不同神经网络架构的“图拉普拉斯矩阵的代数连通度（Spectral Gap）”。用于量化 Transformer（全连通，Gap=1.0）与 MLP/Mamba（断开/带状，Gap$\approx$0.0）在相变中的拓扑阻力。
*   **[`run_spectral_collapse.py`](./run_spectral_collapse.py)**: 结合谱间隙修改无量纲相变常数 $\Lambda_{topo}(t)$，实现 Transformer 与 MLP 的跨架构终极坍缩，生成 **`ultimate_spectral_collapse.png`**。

### 宏观视野：智能三相图
*   **[`sweep_2d_phase_diagram.py`](./sweep_2d_phase_diagram.py)**: 在二维网格上同时扫描“数据纠缠引力 (Data Covariance)”和“系统热噪声 (Temperature)”，绘制出包含临界边界线的宏观 AI 相图，生成 **`2d_phase_diagram.png`**。

---

## 3. 探针与工具包
*   **[`calculate_lambda_crit.py`](./calculate_lambda_crit.py)**: 读取日志，定位相变突变点，并精确计算出诱发相变的临界常数 $\Lambda_{crit}$。
*   **`core/` 目录**: 存放底层的 `hyperbolic_space.py`（庞加莱球映射）和 `metric_alignment.py`（基于 RT 公式的引力/度规对齐损失）。
*   **`probes/` 目录**: 包含 `telm_monitor.py`，用于在训练过程中实时抓取系统的有效秩（$\Phi$）、参数梯度范数等热力学探针数据。

---

## 4. 关键成果图表索引

*   📊 **[Batch Size 万有坍缩 (`universality_collapse.png`)](./universality_collapse.png)**: 证明 $B \propto \frac{1}{T_{sys}}$。
*   📊 **[流形维度坍缩 (`model_size_collapse.png`)](./model_size_collapse.png)**: 证明 $d_{hyp}$ 是相变的核心自由度限制。
*   📊 **[架构族相图分离 (`architecture_collapse_v4.png`)](./architecture_collapse_v4.png)**: 证明同架构族内由于 $D_{param}$ 缩放导致的完美坍缩，同时直观展示了“归纳偏置”在热力学轨迹上的体现。
*   📊 **[终极拓扑大一统 (`ultimate_spectral_collapse.png`)](./ultimate_spectral_collapse.png)**: 通过引入“谱间隙”阻力，实现了 Transformer、Mamba 与 MLP 的究极大一统。
*   🗺️ **[AI 智能的二维三相图 (`2d_phase_diagram.png`)](./2d_phase_diagram.png)**: 展现了数据知识密度与系统计算噪声对抗下的相变全景图。