# 论文大纲：神经规范场与递归热力学网络
# Paper Outline: Neural Gauge Fields & Recursive Thermodynamic Networks

**目标期刊**：NeurIPS 2025 / Nature Machine Intelligence
**核心贡献**：提出一套统一的几何与热力学理论，解决深度学习中的几何泛化与异构架构融合难题。

---

## 1. Introduction (引言)
*   **The Geometric Crisis (几何危机)**：
    *   现代深度学习（CNN/Transformer）缺乏内建的几何不变性。
    *   展示 **Figure 1**（2D t-SNE 流形错位 + 3D W型准确率崩塌），直观展示现有模型在旋转下的脆弱性。
*   **The Architectural Divide (架构割裂)**：
    *   Mamba (SSM) 与 Transformer (Attention) 被视为两种对立的范式，难以高效融合（Jamba 的静态拼接问题）。
*   **Our Solution (本文方案)**：
    *   引入物理学中的 **规范场论 (Gauge Field Theory)**。
    *   **Micro**: 提出 **Neural Gauge Fields (NGF)**，通过协变导数实现几何泛化。
    *   **Macro**: 提出 **Thermodynamic Gated Networks (TGN)**，通过规范场连接器实现 Mamba-Llama 的流形对齐。

## 2. Theoretical Framework (理论框架)
*   **2.1 Geometric Foundations (几何基础)**：
    *   定义特征空间为纤维丛 (Fiber Bundle)。
    *   推导协变导数 $D_\mu = \partial_\mu + i A_\mu$ 与平行移动。
    *   **关键洞见**：不同位置/架构的特征向量不可直接相加，必须先经过联络 (Connection) 的校正。
*   **2.2 Thermodynamic Dynamics (热力学动力学)**：
    *   将架构融合建模为自由能最小化过程 $F = U - TS$。
    *   **Mamba** = 高熵流形 (Sphere)；**Transformer** = 低能流形 (Cone)。
    *   推导 **Entropy-Driven Gating**：仅在局部熵增（不确定性大）时开启连接器。

## 3. Method: Neural Gauge Fields (NGF)
*   **3.1 Gauge-Covariant Layers**：
    *   设计支持群变换（如 $SO(2), SO(3)$）的卷积与线性层。
    *   引入 **Low-Rank Gauge Connector**，解决大模型显存爆炸问题 ($O(d^2) \to O(rd)$)。
*   **3.2 Manifold Alignment Mechanism**：
    *   定义流形对齐损失函数：$\mathcal{L}_{align} = \| \text{Cov}(h_A) - U^T \text{Cov}(h_B) U \|_F^2$。
    *   阐述如何通过相位同步（Phase Stiffness）自发涌现出门控行为。

## 4. Experiment 1: Geometric Generalization (几何泛化实验)
*   **4.1 2D Rotated MNIST**：
    *   **设置**：Train on 0°, Test on 0-360°。
    *   **结果**：NGF 保持 >85% 准确率，Baseline 跌至 10%。
    *   **可视化**：展示 t-SNE 图，证明 NGF 实现了完美的流形对齐。
*   **4.2 3D Geometric Shapes (ModelNet-Toy)**：
    *   **设置**：Train on Upright, Test on SO(3) Rotation。
    *   **结果**：Baseline 呈现“W型崩塌”（对45°旋转失效），NGF 保持 100% 稳健。
    *   **结论**：证明 NGF 能修复离散采样带来的对称性破缺。

## 5. Experiment 2: Heterogeneous Architecture Fusion (异构融合实验)
*   *(待完成)* **Mamba-Llama Alignment**：
    *   展示跨架构的 CKA 热力图。
    *   证明 NGF 连接器能消除 "Layer 7 Hub" 带来的流形摩擦。
*   *(待完成)* **Long-Context Robustness**：
    *   在 GSM8K / Long-Context 任务上，证明 TGN 比 Jamba 更抗幻觉。

## 6. Discussion & Conclusion (讨论与结论)
*   **Unified Perspective**：Mamba 和 Transformer 是同一物理场在不同规范下的特例（微分 vs 积分）。
*   **Future Work**：向 AI for Science (分子/流体) 和 AGI (自适应生长) 的扩展。

---

## 附录 (Appendices)
*   **A. 数学推导**：纤维丛与协变导数的详细证明。
*   **B. 显存分析**：全秩 vs 低秩连接器的复杂度对比。
*   **C. 实验细节**：超参数设置与数据集生成代码。
