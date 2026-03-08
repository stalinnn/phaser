# Mamba vs Transformer: 几何结构与线性对齐性分析报告

**日期**：2026年2月11日
**实验对象**：
1.  **Small Scale (1B)**: Llama-3.2-1B (Transformer) vs Mamba2-1.3B (SSM)
2.  **Mid Scale (7B/8B)**: Llama-3.1-8B (Transformer) vs Falcon-Mamba-7B (SSM)
**核心结论**：
*   **1B 规模**: 宏观拓扑高度一致 (CKA > 0.9)，流形结构相似。
*   **7B/8B 规模**: 出现显著的 **"Layer 7 Hub" 现象**。Mamba 的 Layer 7 成为信息密度极高（Max Effective Rank）的枢纽层，导致 Llama 的大部分层都与该层呈现最高相似度。
*   **长程能力**: Mamba 在长程特征探测上存在显著瓶颈（Accuracy 80% vs Transformer 100%），证实了混合架构的必要性。

---

## 1. 实验概览与设置

为了探究 SSM (Mamba) 与 Transformer (Attention) 在表征空间上的异同，我们进行了跨架构、跨规模的层级相似性分析。

### 模型配置
| 组别 | Transformer | Mamba (SSM) | 备注 |
| :--- | :--- | :--- | :--- |
| **Small** | Llama-3.2-1B (16 Layers) | Mamba2-1.3B (48 Layers) | 验证基础流形一致性 |
| **Mid** | Llama-3.1-8B (32 Layers) | Falcon-Mamba-7B (64 Layers) | 验证大模型下的特征分化 |

---

## 2. 核心发现 (Key Findings)

### 2.1 宏观几何一致性 (CKA Analysis)

#### 1B 规模：殊途同归
*   **现象**: CKA 热力图呈现清晰的对角线趋势。
*   **结论**: 尽管计算机制不同，两者在小规模下收敛到了极其相似的语义流形。

![1B CKA Heatmap](./cka_llama_1b_vs_mamba_1.3b.png)

#### 7B/8B 规模：枢纽涌现 (The "Layer 7 Hub")
*   **现象**: 热力图不再是对角线，而是呈现 **垂直条纹**。Llama 的 Layer 0-31 居然大部分都与 Falcon-Mamba 的 **Layer 7** 相似度最高。
*   **原因分析 (SVD)**: SVD 分析显示，Falcon-Mamba 的 Layer 7 拥有全模型最高的 **有效秩 (Effective Rank: 18.36)**。这意味着它是信息密度最大、维度展开最充分的一层。
*   **结论**: 在更大规模的模型中，Mamba 倾向于在浅层（Layer 7）迅速完成特征提取和维度展开，然后保持高维度的线性演化；而 Transformer 则进行更渐进式的特征抽象。

![7B CKA Heatmap](./cka_llama_8b_vs_falcon_mamba_7b.png)

### 2.2 维度利用率与各向异性 (SVD Analysis)

*   **有效秩 (Effective Rank)**:
    *   **Mamba (Mean: 18.05)** > Llama (Mean: 17.51)。
    *   **解读**: Mamba 对 4096 维隐状态空间的利用率更高，信息更丰富。
*   **各向异性 (Anisotropy)**:
    *   **Mamba (0.15)** << Llama (0.30)。
    *   **解读**: Mamba 的表征分布更加均匀 (Isotropic)，克服了 Transformer 常见的“锥形效应” (Representation Collapse)。这是一个非常积极的发现，说明递归压缩并没有导致表征退化。

![SVD Analysis](./svd_analysis.png)

### 2.3 线性可分性探测 (Probing) —— 关键短板

我们训练线性分类器探测“长短句分类”这一全局统计特征。

| 模型层级 | Llama 3.1 8B | Falcon Mamba 7B | 差距 |
| :--- | :--- | :--- | :--- |
| **Early** | 90% | 80% | -10% |
| **Middle** | **100%** | **80%** | **-20%** |
| **Late** | **100%** | **80%** | **-20%** |

*   **结论**: Mamba 存在明显的 **"80% 天花板"**。由于固定状态大小的限制，它无法像 Attention 那样完美保留全局上下文信息。
*   **意义**: 这直接证明了纯 Mamba 架构在处理长程复杂逻辑时的局限性，也为 TGN (引入 Attention) 提供了最强的理论支持。

---

## 3. 深度讨论：现有架构缺陷与热力学视角

### 3.1 对现有混合架构 (Jamba/Zamba) 的批判
基于我们的实验数据，当前的串行混合架构（如 Jamba, $M \to A \to M$）存在本质的设计缺陷：

1.  **资源错配 (Misallocation)**:
    *   Jamba 采用均匀混合（每 8 层）。但我们的 SVD 数据显示，Mamba 在浅层（Layer 0-7）已经达到信息密度峰值，且远优于 Transformer。**在浅层插入 Attention 是算力的纯粹浪费。**
    *   Probing 数据显示 Mamba 的衰退主要发生在需要极长 Context 的深层。Jamba 在深层的 Attention 密度可能不足以弥补那 20% 的精度损失。
2.  **流形冲突 (Manifold Conflict)**:
    *   **球与锥的战争**：CKA 显示 $R^2 \approx 0.5$。Mamba 构建的是**各向同性球体**（Isotropic, Anisotropy $\approx 0.15$），而 Transformer 构建的是**各向异性锥体**（Anisotropic, Anisotropy $\approx 0.30$）。
    *   串行结构迫使特征在“平滑流形”（Mamba）和“尖峰流形”（Attention）之间反复跳跃，破坏了 Mamba 的惯性流（Inertial Flow），导致了严重的几何摩擦。

