# 投稿必备声明 (Declarations & End Matter Templates)

以下模板适用于 **Nature Machine Intelligence (NMI)** 投稿。请将填写完整后的英文段落添加到 Main Manuscript 的参考文献之前。

---

## 1. 摘要 (Abstract - Nature Style)
*(已按 Nature 格式韵律重写，直接可用)*

**Abstract**
Complex systems, from biological neural networks to artificial high-dimensional models, face a fundamental challenge: maintaining macroscopic order against thermodynamic entropy. Traditional local interaction models struggle with "glassy dynamics" in high-dimensional non-convex landscapes, where relaxation times diverge exponentially. Here we show that the long-range attention mechanism is not merely a computational heuristic but a **geometric anti-dissipative force**. By establishing a unified framework based on Riemannian geometric flow, we demonstrate that attention is physically isomorphic to a non-local heat kernel, which introduces topological shortcuts to maximize the spectral gap and reduce mixing times. This theory is supported by cross-disciplinary evidence: (1) attention-based dynamics exhibit scale-invariant tunneling in spin glass simulations, bypassing the Arrhenius barrier; (2) rank collapse in deep language models is reversed by this geometric force; and (3) fMRI analysis reveals that conscious states rely on maintaining such long-range geometric correlations. Finally, we introduce Thermodynamic Gated Networks (TGN), an architecture that adaptively switches between inertial and geometric modes, demonstrating that intelligence is fundamentally a thermodynamic process of minimizing free energy through geometric renormalization.

---

## 2. 数据可用性声明 (Data Availability Statement)
*(NMI 强制要求)*

**Data Availability**
The fMRI datasets analyzed during the current study are available in the **OpenNeuro repository** under accession number **ds003171** (https://openneuro.org/datasets/ds003171). The **WikiText-103** dataset used for language modeling is publicly available at https://blog.salesforceairesearch.com/the-wikitext-long-term-dependency-language-modeling-dataset/. The synthetic data generation scripts for the spin glass simulations are included in the code repository. Source data for Figures 1–4 are provided with this paper.

---

## 3. 代码可用性声明 (Code Availability Statement)
*(NMI 鼓励提供 GitHub 链接)*

**Code Availability**
The source code for the Thermodynamic Gated Network (TGN), along with scripts to reproduce the spin glass simulations and fMRI geometric rank analysis, is available at **[INSERT GITHUB URL HERE]**. The implementation is based on PyTorch and supports distributed training on NVIDIA GPUs. A simplified demo notebook for the "geometric tunneling" experiment is also provided for educational purposes.

---

## 4. 作者贡献 (Author Contributions)
*(请根据实际情况修改缩写和分工)*

**Author Contributions**
**M.X.** conceived the thermodynamic framework, derived the theoretical proofs, designed the TGN architecture, performed the spin glass and LLM experiments, and wrote the manuscript. **[Co-author Name]** assisted with the fMRI data preprocessing and statistical analysis. **[Co-author Name]** contributed to the mathematical formalization of the heat kernel isomorphism. All authors discussed the results and commented on the manuscript.

---

## 5. 利益冲突声明 (Competing Interests)
*(必须声明)*

**Competing Interests**
The authors declare no competing interests.
*(如果申请了 TGN 相关的专利，请改为: The authors have filed a patent application related to the Thermodynamic Gated Network architecture described in this work.)*

---

## 6. 图注检查清单 (Figure Legend Checklist)
*(NMI 要求图注独立可读，请检查是否包含以下要素)*

*   [ ] **图 1 (Spin Glass)**:
    *   **Bold Title**: **Fig. 1: Geometric tunneling bypasses the glassy landscape.**
    *   Details: 解释坐标轴 (Energy Density vs Time/Scale)。说明绿色曲线代表 Attention Dynamics。注明误差棒代表标准差 (SD, n=50 independent runs)。
*   [ ] **图 2 (LLM Rank)**:
    *   **Bold Title**: **Fig. 2: Reversing rank collapse in deep networks.**
    *   Details: 解释 V 型曲线的物理含义 (Semantic Crystallization vs Geometric Pumping)。说明灰色虚线代表无 Attention 的基线。注明温度 $T=1$ 处的相变点。
*   [ ] **图 3 (fMRI)**:
    *   **Bold Title**: **Fig. 3: Geometric signatures of consciousness.**
    *   Details: 解释纵轴 (Effective Geometric Rank)。对比 Awake, Light Sedation, Deep Sedation。注明统计显著性 (*P* < 0.001, one-way ANOVA)。
*   [ ] **图 4 (TGN)**:
    *   **Bold Title**: **Fig. 4: Spontaneous thermodynamic phase transition in TGN.**
    *   Details: 解释 U 型曲线 (Inertial Collapse -> Geometric Awakening)。说明双流架构如何通过最小化亥姆霍兹自由能自发寻找稀疏解。
