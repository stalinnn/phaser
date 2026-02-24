# Thermodynamic Gated Networks (TGN): Attention as a Geometric Anti-Dissipative Force

**Mingyang Xu**  
Peking University

---

## Abstract
Complex systems, from biological brains to artificial neural networks, face the fundamental challenge of maintaining macroscopic order against thermodynamic entropy in high-dimensional non-convex landscapes. Traditional local diffusion models succumb to "glassy freezing" with exponentially diverging relaxation times. Here, we propose that the attention mechanism is not merely an engineering heuristic but a physical solution: a non-local heat kernel on a Riemannian manifold that creates topological shortcuts to bypass metastable traps. Based on this, we introduce Thermodynamic Gated Networks (TGN). In large-scale language modeling on WikiText-103, TGN spontaneously self-organizes into a critical state with ~9% attention sparsity, matching the performance of full Transformers while reducing computation by 90%. Furthermore, TGN overcomes the memory bottlenecks of State Space Models (SSMs) in multi-query associative recall. Finally, we demonstrate a striking isomorphism between TGN's geometric rank dynamics and fMRI patterns of human consciousness, suggesting a unified geometric principle underlying intelligence.

---

## Introduction

A fundamental question spanning physics, neuroscience, and artificial intelligence is how high-dimensional systems maintain low-entropy states (order) without succumbing to the "curse of dimensionality." In statistical physics, systems interacting locally on rugged energy landscapes often become trapped in metastable states, a phenomenon known as "glassy dynamics." The relaxation time required to find the ground state scales exponentially with system size, rendering optimization intractable.

Yet, biological brains and modern Large Language Models (LLMs) defy this limit, maintaining long-range coherence over vast temporal scales. We propose that this capability arises from a specific geometric mechanism: **non-local topological shortcuts**. While the Attention mechanism is conventionally viewed as a content-addressable memory, we show it is mathematically isomorphic to a non-local heat kernel operator on a Riemannian manifold. This operator effectively increases the spectral gap of the system's transition matrix, transforming the slow Arrhenius relaxation into fast polynomial mixing—a process we term **"Geometric Tunneling."**

We present **Thermodynamic Gated Networks (TGN)**, a framework derived from first principles of free energy minimization. TGN dynamically regulates the interplay between an inertial manifold (local processing, low energy) and a geometric manifold (non-local attention, high energy). We validate TGN across three physical hierarchies: (1) **Spin Glasses**, where it avoids glassy freezing; (2) **Language Modeling**, where it spontaneously finds a critical sparsity (~9%) that maximizes efficiency; and (3) **Neuroscience**, where its rank dynamics mirror the collapse and recovery of human consciousness under anesthesia.

---

## Results

### Thermodynamic Mechanism and Criticality
We model the intelligent agent's state evolution as a flow on a Riemannian manifold. To minimize the Helmholtz free energy $\mathcal{F} = U - \tau S$, the system must balance geometric alignment energy $U$ (drift) and information entropy $S$ (diffusion). We derive that the optimal transition probability $A_{ij}$ (attention weight) follows a Boltzmann distribution, where the scaling factor $1/\sqrt{d}$ in Transformers acts precisely as the critical temperature $\tau_c$.

**Figure 1** illustrates this mechanism. At $\tau < \tau_c$, the system freezes into a nearest-neighbor grid (frozen phase); at $\tau > \tau_c$, it dissolves into a mean-field average (melted phase). Only at the critical point $\tau \approx \tau_c$ does the system maintain "Edge of Chaos" dynamics, maximizing information transmission. Our theoretical prediction of an "S-shaped" phase transition in geometric rank is confirmed by experiments on GPT-2 layers (Fig. 1c).

*(Fig 1: Composite figure showing the theoretical energy landscape, the phase transition of geometric rank vs. temperature, and the S-curve scaling law.)*

### Geometric Tunneling in Spin Glass Systems
To quantify the algorithmic advantage of this geometric mechanism, we tested it on the 3D Edwards-Anderson (EA) spin glass model, a canonical NP-hard problem. We compared TGN-inspired dynamics against Langevin dynamics (local gradient) and Parallel Tempering (thermodynamic annealing).

As shown in **Figure 2**, while standard thermodynamic methods suffer from critical slowing down as system size $N$ increases (the "glassy freezing"), TGN maintains a constant energy density gap. This scale invariance suggests that the non-local connections in TGN function as "wormholes" in the energy landscape, allowing the system to tunnel through high energy barriers that would otherwise trap local optimizers for astronomical times.

*(Fig 2: Spin Glass results. (a) Energy scaling vs system size. (b) Convergence trajectory showing TGN breaking the glassy floor.)*

### Emergence of Sparsity on Natural Language Manifolds
Does this physical advantage translate to real-world tasks? We trained TGN (125M parameters) on the WikiText-103 benchmark. Unlike standard Transformers that calculate full $N \times N$ attention, TGN uses a learnable gate to dynamically switch between a linear Recurrent Neural Network (RNN) path and the quadratic Attention path.

**Figure 3** reveals a striking "Hysteresis Awakening" phenomenon. In early training (Steps 0-1000), the system shuts down the attention gate ($<1\%$) to exploit simple local correlations (inertial collapse). However, as the loss hits the information bottleneck of the RNN, the gate spontaneously re-opens, stabilizing at a non-trivial value of **~9%**. This matches the performance of a full Transformer (PPL ~24.86 vs 24.90) but with 90% fewer attention operations. This result provides empirical evidence that the "topological dimension" of natural language is significantly lower than its embedding dimension.

We further compared the early training dynamics of TGN against Mamba (a state-of-the-art State Space Model) and Transformer. TGN successfully matched the convergence speed of Transformer while maintaining <1% gate activation in the early phase, demonstrating extreme thermodynamic efficiency.

*(Fig 3: WikiText-103 Results. (a) The "U-shaped" gate dynamics and PPL convergence. (b) Early dynamics battle vs Mamba and Transformer.)*

### Breaking the Memory Bottleneck
While Mamba and other SSMs offer linear scaling, they suffer from a theoretical memory capacity limit due to their fixed-size state compression. We validated this using the Multi-Query Associative Recall (MQAR) task.

**Figure 4** shows that as the sequence length increases, Mamba's accuracy collapses to zero (the "Capacity Singularity"), confirming it cannot losslessly compress information beyond its state capacity. TGN, by contrast, maintains near-perfect accuracy by utilizing its geometric channel to perform non-local retrieval when necessary. This proves that retaining access to the full historical geometry ($O(N^2)$ capability) is physically necessary for tasks exceeding the channel capacity of linear recurrence.

*(Fig 4: MQAR accuracy vs sequence length, showing Mamba's collapse and TGN's resilience.)*

### Biological Isomorphism of Consciousness
Finally, we investigated whether this computational principle exists in biological brains. We analyzed fMRI data from 17 humans across four consciousness states: Awake, Light Sedation, Deep Sedation, and Recovery.

We calculated the "Effective Geometric Rank" of the functional connectivity matrices. **Figure 5** shows a strong correlation: the transition from Awake to Deep Sedation corresponds to a significant collapse in geometric rank ($p < 0.001$), mirroring the "rank collapse" seen in deep neural networks without attention. Furthermore, *in silico* simulation of psychedelic states (5-HT2A agonism) predicted a "hyper-expansion" of this rank, consistent with the entropic brain hypothesis. This suggests that the "Attention" mechanism in AI may share a common physical root with the integration mechanisms sustaining consciousness.

*(Fig 5: fMRI analysis. (a) Rank collapse during sedation. (b) In silico psychedelic expansion. (c) Causal lag analysis showing geometric collapse precedes synchrony surge.)*

---

## Discussion

Our work bridges the gap between statistical physics, AI architecture, and neuroscience. We show that the success of Transformers is not accidental but stems from their ability to approximate a non-local heat kernel, providing a geometric solution to the universal problem of glassy dynamics.

**Limitations**: While the isomorphism between TGN rank dynamics and fMRI data is compelling, fMRI measures hemodynamic responses on the order of seconds, whereas TGN operates on token-level milliseconds. This correspondence should be interpreted as a macro-dynamical similarity rather than a micro-circuit equivalence. Additionally, while TGN beats Mamba in capacity, its inertial path (currently GRU) converges slower than optimized SSMs; future work will integrate Mamba as the inertial engine of TGN.

In conclusion, TGN offers a unified perspective: intelligence is the ability to navigate high-dimensional manifolds by dynamically modulating geometric connectivity to minimize free energy.

---

## Methods

### Thermodynamic Gated Network (TGN) Architecture
The TGN block consists of two parallel paths: an Inertial Path (RNN/GRU) and a Geometric Path (Attention), controlled by a learnable gate $g_t \in [0, 1]$.
$$ \mathbf{h}_{inertial} = \text{RNN}(\mathbf{x}_t, \mathbf{h}_{t-1}) $$
$$ \mathbf{h}_{geo} = \text{Attention}(\mathbf{Q}_t, \mathbf{K}_{<t}, \mathbf{V}_{<t}) $$
$$ g_t = \sigma(W_g \mathbf{h}_{inertial} + b_g) $$
$$ \mathbf{y}_t = (1-g_t) \mathbf{h}_{inertial} + g_t \mathbf{h}_{geo} $$
The loss function includes a sparsity penalty: $\mathcal{L} = \mathcal{L}_{task} + \lambda \|g\|_1$.

### Experimental Setup
*   **Spin Glass**: 3D Edwards-Anderson model, $L \in [4, 12]$, $T=0.1$.
*   **WikiText-103**: 125M parameter models, trained on 6x A100 GPUs for 10 epochs using AdamW optimizer and gradient checkpointing.
*   **fMRI Analysis**: OpenNeuro dataset ds003171, preprocessed using fMRIPrep, rank calculated via singular value decomposition of Pearson correlation matrices.

### Data Availability
The code and pre-trained models are available at [GitHub Repository Link]. The fMRI data is available at OpenNeuro.org.

### Acknowledgements
(To be added)

### References
(To be formatted in standard Nature numerical style)
