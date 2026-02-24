# Thermodynamic Gated Networks: Attention as a Geometric Anti-Dissipative Force

**Author:** Mingyang Xu  
**Affiliation:** Peking University  
**Date:** Jan 11, 2026  

---

## Abstract

How complex systems—such as biological neural networks and artificial high-dimensional models—maintain macroscopic order under thermodynamic entropy increase is a fundamental challenge in physics and neuroscience. Traditional local-interaction models (e.g., Langevin dynamics) suffer from exponentially large relaxation times in high-dimensional nonconvex energy landscapes, i.e., the *glassy dynamics* problem. Here we argue that the long-range interactions emergent in biological and artificial systems—*attention*—are not an engineering accident, but a geometric mechanism that counteracts high-dimensional thermodynamic dissipation. By establishing a unified theoretical framework based on Riemannian geometric flows, we show that attention is physically equivalent to a **non-local heat-kernel operator** on a manifold. By introducing **non-local topological shortcuts**, it substantially enlarges the spectral gap and reduces the mixing time. Our analysis further indicates that the Transformer scaling factor \(1/\sqrt{d}\) helps maintain the system at criticality—the **edge of chaos**—between order and disorder. This framework is supported by cross-domain evidence: (1) in physical simulations, attention-like dynamics exhibit superior scaling behavior over conventional thermodynamic annealing in large-scale spin-glass models; (2) in AI, it explains and reverses *rank collapse* in deep large language models; (3) in neuroscience, fMRI analyses show that the maintenance of conscious states correlates significantly with the strength of long-range geometric associations. Finally, we propose a **Thermodynamic Gated Network (TGN)**, suggesting that intelligence can be viewed as a geometric process that counteracts thermodynamic dissipation, whose physical essence follows a scale-invariant renormalization-group flow.

---

## 1. Introduction

Complex systems—whether biological neural networks or artificial deep models—face a fundamental physical challenge: how to sustain low-entropy ordered states in high-dimensional, nonconvex state spaces. According to nonequilibrium statistical physics, systems with purely local interactions (e.g., overdamped Langevin dynamics) confront severe scaling limitations when navigating high-dimensional energy landscapes: as the number of degrees of freedom \(N\) grows, the number of local metastable states increases exponentially, causing relaxation times to diverge and leading to so-called **glassy dynamics** or, in computational terms, **mode collapse**.

Yet modern AI systems (especially Transformer architectures) and biological brains appear to transcend this thermodynamic limitation, maintaining long-range correlations at large scales. We propose that this capability arises from a specific geometric mechanism. While attention is often treated as an efficient engineering module for information retrieval, we argue that it is mathematically isomorphic to a **non-local heat-kernel operator** on a Riemannian manifold.

This isomorphism reveals the physical essence of attention: it does not merely aggregate information, but acts as a **geometric anti-dissipative force**. By introducing content-based non-local couplings, attention effectively rewires the topology of the state space, creating “shortcuts” in spectral geometry. This significantly increases the **spectral gap**, enabling information flow to bypass metastable traps of local gradients via non-local routes, and thereby achieve **fast mixing** in phase space.

Within this framework, we build a unified theoretical model clarifying how intelligent systems use non-local geometric flows to counteract thermodynamic dissipation on information manifolds (i.e., rank collapse). To test universality, we conduct empirical studies across three physical levels:

1. **Statistical physics systems:** In 3D spin-glass models, we quantify the scaling advantage of the geometric mechanism over conventional thermodynamic annealing (e.g., parallel tempering), demonstrating a “geometric tunneling” effect in nonconvex optimization.
2. **Artificial neural networks:** By monitoring microscopic evolution of representation manifolds in LLMs, we confirm that attention reverses the natural decay of information entropy in deep networks, producing a renormalization of eigenvalues (rank renormalization).
3. **Biological neural systems:** Using human fMRI data, we find phase-transition-like behavior in geometric complexity of brain functional networks across consciousness states, consistent with theoretical predictions, suggesting that biological intelligence may follow the same topological dynamical principles.

In summary, we propose the **Thermodynamic Gated Networks (TGN)** framework. Without relying on specific biological or engineering analogies, and starting from first principles, we argue that non-local geometric connections are a physical necessity for high-dimensional intelligent systems to overcome thermodynamic dissipation and maintain critically ordered states.

---

## 2. Theoretical Framework: Energy-Based Riemannian Geometric Flow

To elucidate the physical mechanism of TGN from first principles, we avoid purely biological metaphors and instead adopt the **Energy-Based Model (EBM)** perspective. We formalize inference in an intelligent system as evolution of a gradient flow on a state-space manifold \(\mathcal{M}\).

### 2.1 Variational Principle: Attention as Free-Energy Minimization

Consider a query state \(\mathbf{q}_i\) on \(\mathcal{M}\). We seek an optimal probability distribution \(\mathbf{a}_i\) (attention weights) to aggregate neighborhood information \(\mathbf{K}\). This process is governed by two competing constraints:

1. **Geometric affinity:** the system tends to select features \(\mathbf{k}_j\) aligned with \(\mathbf{q}_i\) in the tangent space. We define the local potential energy as the negative inner product: \(E_{ij} = -\langle \mathbf{q}_i, \mathbf{k}_j \rangle\).
2. **Maximum-entropy principle:** to avoid overfitting (the trivial Dirac solution \(\delta\)), the system must maintain diversity of the distribution.

By the maximum-entropy principle in statistical physics, the problem is equivalent to the following constrained optimization:

$$
\min_{\mathbf{a}_i} \mathcal{F}[\mathbf{a}_i] = \underbrace{\sum_{j} a_{ij} E_{ij}}_{\text{Internal Energy } \mathcal{U}} - \tau \underbrace{\left( -\sum_{j} a_{ij} \log a_{ij} \right)}_{\text{Shannon Entropy } \mathcal{S}}
$$

$$
\text{s.t.} \quad \sum_{j} a_{ij} = 1, \quad a_{ij} \ge 0,
$$

where \(\tau\) is a Lagrange multiplier corresponding to the system’s **computational temperature**.

**Derivation 2.1.** Construct the Lagrangian
\(\mathcal{L} = \sum a_{ij} E_{ij} + \tau \sum a_{ij} \log a_{ij} + \lambda (\sum a_{ij} - 1)\).
Setting \(\frac{\partial \mathcal{L}}{\partial a_{ij}} = 0\) yields

$$
E_{ij} + \tau (1 + \log a_{ij}) + \lambda = 0
\;\Rightarrow\;
\log a_{ij} = -\frac{E_{ij}}{\tau} - \left(1 + \frac{\lambda}{\tau}\right).
$$

Thus the optimal distribution is of Boltzmann form:

$$
a_{ij}^*
= \frac{\exp(-E_{ij}/\tau)}{\sum_k \exp(-E_{ik}/\tau)}
= \text{Softmax}\!\left( \frac{\langle \mathbf{q}_i, \mathbf{k}_j \rangle}{\tau} \right).
$$

**Physical corollary.** Softmax attention in Transformers is not an ad-hoc engineering design, but the unique minimizer of the **Helmholtz free energy** \(\mathcal{F} = \mathcal{U} - \tau \mathcal{S}\) on the simplex. The scaling factor \(1/\sqrt{d}\) plays precisely the role of inverse temperature \(\beta = 1/\tau\).

### 2.2 Dynamics: Inverse-Contractive Geometric Flow

In deep networks, nonlinear activations (e.g., ReLU/Tanh) are typically **contractive**, causing manifold volume shrinkage (i.e., rank collapse). We formalize attention as an **inverse contractive flow** that counteracts this natural dimensional decay.

We interpret the residual update \(\mathbf{h}_{t+1} = \mathbf{h}_t + \text{Attn}(\mathbf{h}_t)\) as a discretization of a continuous-time ODE:

$$
\frac{d\mathbf{h}}{dt} = -\nabla_{\mathbf{h}} \mathcal{V}_{local}(\mathbf{h}) - \nabla_{\mathbf{h}} \mathcal{F}_{global}(\mathbf{h})
$$

- The first term \(-\nabla \mathcal{V}_{local}\) corresponds to local processing by FFN/RNN, typically driving the information manifold toward low-dimensional attractors (dissipation).
- The second term \(-\nabla \mathcal{F}_{global}\) is the non-local gradient flow introduced by attention.

**Mathematical property.** At a critical temperature \(\tau = 1/\sqrt{d}\), the attention matrix \(\mathbf{A}\) approximates a **heat-kernel operator** \(e^{-t\Delta}\) on the manifold. By spectral geometry, the heat kernel both smooths and diffuses; in the eigen-spectrum it lifts small eigenvalues (long-range modes), thereby reversing deterioration of the covariance condition number.

This provides a rigorous definition of the “anti-dissipative force”: **attention is a spectral regularization operator on the tangent bundle that maintains the effective dimension of the manifold by injecting non-local correlations.**

### 2.3 Critical Temperature and Concentration of Measure

We examine the physical meaning of \(\tau\). Let \(\mathbf{q}, \mathbf{k} \in \mathbb{R}^d\) be independent random vectors with components \(\mathcal{N}(0,1)\). The dot product \(X=\mathbf{q}^T\mathbf{k}\) has variance \(d\). For \(d \gg 1\), by concentration of measure, the unscaled dot products concentrate in an \(O(\sqrt{d})\) range.

If \(\tau\) is mis-set, the system deviates from the **edge-of-chaos** regime:

1. **Frozen limit (\(\tau \ll \sqrt{d}\))**: Softmax saturates, gradients vanish, and attention degenerates to one-hot. Physically, the manifold becomes effectively discretized and gradient propagation is blocked.
2. **Melted limit (\(\tau \gg \sqrt{d}\))**: Attention becomes uniform (mean-field); topology is washed out by noise.
3. **Critical state (\(\tau \approx \sqrt{d}\))**: Attention is neither Dirac nor uniform; the system reaches maximal geometric entropy, allowing gradients to propagate effectively through deep networks.

This gives a mean-field explanation of the Transformer scaling \(1/\sqrt{d}\): it counteracts high-dimensional concentration and keeps the system near **isometry**.

### 2.4 Linear Stability Analysis: The Singularity of \(K=1\)

To rigorously explain the “\(K=1\) instability” observed in simulations, we perform a linear stability analysis of the distributed iteration. This resembles classical control instability (overshoot under excessive feedback gain) but, in our distributed-network context, corresponds to a specific **topological obstruction**.

Consider error dynamics near a fixed point \(\mathbf{x}^*\): \(\mathbf{e}_t=\mathbf{x}_t-\mathbf{x}^*\). The discrete update is

$$
\mathbf{e}_{t+1} = (\mathbf{I} - \alpha \hat{\mathbf{G}}_K)\,\mathbf{e}_t,
$$

where \(\hat{\mathbf{G}}_K\) is the effective geometric operator (metric tensor estimate) inferred by \(K\)-step consensus. Stability depends on the spectral radius \(\rho(\mathbf{M})\) of \(\mathbf{M}=\mathbf{I}-\alpha \hat{\mathbf{G}}_K\), requiring \(|\mu_i|<1\) for all eigenvalues \(\mu_i\).

1. **First-order blindness (\(K=1\))**: the agent uses only local gradients, effectively \(\hat{\mathbf{G}}_1 \approx \mathbf{I}\) (identity assumption), mis-estimating the true physical Hessian \(\mathbf{H}\) as constant. With fixed \(\alpha\), high-frequency modes can satisfy \(|1-\alpha \lambda_{\max}|>1\), so errors alternate in sign and grow exponentially—matching the observed overshoot and energy divergence.

   We emphasize that this is a local approximation. In truly nonlinear loss landscapes, the instability manifests as getting trapped in saddle/high-loss regions rather than mere numerical overflow.

2. **Second-order damping (\(K \ge 2\))**: with \(K=2\), neighbor feedback provides sensitivity to spatial variation of \(\nabla V\), approximating \(\hat{\mathbf{G}}_2 \approx \mathbf{H}_{diag}\) or a Laplacian-like surrogate. Physically this introduces a **geometric damper**; modes with large curvature \(\lambda_{\max}\) are pulled back inside the unit circle.

Thus, the transition from \(K=1\) to \(K=2\) is not a quantitative improvement but a **topological phase transition** from a divergent spectral-radius regime to a convergent one.

### 2.5 Topological Necessity: Logarithmizing Arrhenius Barriers

To show attention is not an arbitrary engineering choice but a key topological solution to glassy dynamics, we consider Kramers escape. For any system with only local connectivity (\(K \ll \log N\)), the mean time to cross a barrier \(\Delta E\) follows Arrhenius:

$$
\tau_{local} \propto \exp\!\left(\frac{\Delta E}{k_B T_{sys}}\right).
$$

In nonconvex optimization (spin glasses, deep learning), typical barrier heights scale as \(\Delta E \sim N^\alpha\) with \(\alpha>0\), so \(\tau_{local}\) diverges exponentially with \(N\): glassy freezing / the dimensionality barrier.

Attention introduces content-based non-local coupling, inducing a deformed metric tensor \(\mathbf{G}_{Attn}\). From spectral graph theory, the attention matrix corresponds to adding a low-rank perturbation to a sparse adjacency, greatly increasing **algebraic connectivity** (the second-smallest Laplacian eigenvalue \(\lambda_2\)).

Under this renormalized metric, two points with Euclidean distance \(L\) can have an **effective resistance distance** compressed to \(\text{dist}_{eff} \approx \log L\) (small-world behavior). The expanded spectral gap yields exponential acceleration of mixing. In a large-deviation/path-integral view, the effective barrier is renormalized as

$$
\Delta E_{eff} \approx \frac{\Delta E}{\xi_{Attn}} \approx \frac{\Delta E}{\log N}.
$$

Consequently the relaxation time undergoes a qualitative shift:

$$
\tau_{Attn}
\propto \exp\!\left(\frac{\Delta E_{eff}}{k_B T_{sys}}\right)
= \exp\!\left(\frac{\Delta E}{k_B T_{sys}\cdot \log N}\right)
= N^{\frac{\Delta E}{k_B T_{sys} (\ln N)^2}}
\approx \text{Polynomial}(N).
$$

**Conclusion.** Attention provides a mechanism that can transform exponentially large **thermodynamic times** into tractable **computational times** at effective scales.

### 2.6 Variational Derivation of Sparsity: An Information-Bottleneck View

The spontaneous sparsification observed in TGN experiments (~10% gate rate) can be described as the optimal solution to an **information bottleneck (IB)** problem.

Let the gating decision be a binary random variable \(g_t \in \{0,1\}\). The goal is to minimize prediction error (distortion) while minimizing information flow through the geometric channel (rate/complexity). Define the total Lagrangian:

$$
\mathcal{L}(g_t) = \mathbb{E} \left[ \underbrace{(1-g_t) \ell_{inert} + g_t \ell_{geo}}_{\text{Prediction Loss}} + \underbrace{\lambda \cdot g_t}_{\text{Complexity Cost}} \right]
$$

where:

- \(\ell_{inert} = -\log p(\mathbf{y}\mid \mathbf{h}_{rnn})\) is the negative log-likelihood of the inertial channel.
- \(\ell_{geo} = -\log p(\mathbf{y}\mid \mathbf{h}_{rnn}, \mathbf{h}_{attn})\) is the negative log-likelihood of the geometric channel.
- \(\lambda\) is a sparsity penalty (chemical potential in thermodynamics).

Taking a variation (or derivative) w.r.t. \(g_t\):

$$
\frac{\partial \mathcal{L}}{\partial g_t} = (\ell_{geo} - \ell_{inert}) + \lambda.
$$

The optimal gating strategy takes a **Heaviside step** form:

$$
g_t^* = \mathbb{I}\left( \underbrace{\ell_{inert} - \ell_{geo}}_{\text{Information Gain } \Delta \mathcal{I}} > \lambda \right)
$$

**Physical interpretation.** The gate performs a Maxwell-demon-like operation:

1. **Measurement:** it measures instantaneous information gain \(\Delta \mathcal{I}\) (surprisal reduction) from enabling attention.
2. **Decision:** it opens only when \(\Delta \mathcal{I}\) exceeds the thermodynamic threshold \(\lambda\).

Because natural language and many physical processes follow Zipf/heavy-tailed statistics, high-surprisal events are sparse along time; hence \(P(\Delta \mathcal{I}>\lambda)\) is necessarily small.

**Conclusion.** TGN sparsity is not a manually tuned hyperparameter but an analytic consequence of heavy-tailed signals under a rate–distortion constraint, elevating “sparsity rate” to a measure of intrinsic data dimensionality.

### 2.7 Architectural Reification of the Thermodynamic Formula

We treat the Helmholtz free energy formula \(\mathcal{F} = U - TS\) as the physical blueprint for TGN design. The strict mapping between physical quantities and neural components is as follows:

1. **Internal Energy (\(U \leftrightarrow \mathcal{L}_{pred}\))**:
   - *Physical Meaning*: The potential energy of the system, corresponding to the surprisal caused by prediction error.
   - *Architectural Correspondence*: The hybrid prediction head. The inertial channel (RNN) provides a baseline prediction (low potential state), while the geometric channel (Attention) provides a corrective prediction. Minimizing internal energy drives the system to pursue high accuracy.

2. **Entropy Term (\(S \leftrightarrow -\|g\|_1\))**:
   - *Physical Distinction*: In classical thermodynamics, nature tends toward entropy maximization (\(\min F = U - TS\)). However, in biological and intelligent systems, to maintain a low-entropy ordered structure (Schrödinger's negative entropy), work must be performed to counteract natural dissipation. Thus, within TGN's variational free energy framework (Friston's Free Energy Principle), this term corresponds to **Variational Complexity** or **Negative Entropy (Negentropy)**.
   - *Architectural Correspondence*: We suppress the system's microscopic degrees of freedom (gating entropy) by minimizing \(\lambda \|g\|_1\), which is physically equivalent to introducing a **negentropy flow** to offset the thermodynamic entropy production of computation. Gate fully open (\(g=1\)) corresponds to a high-energy, high-DoF state; gate closed (\(g=0\)) corresponds to a low-energy quiescent state.

3. **Temperature Term (\(T \leftrightarrow \lambda\))**:
   - *Physical Meaning*: An order parameter controlling the system's phase. \(T\) determines the trade-off between "order (inertia)" and "disorder (geometry)".
   - *Architectural Correspondence*: The sparsity regularization coefficient \(\lambda\).
     - **High Temperature (\(\lambda \to 0\))**: Gates open randomly; the system degenerates into a pure geometric flow (Transformer), which has strong fitting power but is in a high-energy chaotic state.
     - **Low Temperature (\(\lambda \to \infty\))**: Gates freeze; the system degenerates into a pure inertial flow (RNN), which is energy-efficient but trapped in a crystalline state of local minima.
     - **Critical State (\(\lambda \approx \lambda_c\))**: TGN adaptively seeks the phase boundary, allowing the system to possess both inertial stability and geometric flexibility.

### 2.8 Control-Theoretic Principle of Adaptive Sparsity

We model the dynamic evolution of the gating mechanism as a constrained control process. The system's total objective function can be expressed as a generalized Lagrangian:

$$
\mathcal{L}_{total} = \mathcal{L}_{task}(\theta) + \lambda(t) \cdot \|g\|_1
$$

where \(\lambda(t)\) is an adaptive penalty coefficient that evolves with the training state. This objective function induces a competition and transition between two dynamical phases during optimization:

1. **Error-Driven Phase**: In the early stages of training or when facing complex tasks, the task loss \(\mathcal{L}_{task}\) dominates (\(\nabla \mathcal{L}_{task} \gg \lambda \nabla \|g\|_1\)). Since the geometric channel (Attention) possesses superior gradient flow properties and long-range modeling capabilities, the system tends to open the gate to rapidly capture key features and minimize prediction error. At this stage, the physical demand for "work" outweighs the demand for "energy conservation".

2. **Energy-Constrained Phase**: As model performance improves, the task error gradient gradually decays (\(\nabla \mathcal{L}_{task} \to 0\)), and the relative or absolute influence of the sparsity penalty term \(\lambda \|g\|_1\) rises. This generates a sustained "thermodynamic pressure" that forces the system to eliminate redundant geometric connections as much as possible, provided that the manifold structure (i.e., model performance) is maintained.

Mathematically, this process can be viewed as a form of **Homotopy Continuation**: the optimization trajectory first searches for the basin of attraction of the global optimum in a relaxed energy landscape, and then, as the constraints tighten, the solution trajectory spontaneously contracts towards a low-entropy, high-sparsity manifold subspace. This mechanism ensures that the model can adaptively find the optimal balance between "computational performance" and "energy efficiency" based on task difficulty and training stage.

---

## 3. Simulation and Results

### 3.1 Thermodynamic Stability of a Linear System

We first simulate a linearly coupled oscillator system. As shown in Fig. 1, the scalar system (red) exhibits an error that grows approximately linearly with temperature \(T\). In contrast, the tensor system (green), based on distributed consensus (\(K=20\)), maintains an extremely low error rate. Notably, the tensor dynamics here are realized purely via local communication, without any global matrix inversion, demonstrating the physical robustness of the mechanism.

![Fig. 1: Thermodynamic stability comparison](../figures/first_principles_network_dynamics_fixed.png)
*Fig. 1: (Left) Convergent dynamics in a high-entropy environment (\(T=2.5\)). (Right) Scaling of robustness with noise temperature. The green region indicates the negative-entropy gain achieved via local consensus.*

### 3.2 Complexity Barrier and Finite-Size Scaling

To test the “trading time for space” hypothesis, we perform a finite-size scaling analysis on a 1D ring lattice. Fig. 2 shows that as the system size \(N\) increases from 20 to 150:

1. **Scalar coordination (red):** due to diffusion slow-down, the error diverges as a power law \(\sim N^{1.5}\).
2. **Tensor coordination (green):** we set the consensus steps to \(K \approx N/2\). The coordination error is successfully suppressed to an approximately constant level.

![Fig. 2: Finite-size scaling](../figures/finite_size_scaling_ring.png)
*Fig. 2: Finite-size scaling on a ring lattice. The divergence of scalar coordination error reveals a “complexity barrier,” while linearly increasing consensus depth (Distributed Tensor) maintains scale-invariant coordination. This validates the theoretical prediction in Eq. (13) that increasing the spectral gap suppresses thermodynamic dissipation.*

### 3.3 Non-monotonic Phase Transition: The Coordination Instability Trap

Our high-resolution critical scan reveals a **non-trivial physical phenomenon**: the coordination order parameter \(R\) is not monotonic in the consensus depth \(K\), but exhibits a pronounced **non-monotonic re-entrant behavior**.

As shown in Fig. 3, the system displays three distinct dynamical phases:

1. **Thermodynamic blindness (\(K=0\))**: the system operates in an overdamped scalar diffusion mode. While it cannot establish long-range order (\(R \approx 0.4\)), each agent responds only to local noise, yielding a form of “mediocre stability.”
2. **First-order instability (\(K=1\))**: this is the key discovery. When agents attempt to incorporate nearest-neighbor information (a first-order gradient correction), performance does not improve; it collapses. The order parameter falls below random level (\(R < 0.1\)), and system energy surges.
   - **Physical mechanism:** this corresponds to resonant amplification of high-frequency modes. With only one communication step, agents obtain directional gradient information (where to go) but lack information about gradient variation (how curved the terrain is). In strongly coupled networks, this one-sided local greedy optimization leads to **collective overshoot**.
3. **Geometric ordered phase (\(K \ge 2\))**: once the consensus depth crosses the critical threshold \(K_c=2\), the system undergoes a **first-order phase transition**, and \(R\) jumps to \(R \approx 0.9\). The physical meaning of \(K=2\) is **curvature sensing**: via neighbor feedback, agents perceive second-order geometric structure (the Hessian). Attention introduces this second-order information as a **geometric damper**, suppressing the \(K=1\) oscillation and allowing the system to safely traverse the instability trap.

![Fig. 3: Non-monotonic phase transition](../figures/criticality_search.png)
*Fig. 3: Coordination order parameter (left) and system energy (right) vs. consensus depth \(K\). Note the energy divergence and the sharp drop in \(R\) at \(K=1\). This directly visualizes the high-frequency oscillation implied by Eq. (16) when the spectral radius \(\rho > 1\).*

### 3.4 Geometric Tunneling: Escaping the Glassy Trap

To directly validate the central role of attention in nonconvex optimization, we construct a **Rastrigin glassy potential landscape** model.

As shown in Fig. 4, we test three dynamical mechanisms for escaping local minima:

1. **SGD + momentum (local):** even with a momentum term (\(\beta=0.9\)) to enhance barrier crossing, particles remain trapped near the initial local basin.
2. **Local averaging (\(K=5\))**: although neighborhood information is introduced, the receptive field is limited; particles merely oscillate within a slightly larger local basin and still cannot escape metastability.
3. **Geometric attention (zero-shot geometric transfer):** to address the concern of “overfitting the objective function,” we design a strict **zero-shot transfer** experiment. We pretrain a Transformer only on a geometric denoising task.
   - **Tunneling effect:** results (Fig. 4) show that although the model only learns to “smooth geometric noise,” this smoothing is physically equivalent to suppressing high-frequency barrier oscillations. In a zero-shot setting, particles successfully escape Rastrigin local traps via **geometric tunneling**. This strongly supports attention as a **universal geometric low-pass filter** whose physical function is generic and need not be specialized for a particular landscape.

![Fig. 4: Geometric tunneling](../figures/tunneling_proof.png)
*Fig. 4: Zero-shot validation of geometric tunneling. (Left) SGD + momentum: trapped in local minima. (Right) Zero-shot geometric attention: tunnels toward the global minimum.*

### 3.5 Physical Breakthrough of the Complexity Barrier: From Thermodynamic Annealing to Geometric Tunneling

To validate the algorithmic efficiency advantage of attention in nonconvex optimization, we study the classical **3D Edwards–Anderson (EA) spin-glass model**, a widely recognized nonconvex benchmark with Hamiltonian
\(H(\mathbf{s}) = -\sum_{\langle i,j \rangle} J_{ij} s_i s_j + \sum_i (s_i^2 - 1)^2\).

We compare three dynamics corresponding to different physical mechanisms:

1. **Langevin dynamics (gradient baseline):** local gradient-based search.
2. **Parallel tempering (thermodynamic baseline):** barrier crossing via thermal fluctuations (replica exchange). Although highly optimized PT variants exist on specialized hardware (e.g., Janus), we focus on the standard algorithmic form to probe the intrinsic limit of the thermodynamic mechanism.
3. **Attention dynamics (geometric tunneling):** our geometric-smoothing algorithm, representing a search strategy based on non-local topological renormalization.

To expose scaling behavior with system size, we increase lattice size \(L\) from 4 to 12 (total spins \(N=1728\)). Results (Fig. 5a–b) reveal clear dynamical differences:

- **Glassy freezing of the thermodynamic baseline:** as size increases, standard parallel tempering—though better than Langevin—shows degraded ground-state energy density. This indicates that within finite time, relying solely on thermal fluctuations is constrained by Arrhenius scaling.
- **Scale invariance of the geometric mechanism:** in contrast, attention dynamics maintains a low energy density (\(\sim 0.65\)) across tested scales; at \(L=12\) it demonstrates an advantage over the thermodynamic baseline.

These results demonstrate a “geometric route” distinct from thermodynamic annealing: by changing the topology of the manifold, optimization shifts from “climbing over barriers” to “passing through barriers via non-local connections” (geometric tunneling), thereby improving search efficiency. We note that while the advantage is clear at \(N=1728\), universality in the thermodynamic limit (\(N\to\infty\)) requires larger-scale simulations.

We also plot the energy relaxation trajectories at \(N=1728\) (Fig. 5b) to visualize the dynamical contrast.

![Fig. 5a: 3D spin-glass scaling comparison](../figures/spin_glass_scaling_energy.png)
*Fig. 5a: Scaling analysis in the 3D Edwards–Anderson spin-glass model. (Red/blue) Traditional gradient/thermal-fluctuation methods exhibit glassy freezing as lattice size \(L\) increases, raising energy density. (Green) Attention dynamics shows striking scale invariance, indicating that geometric tunneling bypasses the entropy barrier that grows with dimension.*

![Fig. 5b: Large-scale relaxation dynamics](../figures/spin_glass_convergence_large.png)
*Fig. 5b: Relaxation dynamics in a large spin-glass system (\(N=1728\)) for \(L=12\). (Red) Langevin quickly falls into local metastable states. (Blue) Parallel tempering crosses some barriers but remains constrained by the entropy barrier, showing step-like glassy slow relaxation. (Green) Attention dynamics builds non-local tunneling paths via geometric smoothing, breaking the baseline plateau and reaching deeper ground states. The trajectory differences reflect a qualitative change in search efficiency.*

---

## 4. Zero-Shot Generalization in AI: Long-Range Geometric Generalization vs. GCN

Since physical simulations confirm the “geometric tunneling” advantage of attention in nonconvex optimization, a natural question follows: can existing neural architectures *spontaneously* learn this physical advantage? Put differently, do Transformers implicitly learn to exploit geometric shortcuts? To answer this, we shift from physical lattices to the learning dynamics of graph neural networks.

To address concerns of “circular reasoning” (i.e., AI merely memorizing specific training topologies) and to validate the physical advantage of Transformers on long-range dependency tasks, we design two tiers of experiments: (1) **zero-shot topological inference**, and (2) **Long-Range Arena benchmarks**.

### 4.1 Zero-Shot Topological Inference

We construct a dataset with diverse random topologies to train a Transformer to predict dynamics purely from node-state time series \(X_t\).

Results (Fig. 6) show that even when the input is a flattened 1D sequence, the Transformer spontaneously learns to reconstruct the local connectivity of a 2D grid. The model’s attention matrix is significantly correlated with the theoretical Green’s function (the inverse Laplacian) on the 2D grid (correlation \(\rho \approx 0.4\)).

![Fig. 6: AI geometric generalization](../figures/ai_generalization_proof.png)
*Fig. 6: Zero-shot geometric generalization. (Left) Attention heatmap generated by the model on an unseen 2D grid task. (Middle) Theoretical Green’s function computed from physics (inverse Laplacian). (Right) Field reconstruction shows clear localization. The comparison indicates that Transformers outperform locality-constrained GCNs in dynamical prediction due to non-locality.*

### 4.2 Physical Benchmark: Pure Attention vs. Hybrid SOTA (Benchmark: Horizon Collapse vs. Tunneling)

To quantify the physical limits of local diffusion vs. non-local attention, we construct a **long-range graph distance prediction task**. On a ring lattice, we randomly mark two nodes and ask the model to predict the shortest-path distance. We increase the graph diameter \(N\) from 20 to 100 and compare:

1. **GCN (3 layers):** classical local message passing.
2. **GraphGPS (1 layer):** a current SOTA hybrid (MPNN + linear attention).
3. **Transformer (1 layer):** pure global attention.

Results (Fig. 7) reveal striking physical phenomena:

1. **GCN receptive-field limitation:** as \(N\) grows, the GCN error (red) rapidly increases and saturates at random-guess level. Because the receptive field is limited by depth \(L\), for \(N \gg L\) it falls within an **information horizon**.
2. **Pure Transformer optimization bottleneck:** the pure Transformer (blue) performs well for \(N<60\), but errors rise at the extreme \(N=100\). This matches physical intuition: pure attention lacks topological inductive bias and must learn geometry from scratch. As \(N\) grows, the search space explodes, increasing optimization difficulty nonlinearly. Nevertheless, it remains far better than GCN, supporting that attention is necessary to break the horizon limit.
3. **GraphGPS critical hybrid advantage:** GraphGPS (green dashed) combines local inductive bias (MPNN-like) with global attention, achieving the best performance. This suggests a **critical strategy**: use locality to reduce optimization complexity while retaining long-range attention to break the horizon. Attention provides *possibility* (breaking the horizon), while local structure provides *feasibility* (reducing search space).

![Fig. 7: Long-range dependency benchmark](../figures/long_range_benchmark.png)
*Fig. 7: Long-range graph distance prediction. The GCN (red) fails due to locality-limited information. The pure Transformer (blue) breaks the horizon but faces optimization bottlenecks at very large scales due to lack of inductive bias. GraphGPS (green dashed) achieves the best performance by combining both, supporting a “local structure + global attention” critical hybrid strategy as the optimal response to complex systems.*

### 4.3 Geometric Dynamics in Real Language Models: Rank Renormalization

If GraphGPS demonstrates hybrid superiority on toy tasks, does a similar geometric advantage exist in industrial-scale LLMs? To open this black box, we introduce “geometric rank” as a microscopic probe to directly observe the evolution trajectory of information manifolds through depth.

To test universality in complex cognitive tasks, we measure the evolution of geometric dimension of internal representations in a pretrained language model (GPT-2).

In deep networks, **rank collapse** is widely observed: as depth increases, feature vectors converge to a few principal directions, causing the effective rank \(R_{eff}\) of the covariance matrix to decay rapidly. Physically, this corresponds to thermodynamic dissipation of information along the propagation path.

However, our theory predicts that attention acts as an anti-dissipative operator, maintaining high-dimensionality by injecting orthogonal contextual information.

To validate this, we compute the **effective rank** \(R_{eff}\) of the GPT-2 hidden-state matrix \(\mathbf{H}_l\) at each layer (defined as the exponential of the Shannon entropy of the normalized singular-value distribution, \(R_{eff}=\exp(H)\); it satisfies \(R_{eff}\le \min(\text{seq\_len}, d)\)). With fixed \(\text{seq\_len}=128\), the results (Fig. 8) reveal a pronounced “V-shaped inversion”:

1. **Compression phase (Layers 0–3):** the effective rank drops rapidly from about **40** to about **18**. This corresponds to early semantic clustering and denoising (“semantic crystallization”).
2. **Expansion phase (Layers 4–11):** this is a striking finding. In deeper layers, the effective rank does not continue to collapse; instead, it rises against the trend to about **50+**, exhibiting clear “geometric pumping / manifold fluctuations.”
   - **Limitations of ResNets and GCNs:** we emphasize that although residual connections (\(x+f(x)\)) often help “preserve rank” by preventing singular values from decaying too quickly, they do not explain such a strong rebound (from ~18 up to ~50+).
   - **Ablation confirms attention is the main manifold pump:** to exclude potential confounding effects from FFNs, we perform inference-time ablations. When we mask out the attention module during forward passes (keeping only FFNs), the rank rebound in Layers 4–11 almost disappears (rank gain collapses from 20.86 to 2.18). This provides strong evidence that FFNs and residual connections contribute only weak local rank preservation, while attention contributes the dominant geometric dimension restoration, acting as the core mechanism stabilizing dimensionality.

This “rank recovery” phenomenon constitutes key empirical support for our theory: attention successfully reverses the natural dissipative trend of information (rank collapse).

![Fig. 8: Effective-rank dynamics and ablation](../figures/llm_rank_ablation.png)
*Fig. 8: Geometric dynamics of GPT-2 representations (\(\text{seq\_len}=128\), \(R_{eff}=\exp(H)\)). (Left) The full model with attention (purple) shows a clear V-shaped inversion (“compress then rebound”); removing attention (red) significantly weakens the deep-layer rebound. (Right) Treating layer depth and activity magnitude (standard deviation, as an energy proxy) together with effective rank as phase-space coordinates yields an evolution trajectory of the representation manifold.*

### 4.5 Universality: Large-Scale Geometric Dynamics across Architectures

To show that “geometric manifold expansion” is not a peculiarity of GPT-2 (Post-LN, ReLU), we extend the experiment to a modern LLM architecture: TinyLlama-1.1B (Pre-LN, SiLU, RoPE).

Results (Figs. 9a–b) indicate that two very different architectures converge on the same anti-dissipative essence of attention:

1. **GPT-2 manifold fluctuations:** the blue curve exhibits the classic V-shaped inversion. Due to the causal mask, GPT-2 is forced to compress historical information into a low-dimensional manifold in early layers, and then strongly restores manifold dimensionality in deeper layers via attention’s non-local pumping.
2. **Generalization in a modern LLM (TinyLlama):** we reproduce the experiment on TinyLlama-1.1B (Fig. 9b). Despite different positional encoding (RoPE) and activation functions (SwiGLU), its deep layers still show a pronounced rank rebound. This supports attention’s anti-dissipative property as **architecture-agnostic**.
3. **BERT high-dimensional maintenance:** leveraging bidirectionality, BERT keeps the manifold at very high dimension throughout, effectively blocking the rank-collapse trend common in deep networks.

![Fig. 9a: Cross-architecture effective-rank dynamics](../figures/universal_rank_proof.png)
*Fig. 9a: Large-scale universality test (N=1000, WikiText-2). (Blue) GPT-2 shows pronounced “compression → expansion” dynamics. (Red) BERT, aided by bidirectionality, continuously maintains a high-dimensional manifold via attention.*

![Fig. 9b: Modern-architecture validation](../figures/llm_modern_rank_proof.png)
*Fig. 9b: Modern-architecture validation. (Red) TinyLlama-1.1B likewise exhibits a pronounced rank rebound, demonstrating that the mechanism remains robust under modern stacks including Pre-LN and RoPE.*

### 4.6 Long-Range Order: Near-Criticality and Correlation Extension

To quantify the physical efficacy of attention in suppressing information dissipation, we measure the **two-point correlation function** of GPT-2 generated text:

$$
C(k) = \langle \mathbf{v}_t \cdot \mathbf{v}_{t+k} \rangle.
$$

Our measurements (Fig. 10) show that while the tail remains approximately exponential (indicating the system is not perfectly scale-free critical), the characteristic correlation length reaches a remarkable \(\xi \approx 596\) tokens—two orders of magnitude larger than a traditional Markov process.

Physically, this state can be described as **near-criticality**. Attention acts as a **renormalization operator**, dramatically extending microscopic short-range correlations so that within the effective scale of human cognition (hundreds of words), the system exhibits effective long-range order.

![Fig. 10: Near-critical correlations](../figures/llm_criticality.png)
*Fig. 10: Thermodynamic correlation analysis of a language system. GPT-2 exhibits an extremely long characteristic correlation length (\(\xi \approx 596\)). Although correlations eventually decay exponentially, attention pushes the system toward a near-critical “edge-of-chaos” regime, maintaining macroscopic order over effective scales.*

### 4.7 Theoretical Prediction and Verification: Transformer at the Edge of Chaos

A complete physical theory should not only explain known phenomena but also **predict the unknown**. According to our geometric-renormalization theory, the attention scaling factor \(1/\sqrt{d}\) is not merely for numerical stability; it places the system precisely at the **critical point between order and disorder**.

We therefore make a counterintuitive prediction: if we manually change the attention “temperature” \(T\) (i.e., the scaling becomes \(1/(T\sqrt{d})\)), the geometric rank \(R_{eff}\) will not change linearly. Instead, it should exhibit an S-shaped curve akin to a ferromagnetic phase transition; moreover, the model’s optimal performance point (minimum perplexity) should coincide with the **inflection point** of this transition.

To test this, we perform **inference-time intervention** on a pretrained GPT-2 model, scanning \(T \in [0.1, 10]\).

Results (Fig. 11) strongly match the prediction:

1. **Phase-transition behavior:** the geometric rank (blue) indeed shows a clear transition-like pattern.
2. **Critical optimality:** the perplexity (red; loss proxy) attains its minimum near **\(T=1.0\)**, precisely around the point of maximum curvature of the rank curve (the most rapidly changing region).

This demonstrates that Transformers are effectively trained at the **edge of chaos**. Without explicit physics supervision, optimization for language tasks drives the model to discover the best physical temperature for maintaining high-dimensional geometric complexity.

![Fig. 11: Temperature phase-transition test](../figures/llm_criticality_test.png)
*Fig. 11: Verification of the theoretical prediction. (Blue) Geometric rank vs. temperature \(T\) shows a phase-transition-like curve. (Red) Model performance is optimized near \(T=1\), aligning with the critical region of the geometric transition. This supports that \(1/\sqrt{d}\) maintains the system at the edge of chaos.*

### 4.8 Complementary Perspective: Biological Unification of Inertia and Geometry

State-space models (SSMs, e.g., Mamba/S4) are often viewed as Transformer competitors due to their \(O(N)\) linear-time inference. From a biophysical perspective, however, we argue this is not a zero-sum game: they correspond to two fundamental dynamical modes of neural systems—**inertial dynamics** and **geometric dynamics**.

- **SSM (inertia / local loops):** biologically, the recurrence \(h_t = Ah_{t-1} + Bx_t\) resembles local neural circuits or synaptic integration. It efficiently processes continuous, low-frequency local information streams and maintains an “inertial” state, advantageous for highly correlated signals such as speech and motor control (low energy, continuity). Geometrically, it compresses history into a fixed-dimensional tube, acting as a **low-pass filter**.
- **Attention (geometry / long-range connections):** attention corresponds to long-range functional connectivity or re-entrant processing in the brain. Although energy-expensive (\(O(N^2)\)), it breaks spatiotemporal constraints and creates instantaneous **high-dimensional topological shortcuts**, performing global manifold renormalization.

Our theory suggests that intelligence emerges from a **critical mixture** of these two mechanisms. A system should not pursue extreme \(O(N)\) efficiency alone (which would cause geometric collapse in complex reasoning, as indicated by the red curve in Fig. 12), nor should it continuously spend \(O(N^2)\) energy. TGN results indicate that a system can rely on SSM-like inertia for ~90% of the time, activating attention-like geometry only when it must break a local “information horizon.”

Results (Fig. 12) further validate this complementarity:

- **Mamba proxy (red):** in deep networks, pure recurrence lacks geometric pumping, and effective rank gradually decays (~41), reflecting the physical limitation of local dynamics in maintaining long-range high-dimensional correlations.
- **Transformer (blue):** global attention maintains a higher geometric rank (~53) at the cost of sustained high energy.

This suggests that an ideal neural architecture (and possibly the biological brain) is a **thermodynamic hybrid**: use SSMs as the inertial base to sustain an inertial manifold, and use attention as a gated operator for geometric jumps.

![Fig. 12: SSM vs. attention effective-rank dynamics](../figures/ssm_vs_attention_rank.png)
*Fig. 12: Evidence for geometric necessity. (Red) An SSM (Mamba proxy) undergoes significant rank collapse in deep networks due to state compression. (Blue) Global attention maintains a high-dimensional manifold via \(O(N^2)\) non-local connectivity, indicating that for tasks requiring high geometric complexity, global attention has an irreplaceable physical advantage.*

### 4.9 From Toy Models to Giants: Validation of Geometric Scaling Laws

Based on our geometric thermodynamic theory, we make a **strong prediction**: if attention is truly an anti-dissipative engine, then as model scale (degrees of freedom) increases, the system must perform more intense geometric renormalization to counteract the exponentially growing phase-space entropy. Equivalently, the amplitude of the “V-shaped” curve should grow nonlinearly with parameter count.

To validate this prediction, we conduct an empirical study spanning three orders of magnitude, covering SOTA models from 0.1B to 7B parameters:

1. **GPT-2 Small** (0.1B, 12 layers)
2. **TinyLlama** (1.1B, 22 layers)
3. **Qwen2.5-7B** (7B, 28 layers; comparable to Llama-3 class)

Measurements (Fig. 13) strongly confirm the prediction and reveal a scale-amplification effect of **manifold fluctuations**:

- **Semantic crystallization (physical analog: dimensional compression / cooling):** as scale grows, early-layer geometric compression becomes extremely sharp. Qwen-7B compresses effective rank from ~140 down to **~8.9** at Layer 4, indicating strong feature extraction that compresses massive text information into a single-digit geometric dimension.
- **Geometric hyper-expansion (physical analog: manifold tunneling / reprojection):** in deep layers, attention-induced rank recovery not only persists but grows substantially. Qwen-7B rebounds from ~8.9 up to ~274.6, achieving >30× geometric tunneling.

This “deep V-shaped” dynamics indicates that attention’s anti-dissipative role is not a small-model patch but the geometric engine behind intelligence emergence in large models. Over-parameterization does not eliminate the need for geometric tunneling; it provides room for larger-scale dimension manipulation.

![Fig. 13: Empirical geometric scaling law](../figures/llm_scaling_law_empirical.png)
*Fig. 13: Cross-scale empirical geometric dynamics. (Blue/orange) Smaller models (0.1B/1.1B) show mild rank recovery. (Red) A SOTA large model (Qwen-7B) exhibits dramatic “semantic crystallization” followed by “geometric hyper-expansion.” This indicates that as model scale increases, the system’s capacity to renormalize the manifold via attention is amplified.*

---

## 5. Cross-Species Validation: Geometric Rank Recovery in Biological Brains

So far, we have validated the universality of the TGN principle in physical simulations and artificial neural networks. A deeper question then arises: if attention is truly an “evolutionary advantage” against high-dimensional dissipation, has the biological brain—shaped by millions of years of evolution—also developed similar geometric structure? To address this, we shift from silicon models to carbon-based brains.

To extend the theory from artificial networks to biological entities, we analyze a public fMRI dataset from **OpenNeuro (ds003171)**. The dataset records whole-brain activity from 17 healthy participants across four states: **Awake**, **Light Sedation**, **Deep Sedation**, and **Recovery**.

Using the Schaefer 2018 atlas, we parcellate the brain into 100 functional regions and compute the **effective geometric rank** of the neural manifold under different consciousness states.

### 5.1 Empirical Evidence: Geometric Rank Evolution across Consciousness States

The empirical results (Fig. 14) reveal a clear geometric phase-transition-like trend:

1. **Awake:** the brain maintains the highest geometric complexity (\(R_{eff} \approx 11.23\)). This aligns with the Global Workspace view of whole-brain information integration and suggests that attention-like long-range coupling may help sustain a high-dimensional manifold.
2. **Sedation (rank collapse):** as sedation deepens, 15 participants show a significant rank decrease (Awake vs. Deep, \(p < 0.001\)). This matches our core hypothesis: when the long-range “geometric pump” is chemically suppressed, the system may regress to a low-dimensional, locally diffusive mode.
3. **Recovery (geometric rebound):** during recovery, geometric rank not only rebounds but in some participants even overshoots the baseline (\(R_{eff} \approx 11.55\)). Such non-monotonic elastic dynamics is typical in geometric/topological systems.

Overall, this biological evidence parallels the rank-recovery patterns in AI models, indicating a strong association between consciousness-state transitions and geometric-rank dynamics, and providing a computational perspective on attention-like mechanisms in biological intelligence.

![Fig. 14: Effective geometric rank in real human fMRI](../figures/long_range_specific_collapse.png)
*Fig. 14: Geometric rank evolution from OpenNeuro ds003171 (N=17). (A) Awake maintains high rank. (B) Deep sedation reduces the mean geometric rank, indicating contraction of consciousness-related associative scale. Despite substantial inter-individual variability (error bars), the overall trend supports the physical hypothesis that a high-dimensional geometric state may be a necessary condition for consciousness.*

### 5.2 Mechanism: Computational Simulation of an In Silico Psychedelic State

To probe the microscopic mechanism behind these geometric changes, we propose a **falsifiable, counterintuitive prediction**: if anesthetics (GABA agonists) induce geometric rank collapse, then classical psychedelics (5-HT2A agonists), as disinhibitors, should cause the **opposite**—a geometric **hyper-expansion**, potentially exceeding the awake baseline.

We build an **in silico psychedelic simulation** on a modular brain network. We model 5-HT2A agonism as two independent geometric parameter changes:

1. **Enhanced entropic drive:** increased neuronal gain raises noise level (REBUS-like mechanism).
2. **Long-range disinhibition:** selective strengthening of long-range geometric connections (increased global functional connectivity).

Simulation results (Fig. 15) reveal a pronounced **geometric transition**:

- **Awake baseline:** geometric rank remains near a critical level (\(R_{eff} \approx 3.4\)).
- **Psychedelic state:** during the drug window (steps 500–1500), geometric rank expands sharply, peaking at **2.55×** the baseline (\(R_{eff} \approx 8.65\)).

This suggests that psychedelic experience may correspond, in mathematical essence, to **dimensional expansion of the state manifold**. Such geometric over-expansion offers an explanation for subjective reports of “expanded consciousness,” synesthesia, and dissolving semantic boundaries: the system breaks conventional low-dimensional constraints and accesses high-dimensional state spaces that are normally unreachable. This computational result is consistent with 5-HT2A mechanisms and provides quantitative topological support for Carhart-Harris’ “entropic brain” hypothesis.

![Fig. 15: In silico psychedelic geometric simulation](../figures/psychedelic_rank_expansion.png)
*Fig. 15: In silico psychedelic simulation. (Top) Neural activity raster shows global avalanche-like activity during the drug window (steps 500–1500). (Bottom) Effective geometric rank over time. During the psychedelic window, rank does not collapse; it undergoes significant **hyper-expansion**, exceeding 2.5× the baseline.*

### 5.3 Computational Mechanism Analysis: The Geometric Precursor Effect

To study the temporal relationship between geometric structural change and state transition, we perform **temporal lag analysis**. Note that while Granger causality is not equivalent to physical causality, temporal precedence can still provide mechanistic clues.

If our hypothesis is correct—namely, that long-range geometric connections are the physical substrate for maintaining a high-dimensional manifold—then during anesthesia induction, the rupture of long-range geometry (geometric collapse) should precede the burst of global synchrony (a marker of loss of consciousness). We call this the **geometric precursor**.

We construct a detailed dynamical causal model simulating a linear rise of anesthetic concentration. Results (Fig. 16) show a significant temporal lag:

1. **Geometric collapse (blue):** as drug concentration increases, \(R_{eff}\) responds first to small perturbations in long-range coupling and begins to drop rapidly.
2. **Synchrony surge (red):** global synchrony (order parameter) responds later and displays inertia.
3. **Time difference \(\Delta t\):** the maximal collapse-rate point of geometric rank occurs **425 time steps earlier** than the maximal rise-rate point of synchrony.

The clear \(\Delta t > 0\) is consistent with the hypothesis that “geometric structure collapse may precede the macroscopic phase transition.” While simulations cannot fully replace biological experiments, they provide mechanism-consistent computational evidence, suggesting a potential dynamic driver from microscopic geometry to macroscopic consciousness state.

#### Statistical Analysis Note

For fMRI correlation analyses, we use Pearson correlation and apply Bonferroni correction across \(N=17\) participants (\(p < 0.05/100\)). In silico simulation results are averaged over 50 independent random seeds; shaded regions indicate 95% confidence intervals.

![Fig. 16: Causal geometric precursor analysis](../figures/causal_lead_lag.png)
*Fig. 16: Temporal lag analysis during anesthesia induction. (Blue) Geometric rank (manifold dimension proxy) collapses first. (Red dashed) Global synchrony (loss-of-consciousness proxy) shows a pronounced delay. The lag \(\Delta t > 0\) suggests that microscopic geometric changes may serve as a precursor to macroscopic consciousness-state transitions.*

### 5.4 Virtual TMS Intervention: Counterfactual Simulation

To further test the dependence between geometric structure and system state, we design a computational **counterfactual intervention** experiment.

Using the in silico brain model, we simulate:

1. **Induction:** the system enters a low-rank, low-consciousness state.
2. **Maintenance:** drug concentration is held constant (metabolism suppressed).
3. **Virtual TMS awakening:** without changing drug concentration, we apply strong external drive (simulated TMS) to forcibly activate long-range geometric connections.

Prediction: if geometric structure is a necessary carrier for high-dimensional information flow, then TMS should rapidly restore geometric rank even when metabolism remains suppressed.

Results (Fig. 17, right) support this prediction: during the TMS window, despite the continued “anesthetic” environment, geometric rank rebounds rapidly to the awake level (\(R_{eff}\) jumps from ~1.5 to ~8.0). This counterfactual result is **mechanistically consistent** with the hypothesis that topology itself may be a key factor for sustaining high-dimensional information flow, echoing optogenetic findings such as Redinbaugh et al. (2020).

![Fig. 17: Geometric mechanism and causal intervention](../figures/tms_causality_proof.png)
*Fig. 17: (Left) Virtual TMS intervention experiment. The yellow region indicates the TMS window. Under persistent anesthetic conditions, forcibly restoring long-range connections rapidly restores geometric rank, supporting the hypothesis that topology plays an important role in maintaining high-dimensional states.*

---

## 6. Architecture Innovation and Engineering Validation: Thermodynamic Gated Networks

Theory has provided a direction and empirical analyses have validated the principle; now it is time to turn science into engineering. In this section we propose the **Thermodynamic Gated Network (TGN)** and systematically validate its engineering advantages following the logic **“microscopic mechanism \(\to\) mesoscopic scaling \(\to\) macroscopic emergence \(\to\) physical limits.”**

We do not construct the model in a vacuum; rather, we treat TGN as an ultimate experimental platform to test the physical theory above. We aim to answer five core questions:

1. **Mechanism validation:** does it really function like thermodynamic gating? (6.1)
2. **Limit testing:** can it solve problems that linear models cannot? (6.2)
3. **Scaling behavior:** as the system scales up, does it exhibit the expected phase transition? (6.3)
4. **Natural emergence:** on real data manifolds, does it spontaneously find sparse solutions? (6.4)
5. **Boundary probing:** compared with Mamba, where is its physical limit? (6.5)

Based on the theoretical findings, we propose a new neural architecture—**Thermodynamic Gated Networks (TGN)**. By introducing an adaptive phase-transition mechanism based on geometric entropy, it aims to resolve the classic dilemma in hybrid models between long-range reasoning and computational efficiency.

### 6.1 Adaptive Hybrid Architecture: Homeostasis of Micro-Dynamics

To validate engineering feasibility, we build a hybrid prototype based on a **hysteresis-based thermodynamic regulation** mechanism. Unlike fixed hybrid architectures (e.g., Jamba), this mechanism allows an adaptive bistable switch between an **inertial manifold** (low rank) and an **excited manifold** (high rank).

We design a **Hybrid Nano-Architecture** consisting of an inertial unit (GRU) and a geometric pump unit (multi-head attention). The system includes a real-time “entropy sensor” that monitors the effective rank of the hidden-state manifold.

#### 6.1.1 Design Goal and Mechanism Overview

The core goal of TGN is: for most time steps, process local dependencies using a **low-energy inertial channel** (RNN/GRU), and only when information is about to undergo “geometric collapse,” trigger a **high-energy geometric pump** (causal attention) once to restore dimension. From first principles, this is equivalent to asking: in a system with \(N\) nodes, what aggregation weights \(p_{ij}\) should node \(i\) use to minimize an energy–entropy trade-off? Under a temperature constraint, scaled dot-product attention is a canonical solution to this variational problem.

#### 6.1.2 Algorithm 1: Core Mechanism of Thermodynamic Gated Networks (TGN)

TGN outputs \(g_{1:T}\in(0,1)\) via a lightweight gating network (“entropy regulator”). During training, it uses **soft gated mixing** for stability and differentiability; during inference, one can threshold \(g\) to obtain sparse acceleration.

```text
Input:  token sequence x_{1:T}, embedding E(·), inertial unit GRU,
        causal attention Attn, gating network Gate(·), output head Head
Output: predicted logits, gates g_{1:T}, attention weights A (optional)

1) Pre-normalization:    z_{1:T} ← LayerNorm(x_{1:T})
2) Inertial channel:     h^{rnn}_{1:T} ← GRU(z_{1:T})
3) Geometric pump:       h^{attn}_{1:T} ← Attn(z_{1:T})   (computed in parallel with RNN)
4) Compute gate:         g_{1:T} ← Gate(h^{rnn}_{1:T})
5) Gated fusion:         Δh_{1:T} ← (1 - g_{1:T}) ⊙ h^{rnn}_{1:T} + g_{1:T} ⊙ h^{attn}_{1:T}
6) Residual connection:  x'_{1:T} ← x_{1:T} + Δh_{1:T}
7) Feed-forward:         x_{next} ← x'_{1:T} + FFN(LayerNorm(x'_{1:T}))

Training loss:
L = CE(logits, y) + λ · mean(g_{1:T})   (λ>0 penalizes attention “energy cost,” encouraging ultra-sparse gating)
```

#### 6.1.3 Complexity Analysis (Sequence Length \(T\), Hidden Dimension \(D\))

To avoid the misconception “sparse gating \(\Rightarrow\) guaranteed \(100\times\) speedup,” we state the dominant complexity under different implementation settings. Let the expected gate activation rate be \(p=\mathbb{E}[\mathbb{1}(g>\tau)]\).

- **Inertial channel (GRU/RNN):** one recurrent update per step; dominated by matrix multiplications. Time \(\approx O(TD^2)\); if storing all hidden states, space \(O(TD)\).
- **Full-sequence self-attention (standard Transformer):** attention matrix is \(T\times T\). Time \(O(T^2D)\); space \(O(T^2)\) (ignoring projection constants).
- **This paper’s implementation setting (predict only at the last step):** we perform a causal-attention readout only at the final step, so the attention part is \(O(TD)\); overall cost is typically dominated by projections and RNN updates (effectively \(\sim O(TD^2)\)).
- **Expected complexity of TGN:**
  - **Training (soft gating):** for differentiability and batching stability, attention outputs are still computed and mixed; complexity remains comparable to the sum of “inertial + attention” channels.
  - **Inference (hard gating):** with thresholded gating, the expensive geometric pump is triggered only when \(g>\tau\). For full-sequence attention, an expected complexity is

  $$
  O(TD^2)\;+\;p\cdot O(T^2D).
  $$

  Hence TGN’s computational benefit comes from sparsifying the expensive global-mixing computation by a factor of \(p\). In wall-clock time, realized speedup still depends on whether the baseline attention is truly \(O(T^2)\) and whether the sparse operator is efficiently implemented.

#### 6.1.4 Architectural Variants: Sequential Bottleneck vs. Parallel Rescue

TGN supports two topological coupling modes, corresponding to different physical hypotheses and application scenarios:

1. **Sequential coupling (Sequential TGN, \(\mathbf{h}=\text{Attn}(\text{RNN}(\mathbf{x}))\))**: the attention module can only access the RNN output. This design imposes an information bottleneck, forcing gating to choose between “accepting lossy compression” and “turning on high-energy computation.” It is suitable for **ablation studies** to test gating sensitivity.
2. **Parallel coupling (Parallel TGN, \(\mathbf{h}=\text{Gate}\cdot \text{Attn}(\mathbf{x}) + (1-\text{Gate})\cdot \text{RNN}(\mathbf{x})\))**: attention directly accesses the raw history. This allows the geometric channel to bypass inertial dissipation and achieve true “dimension restoration.” It is suitable for **high-performance benchmarks** aiming for best convergence speed and memory capacity.

In subsequent experiments, we adopt **sequential mode** in **Section 6.2 (limit ablation)** and **Section 6.3 (large-model scaling)** to validate effectiveness and scaling under an information bottleneck; and we adopt **parallel mode** in **Section 6.4 (language modeling)** and **Section 6.5 (physical limit testing)** to pursue maximal memory capacity and inference speed.

#### 6.1.5 Prototype Validation: Phase-Transition "Geometric Rescue" and Homeostatic Cycle

Experimental results (Fig. 18) reveal a striking phenomenon of **macroscopic "Cognitive Breathing"**:

1. **Information Overload & Collapse:** As sequence length increases, the effective rank of the GRU hidden state exhibits a natural entropy-reduction trend, gradually falling below a critical threshold ($\approx 0.55$).
2. **Geometric Rescue:** Once the sensor detects rank collapse, the Attention gate is instantly activated. We observe that the Attention mechanism does not simply superimpose information; instead, via non-local retrieval, it instantaneously **pulls** the rank back to a safe zone (jumping from 0.55 to 0.65+).
3. **Homeostatic Cycle:** This "collapse-rescue-cooling" cycle emerges spontaneously in long sequences, demonstrating that the system can maintain long-range macroscopic order with minimal computational cost (activating Attention in only ~15% of time steps).

This result directly supports our core hypothesis: intelligence is not a static structure, but a non-equilibrium steady state dynamically balanced between **dissipation** and **anti-dissipation**.

![Fig. 18: Adaptive phase-transition validation](../figures/real_hybrid_experiment_polished.png)
*Fig. 18: Adaptive phase-transition dynamics in a real hybrid network. (Top) The effective rank of the GRU hidden state decays naturally with sequence length. When it falls below the red dashed threshold, the Attention mechanism intervenes (red vertical bars), causing an instantaneous rebound of the rank (Geometric Rescue). (Bottom) Gating states show that Attention opens only at critical moments, achieving a Pareto optimum between energy saving and intelligence.*

### 6.2 Limit Testing: From Hysteresis Awakening to Emergence of Intelligence

To fully dissect the interaction between inertia and geometry in TGN, we conduct a stringent ablation study on an **extreme long-range (SeqLen = 1024)** associative recall task. To force the system to expose its physical limits, we use the **sequential coupling** architecture and apply a strong adaptive thermodynamic penalty (\(\lambda_{max}=0.02\)).

Results (Fig. 19) reveal essential differences among three physical mechanisms:

1. **Horizon collapse of the inertial manifold:** as shown by the red dashed line, a pure RNN model completely fails at length \(L=1024\), with accuracy stuck at random-guess level. This confirms that gradient signals cannot penetrate a “thermodynamic horizon” that decays exponentially with time.
2. **Full-attention computation on the geometric manifold:** a pure Transformer (blue dashed) converges extremely fast (reaching 100% by Iter 200), but only with **100% attention activation**, incurring high energy cost and offering no computational economy.
3. **Thermodynamic phase transition in TGN:** TGN (green solid) exhibits **hysteretic phase-transition** dynamics, fully displaying the three-stage behavior of self-organized criticality:
   - **Incubation (Iter 0–400):** constrained by the sparsity penalty, the system maintains a low gate rate (orange dashed), and accuracy remains low.
   - **Phase transition (Iter 400–800):** as accumulated gradient potential crosses a thermodynamic threshold, the system undergoes a cascade transition; gate rate rises quickly and drives accuracy to 90%+.
   - **Sparsification (Iter 1000+):** this is the key moment of intelligence emergence. After learning the rule, the system spontaneously performs “structured pruning.” For example, in the best-performing Seed 1024, the model keeps 100% accuracy at Iter 1900 while reducing the gate rate to **5.5%**.

These results show that TGN not only solves long-range dependency tasks, but also exhibits a form of “meta-cognitive” capability: it self-organizes toward a **compute/accuracy Pareto frontier**. With <6% geometric-entropy cost, it achieves causal connections across thousand-step sequences.

![Fig. 19: Hysteresis awakening dynamics](../figures/ablation_associative_rigorous.png)
*Fig. 19: Hysteresis awakening and adaptive sparsification in long-range associative recall (L=1024). (Red) Pure RNN fails. (Blue) Transformer is fast but energy-expensive. (Green) TGN exhibits nonlinear “incubation → phase transition → sparsification” dynamics. Note the gate rate (orange dashed; right axis) spontaneously decreases after task acquisition; Seed 1024 achieves an extremely sparse solution at 5.5%.*

### 6.3 Meso-Scaling: Sigmoidal Growth of Geometric Intelligence

To determine the theoretical limit of TGN gating in very large models (>100B), we cross-validate our experimental findings with the sparse-attention literature.

#### 6.3.1 Limitations of Existing Sparse Paradigms

Current sparse-attention methods can be grouped into three classes, each reflecting a different sparsity trade-off:

1. **Static patterns:** e.g., Sparse Transformer (Child et al., 2019) and BigBird (Zaheer et al., 2020). They use fixed sparse masks (e.g., local window + global tokens), typically with density **15%–25%**.
   - *Limitation:* such rigid geometry cannot adapt to dynamic data. For tasks requiring precise long-range retrieval (e.g., code completion), static sparsity often causes performance collapse.
2. **Learnable clustering:** e.g., Reformer (Kitaev et al., 2020). It uses LSH to cluster similar queries/keys and can in principle reduce density to **<5%**.
   - *Limitation:* clustering introduces heavy overhead and in practice often yields significant accuracy degradation; it is difficult to reproduce BERT/GPT-level performance.
3. **Hardware-aware pruning:** e.g., H2O (Zhang et al., 2023), a current SOTA approach that dynamically drops low-score KV cache entries at inference.
   - *Empirical limit:* H2O reports near-lossless performance at **20%** KV cache, but perplexity degrades significantly once below **10%**.

#### 6.3.2 TGN’s Sigmoidal Scaling Prediction

Combining TGN observations with these empirical limits, we construct a clean and conservative argument that TGN scaling exhibits a **sigmoidal growth** trend:

![Fig. 20: Sigmoidal growth of geometric intelligence](../figures/s_curve_final.png)
*Fig. 20: S-shaped growth of TGN gate rate with model scale. Solid points are empirical observations; the dashed curve is a conservative theoretical projection anchored by literature limits such as H2O. The curve suggests a phase transition from “inertia-dominated” to “geometry-dominated” intelligence.*

1. **Observation 1 (initial adaptation, Small, 21M):** gate rate \(\approx 1\%\). Limited capacity prevents effective global utilization; the system is inertia (RNN) dominated.
2. **Observation 2 (rapid adaptation, Large, 454M):** gate rate \(\approx 6\%\). With increased capacity, the system undergoes a phase transition and spontaneously demands long-range geometric connections.
3. **Literature anchor:** frontier studies such as H2O suggest that for very large LMs (>7B), compressing KV cache below **20%** begins to harm performance, setting an empirical lower bound on information density.
4. **Inference:** TGN uses a **training-aware**, proactive sparsification strategy, which should be more information-efficient than inference-only dropping in H2O. We therefore infer that TGN’s gate-rate limit likely lies in **6%–20%**, plausibly converging to **10%–15%**.

This suggests that TGN’s adaptive gating is effectively searching for a **physical limit**. Unlike H2O’s post hoc remedy, TGN spontaneously converges to this golden interval during training. In this view, **10%–15% geometric sparsity is not an engineering knob but a projection of the intrinsic “topological dimension” of human language tasks.**

### 6.4 Emergence of Sparsity on Natural Language Manifolds

To test whether TGN gating can spontaneously find a thermodynamic optimum under **natural data distributions**, we study TGN on the industrial-scale language-modeling benchmark **WikiText-103**. We train a **125M-parameter** model (comparable to GPT-Neo Small) on a 6×A100 cluster.

Results (Fig. 21) reveal two salient physical phenomena:

1. **Convergence-dynamics advantage:** early in training (Steps 0–1000), TGN undergoes strong **inertial collapse**, with gate rate dropping below 1%. This corresponds to the model prioritizing RNN learning of local phrase structures (e.g., n-grams).
2. **Hysteresis awakening and natural sparsity:** as perplexity (PPL) falls below 100, the model hits an information bottleneck that an RNN alone cannot overcome. The system then undergoes a **geometric phase transition**; gate rate rebounds (hysteresis awakening) and stabilizes near **~8.9%**.
   - **Final performance:** the model reaches **PPL 24.86**, comparable to a same-scale full-attention Transformer. Crucially, its gate rate stabilizes at **~9%**. This emergent sparse constant suggests that, in principle, TGN may maintain comparable intelligence with only **~10%** of Transformer attention FLOPs. While current PyTorch implementations lack hardware optimizations for dynamic sparsity, the finding provides a physical basis for next-generation efficient inference via sparse tensor cores.

![Fig. 21: Thermodynamic annealing on natural language manifolds](../figures/lm_efficiency_cloud.png)
*Fig. 21: Thermodynamic annealing on WikiText-103 (125M). (Red) PPL decreases approximately log-linearly with training steps and converges to 24.86. (Green) Gate rate exhibits classic U-shaped phase dynamics: first collapsing under local optimization, then awakening due to long-range demand, and finally stabilizing at a sparse equilibrium near ~9%.*

### 6.5 Physical Limit Testing: Breaking the Information-Theoretic Bottleneck

To establish TGN’s absolute advantage at the level of computational physics relative to SOTA models (Transformer, Mamba), we conduct two stress tests: an **early-training dynamics battle** on WikiText-103 and an **MQAR memory-capacity probe**.

#### 6.5.1 Comparison of Early Training Dynamics

On **WikiText-103** (350M parameters), we compare Mamba (SOTA SSM; specifically Mamba-1), Transformer (baseline), and TGN (ours) under strictly matched parameter counts and dataset conditions. While newer architectures like Mamba-2 include engineering optimizations (e.g., SSD), Mamba-1 remains a representative physical baseline for SSMs.

Results (Fig. 22) reveal clear dynamical differences under different inductive biases:

1. **Mamba’s rapid fitting:** as shown by the purple curve, Mamba’s sequential inductive bias yields extremely fast loss reduction early in training (Steps 0–500), quickly dropping below 1.0—confirming SSMs’ advantage in capturing local patterns.
2. **TGN’s convergence behavior:** although TGN starts slower due to GRU inertia, its loss reduction rate gradually surpasses the Transformer (blue). Notably, at Step 500, TGN’s loss (6.89) is already better than the Transformer’s (7.30).
3. **Energy-efficiency analysis:** this convergence is achieved at extremely low cost—TGN’s gate rate (green dashed) spontaneously converges to **<1%** (0.0085). That is, with attention almost entirely off, a lightweight inertial engine reaches Transformer-like performance, demonstrating strong potential for compute energy efficiency.

![Fig. 22: Early dynamics battle—TGN vs. SOTA](../figures/sota_comparison_battle.png)
*Fig. 22: Early training dynamics on WikiText-103. (Purple) Mamba rapidly fits via linear-RNN advantages. (Red) TGN catches up to and surpasses the Transformer (blue) early in training, while its attention gate rate is only 0.9% (green dashed), implying a 100× sparsification of attention compute.*

#### 6.5.2 Associative Memory Capacity Analysis

Results (Fig. 23a) reveal an essential physical contrast:

- **Mamba capacity limit:** the blue dashed line shows that a SOTA Mamba model is limited on MQAR accuracy. This follows from SSMs compressing history into a fixed-size hidden state \(h_t\); such compression faces an information-theoretic bottleneck for high-entropy, nonlocal dense storage.
- **TGN geometric advantage:** by contrast, TGN (orange solid) uses non-local geometric access and breaks the bottleneck. This indicates that for high-complexity tasks, **preserving non-local access** is an effective way to maintain high-dimensional information.

![Fig. 23a: MQAR capacity test](../figures/mqar_benchmark.png)
*Fig. 23a: MQAR physical-limit probe. (Blue) Mamba collapses to 0% accuracy due to state-capacity limits and an information-theoretic bottleneck. (Orange) TGN leverages the geometric channel and learns to break the bottleneck.*

#### 6.5.3 Dynamic Stability in Deep Recurrence

To test numerical stability under ultra-long sequences, we extend the hierarchical reasoning task ListOps to **L=2048**. Results (Fig. 23b) show:

- **Sensitivity of pure recurrence:** Mamba (blue dashed) shows numerical fluctuations in late training, potentially due to deep recurrent dynamics’ sensitivity to parameter perturbations.
- **Robustness of TGN:** TGN (orange solid) remains in the **90%–100%** range throughout, indicating good stability. This suggests that attention’s non-local connectivity may provide a more direct pathway for gradient flow, enhancing training robustness.

![Fig. 23b: ListOps resilience test](../figures/lra_benchmark.png)
*Fig. 23b: Ultra-long ListOps (L=2048) resilience test. (Blue) Mamba suffers numerical collapse, demonstrating the fragility of pure recurrence at extreme depth. (Orange) TGN, aided by a parallel geometric channel, shows excellent robustness.*

### 6.6 Engineering Feasibility: Chunked Gating and Throughput Benchmark

Since TGN currently lacks low-level hardware optimization for sparse tensors (e.g., Triton kernels), we design a **Chunked-TGN** variant to validate practical feasibility on existing hardware.

We relax the gating granularity from per-token to **chunks (128 tokens)** and use Mamba as the inertial backbone. Concretely, the system processes ~85% of chunks at \(O(L)\) cost and triggers global attention only for ~15% high-surprisal chunks.

We benchmark inference throughput on an NVIDIA A800 GPU from 1K to 32K sequence length. Results (Fig. 24) show:

1. **Breaking the quadratic bottleneck:** as sequence length increases to 32K, the standard Transformer throughput drops sharply due to \(O(L^2)\) complexity, down to ~28k tokens/sec.
2. **Preserving linear advantage:** with 15% sparsity, Chunked-TGN sustains ~167k tokens/sec at 32K—**6×** faster than the Transformer.
3. **A sweet spot:** while still below pure Mamba (~1.2M tokens/sec), TGN provides a long-range recall capability that Mamba lacks, demonstrating that coarse-grained sparsity can yield significant acceleration without specialized hardware.

![Fig. 24: Inference throughput comparison](../figures/throughput_comparison.png)
*Fig. 24: Throughput vs. sequence length (log–log). (Blue) Transformer suffers severe degradation as length increases. (Purple) Mamba maintains nearly constant linear throughput. (Red) Chunked-TGN (15% sparsity) achieves intermediate performance on current GPUs, reaching 6× Transformer throughput at 32K, validating the engineering value of sparse computation.*

### 6.7 Discussion and Future Work

We summarize the profound implications of TGN along three core dimensions: physical nature, cognitive insights, and computational paradigms.

#### 6.7.1 Physical Nature: Gating as Maxwell's Demon
Although TGN's gate is implemented as a simple nonlinear projection, its physical significance is nontrivial.
- **Thermodynamic view:** In information thermodynamics, prediction error is equivalent to **instantaneous entropy production**. RNN's inertial processing corresponds to an **adiabatic process**. The gate effectively acts as a **computational Maxwell's demon**: it monitors the "heat" (surprisal) of microscopic states in real time. Only when a "hot spot" (inertial prediction failure) is detected does it open the valve, injecting expensive computational energy (Attention) to introduce a negentropy flow. The formula \(g^* \approx \mathbb{I}(\Delta \mathcal{L} > \lambda)\) precisely describes the critical condition for this **energy-information conversion**.
- **Geometric view:** From Riemannian geometry, opening the gate performs an instantaneous **topological reconstruction** of the manifold. The closed state corresponds to geodesic evolution on a low-dimensional flow tube; the open state is akin to "sewing" a **high-dimensional handle** or wormhole between the present and the past. Thus, TGN inference is a process where the manifold itself breathes over time (topological fluctuations).
- **Future direction:** We envision hard-coding physical conservation laws (e.g., energy, momentum) as inductive biases into the gating function to build **"Physics-Consistent Gating"**, fusing data-driven intuition with axiom-driven truth.

#### 6.7.2 Cognitive Implications: From Slow Thinking to Automated Scientific Discovery
TGN's dual-channel architecture provides a precise computational model for cognitive science.
- **System 1 vs. System 2:** We map the RNN inertial manifold to fast thinking (intuition, low energy) and the Attention geometric manifold to slow thinking (logic, high energy). This reveals the **vertical dimension** of slow thinking—counteracting information dissipation via instantaneous geometric renormalization within a single inference step.
- **Scientific Revolution Detector:** Current TGN is passive. Looking forward, we propose an **active TGN-Explorer**. It uses reinforcement learning to actively perform **counterfactual topological perturbations** in the Attention matrix (e.g., forcibly connecting "apple" and "moon"). When a specific long-range link causes a **catastrophic collapse** in the system's effective rank (explaining complex phenomena with a minimal law), this marks a mathematical moment of "scientific revolution." This suggests AI has the potential to evolve from "data fitters" to "theory discoverers."

#### 6.7.3 Computational Paradigm: Thermodynamic Computing and Hardware Co-Design
TGN's dynamic sparsity points to a new paradigm of **"Thermodynamic Computing."**
- **Compute on Demand:** Unlike the brute-force full connectivity of Transformers, TGN deploys high compute power only when processing **"local high-entropy excitations"** (information solitons), perfectly aligning with the extreme energy-efficiency demands of edge devices and embodied intelligence.
- **Software-Hardware Co-Design:** Future AI accelerators should adopt a **heterogeneous thermodynamic architecture**: low-power NPUs handling inertial flow, while photonic chips or analog computing cores (adept at global relaxation) handling geometric flow. This biomimetic design offers a physical path to breaking the von Neumann bottleneck and extending Moore's Law.

#### 6.7.4 Conclusion: Toward Physics-Driven Intelligent Architectures

In summary, this work does not merely propose a neural-network variant; it seeks a first-principles framework connecting nonequilibrium thermodynamics, Riemannian geometry, and AI. Our results suggest that Transformers and attention are not purely engineering heuristics, but a necessary geometric solution that emerges when intelligent systems must counteract thermodynamic dissipation and maintain long-range order in nonconvex energy landscapes. **Thermodynamic Gated Networks (TGN)** provide an engineering instantiation of this theory: by mimicking the brain’s “inertia–geometry” dual dynamics, we can break the physical limits of existing linear models without sacrificing long-range reasoning, achieving simultaneous gains in computational efficiency and intelligence density. As Moore’s law slows and model scale continues to grow, this paradigm shift—**computing from physics**—may be a key path toward next-generation efficient general intelligence.

---

## Appendix A: Rigorous Proof of Isomorphism between Attention and Heat Diffusion

To address concerns that our claims are merely “analogies,” this appendix—based on diffusion maps (Coifman & Lafon, 2006) and the geometric constraint induced by LayerNorm—shows that softmax attention is, under specific conditions, **strictly equivalent** to a heat-diffusion operator on a Riemannian manifold.

### A.1 Geometric Constraint: From Inner Product to Euclidean Distance

In Transformers, LayerNorm effectively constrains feature vectors \(\mathbf{x}\) near a tangent layer of a \(d\)-dimensional hypersphere. For strictly normalized vectors \(\|\mathbf{q}\|=\|\mathbf{k}\|=R\) (with \(R=\sqrt{d}\)), the cosine law gives a one-to-one mapping between Euclidean distance and dot product:

$$
\|\mathbf{q} - \mathbf{k}\|^2
= \|\mathbf{q}\|^2 + \|\mathbf{k}\|^2 - 2\mathbf{q}^T \mathbf{k}
= 2R^2 - 2\mathbf{q}^T \mathbf{k}.
$$

Thus the dot product (similarity) can be converted directly into distance (dissimilarity):

$$
\mathbf{q}^T \mathbf{k} = R^2 - \frac{1}{2}\|\mathbf{q} - \mathbf{k}\|^2.
$$

Substituting into the standard softmax attention formula:

$$
A_{ij}
= \frac{\exp(\mathbf{q}_i^T \mathbf{k}_j / \tau)}{\sum_l \exp(\mathbf{q}_i^T \mathbf{k}_l / \tau)},
$$

we obtain

$$
A_{ij}
= \frac{\exp(R^2/\tau)\cdot \exp\!\left(-\frac{\|\mathbf{q}_i - \mathbf{k}_j\|^2}{2\tau}\right)}
{\sum_l \exp(R^2/\tau)\cdot \exp\!\left(-\frac{\|\mathbf{q}_i - \mathbf{k}_l\|^2}{2\tau}\right)}
= \frac{\exp\!\left(-\frac{\|\mathbf{q}_i - \mathbf{k}_j\|^2}{2\tau}\right)}
{\sum_l \exp\!\left(-\frac{\|\mathbf{q}_i - \mathbf{k}_l\|^2}{2\tau}\right)}.
$$

**Conclusion A.1 (kernel equivalence).** Under the LayerNorm constraint, attention weights \(A_{ij}\) are **mathematically identical** to a normalized Gaussian heat kernel
\(K_\tau(x,y)=\exp(-\|x-y\|^2/2\tau)\).
This removes the ambiguity of mere “formal similarity.”

### A.2 Operator Convergence: From Discrete Summation to a Differential Operator

Diffusion maps theory shows that when the number of samples \(N\to\infty\) and the bandwidth \(\tau\to 0\), the normalized graph Laplacian \(\mathbf{L}\) built from a Gaussian kernel converges to the Laplace–Beltrami operator \(\Delta_\mathcal{M}\) on manifold \(\mathcal{M}\).

Consider a single attention update under a residual connection:

$$
\mathbf{x}_i^{t+1}
= \mathbf{x}_i^t + \sum_j A_{ij}(\mathbf{x}_j^t - \mathbf{x}_i^t)
\quad
(\text{assume } \mathbf{W}_V \approx \mathbf{I} \text{ to reveal the core dynamics}).
$$

Here the subtraction of \(\mathbf{x}_i^t\) is implicit in the convex-combination property of softmax: \(\mathbf{x}^{t+1}=\sum_j A_{ij}\mathbf{x}_j = \mathbf{x}_i + \sum_j A_{ij}(\mathbf{x}_j-\mathbf{x}_i)\).

This is exactly the **explicit Euler discretization** of the heat equation \(\partial_t \mathbf{u}=\Delta \mathbf{u}\):

$$
\frac{\mathbf{x}^{t+1}-\mathbf{x}^t}{\Delta t} \approx \Delta_{\mathcal{M}} \mathbf{x}^t,
$$

with time step \(\Delta t = 1\).

**Conclusion A.2 (dynamical isomorphism).** The “residual + attention” layer is not an arbitrary engineering choice; physically, it corresponds precisely to **one step of heat-flow evolution** on the manifold.

### A.3 Energy Perspective: Dirichlet Energy Minimization

This geometric smoothing corresponds physically to minimizing the system’s **Dirichlet energy**, i.e., microscopic disorder:

$$
E(\mathbf{X}) = \frac{1}{2}\sum_{i,j} A_{ij}\,\|\mathbf{x}_i - \mathbf{x}_j\|^2.
$$

Gradient descent on this functional, \(\mathbf{x}_i \leftarrow \mathbf{x}_i - \eta \nabla_{\mathbf{x}_i} E\), yields the attention aggregation operation.

**Physical meaning:**

1. **Denoising:** heat flow smooths high-frequency noise, making the manifold smoother.
2. **Rank collapse:** as \(t\to\infty\), heat flow drives all points to the mean (heat death), corresponding to rank collapse in deep networks.
3. **Role of TGN:** by introducing gating, TGN activates heat flow only when geometric smoothing is needed, balancing manifold high-dimensionality maintenance (preventing heat death) and long-range association (using heat-flow tunneling).

---

## Appendix B: Random Matrix Theory Derivation of the \(1/\sqrt{d}\) Scaling

To derive from first principles why the scaling \(\lambda = 1/\sqrt{d}\) is necessary, we analyze the spectral behavior of the attention matrix \(\mathbf{A}=\text{softmax}(\lambda \mathbf{Q}\mathbf{K}^T)\) at initialization.

Assume \(\mathbf{Q},\mathbf{K}\in\mathbb{R}^{N\times d}\) have i.i.d. entries \(\mathcal{N}(0,1)\). By the Marchenko–Pastur law, the eigenvalue distribution of the unscaled product \(\mathbf{M}=\mathbf{Q}\mathbf{K}^T\) is highly spread.

Consider the dot product
\(z_{ij} = \mathbf{q}_i^T\mathbf{k}_j=\sum_{l=1}^d q_{il}k_{jl}\).
Since \(q,k\sim\mathcal{N}(0,1)\), by the CLT as \(d\to\infty\):

$$
z_{ij} \sim \mathcal{N}(0,d),
$$

so \(\text{Var}(z_{ij})=d\).

Without scaling (\(\lambda=1\)), softmax inputs follow \(\mathcal{N}(0,d)\). For large \(d\), extreme-value theory implies \(\max_j z_{ij}\) grows with \(d\), pushing softmax into saturation: gradients vanish, attention becomes one-hot, and the effective spectral rank collapses to 1.

With scaling \(\lambda\), the input distribution becomes \(\mathcal{N}(0,\lambda^2 d)\). To keep the system at a “critical state” where softmax is neither saturated nor uniform, we require the input variance to be \(O(1)\), say \(C\approx 1\):

$$
\text{Var}(\lambda z_{ij}) = \lambda^2 d = 1
\;\Rightarrow\;
\lambda = \frac{1}{\sqrt{d}}.
$$

**RMT numerical verification.** Random-matrix simulations (Fig. B1) confirm:

1. **Frozen phase:** \(\lambda \gg 1/\sqrt{d}\), \(\mathbf{A}\to\mathbf{I}\); effective rank collapses rapidly and capacity is lost.
2. **Chaotic phase:** \(\lambda \ll 1/\sqrt{d}\), \(\mathbf{A}\to \frac{1}{N}\mathbf{1}\mathbf{1}^T\); the spectrum becomes unstructured, akin to white noise.
3. **Critical phase:** at \(\lambda=1/\sqrt{d}\), the singular-value distribution exhibits a power-law long tail, and the effective-rank curve lies near the **inflection point** of the transition. Here the Laplacian-like operator \(\Delta_G\) induced by attention maintains global connectivity (nonzero spectral gap) while maximizing feature-space dimensionality (high effective rank), placing the system on a Pareto frontier between “fastest mixing” and “maximum capacity.”

![Fig. B1: RMT phase transition validation](../figures/rmt_phase_transition.png)
*Fig. B1: (Left) Singular-value spectral density of the attention matrix under different scalings. \(\alpha=1.0\) (green) shows a critical long-tail distribution that is neither white noise nor a single spike. (Right) Effective rank vs. scaling \(\alpha\), showing that \(1/\sqrt{d}\) lies exactly at the boundary between disorder and order.*

This indicates that Transformer design implicitly uses an “inverse law of large numbers” to counteract concentration of measure in high-dimensional space; this is the mathematical nature of the scaling.

**Connection to standard initialization theory.** Classical initialization theories (e.g., Xavier/He) also yield similar scaling to preserve variance and avoid vanishing/exploding gradients. Our thermodynamic derivation does not negate this; it provides a complementary geometric view: \(1/\sqrt{d}\) is not only a numerical-stability condition, but a critical physical temperature that **maximizes geometric entropy** while preserving **structural discriminability**.

---

## Appendix C: Complexity Analysis under the Information Sparsity Hypothesis

To rigorously answer whether TGN’s computational complexity is fundamentally still \(O(N^2)\), we must introduce a hypothesis about the distribution of information entropy. If information is uniformly distributed over time (e.g., white noise), no algorithm can break \(O(N^2)\). However, natural signals often follow Zipf’s law or power-law statistics.

**Assumption C.1 (information sparsity).** In a sequence of length \(N\), the probability \(p(t)\) that token \(t\) must retrieve distant history is not constant; it varies with semantic depth or surprisal. Suppose only \(K\) “critical moments” require attention.

**Assumption C.2 (gating oracle).** Suppose the TGN gate \(g_t\) perfectly learns this distribution, such that \(\sum_t g_t = K\).

**Derivation.** The total compute (FLOPs) can be approximated as:

$$
C_{total} \approx N\cdot O(d^2) + \sum_{t \in \text{Critical}} O(t\cdot d).
$$

In the worst case (all critical points occur near the end),
\(C_{total} \approx N\cdot d^2 + K\cdot (N\cdot d)\).

**Case discussion:**

1. **Dense information ( \(K \propto N\) )** (e.g., machine translation): \(C_{total} \approx O(N^2)\). TGN degenerates to a Transformer.
2. **Sparse information ( \(K \propto \log N\) )** (e.g., QA, retrieval): \(C_{total} \approx O(N\log N)\).
3. **Extremely sparse (“needle”, \(K \approx C\))** (needle-in-a-haystack): \(C_{total} \approx O(N)\).

---

## Appendix D: Mathematical Formalization of Geometric Definitions

### D.1 Effective Metric Tensor and Metric Deformation

We model the state space of an intelligent system as a Riemannian manifold \((M,g)\). Without attention, the baseline metric \(g_0\) is defined by Euclidean distance or local connectivity. Introducing attention is equivalent to adding a content-dependent perturbation tensor \(\delta g(\mathbf{x})\), yielding an effective metric:

$$
g_{eff}(\mathbf{x}) = g_0 + \delta g(\mathbf{x}).
$$

Here \(\delta g(\mathbf{x})\) is the bilinear form induced by attention weights \(A_{ij}\). On the tangent space \(T_{\mathbf{x}}M\), for any two tangent vectors \(u,v\), the attention-induced inner-product correction can be written as:

$$
\langle u, v \rangle_{Attn}
= \sum_{i,j} A_{ij}(\mathbf{x}) \langle u_i, v_j \rangle.
$$

Physically, this metric deformation manifests as **spatial contraction**: two states that are far apart under \(g_0\) (e.g., the beginning and end of a sequence) can become effectively close under \(g_{eff}\) if they are semantically related (high attention weight). This is the precise Riemannian-geometric definition of the “topological shortcut.”

### D.2 Geometric Rank and Shannon Entropy

The “effective geometric rank” \(R_{eff}\) used in this paper is a spectral-entropy measure based on the covariance eigen-spectrum. For a covariance matrix \(\Sigma\) with eigenvalues \(\lambda_1,\dots,\lambda_d\), define the normalized eigenvalue distribution \(p_i=\lambda_i/\sum_j \lambda_j\). The effective rank is:

$$
R_{eff}(\Sigma) = \exp\!\left( -\sum_{i=1}^d p_i \log p_i \right).
$$

Physically, this corresponds to the effective volume of the partition function in statistical mechanics. When \(p_i\) is uniform (isotropic, maximum entropy), \(R_{eff}=d\); when \(p_i\) collapses to a Dirac distribution (fully anisotropic, zero entropy), \(R_{eff}=1\).

Therefore, changes in geometric rank \(\Delta R_{eff}\) directly reflect expansion or contraction of manifold volume—i.e., information dissipation or generation.
