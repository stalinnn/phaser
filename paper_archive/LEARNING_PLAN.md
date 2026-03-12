# Learning Plan (8 Weeks, 6–10h/week)

This plan is designed for an AI engineer to **rapidly** close the gaps needed to defend and improve the paper:
`paper_archive/manuscripts/Paper_Final_EN.md` (Thermodynamic Gated Networks, TGN).

Principle: **read 2 pieces + write 1 page + patch the paper** every week.

---

## Weekly Rhythm (recommended)

- **3–5h**: read (2 items: 1 lecture note + 1 short paper/review)
- **1–2h**: write 1-page notes (“Definitions / Assumptions / Scope / Failure modes”)
- **1–3h**: patch the manuscript (add Assumptions boxes, caveats, symbol table, limitations)

---

## Week 1 — Linear Algebra & Spectra (Rank, Spectral Gap)

### Goal
Be able to defend every use of “rank / effective rank / spectral radius / spectral gap”.

### Study
- SVD, eigenvalues, spectral radius
- Spectral gap intuition (mixing speed, connectivity, stability)
- Effective rank (entropy of singular values) vs alternatives (participation ratio)

### Output (1 page)
- Definitions: effective rank, participation ratio
- Why effective rank is used in this paper, when it can fail

### Paper mapping
- Effective-rank claims: **Section 4.3** (“Rank Collapse/Recovery”), **Figure 9**
- “Spectral gap” claims: **Section 2.3** (“Spectral Gap”), heat-kernel interpretation

---

## Week 2 — Heat Equation, Heat Kernel, Laplace–Beltrami

### Goal
Make the “Attention ≈ heat kernel” mapping **precise and safely stated**.

### Study
- Heat equation, Green’s function, heat-kernel semigroup
- Laplacian vs Laplace–Beltrami: definition + geometric meaning
- Short-time heat-kernel asymptotics (what is an approximation vs identity)

### Output (Assumptions box)
Add an explicit **Assumptions & Scope** box:
- normalization / hypersphere assumption
- short-time kernel approximation
- metric approximation (what is assumed about geometry)
- when the mapping is “effective analogy” rather than strict equivalence

### Paper mapping
- Heat-kernel mapping: **Section 2.3** (“The Mapping”)
- Appendix equivalence: **Appendix (Beltrami Flow)** (mapping to Gaussian kernel form)

---

## Week 3 — Stochastic Processes: Langevin & Fokker–Planck

### Goal
Avoid symbol/physics mismatch and make dynamics equations standard.

### Study
- Overdamped Langevin dynamics: drift vs diffusion terms
- Fokker–Planck equation and relation to Langevin
- Stationary distributions and assumptions (temperature/noise)

### Output
- A short **symbol table** (units/meaning): \(T, \beta, \gamma, \lambda\)
- Rewrite the key dynamics into a standard form (even if “effective”)

### Paper mapping
- Dynamics equation: **Section 2.3** (“Geometric Interaction Equation”)
- Discussion: where “effective dynamics” is claimed

---

## Week 4 — Statistical Physics: Free Energy → Softmax/Gibbs

### Goal
Defend “softmax as a canonical solution” without overclaiming “necessity”.

### Study
- Maximum entropy principle
- Gibbs/Boltzmann distribution and free energy \(F = U - TS\)
- When softmax emerges (constraints, regularization, normalization)

### Output (patch)
- Replace strong words (“must/inevitable”) with accurate ones:
  “canonical solution under stated constraints”
- List constraints explicitly (normalization, entropy term, choice of energy)

### Paper mapping
- Free energy functional: **Section 2.2**
- Softmax interpretation: **Section 2.2–2.3**

---

## Week 5 — Arrhenius/Kramers/Glassy Slowing Down

### Goal
Make the “exponential → polynomial” story scientifically strong and caveated.

### Study
- Arrhenius law, Kramers escape
- Conditions for glassy slowing down (barrier distributions, rugged landscapes)
- What can be proven vs scaling heuristics

### Output (caveats box)
Add a “Scaling Argument / Upper Bound / Caveats” box:
- which steps are heuristic
- when log-distance compression fails
- what empirical predictions can falsify the claim

### Paper mapping
- Barrier argument: **Section 2.5** (“Logarithmization of Arrhenius Barriers”)
- Spin-glass scaling: **Section 3.5** (scaling figures)

---

## Week 6 — Renormalization: What is strict RG vs effective analogy

### Goal
Use “renormalization / wormhole / Einstein–Rosen bridge” safely.

### Study
- RG essentials: coarse-graining, flow, fixed points
- ML usage of “renormalization” (often effective, not strict)

### Output (terminology patch)
Add a short paragraph:
“What we mean by renormalization in this paper”
and add qualifiers:
effective / interpret / analogy where appropriate.

### Paper mapping
- “renormalization” language: Abstract + **Section 1–2**
- “wormhole/shortcut” language: Introduction and mapping sections

---

## Week 7 — Causality: Granger, Lead–Lag, Intervention vs Correlation

### Goal
Prevent the most common reviewer attack: “correlation ≠ causation”.

### Study
- Granger causality assumptions (linearity, stationarity, confounding)
- Temporal precedence vs causal intervention
- How to write claims as “evidence consistent with…”

### Output (rewrite)
Rewrite any “causal proof” phrasing into:
“evidence consistent with temporal precedence + mechanistic plausibility”
and add explicit limitations.

### Paper mapping
- Causal lead–lag: **Section 6.3 (Figure 16)**
- “Virtual TMS” intervention: **Section 6.4 (Figure 14C)**

---

## Week 8 — Consciousness & Anesthesia: minimal, defensible neuroscience

### Goal
Place the fMRI/anesthesia results in a credible framework and tone down overclaims.

### Study
- Global Workspace and anesthesia/network integration (review-level)
- Typical consciousness biomarkers and limitations

### Output (positioning)
Add a “Positioning” paragraph:
- what your metric corresponds to (integration/complexity)
- what you do *not* claim
- what experiments could falsify or strengthen the claim

### Paper mapping
- Cross-species section: **Section 6**
- Discussion limitations: **Section 7**

---

## Deliverables after 8 weeks

- A “Definitions / Assumptions / Failure Modes” appendix (1–2 pages)
- Clean symbol table for all key equations
- A stronger “Limitations” section that preempts reviewer attacks
- A more defensible and precise mapping between Attention and geometric operators

