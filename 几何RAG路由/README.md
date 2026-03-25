# Geometric RAG Router

**Geometric RAG Router** is a lightweight middleware for Retrieval-Augmented Generation (RAG) systems. It intelligently analyzes the intrinsic geometry (manifold curvature) of your dataset and automatically selects the optimal retrieval strategy:

- **Tree-like / Hierarchical Data** $\rightarrow$ Routes to **Hyperbolic RAG**
- **Flat / Semantic Data** $\rightarrow$ Routes to **Standard RAG (Euclidean)**
- **Mixed / Clustered Data** $\rightarrow$ Routes to **Holo-Hybrid Fusion RAG**

## Why?

Most developers blindly use GraphRAG for complex logic or standard RAG for QA, regardless of whether the underlying data topology actually requires it. This router uses the **Gromov $\delta$-Hyperbolicity** measure to sample your document embeddings and scientifically determine how "tree-like" your data is.

## Installation

```bash
pip install -e .
```

## Quick Start

```python
import numpy as np
from geom_rag_router import ManifoldCurvatureDetector

# 1. Load your document embeddings (e.g., from BGE-M3 or OpenAI)
embeddings = np.random.rand(1000, 1024) # Example flat data

# 2. Initialize and analyze
detector = ManifoldCurvatureDetector()
stats = detector.analyze(embeddings)

print(f"Topology: {stats['topology']}")
print(f"Mean Delta: {stats['mean_delta']:.4f}")
print(f"Recommended RAG Mode: {stats['recommended_mode']}")
print(f"Fusion Weights (Hyp, Euc): {stats['fusion_weights']}")
```

## Integration with RRF (Reciprocal Rank Fusion)
The router outputs fusion weights `(w_hyp, w_euc)` that can be directly plugged into an RRF algorithm to seamlessly blend Hyperbolic and Euclidean search results based on the data's true geometric nature.
