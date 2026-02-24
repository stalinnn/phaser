import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import os

"""
Experiment: Geometric Rank Analysis
-----------------------------------
Critique: "Negative correlation" != "Orthogonality".
Response: Measure the Effective Rank (Stable Rank) of the representation manifold.

Metric: Effective Rank (Entropy of Singular Values)
R_eff = exp( - sum p_i log p_i ) where p_i = sigma_i / sum(sigma)
This tells us the "Effective Dimensionality" of the semantic space.

Hypothesis: 
Deep networks usually suffer from "Rank Collapse" (convergence to mean).
Attention's job is to maintain High Rank (Manifold Expansion).
"""

def effective_rank(matrix):
    # matrix: [N, D]
    # Compute Singular Values
    # Use float32 or 64 for precision
    if isinstance(matrix, torch.Tensor):
        matrix = matrix.float()
        _, S, _ = torch.linalg.svd(matrix, full_matrices=False)
        S = S.cpu().numpy()
    else:
        _, S, _ = np.linalg.svd(matrix, full_matrices=False)
    
    # Normalize Singular Values to probability distribution
    # p_i = sigma_i / sum(sigma_i) -> This is for Nuclear Norm
    # Standard Stable Rank definition: sum(sigma^2) / max(sigma^2) ?
    # Or Entropy based?
    # Let's use Entropy based "Effective Dimension" as it's more sensitive to distribution.
    
    S_norm = S / np.sum(S)
    entropy = -np.sum(S_norm * np.log(S_norm + 1e-12))
    rank = np.exp(entropy)
    return rank

def measure_layer_ranks():
    print("Loading GPT-2...")
    model_name = 'gpt2'
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.eval()
    
    text = "The universe is defined by the fundamental laws of physics which govern the interactions of matter and energy across space and time." * 10
    inputs = tokenizer(text, return_tensors='pt')
    
    print("Forward pass to get hidden states...")
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        # hidden_states is a tuple of (Layer_0, Layer_1, ... Layer_12)
        # Shape: [Batch, Seq, Hidden]
        
    hidden_states = outputs.hidden_states
    
    ranks = []
    layers = list(range(len(hidden_states)))
    
    print("Computing Effective Rank per layer...")
    for i, h in enumerate(hidden_states):
        # h: [1, Seq, 768]
        # Reshape to [Seq, Hidden] matrix
        matrix = h[0] 
        r = effective_rank(matrix)
        ranks.append(r)
        print(f"Layer {i}: Effective Rank = {r:.2f}")
        
    # Baseline: Random Gaussian Matrix (Theoretical Max for this shape)
    seq_len, dim = matrix.shape
    rand_mat = torch.randn(seq_len, dim)
    rand_rank = effective_rank(rand_mat)
    print(f"Random Baseline Rank: {rand_rank:.2f}")

    # Plot
    os.makedirs('figures', exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(layers, ranks, 'o-', linewidth=2, color='#8e44ad', label='GPT-2 Representation')
    plt.axhline(y=rand_rank, color='gray', linestyle='--', label='Isotropic Gaussian (Max Entropy)')
    
    plt.title("Geometric Rank Maintenance in Deep Transformer")
    plt.xlabel("Layer Depth")
    plt.ylabel("Effective Rank (Dimension)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig('figures/llm_rank_evolution.png', dpi=300)
    print("Saved plot to figures/llm_rank_evolution.png")

if __name__ == "__main__":
    measure_layer_ranks()
