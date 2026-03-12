import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import GPT2Model, GPT2Tokenizer
import os

"""
Thermodynamics of Semantics: Does Attention Reconstruct Meaning?
----------------------------------------------------------------
Hypothesis: 
In a Language Model (GPT-2), the Attention Mechanism is not just 
"looking back", but actively reconstructing the "Semantic Geometry".

If true, Attention A_ij should be proportional to the 
Pointwise Mutual Information (PMI) or Semantic Similarity between tokens,
even if they are far apart in the sequence.

A_ij ~ Semantic_Sim(w_i, w_j) ~ exp(-Dissipation)
"""

def analyze_llm_thermodynamics():
    print("Loading GPT-2 (The 'Silicon Brain')...")
    model_name = 'gpt2' # Small enough for quick analysis
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2Model.from_pretrained(model_name, output_attentions=True)
    model.eval()
    
    # A complex sentence with long-range dependencies
    text = "The scientist, who had been working on the unified theory for twenty years, finally published his results in Nature."
    inputs = tokenizer(text, return_tensors="pt")
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    
    print(f"Analyzing sequence: {text}")
    print(f"Tokens: {tokens}")
    
    with torch.no_grad():
        outputs = model(**inputs)
        
    # Get Attention: Tuple of [Batch, Heads, Seq, Seq] for each layer
    attentions = outputs.attentions 
    # Get Embeddings: [Batch, Seq, Hidden]
    embeddings = outputs.last_hidden_state
    
    # -------------------------------------------------------------------------
    # 1. Compute Semantic Geometry (Ground Truth)
    # -------------------------------------------------------------------------
    # We use the Cosine Similarity of the contextual embeddings as a proxy for
    # "Semantic Closeness" (Hessian). 
    # High Similarity = Strong Interaction = High Curvature.
    
    seq_len = embeddings.shape[1]
    semantic_sim = np.zeros((seq_len, seq_len))
    
    emb_norm = torch.nn.functional.normalize(embeddings[0], p=2, dim=1)
    # Cosine Similarity Matrix: S = E * E^T
    semantic_sim = torch.matmul(emb_norm, emb_norm.t()).numpy()
    
    # -------------------------------------------------------------------------
    # 2. Compute Attention Geometry (The Mechanism)
    # -------------------------------------------------------------------------
    # We average attention across all heads in the LAST layer (most abstract)
    last_layer_attn = attentions[-1][0].mean(dim=0).numpy() # [Seq, Seq]
    
    # -------------------------------------------------------------------------
    # 3. Visualization & Correlation
    # -------------------------------------------------------------------------
    os.makedirs('figures', exist_ok=True)
    plt.figure(figsize=(14, 6))
    
    # Plot 1: Semantic Geometry (What NEEDS to be connected)
    plt.subplot(1, 2, 1)
    sns.heatmap(semantic_sim, cmap='viridis', xticklabels=tokens, yticklabels=tokens)
    plt.title("Semantic Geometry (Embedding Similarity)")
    plt.xticks(rotation=45)
    
    # Plot 2: Attention Mechanism (How it connects)
    plt.subplot(1, 2, 2)
    sns.heatmap(last_layer_attn, cmap='viridis', xticklabels=tokens, yticklabels=tokens)
    plt.title("Attention Mechanism (Thermodynamic Operator)")
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('figures/llm_semantic_geometry.png', dpi=300)
    print("Saved analysis to figures/llm_semantic_geometry.png")
    
    # -------------------------------------------------------------------------
    # 4. Thermodynamic Check: Correlation
    # -------------------------------------------------------------------------
    # Flatten and correlate (ignoring diagonal self-similarity)
    mask = ~np.eye(seq_len, dtype=bool)
    sim_flat = semantic_sim[mask]
    attn_flat = last_layer_attn[mask]
    
    corr = np.corrcoef(sim_flat, attn_flat)[0,1]
    print(f"\n>>> THERMODYNAMIC ALIGNMENT SCORE: {corr:.4f} <<<")
    
    if corr > 0.3:
        print("SUCCESS: Attention aligns with Semantic Geometry!")
        print("The model is actively routing information based on meaning, not just position.")
    else:
        print("FAILURE: No correlation found.")

if __name__ == "__main__":
    analyze_llm_thermodynamics()
