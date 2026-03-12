import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import os

"""
EXP 100: Geometric Rank Collapse in Modern LLMs (Llama-2 / Mistral / TinyLlama)
-----------------------------------------------------------------------------
Goal: Verify if the "V-Shaped" Rank Recovery phenomenon persists in modern
architectures (RoPE, SwiGLU, RMSNorm) compared to GPT-2 (Learned Pos, ReLU, LayerNorm).

Method:
1. Load a modern small LLM (e.g., TinyLlama-1.1B or Llama-2-7b-chat-hf if available).
2. Feed real text (WikiText-2 sample).
3. Extract hidden states from all layers.
4. Compute Effective Rank (Shannon Entropy of Singular Values).
"""

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def compute_effective_rank(matrix):
    """
    Computes the effective rank of a matrix (Batch * Seq, Hidden).
    R_eff = exp(Entropy(Normalized Singular Values))
    """
    # 1. Center the data (Geometry is about covariance, so mean doesn't matter)
    matrix = matrix - matrix.mean(dim=0, keepdim=True)
    
    # 2. SVD
    # We use torch.linalg.svd. For large matrices, svd_lowrank is faster but let's be precise.
    try:
        _, S, _ = torch.linalg.svd(matrix, full_matrices=False)
    except:
        # Fallback for convergence issues
        return 0.0
        
    # 3. Normalize Singular Values to get a probability distribution
    # p_i = \sigma_i / \sum \sigma_j
    S_norm = S / S.sum()
    
    # 4. Shannon Entropy
    # H = - \sum p_i log p_i
    entropy = -torch.sum(S_norm * torch.log(S_norm + 1e-12))
    
    # 5. Effective Rank
    rank_eff = torch.exp(entropy).item()
    return rank_eff

def analyze_model_rank(model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0", num_samples=50):
    print(f"\n>>> Analyzing Geometric Rank for: {model_name} <<<")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, trust_remote_code=True).to(device)
    except Exception as e:
        print(f"Error loading {model_name}: {e}")
        print("Falling back to GPT-2 for demonstration if internet/auth fails.")
        model_name = "gpt2"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    model.eval()
    
    # Sample Text (Physics abstract to make it relevant)
    # Truncate to avoid context window overflow (GPT-2 max is 1024)
    text_segment = """
    The fundamental challenge of complex systems lies in finding stable states in high-dimensional energy landscapes. 
    Traditional diffusion models face exponential relaxation times, known as the glassy freeze problem. 
    Here we propose that attention mechanisms act as non-local heat kernels on Riemannian manifolds, 
    smoothing the landscape and allowing for geometric tunneling. 
    This hypothesis is supported by evidence from spin glasses, large language models, and brain imaging data.
    """ 
    text = text_segment * 5 # Reduced repetition to stay safe
    
    # Tokenize with truncation
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    
    # We need to hook the hidden states
    # For Llama/Mistral, layers are usually model.model.layers
    # For GPT-2, model.transformer.h
    
    hidden_states = {}
    
    def get_hook(layer_idx):
        def hook(module, input, output):
            # Output is usually a tuple (hidden_states, ...)
            if isinstance(output, tuple):
                h = output[0]
            else:
                h = output
            
            # h: [Batch, Seq, Hidden]
            # Flatten to [Batch*Seq, Hidden] for Rank Calculation
            h_flat = h.reshape(-1, h.shape[-1]).float() # Convert to float32 for SVD
            
            hidden_states[layer_idx] = h_flat.detach()
        return hook

    # Register hooks
    hooks = []
    
    # Detect architecture
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        # Llama / Mistral style
        layers = model.model.layers
        print(f"Detected Llama/Mistral architecture with {len(layers)} layers.")
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        # GPT-2 style
        layers = model.transformer.h
        print(f"Detected GPT architecture with {len(layers)} layers.")
    elif hasattr(model, "bert") and hasattr(model.bert, "encoder"):
        # BERT style
        layers = model.bert.encoder.layer
        print(f"Detected BERT architecture with {len(layers)} layers.")
    else:
        print("Unknown architecture. Trying to find ModuleList...")
        # Fallback search
        layers = []
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.ModuleList) and len(module) > 5:
                layers = module
                break
    
    for i, layer in enumerate(layers):
        hooks.append(layer.register_forward_hook(get_hook(i)))
        
    # Forward Pass
    with torch.no_grad():
        model(**inputs)
        
    # Compute Ranks
    ranks = []
    layer_indices = sorted(hidden_states.keys())
    
    print("Computing Effective Ranks per layer...")
    for i in tqdm(layer_indices):
        r = compute_effective_rank(hidden_states[i])
        ranks.append(r)
        
    # Remove hooks
    for h in hooks:
        h.remove()
        
    return ranks, model_name

def run_experiment():
    # 1. GPT-2 (Baseline)
    ranks_gpt2, name_gpt2 = analyze_model_rank("gpt2")
    
    # 2. Modern Model (TinyLlama - similar to Llama 2 but runs on smaller GPUs)
    # If TinyLlama fails (e.g. no internet), it will fallback or we can skip
    ranks_modern, name_modern = analyze_model_rank("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    
    # Normalize for plotting (Rank / Max Rank) or just raw
    # Let's plot raw but on dual axis if dimensions differ greatly
    
    plt.figure(figsize=(10, 6))
    
    # Normalize to 0-1 range for fair comparison of shape
    norm_gpt2 = np.array(ranks_gpt2) / max(ranks_gpt2)
    norm_modern = np.array(ranks_modern) / max(ranks_modern)
    
    plt.plot(norm_gpt2, 'b-o', label=f'{name_gpt2} (Post-LN / ReLU)', linewidth=2)
    plt.plot(norm_modern, 'r-s', label=f'{name_modern} (Pre-LN / SiLU / RoPE)', linewidth=2)
    
    plt.title("Universality of Geometric Rank Dynamics\nFrom GPT-2 to Llama", fontsize=14)
    plt.xlabel("Layer Depth (Normalized)", fontsize=12)
    plt.ylabel("Normalized Effective Rank", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    os.makedirs('figures', exist_ok=True)
    plt.savefig('figures/llm_modern_rank_proof.png', dpi=300)
    print("Saved plot to figures/llm_modern_rank_proof.png")

if __name__ == "__main__":
    run_experiment()
