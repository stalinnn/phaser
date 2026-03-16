import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm
import os
import math
import gc

# Device: GPU is available (A800 80GB)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def compute_effective_rank(matrix):
    """
    Computes the effective rank of a matrix (Batch * Seq, Hidden).
    R_eff = exp(Entropy(Normalized Singular Values))
    """
    # Move to float32 for SVD precision
    matrix = matrix.to(torch.float32)
    matrix = matrix - matrix.mean(dim=0, keepdim=True)
    try:
        _, S, _ = torch.linalg.svd(matrix, full_matrices=False)
        S_norm = S / S.sum()
        entropy = -torch.sum(S_norm * torch.log(S_norm + 1e-12))
        return torch.exp(entropy).item()
    except Exception as e:
        print(f"SVD Error: {e}")
        return 0.0

def compute_anisotropy(matrix):
    """
    Computes the anisotropy of the representation.
    Average cosine similarity between all pairs of vectors.
    """
    matrix = matrix.to(torch.float32)
    matrix = matrix - matrix.mean(dim=0, keepdim=True)
    norm = torch.norm(matrix, p=2, dim=-1, keepdim=True)
    matrix_norm = matrix / (norm + 1e-8)
    # Cosine similarity matrix [N, N]
    cos_sim = torch.matmul(matrix_norm, matrix_norm.t())
    # Average off-diagonal elements
    n = cos_sim.shape[0]
    avg_sim = (cos_sim.sum() - n) / (n * (n - 1))
    return avg_sim.item()

def calculate_cka(X, Y):
    """
    Centered Kernel Alignment (CKA) between X and Y.
    """
    X = X.to(torch.float32).to(device)
    Y = Y.to(torch.float32).to(device)
    
    # Handle different number of tokens (N) due to different tokenizers
    n1, d1 = X.shape
    n2, d2 = Y.shape
    n = min(n1, n2)
    X = X[:n, :]
    Y = Y[:n, :]
    
    # Centering
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)
    
    # Linear CKA
    # trace(K_X * K_Y) = ||X^T * Y||_F^2
    dot_prod = torch.norm(torch.matmul(X.t(), Y), p='fro') ** 2
    norm_x = torch.norm(torch.matmul(X.t(), X), p='fro') ** 2
    norm_y = torch.norm(torch.matmul(Y.t(), Y), p='fro') ** 2
    
    return (dot_prod / (torch.sqrt(norm_x * norm_y) + 1e-8)).item()

def get_activations(model_path, text, layers_attr="model.layers"):
    print(f"\n>>> Extracting activations from: {model_path} <<<")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # Load in float16 for efficiency
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        torch_dtype=torch.float16, 
        low_cpu_mem_usage=True, 
        trust_remote_code=True
    ).to(device)
    model.eval()

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(device)
    
    activations = {}
    hooks = []
    
    def get_hook(idx):
        def hook(module, input, output):
            if isinstance(output, tuple):
                h = output[0]
            else:
                h = output
            # [B, S, D] -> [B*S, D]
            # Store on CPU to save GPU memory if we have many layers
            activations[idx] = h.detach().reshape(-1, h.shape[-1]).cpu()
        return hook

    # Find the layers
    curr = model
    for attr in layers_attr.split("."):
        curr = getattr(curr, attr)
    layers = curr
    
    for i, layer in enumerate(layers):
        hooks.append(layer.register_forward_hook(get_hook(i)))
        
    with torch.no_grad():
        model(**inputs)
        
    for h in hooks:
        h.remove()
        
    # Free model from memory
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    return activations

def run_comprehensive_analysis():
    # Use a more complex prompt for better representation
    text = "In the field of deep learning, comparing different architectures like Transformers and State Space Models is crucial. " * 10
    
    models_to_run = [
        {"name": "Llama-3.2-1B", "path": "/gz-data/Llama-3.2-1B", "attr": "model.layers"},
        {"name": "Mamba2-1.3B", "path": "/gz-data/Mamba2-1.3B", "attr": "backbone.layers"},
        {"name": "Llama-3.1-8B", "path": "/gz-data/Meta-Llama-3.1-8B", "attr": "model.layers"},
        {"name": "Falcon-Mamba-7B", "path": "/gz-data/Falcon3-Mamba-7B-Instruct", "attr": "backbone.layers"},
        {"name": "Qwen2-7B", "path": "/gz-data/Qwen2-7B", "attr": "model.layers"}
    ]
    
    results = {}
    
    for m in models_to_run:
        if not os.path.exists(m["path"]):
            print(f"Skipping {m['name']}, path not found: {m['path']}")
            continue
        try:
            acts = get_activations(m["path"], text, m["attr"])
            # Use tqdm for rank calculation
            ranks = [compute_effective_rank(acts[i]) for i in tqdm(sorted(acts.keys()), desc=f"Ranks {m['name']}")]
            anisotropy = [compute_anisotropy(acts[i]) for i in tqdm(sorted(acts.keys()), desc=f"Anisotropy {m['name']}")]
            results[m["name"]] = {
                "ranks": ranks,
                "anisotropy": anisotropy,
                "activations": acts 
            }
        except Exception as e:
            print(f"Error processing {m['name']}: {e}")

    # CKA Cross-Comparison
    cka_results = {}
    pairs = [
        ("Llama-3.2-1B", "Mamba2-1.3B"),
        ("Llama-3.1-8B", "Falcon-Mamba-7B"),
        ("Llama-3.1-8B", "Qwen2-7B")
    ]
    
    for n1, n2 in pairs:
        if n1 in results and n2 in results:
            acts1 = results[n1]["activations"]
            acts2 = results[n2]["activations"]
            l1, l2 = len(acts1), len(acts2)
            cka_map = np.zeros((l1, l2))
            for i1 in tqdm(range(l1), desc=f"CKA {n1} vs {n2}"):
                for j1 in range(l2):
                    cka_map[i1, j1] = calculate_cka(acts1[i1], acts2[j1])
            cka_results[f"{n1}_vs_{n2}"] = cka_map

    # Plotting
    num_plots = 2 + len(cka_results)
    plt.figure(figsize=(6 * num_plots, 6))
    
    # 1. Effective Rank Plot
    plt.subplot(1, num_plots, 1)
    for name, data in results.items():
        x = np.linspace(0, 1, len(data["ranks"]))
        plt.plot(x, data["ranks"], label=name, marker='o' if '1B' in name else 's', markersize=4)
    plt.title("Effective Rank ($R_{eff}$) across Layers", fontsize=14)
    plt.xlabel("Normalized Depth", fontsize=12)
    plt.ylabel("$R_{eff}$", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Anisotropy Plot
    plt.subplot(1, num_plots, 2)
    for name, data in results.items():
        x = np.linspace(0, 1, len(data["anisotropy"]))
        plt.plot(x, data["anisotropy"], label=name, marker='o' if '1B' in name else 's', markersize=4)
    plt.title("Anisotropy (Avg CosSim) across Layers", fontsize=14)
    plt.xlabel("Normalized Depth", fontsize=12)
    plt.ylabel("Anisotropy", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. CKA Heatmaps
    for i, (pair, cka_map) in enumerate(cka_results.items()):
        plt.subplot(1, num_plots, 3 + i)
        plt.imshow(cka_map, origin='lower', aspect='auto', cmap='magma')
        plt.colorbar(label="CKA Similarity")
        plt.title(f"CKA: {pair}", fontsize=14)
        n1, n2 = pair.split('_vs_')
        plt.xlabel(f"{n2} Layers", fontsize=12)
        plt.ylabel(f"{n1} Layers", fontsize=12)

    plt.tight_layout()
    os.makedirs("layer/results", exist_ok=True)
    plt.savefig("layer/results/paper_metrics_gpu.png", dpi=300)
    print("\nAnalysis complete. Saved plot to layer/results/paper_metrics_gpu.png")
    
    # Save raw results for later if needed
    # np.savez("layer/results/metrics_data.npz", results=results, cka_results=cka_results)
    
    return results, cka_results

if __name__ == "__main__":
    try:
        run_comprehensive_analysis()
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Main Error: {e}")
