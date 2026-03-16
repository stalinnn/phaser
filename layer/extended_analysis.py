import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import os
import gc

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def compute_effective_rank(matrix):
    matrix = matrix.to(torch.float32)
    matrix = matrix - matrix.mean(dim=0, keepdim=True)
    try:
        _, S, _ = torch.linalg.svd(matrix, full_matrices=False)
        S_norm = S / (S.sum() + 1e-8)
        entropy = -torch.sum(S_norm * torch.log(S_norm + 1e-12))
        return torch.exp(entropy).item()
    except:
        return 0.0

def calculate_cka(X, Y):
    X = X.to(torch.float32).to(device)
    Y = Y.to(torch.float32).to(device)
    n = min(X.shape[0], Y.shape[0])
    X, Y = X[:n] - X[:n].mean(0), Y[:n] - Y[:n].mean(0)
    dot_prod = torch.norm(X.t() @ Y, p='fro')**2
    norm_x = torch.norm(X.t() @ X, p='fro')**2
    norm_y = torch.norm(Y.t() @ Y, p='fro')**2
    return (dot_prod / (torch.sqrt(norm_x * norm_y) + 1e-8)).item()

def get_activations_with_perturbation(model, tokenizer, text, perturb_std=0.0):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(device)
    
    # Perturb the embeddings
    original_embeds = model.get_input_embeddings()(inputs['input_ids'])
    if perturb_std > 0:
        noise = torch.randn_like(original_embeds) * perturb_std
        inputs['inputs_embeds'] = original_embeds + noise
        del inputs['input_ids']
    else:
        inputs['inputs_embeds'] = original_embeds
        del inputs['input_ids']

    activations = {}
    hooks = []
    def get_hook(idx):
        def hook(m, i, o):
            res = o[0] if isinstance(o, tuple) else o
            activations[idx] = res.detach().reshape(-1, res.shape[-1]).cpu()
        return hook

    # Target layers based on architecture
    layers = model.model.layers if hasattr(model, 'model') else model.backbone.layers
    for i, layer in enumerate(layers):
        hooks.append(layer.register_forward_hook(get_hook(i)))
        
    with torch.no_grad():
        model(**inputs)
    for h in hooks: h.remove()
    return activations

def run_extended_experiments():
    models_cfg = [
        {"name": "Llama-3.1-8B", "path": "/gz-data/Meta-Llama-3.1-8B"},
        {"name": "Falcon-Mamba-7B", "path": "/gz-data/Falcon3-Mamba-7B-Instruct"}
    ]
    
    results = {}
    
    for cfg in models_cfg:
        print(f"\n>>> Analyzing {cfg['name']} <<<")
        model = AutoModelForCausalLM.from_pretrained(cfg['path'], torch_dtype=torch.float16, trust_remote_code=True).to(device)
        tokenizer = AutoTokenizer.from_pretrained(cfg['path'], trust_remote_code=True)
        
        # Exp 1: Context Length Scaling
        lengths = [128, 512, 1024, 2048]
        rank_scaling = {}
        for L in lengths:
            text = "The evolution of intelligence requires geometric stability. " * (L // 8)
            acts = get_activations_with_perturbation(model, tokenizer, text)
            ranks = [compute_effective_rank(acts[i]) for i in sorted(acts.keys())]
            rank_scaling[L] = ranks
            print(f"  - Length {L} processed.")
            
        # Exp 2: Robustness Probing (Noise epsilon=0.01)
        text_std = "Information geometry is the study of probability manifolds." * 20
        clean_acts = get_activations_with_perturbation(model, tokenizer, text_std, perturb_std=0.0)
        noisy_acts = get_activations_with_perturbation(model, tokenizer, text_std, perturb_std=0.01)
        
        robustness_scores = []
        for i in sorted(clean_acts.keys()):
            # CKA between clean and noisy activations: higher is more robust
            score = calculate_cka(clean_acts[i], noisy_acts[i])
            robustness_scores.append(score)
        
        results[cfg['name']] = {
            "rank_scaling": rank_scaling,
            "robustness": robustness_scores
        }
        
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    # --- Plotting ---
    plt.figure(figsize=(16, 6))
    
    # Plot 1: Rank Scaling (Mamba vs Llama at 128 vs 2048)
    plt.subplot(1, 2, 1)
    for name, data in results.items():
        # Plot shortest and longest
        x = np.linspace(0, 1, len(data["rank_scaling"][128]))
        plt.plot(x, data["rank_scaling"][128], label=f"{name} (L=128)", linestyle='--')
        plt.plot(x, data["rank_scaling"][2048], label=f"{name} (L=2048)", linewidth=2)
    plt.title("Rank Decay under Context Scaling", fontsize=14)
    plt.xlabel("Normalized Depth")
    plt.ylabel("Effective Rank")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 2: Robustness (Sensitivity to Noise)
    plt.subplot(1, 2, 2)
    for name, data in results.items():
        x = np.linspace(0, 1, len(data["robustness"]))
        # Sensitivity = 1 - CKA
        sensitivity = 1.0 - np.array(data["robustness"])
        plt.plot(x, sensitivity, label=name, linewidth=2)
    plt.title("Geometric Sensitivity to Input Perturbation", fontsize=14)
    plt.xlabel("Normalized Depth")
    plt.ylabel("Sensitivity (1 - CKA)")
    plt.yscale('log') # Use log scale to see divergence in deep layers
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs("layer/results", exist_ok=True)
    plt.savefig("layer/results/scaling_robustness_study.png", dpi=300)
    print("\nExtended analysis complete. Saved to layer/results/scaling_robustness_study.png")

if __name__ == "__main__":
    run_extended_experiments()
