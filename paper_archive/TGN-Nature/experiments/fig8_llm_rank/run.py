import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import os
from tqdm import tqdm

"""
Experiment 5.2 (Revised): Ablation Study & 3D Manifold Evolution
----------------------------------------------------------------
1. Error Bands: Run multiple sequences to get Mean +/- Std of Rank.
2. 3D Trajectory: Plot (Layer, Rank, SingularValueSpread) to show the geometric path.
"""

def effective_rank(matrix):
    # matrix: [N, D]
    if isinstance(matrix, torch.Tensor):
        matrix = matrix.float()
        _, S, _ = torch.linalg.svd(matrix, full_matrices=False)
        S = S.detach().cpu().numpy()
    else:
        _, S, _ = np.linalg.svd(matrix, full_matrices=False)
    
    S_norm = S / np.sum(S)
    entropy = -np.sum(S_norm * np.log(S_norm + 1e-12))
    return np.exp(entropy)

def _build_fixed_length_input(tokenizer, text: str, target_len: int) -> torch.Tensor:
    """
    Build a fixed-length GPT-2 input_ids tensor of shape [1, target_len].
    Note: GPT-2 has no pad token by default; we deterministically repeat+truncate.
    """
    ids = tokenizer.encode(text)
    if len(ids) == 0:
        raise ValueError("Tokenizer produced empty input_ids.")
    if len(ids) < target_len:
        reps = (target_len // len(ids)) + 1
        ids = (ids * reps)[:target_len]
    else:
        ids = ids[:target_len]
    return torch.tensor([ids], dtype=torch.long)

class AblationWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.ablation_mode = 'none' 
        
    def forward(self, input_ids):
        # Hooks are cleaner but manual iteration allows exact control
        x = self.model.transformer.wte(input_ids) + self.model.transformer.wpe(torch.arange(input_ids.shape[1], device=input_ids.device))
        
        ranks = []
        hidden_states_summary = [] # Store mean/std of hidden states for 3D plot
        
        for i, block in enumerate(self.model.transformer.h):
            # Capture state before block (Residual stream)
            r = effective_rank(x[0])
            ranks.append(r)
            
            # For 3D plot: Capture simplified state stats (Mean Act, Std Act)
            h_mean = x[0].mean().item()
            h_std = x[0].std().item()
            hidden_states_summary.append((h_mean, h_std))
            
            residual = x
            hidden = block.ln_1(x)
            
            # ATTENTION PATH
            if self.ablation_mode == 'no_attn':
                attn_out = torch.zeros_like(hidden)
            elif self.ablation_mode == 'random_mixer':
                # Strategy C: Random Fixed Mixing (Blind Global Connection)
                # Simulates a fixed "Small World" or MLP-Mixer without content awareness
                batch, seq, dim = hidden.shape
                # Create a fixed random mixing matrix for this sequence length
                # In a real Mixer, this is learned. Here we test if "structure" alone helps.
                if not hasattr(self, 'random_matrix') or self.random_matrix.shape[1] != seq:
                     self.random_matrix = torch.randn(seq, seq, device=hidden.device) / np.sqrt(seq)
                
                # Apply fixed mixing: X = A_rand * X
                attn_out = torch.matmul(self.random_matrix, hidden)
            else:
                attn_out = block.attn(hidden)[0]
                
            x = residual + attn_out
            
            residual = x
            hidden = block.ln_2(x)
            
            # FFN PATH
            if self.ablation_mode == 'no_ffn':
                ffn_out = torch.zeros_like(hidden)
            else:
                ffn_out = block.mlp(hidden)
                
            x = residual + ffn_out
            
        # Last layer
        ranks.append(effective_rank(x[0]))
        h_mean = x[0].mean().item()
        h_std = x[0].std().item()
        hidden_states_summary.append((h_mean, h_std))
        
        return np.array(ranks), np.array(hidden_states_summary)

def run_ablation_experiment():
    print("Loading GPT-2 for Robust Analysis...")
    model_name = 'gpt2'
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    hf_model = GPT2LMHeadModel.from_pretrained(model_name)
    hf_model.eval()
    
    model = AblationWrapper(hf_model)
    
    # Multiple sequences for Error Bands
    # IMPORTANT: Effective rank <= min(seq_len, hidden_dim). The paper's discussion
    # uses ranks in the ~10-40 range, so we must feed sufficiently long sequences.
    target_seq_len = 128
    texts = [
        (
            "In high-dimensional systems, local diffusion is slowed by rugged energy landscapes. "
            "Attention introduces content-dependent non-local couplings, which can be viewed as a "
            "geometric smoothing operator on a manifold, effectively creating topological shortcuts."
        ),
        (
            "From a thermodynamic perspective, the system minimizes a free-energy functional that "
            "balances alignment energy with entropy. The scaled dot-product (1/sqrt(d)) corresponds "
            "to a critical temperature that avoids both frozen and fully mixed regimes."
        ),
        (
            "In deep neural networks, representations often suffer rank collapse as depth increases. "
            "Ablating attention should remove the main source of high-dimensional context injection, "
            "leading to sustained decay of effective rank in later layers."
        ),
        (
            "Spin-glass dynamics, long-range dependency tasks, and brain-wide integration provide "
            "cross-domain evidence that non-local interactions can counteract dissipation and sustain "
            "macroscopic order in complex adaptive systems."
        ),
        (
            "We interpret the evolution across layers as a trajectory in phase space: depth as time, "
            "activity statistics as an energy proxy, and entropy-based effective rank as a measure of "
            "geometric dimensionality of the residual stream."
        ),
    ] * 2  # 10 samples
    
    all_ranks_normal = []
    all_ranks_no_attn = []
    all_ranks_random = [] # Strategy C
    all_states_normal = [] # For 3D plot
    
    print("Running batch processing...")
    for text in tqdm(texts):
        input_ids = _build_fixed_length_input(tokenizer, text=text, target_len=target_seq_len)
        
        # Normal
        model.ablation_mode = 'none'
        r, s = model(input_ids)
        all_ranks_normal.append(r)
        all_states_normal.append(s)
        
        # No Attn
        model.ablation_mode = 'no_attn'
        r, _ = model(input_ids)
        all_ranks_no_attn.append(r)

        # Random Mixer (Strategy C)
        model.ablation_mode = 'random_mixer'
        r, _ = model(input_ids)
        all_ranks_random.append(r)
        
    # Process Stats
    ranks_normal_mean = np.mean(all_ranks_normal, axis=0)
    ranks_normal_std = np.std(all_ranks_normal, axis=0)
    
    ranks_no_attn_mean = np.mean(all_ranks_no_attn, axis=0)
    ranks_no_attn_std = np.std(all_ranks_no_attn, axis=0)

    ranks_random_mean = np.mean(all_ranks_random, axis=0)
    ranks_random_std = np.std(all_ranks_random, axis=0)
    
    # Plotting
    # Save into paper_archive/figures (so manuscripts/../figures resolves correctly)
    out_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "figures"))
    os.makedirs(out_dir, exist_ok=True)
    
    # COMPOSITE FIGURE: 2D Error Bands + 3D Trajectory
    fig = plt.figure(figsize=(16, 6))
    
    # 1. 2D Error Bands (Left Panel)
    ax1 = fig.add_subplot(1, 2, 1)
    layers = range(len(ranks_normal_mean))
    
    # Normal
    ax1.plot(layers, ranks_normal_mean, 'o-', color='#8e44ad', linewidth=2, label='Attention (Dynamic Geometry)')
    ax1.fill_between(layers, ranks_normal_mean - ranks_normal_std, ranks_normal_mean + ranks_normal_std, color='#8e44ad', alpha=0.2)
    
    # No Attn
    ax1.plot(layers, ranks_no_attn_mean, 's--', color='#e74c3c', linewidth=2, label='No Attention (Local Decay)')
    ax1.fill_between(layers, ranks_no_attn_mean - ranks_no_attn_std, ranks_no_attn_mean + ranks_no_attn_std, color='#e74c3c', alpha=0.2)

    # Random Mixer
    ax1.plot(layers, ranks_random_mean, 'x:', color='gray', linewidth=2, label='Random Mixer (Blind Global)')
    ax1.fill_between(layers, ranks_random_mean - ranks_random_std, ranks_random_mean + ranks_random_std, color='gray', alpha=0.1)

    
    ax1.set_title(f"Geometric Rank Renormalization (Robustness Check, seq_len={target_seq_len})")
    ax1.set_xlabel("Layer Depth")
    ax1.set_ylabel("Effective Rank (exp entropy of singular values)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 3D Evolution Trajectory (Right Panel)
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    
    # Use the mean trajectory of the Normal case
    # X: Layer, Y: Activity Std (Energy), Z: Rank (Entropy)
    mean_states = np.mean(all_states_normal, axis=0) # [Layers, 2]
    activity_std = mean_states[:, 1]
    
    # Plot the line
    # Map color to layer depth
    p = ax2.scatter(layers, activity_std, ranks_normal_mean, c=layers, cmap='viridis', s=50, depthshade=True)
    ax2.plot(layers, activity_std, ranks_normal_mean, color='gray', alpha=0.5)
    
    ax2.set_xlabel('Layer Depth')
    ax2.set_ylabel('Activity Std (Energy proxy)')
    ax2.set_zlabel('Effective Rank (Entropy, exp form)')
    ax2.set_title('Manifold Trajectory in Phase Space')
    
    # Add arrows to show flow?
    # Simple arrows
    for i in range(len(layers)-1):
        ax2.plot([layers[i], layers[i+1]], 
                 [activity_std[i], activity_std[i+1]], 
                 [ranks_normal_mean[i], ranks_normal_mean[i+1]], color='black', alpha=0.3)
    
    cbar = plt.colorbar(p, ax=ax2, pad=0.1)
    cbar.set_label('Layer Depth')
    
    plt.tight_layout()
    out_path = os.path.join(out_dir, "llm_rank_ablation.png")
    plt.savefig(out_path, dpi=300)
    print(f"Saved composite figure to {out_path}")

if __name__ == "__main__":
    run_ablation_experiment()
