import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import json
import math
import matplotlib.pyplot as plt

from core.hyperbolic_space import clamp_to_ball, make_poincare_ball
from core.metric_alignment import metric_alignment_loss, upper_triangle_mask
from emerge_holographic_bulk import set_seed, load_texts, encode_euclidean, build_target_proxy
from emerge_arch_holographic import HolographicArchitecture, count_parameters
from probes.telm_monitor import collect_reading

def get_calibrated_lambda_crit():
    # We calibrate using the previously run MLP (L=2, D=64)
    # We find the Lambda value when the absolute rank drops below a stable threshold.
    # In V4, MLP starts at rank ~6-8, but its phase transition drops it. 
    # Actually, let's look at the data dynamically.
    run_dir = Path("runs/arch_mlp_l2_d64")
    ckpt_path = run_dir / "checkpoint.pt"
    jsonl_path = run_dir / "telm_readings.jsonl"
    
    data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = data["config"]
    d_param = cfg["d_param"]
    
    euc_emb = data["euc_emb"]
    euc_emb_centered = euc_emb - euc_emb.mean(dim=0, keepdim=True)
    cov_matrix = (euc_emb_centered.T @ euc_emb_centered) / euc_emb.size(0)
    cov = float(torch.linalg.matrix_norm(cov_matrix, ord='fro').item())
    
    # Let's find the maximum negative gradient of the rank (the sharpest drop)
    ranks = []
    lambdas = []
    with open(jsonl_path, "r") as f:
        for line in f:
            if not line.strip(): continue
            rec = json.loads(line)
            ranks.append(rec["effective_rank"])
            grad_var = rec["grad_variance"]
            T_sys = (cfg["lr"] / cfg["batch_size"]) * max(grad_var, 1e-12)
            lambdas.append(cov / (T_sys * d_param * 16.0))
            
    # The initial gradient variance is tiny, so Lambda is huge initially, which is a numerical artifact of the first step.
    # We should ignore the first 5 steps of warmup.
    ranks = np.array(ranks[5:])
    lambdas = np.array(lambdas[5:])
    
    # Let's use moving average of ranks to find the true structural drop
    smooth_ranks = np.convolve(ranks, np.ones(5)/5, mode='valid')
    drops = np.diff(smooth_ranks)
    crit_idx = np.argmin(drops) + 2 # adjust for valid convolution offset
    
    # Actually, the sharpest drop might be just an early artifact.
    # We want the point where it first drops below the median of its dynamic range
    # Or specifically where Lambda is around the theoretical critical point we found.
    # In earlier experiments, the theoretical critical log10_lambda for MLP was ~3-5.
    # Let's just find when it first drops below rank 6.8 (a solid indicator of structure forming)
    crit_lambda = None
    for r, l in zip(ranks, lambdas):
        if r < 6.8:
            crit_lambda = l
            break
            
    if crit_lambda is None:
        crit_lambda = lambdas[np.argmin(drops)]
    return crit_lambda, cov

def live_prediction_experiment(crit_lambda, cov):
    print(f"\n[SYSTEM] Calibrated Base Critical Constant \Lambda_crit = {crit_lambda:.2e}")
    
    set_seed(100) # New seed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    texts = load_texts(None, 256)
    euc_emb = encode_euclidean(texts, device, force_synthetic=True)
    target = build_target_proxy(euc_emb)
    mask = upper_triangle_mask(256, device=device)
    
    # Define three unseen models to predict
    experiments = [
        {"arch": "mlp", "L": 5, "D": 128, "gap": 0.000001, "lr": 0.005},
        {"arch": "transformer", "L": 2, "D": 128, "gap": 1.003922, "lr": 0.001},
        {"arch": "mamba", "L": 4, "D": 64, "gap": 0.000137, "lr": 0.005}
    ]
    
    plt.figure(figsize=(15, 5))
    
    for idx, exp in enumerate(experiments):
        print(f"\n=======================================================")
        print(f"[SYSTEM] Starting blind prediction on UNSEEN {exp['arch'].upper()} (L={exp['L']}, D={exp['D']})")
        
        model = HolographicArchitecture(arch_type=exp["arch"], input_dim=128, hidden_dim=exp["D"], hyp_dim=16, num_layers=exp["L"]).to(device)
        d_param = count_parameters(model)
        print(f"[SYSTEM] New Model Parameters: {d_param} | Spectral Gap: {exp['gap']:.4f}")
        
        lr = exp["lr"]
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        
        ema_grad_var = 0.0
        ema_beta = 0.9
        
        history_rank = []
        history_lambda = []
        history_accuracy = []
        predicted_epoch = -1
        
        # Calculate topological resistance penalty for this specific architecture
        # (Same formula used in ultimate_spectral_collapse)
        # Using the penalty formula: Lambda_topo = Lambda_base / (1 + gap * 2000)
        # Therefore, the threshold for base Lambda to cross is:
        # Base_Threshold = Calibrated_Lambda_Crit * (1 + gap * 2000)
        topological_penalty = (1.0 + exp["gap"] * 2000.0) 
        local_crit_lambda = crit_lambda * topological_penalty
        print(f"[RADAR] Topologically Adjusted \Lambda_crit for {exp['arch'].upper()} = {local_crit_lambda:.2e}")
        
        # We need a stable moving average for Live Lambda because gradient variance is noisy
        live_lambdas_window = []
        
        # Actually, if Lambda < local_crit_lambda, it means the system is still too HOT (high noise, low Lambda).
        # Phase transition happens when the system COOLS DOWN and Lambda crosses ABOVE the threshold.
        # But wait, in early training, gradients are tiny (cold), so Lambda is artificially huge. 
        # Then gradients explode (heat up), Lambda drops. 
        # Then gradients settle down (cools), Lambda rises again.
        # We want to detect when Lambda RISES and crosses the threshold from below, AFTER the initial warmup phase.
        
        for epoch in range(1, 201):
            optimizer.zero_grad()
            
            h = model.input_proj(euc_emb).unsqueeze(0)
            if model.arch_type == "mlp":
                for layer in model.layers:
                    h = layer(h) + h
                h = h.squeeze(0)
            elif model.arch_type == "transformer":
                h_out = model.layers[0](h)
                h = h + h_out
                h = h.squeeze(0)
            elif model.arch_type == "mamba":
                for layer in model.layers:
                    h = layer(h)
                h = h.squeeze(0)
                
            z_euc = model.out_proj(h) * 0.05
            
            z_hyp = model.manifold.expmap0(z_euc)
            z_hyp = clamp_to_ball(z_hyp)
            
            loss = metric_alignment_loss(model.manifold, z_hyp, target, mask=mask)
            loss.backward()
            
            # --- NEW PROOF: Logical Accuracy ---
            # We measure how well the pairwise distances actually match the target matrix
            # If phase transition happens, this accuracy should suddenly shoot up.
            with torch.no_grad():
                pred_dist = model.manifold.dist(z_hyp.unsqueeze(1), z_hyp.unsqueeze(0))
                # target is the expected exp(-d), so expected d = -ln(target)
                expected_dist = -torch.log(target + 1e-12)
                
                # We calculate R-squared (coefficient of determination) as accuracy
                mse = ((pred_dist - expected_dist) ** 2 * mask).sum()
                var = ((expected_dist - expected_dist.mean()) ** 2 * mask).sum()
                r2_accuracy = max(0.0, 1.0 - (mse / (var + 1e-12)).item())
                history_accuracy.append(r2_accuracy)
            
            grad_vec = []
            for p in model.parameters():
                if p.grad is not None:
                    grad_vec.append(p.grad.view(-1))
            if grad_vec:
                grad_vec = torch.cat(grad_vec)
                current_grad_var = grad_vec.var().item()
                ema_grad_var = ema_beta * ema_grad_var + (1 - ema_beta) * current_grad_var
                
            optimizer.step()
            
            # --- THE BLIND PREDICTION RADAR ---
            T_sys = lr * max(ema_grad_var, 1e-12)
            # Base lambda without topological adjustment
            live_base_lambda = cov / (T_sys * d_param * 16.0)
            live_lambdas_window.append(live_base_lambda)
            if len(live_lambdas_window) > 5:
                live_lambdas_window.pop(0)
            
            smoothed_live_lambda = np.mean(live_lambdas_window)
            
            reading = collect_reading(step=epoch, loss=loss.item(), z_hyp=z_hyp.detach(), params_with_grad=None)
            actual_rank = reading.effective_rank
            
            history_rank.append(actual_rank)
            history_lambda.append(smoothed_live_lambda)
            
            # We trigger the alarm when the system cools down enough that Lambda RISES above the threshold.
            # We wait until epoch > 15 to skip the initial artificial "cold" state where gradients are zero.
            if smoothed_live_lambda > local_crit_lambda and predicted_epoch == -1 and epoch > 15:
                # Need to also ensure we are in the "cooling" phase, i.e., lambda is rising.
                if len(history_lambda) > 2 and history_lambda[-1] > history_lambda[-2]:
                    print(f"🚨 [RADAR ALERT] At Epoch {epoch}, Smoothed Live \Lambda ({smoothed_live_lambda:.2e}) crossed Threshold ({local_crit_lambda:.2e})!")
                    predicted_epoch = epoch
                
        # Plotting the verification for this architecture
        plt.subplot(1, 3, idx+1)
        
        fig, ax1 = plt.subplots(figsize=(6, 5)) if idx == -1 else plt.gca(), plt.gca() # Dummy for context
        
        # We need to plot two y-axes on the same subplot for the new metric
        ax1 = plt.subplot(1, 3, idx+1)
        
        color = 'tab:blue'
        ax1.set_xlabel('Training Epochs')
        if idx == 0:
            ax1.set_ylabel('Absolute Effective Rank ($\Phi$)', color=color)
        ax1.plot(range(1, 201), history_rank, color=color, linewidth=2, label="Actual $\Phi$")
        ax1.tick_params(axis='y', labelcolor=color)
        
        if predicted_epoch != -1:
            ax1.axvline(x=predicted_epoch, color='r', linestyle='--', linewidth=2, 
                        label=f"Predicted Emergence")
            ax1.scatter([predicted_epoch], [history_rank[predicted_epoch-1]], color='r', s=100, zorder=5)
            
        # Instantiate a second axes that shares the same x-axis
        ax2 = ax1.twinx()  
        color2 = 'tab:green'
        if idx == 2:
            ax2.set_ylabel('Logical Tree Accuracy ($R^2$)', color=color2)
        ax2.plot(range(1, 201), history_accuracy, color=color2, linewidth=2, alpha=0.6, label="Logical Accuracy")
        ax2.tick_params(axis='y', labelcolor=color2)
        
        plt.title(f"{exp['arch'].upper()} (L={exp['L']}, D={exp['D']})")
        
        # Combine legends from both axes
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right")
        
        ax1.grid(True, alpha=0.3)
        
    plt.tight_layout()
    plt.savefig("multi_arch_emergence_prediction.png", dpi=150)
    print("\n[VERIFICATION] Saved multi_arch_emergence_prediction.png")
    
if __name__ == "__main__":
    crit_lambda, cov = get_calibrated_lambda_crit()
    live_prediction_experiment(crit_lambda, cov)
