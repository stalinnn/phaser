import json
from pathlib import Path
import numpy as np

def analyze_log(name, dir_name):
    print(f"\n{'='*50}")
    print(f"Analyzing {name} ({dir_name})")
    print(f"{'='*50}")
    log_path = Path(dir_name) / "telm_readings.jsonl"
    if not log_path.exists():
        print("Log file not found.")
        return
        
    steps, ranks, Ts, losses = [], [], [], []
    with open(log_path, 'r') as f:
        for line in f:
            if not line.strip(): continue
            d = json.loads(line)
            steps.append(d["step"])
            ranks.append(d["effective_rank"])
            Ts.append(d["physics_T"])
            losses.append(d["loss"])
            
    print(f"Step 1: Rank = {ranks[0]:.2f}, T = {Ts[0]:.4f}, Loss = {losses[0]:.4f}")
    
    # Find max rank and when it drops
    max_rank = max(ranks)
    max_idx = ranks.index(max_rank)
    print(f"Max Rank = {max_rank:.2f} at Step {steps[max_idx]}, T = {Ts[max_idx]:.4f}")
    
    # Calculate drops
    rank_diffs = np.diff(ranks)
    min_diff_idx = np.argmin(rank_diffs)
    print(f"Biggest Rank Drop: {ranks[min_diff_idx]:.2f} -> {ranks[min_diff_idx+1]:.2f} "
          f"(Drop of {rank_diffs[min_diff_idx]:.2f}) at Step {steps[min_diff_idx+1]}, T = {Ts[min_diff_idx+1]:.4f}")
          
    print(f"Final Step {steps[-1]}: Rank = {ranks[-1]:.2f}, T = {Ts[-1]:.4f}, Loss = {losses[-1]:.4f}")
    
    # Sample every 50 steps
    print("\nTrajectory sample:")
    for i in range(0, len(steps), max(1, len(steps)//5)):
        print(f"  Step {steps[i]:>3}: Rank = {ranks[i]:>6.2f}, T = {Ts[i]:.4f}")

analyze_log("MLP (2L, 64D)", "runs/arch_mlp_l2_d64")
analyze_log("Transformer (2L, 64D)", "runs/arch_tf_l2_d64")
analyze_log("Transformer (4L, 128D)", "runs/arch_tf_l4_d128")
analyze_log("Mamba (3L, 64D)", "runs/arch_mamba_l3_d64")

