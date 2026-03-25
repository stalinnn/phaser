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
        
    steps, ranks, Ts = [], [], []
    with open(log_path, 'r') as f:
        for line in f:
            if not line.strip(): continue
            d = json.loads(line)
            steps.append(d["step"])
            ranks.append(d["effective_rank"])
            Ts.append(d["physics_T"])
            
    print("\nTrajectory sample:")
    for i in range(0, len(steps), max(1, len(steps)//5)):
        print(f"  Step {steps[i]:>3}: Rank = {ranks[i]:>6.2f}, T = {Ts[i]:.4f}")

analyze_log("MLP (2L, 64D)", "runs/arch_mlp_l2_d64")
analyze_log("Transformer (2L, 64D)", "runs/arch_tf_l2_d64")
analyze_log("Transformer (3L, 64D)", "runs/arch_tf_l3_d64")
analyze_log("Mamba (3L, 64D)", "runs/arch_mamba_l3_d64")

