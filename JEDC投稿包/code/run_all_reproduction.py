import os
import time

def run_script(script_name, description):
    print(f"\n{'='*60}")
    print(f"RUNNING: {description}")
    print(f"SCRIPT: {script_name}")
    print(f"{'='*60}")
    
    start_time = time.time()
    exit_code = os.system(f"python {script_name}")
    duration = time.time() - start_time
    
    if exit_code == 0:
        print(f"\n>>> SUCCESS ({duration:.2f}s)")
    else:
        print(f"\n>>> FAILED (Exit Code: {exit_code})")
        
if __name__ == "__main__":
    print("JEDC REPRODUCTION SUITE")
    print("Starting full reproduction pipeline...\n")
    
    # 1. Theoretical Foundation
    run_script("code/run_abm_simulation.py", "Agent-Based Model Simulation (Figure 2A)")
    
    # 2. Core Model & Orthogonality
    run_script("code/core_model.py", "Core Model Validation & Figure 1 Generation")
    
    # 3. Benchmarking (NEW)
    run_script("code/calculate_absorption_ratio.py", "Calculate Absorption Ratio Data")
    run_script("code/plot_tfi_vs_benchmark.py", "Generate Figure 1C (TFI vs Absorption Ratio)")

    # 4. Econometric Tests
    run_script("code/run_asset_pricing_test.py", "Kurtosis Prediction Regression (Table 1)")
    
    # 5. Strategy Backtests
    run_script("code/backtest_parisi_strategy.py", "US Strategy Walk-Forward Backtest (Table 2 & Figure 4)")
    run_script("code/backtest_china_a_share.py", "China A-Share Robustness Check (Figure A1)")
    
    # 6. Event Studies
    run_script("code/plot_lead_time_event_study.py", "2020 COVID Crash Event Study (Figure 5)")
    
    print("\n" + "="*60)
    print("ALL TASKS COMPLETED.")
    print("Please check the 'figures/' and 'results/' directories for outputs.")
    print("="*60)
