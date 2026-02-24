import torch
import time
import argparse
from run_cloud_sota_battle import UniversalModel, Config, GeometricGate, CausalSelfAttention
# Ensure Mamba is importable
try:
    from mamba_ssm import Mamba as FastMamba
except ImportError:
    print("Mamba not found, skipping Mamba benchmark")

def benchmark(model_type, batch_size=32, seq_len=1024, d_model=1024):
    device = "cuda"
    # Create dummy config
    config = Config(model_size='medium') # 350M params
    config.n_embd = d_model
    
    # Initialize model
    model = UniversalModel(config, model_type=model_type).to(device).half() # FP16
    model.eval()
    
    # Dummy input
    x = torch.randint(0, 50000, (batch_size, seq_len)).to(device)
    
    # Warmup
    print(f"Warmup {model_type}...")
    with torch.no_grad():
        for _ in range(5):
            _ = model(x)
            
    torch.cuda.synchronize()
    
    # Benchmark
    print(f"Benchmarking {model_type}...")
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    steps = 50
    with torch.no_grad():
        for _ in range(steps):
            _ = model(x)
    end_event.record()
    torch.cuda.synchronize()
    
    elapsed_time_ms = start_event.elapsed_time(end_event)
    total_tokens = batch_size * seq_len * steps
    throughput = total_tokens / (elapsed_time_ms / 1000)
    
    # Memory
    mem = torch.cuda.max_memory_allocated() / 1024**3
    
    print(f"Result for {model_type}:")
    print(f"Throughput: {throughput:.2f} tokens/s")
    print(f"Peak Memory: {mem:.2f} GB")
    print("-" * 30)
    return throughput, mem

if __name__ == "__main__":
    print("Running Inference Benchmark on A800 (FP16)...")
    
    res = {}
    
    try:
        res['transformer'] = benchmark('transformer')
    except Exception as e:
        print(f"Transformer failed: {e}")

    try:
        res['mamba'] = benchmark('mamba')
    except Exception as e:
        print(f"Mamba failed: {e}")
        
    try:
        res['tgn'] = benchmark('tgn')
    except Exception as e:
        print(f"TGN failed: {e}")
        
    print("\nFINAL SUMMARY TABLE:")
    print(f"{'Model':<15} | {'Speed (tok/s)':<15} | {'Memory (GB)':<15}")
    for k, v in res.items():
        print(f"{k:<15} | {v[0]:<15.2f} | {v[1]:<15.2f}")
