import torch
import time
import pandas as pd
import matplotlib.pyplot as plt
from tgn_triton_ops import tgn_block_sparse_attention

def benchmark_speed(func, args, n_warmup=10, n_repeat=100):
    # Warmup
    for _ in range(n_warmup):
        func(*args)
    torch.cuda.synchronize()
    
    # Measure
    start = time.time()
    for _ in range(n_repeat):
        func(*args)
    torch.cuda.synchronize()
    end = time.time()
    
    return (end - start) / n_repeat * 1000 # ms

def run_benchmark():
    device = "cuda"
    dtype = torch.float16
    
    # Configs
    BATCH = 4
    HEAD = 8
    DIM = 128
    SEQLENS = [2048, 4096, 8192, 16384] # 可根据显存调整
    SPARSITIES = [0.0, 0.5, 0.8, 0.9, 0.95] # 0% empty -> 95% empty
    
    results = []

    print(f"{'SeqLen':<8} | {'Sparsity':<8} | {'Torch (ms)':<10} | {'TGN (ms)':<10} | {'Speedup':<8}")
    print("-" * 60)

    for L in SEQLENS:
        q = torch.randn(BATCH, HEAD, L, DIM, device=device, dtype=dtype)
        k = torch.randn(BATCH, HEAD, L, DIM, device=device, dtype=dtype)
        v = torch.randn(BATCH, HEAD, L, DIM, device=device, dtype=dtype)
        
        # PyTorch SDPA Baseline
        # PyTorch 无法利用 Gate 加速，只能跑全量
        # 我们用 SDPA (FlashAttn backend) 作为最强基线
        def torch_sdpa(q, k, v):
            return torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
            
        time_torch = benchmark_speed(torch_sdpa, (q, k, v))
        
        for s in SPARSITIES:
            # Generate Gate
            n_blocks = (L + 127) // 128
            # Sparsity s means s% are ZERO. So we want (1-s)% to be ONE.
            # prob of 1 is (1-s)
            gate = (torch.rand(BATCH, n_blocks, n_blocks, device=device) < (1-s)).float()
            
            # TGN Triton Kernel
            # 注意: TGN Kernel 内部有 Overhead，只有在高稀疏度下才能赢 FlashAttn
            time_tgn = benchmark_speed(tgn_block_sparse_attention, (q, k, v, gate))
            
            speedup = time_torch / time_tgn
            
            print(f"{L:<8} | {s:<8.2f} | {time_torch:<10.2f} | {time_tgn:<10.2f} | {speedup:<8.2f}x")
            
            results.append({
                "SeqLen": L,
                "Sparsity": s,
                "Torch_ms": time_torch,
                "TGN_ms": time_tgn,
                "Speedup": speedup
            })

    # Save
    df = pd.DataFrame(results)
    df.to_csv("benchmark_results.csv", index=False)
    print("\nResults saved to benchmark_results.csv")

if __name__ == "__main__":
    run_benchmark()