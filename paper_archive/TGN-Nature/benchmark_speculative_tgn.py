import torch
import time
import matplotlib.pyplot as plt
import numpy as np
try:
    from tgn_triton_ops import tgn_block_sparse_attention
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

def benchmark_speculative():
    if not HAS_TRITON:
        print("Triton required.")
        return

    device = "cuda"
    dtype = torch.float16
    
    BATCH = 1
    HEAD = 8
    DIM = 128
    CHUNK_SIZE = 128
    SEQLENS = [8192, 16384, 32768]
    ACCEPT_RATES = [1.0, 0.9, 0.8, 0.7] # 模拟 Mamba 的准确率
    
    print(f"Benchmarking Speculative TGN (Chunk={CHUNK_SIZE})...")
    
    results = {} # {len: {rate: speedup}}

    # 1. Measure Baselines first
    # Mamba Step Time (O(1))
    mamba_step_time = 0.02 # ms (Empirical from prev bench)
    
    # Transformer Step Time (O(N)) - Pre-measured or approximate
    # From previous run: 32k -> ~1.76ms
    def get_transformer_time(L):
        # Linear approximation from our data
        # 1k=0.06, 32k=1.76. Slope approx 0.05 ms per 1k tokens
        return 0.05 + (L / 1000) * 0.054 

    # 2. Measure Triton Verify Time (O(N) but sparse)
    # We need to measure this for each length
    verify_times = {}
    
    for L in SEQLENS:
        q = torch.randn(BATCH, HEAD, CHUNK_SIZE, DIM, device=device, dtype=dtype)
        k = torch.randn(BATCH, HEAD, L, DIM, device=device, dtype=dtype)
        v = torch.randn(BATCH, HEAD, L, DIM, device=device, dtype=dtype)
        
        # Gate: 90% Sparsity
        n_q_blocks = (CHUNK_SIZE + 127) // 128
        n_k_blocks = (L + 127) // 128
        gate = (torch.rand(BATCH, n_q_blocks, n_k_blocks, device=device) > 0.9).float()
        
        # Warmup
        for _ in range(5): tgn_block_sparse_attention(q, k, v, gate)
        torch.cuda.synchronize()
        
        start = time.time()
        for _ in range(20):
            tgn_block_sparse_attention(q, k, v, gate)
        torch.cuda.synchronize()
        
        verify_time = (time.time() - start) / 20 * 1000 # ms per chunk verification
        verify_times[L] = verify_time
        print(f"L={L}: Verify Time = {verify_time:.2f} ms")

    # 3. Calculate Speculative Speedups
    print("\n--- Speedup Analysis ---")
    print(f"{'Len':<8} | {'Accept%':<8} | {'TGN(ms)':<8} | {'Trans(ms)':<8} | {'Speedup':<8}")
    
    for L in SEQLENS:
        trans_time_per_token = get_transformer_time(L)
        
        for alpha in ACCEPT_RATES:
            # Formula for Speculative Decoding Latency:
            # Time = (Draft_Time + Verify_Time) / (Chunk_Size * alpha + 1) roughly
            # More precisely:
            # We always pay Draft + Verify cost.
            # If accepted (prob alpha), we generated CHUNK_SIZE tokens.
            # If rejected (prob 1-alpha), we generated approx 1 token (and discard rest).
            # Expected Tokens per Step = alpha * CHUNK_SIZE + (1-alpha) * 1
            
            total_cost_per_step = (CHUNK_SIZE * mamba_step_time) + verify_times[L]
            expected_tokens = alpha * CHUNK_SIZE + (1 - alpha) * 1
            
            tgn_time_per_token = total_cost_per_step / expected_tokens
            
            speedup = trans_time_per_token / tgn_time_per_token
            
            print(f"{L:<8} | {alpha:<8.1f} | {tgn_time_per_token:<8.3f} | {trans_time_per_token:<8.3f} | {speedup:<8.2f}x")

if __name__ == "__main__":
    benchmark_speculative()
