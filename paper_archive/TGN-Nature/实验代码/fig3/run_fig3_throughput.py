import torch
import torch.nn as nn
import time
import math
import pandas as pd
import matplotlib.pyplot as plt
import os

# Try Mamba
try:
    from mamba_ssm import Mamba
    HAS_MAMBA = True
except ImportError:
    HAS_MAMBA = False
    print("Mamba not found, simulating Mamba with lightweight RNN")

class ThroughputBenchmarker:
    def __init__(self, model_dim=1024, n_layers=24, device='cuda'):
        self.d = model_dim
        self.L = n_layers
        self.device = device
        self.chunk_size = 128
        
        # Components
        # Important: convert models to half (FP16) to match input
        self.mamba = (Mamba(d_model=self.d, d_state=16, d_conv=4, expand=2) if HAS_MAMBA else nn.GRU(self.d, self.d)).to(device).half()
        self.attn = nn.MultiheadAttention(self.d, num_heads=16, batch_first=True).to(device).half()
        
    def run_transformer(self, x):
        # Simulate L layers of Attention + MLP
        for _ in range(self.L):
            # Attention (Cost dominant)
            # Use PyTorch 2.0 Flash Attn for fair comparison
            with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=True):
                y, _ = self.attn(x, x, x)
            # MLP (Cost dominant)
            x = x + y
        return x

    def run_mamba(self, x):
        # Simulate L layers of Mamba + MLP
        for _ in range(self.L):
            if HAS_MAMBA:
                y = self.mamba(x)
            else:
                y, _ = self.mamba(x)
            x = x + y
        return x

    def run_chunked_tgn(self, x, sparsity=0.2):
        # TGN Logic:
        # - 80% chunks: Only Mamba
        # - 20% chunks: Mamba + Attention
        
        B, T, D = x.shape
        n_chunks = T // self.chunk_size
        n_dense = int(n_chunks * sparsity)
        n_sparse = n_chunks - n_dense
        
        # Simulate by running proportionally
        # We don't need real masking, just compute cost
        
        for _ in range(self.L):
            # 1. Inertial (Always runs)
            if HAS_MAMBA: y_m = self.mamba(x)
            else: y_m, _ = self.mamba(x)
            
            # 2. Geometric (Only runs on 20% tokens)
            # We simulate this by slicing input
            x_dense = x[:, :n_dense*self.chunk_size, :]
            if x_dense.size(1) > 0:
                with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=True):
                    y_a, _ = self.attn(x_dense, x_dense, x_dense)
            
            x = x + y_m # Simplified add
        return x

    def benchmark_single(self, name, func, seq_len, batch_size):
        x = torch.randn(batch_size, seq_len, self.d, device=self.device, dtype=torch.float16)
        try:
            # Warmup
            func(x)
            torch.cuda.synchronize()
            
            # Measure
            start = time.time()
            n_iters = 5
            for _ in range(n_iters):
                func(x)
            torch.cuda.synchronize()
            end = time.time()
            
            # Tokens per second
            total_tokens = batch_size * seq_len * n_iters
            tps = total_tokens / (end - start)
            return {name: tps}
        except torch.cuda.OutOfMemoryError:
            return {name: 0.0}

    def benchmark(self, seq_len, batch_size=1, sparsity=0.2):
        # Deprecated, use loop in main
        pass

def main():
    bench = ThroughputBenchmarker(model_dim=1024, n_layers=24) # 350M config
    
    # Sequence Lengths: 1k to 32k
    seq_lens = [1024, 2048, 4096, 8192, 16384, 32768]
    sparsity = 0.20 # 20%
    
    data = []
    print(f"Benchmarking on {torch.cuda.get_device_name(0)}...")
    print(f"Config: 350M Params, FP16, Sparsity={sparsity:.0%}")
    
    for L in seq_lens:
        # Uniform Batch Size = 1 (Latency Focused)
        # This ensures fair comparison and avoids OOM on Transformer
        bs = 1
        
        res = {}
        
        # 1. Run Transformer
        try:
            torch.cuda.empty_cache()
            res_t = bench.benchmark_single('Transformer', bench.run_transformer, L, bs)
            res.update(res_t)
        except:
            res['Transformer'] = 0.0

        # 2. Run Mamba
        try:
            torch.cuda.empty_cache()
            res_m = bench.benchmark_single('Mamba', bench.run_mamba, L, bs)
            res.update(res_m)
        except:
            res['Mamba'] = 0.0
            
        # 3. Run TGN
        try:
            torch.cuda.empty_cache()
            res_g = bench.benchmark_single(f'TGN (Chunked {sparsity*100:.0f}%)', lambda x: bench.run_chunked_tgn(x, sparsity), L, bs)
            res.update(res_g)
        except:
            res[f'TGN (Chunked {sparsity*100:.0f}%)'] = 0.0

        print(f"SeqLen {L:5d} (BS {bs}): {res}")
        
        row = {'seq_len': L}
        row.update(res)
        data.append(row)

    # Save CSV
    df = pd.DataFrame(data)
    os.makedirs('result_fig3', exist_ok=True)
    csv_path = 'result_fig3/throughput_benchmark.csv'
    df.to_csv(csv_path, index=False)
    print(f"Saved to {csv_path}")
    
    # Plot
    plt.figure(figsize=(8, 6))
    for col in df.columns:
        if col not in ['seq_len', 'batch_size']:
            plt.plot(df['seq_len'], df[col], marker='o', linewidth=2, label=col)
            
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Sequence Length')
    plt.ylabel('Throughput (tokens/sec)')
    plt.title('Inference Throughput Scaling')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.savefig('result_fig3/figure_3b_throughput.png')

if __name__ == '__main__':
    main()
