import torch
try:
    import triton
    import triton.language as tl
except ImportError:
    print("Triton not installed. Ops will fail.")
    triton = None
    tl = None

# =========================================================
# 1. Triton Kernel: TGN Block-Sparse Flash Attention
# =========================================================

# 启用 Autotune 以寻找最佳性能配置
@triton.autotune(
    configs=[
        # 强制固定 BLOCK_M=128, BLOCK_N=128 以匹配 Gate 的粒度
        # 只搜索 num_stages 和 num_warps
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_stages=4, num_warps=8),
    ],
    key=[], # 空列表或不传，让 Triton 使用所有非 Tensor 参数作为 Key
)
@triton.jit
def _tgn_fwd_kernel(
    Q, K, V, Gate, Out,
    stride_b, stride_h, stride_n, stride_d,
    stride_gate_b, stride_gate_m, stride_gate_n,
    stride_out_b, stride_out_h, stride_out_n, stride_out_d,
    Z, H, 
    N_CTX_Q, # 分离 Q 和 K 的长度
    N_CTX_K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    # --- 1. Grid 索引 ---
    start_m = tl.program_id(0)
    off_z = tl.program_id(1)
    off_h = tl.program_id(2)

    # --- 2. 边界参数计算 ---
    num_m_blocks = (N_CTX_Q + BLOCK_M - 1) // BLOCK_M
    num_n_blocks = (N_CTX_K + BLOCK_N - 1) // BLOCK_N
    
    if start_m >= num_m_blocks:
        return

    # --- 3. 指针偏移初始化 ---
    q_offset = (off_z * stride_b) + (off_h * stride_h) + (start_m * BLOCK_M * stride_n)
    gate_base = (off_z * stride_gate_b) + (start_m * stride_gate_m)
    out_offset = (off_z * stride_out_b) + (off_h * stride_out_h) + (start_m * BLOCK_M * stride_out_n)

    # --- 4. 加载 Q ---
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, HEAD_DIM)
    q_ptrs = Q + q_offset + (tl.arange(0, BLOCK_M)[:, None] * stride_n) + (offs_k[None, :] * stride_d)
    
    mask_m = offs_m[:, None] < N_CTX_Q
    q = tl.load(q_ptrs, mask=mask_m, other=0.0)

    # --- 5. 初始化 Online Softmax ---
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    qk_scale = 1.44269504 * (1.0 / (HEAD_DIM ** 0.5))

    # --- 6. 循环遍历 K/V ---
    lo = 0
    # 对于 Causal Mask，只有当 Q 和 K 是同一个序列时 (Self-Attn)，我们需要限制 hi
    # 如果是 Cross-Attn (Streaming)，通常我们要看所有的历史 K，直到当前 Q 对应的位置
    # 这里我们假设 Streaming 模式下 K 已经包含了 Q，或者 IS_CAUSAL 会正确处理
    # 简单起见，如果 N_CTX_Q != N_CTX_K，我们假设 K 是历史缓存，全看 (hi = N_CTX_K)
    # 如果是 Self-Attn (Training)，保持原逻辑
    if IS_CAUSAL and N_CTX_Q == N_CTX_K:
        hi = (start_m + 1) * BLOCK_M
    else:
        hi = N_CTX_K # 否则看全部 K (通常是 Past + Current)
    
    for start_n in range(lo, hi, BLOCK_N):
        block_n_idx = start_n // BLOCK_N
        if block_n_idx < num_n_blocks:
             gate_val = tl.load(Gate + gate_base + (block_n_idx * stride_gate_n))
        else:
             gate_val = 0.0

        if gate_val >= 0.5:
            offs_n = start_n + tl.arange(0, BLOCK_N)
            k_offset = (off_z * stride_b) + (off_h * stride_h) + (start_n * stride_n)
            k_ptrs = K + k_offset + (tl.arange(0, BLOCK_N)[None, :] * stride_n) + (offs_k[:, None] * stride_d)
            mask_n = offs_n[None, :] < N_CTX_K
            k = tl.load(k_ptrs, mask=mask_n, other=0.0)

            qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
            qk += tl.dot(q, k)
            qk *= qk_scale

            # Causal Mask (Only if Self-Attention context)
            if IS_CAUSAL:
                if N_CTX_Q == N_CTX_K:
                    if start_n == start_m * BLOCK_M:
                        mask_causal = offs_m[:, None] >= offs_n[None, :]
                        qk = qk + tl.where(mask_causal, 0.0, float("-inf"))
            
            if start_n + BLOCK_N > N_CTX_K:
                 qk = qk + tl.where(mask_n, 0.0, float("-inf"))

            m_curr = tl.max(qk, 1)
            p = tl.math.exp2(qk - m_curr[:, None])
            m_new = tl.maximum(m_i, m_curr)
            alpha = tl.math.exp2(m_i - m_new)
            p_alpha = p * tl.math.exp2(m_curr - m_new)[:, None]
            
            v_offset = (off_z * stride_b) + (off_h * stride_h) + (start_n * stride_n)
            v_ptrs = V + v_offset + (tl.arange(0, BLOCK_N)[:, None] * stride_n) + (offs_k[None, :] * stride_d)
            v = tl.load(v_ptrs, mask=mask_n.T, other=0.0)

            acc = acc * alpha[:, None] + tl.dot(p_alpha.to(tl.float16), v)
            l_i = l_i * alpha + tl.sum(p_alpha, 1)
            m_i = m_new

    # --- 7. 写回 ---
    l_i = tl.where(l_i == 0.0, 1.0, l_i)
    acc = acc / l_i[:, None]

    out_ptrs = Out + out_offset + (tl.arange(0, BLOCK_M)[:, None] * stride_out_n) + (tl.arange(0, HEAD_DIM)[None, :] * stride_out_d)
    tl.store(out_ptrs, acc.to(tl.float16), mask=mask_m)

# =========================================================
# 2. Python Wrapper
# =========================================================

def tgn_block_sparse_attention(q, k, v, gate, causal=True):
    """
    Args:
        q: [Batch, Head, LenQ, Dim]
        k: [Batch, Head, LenK, Dim]
        v: [Batch, Head, LenK, Dim]
        gate: [Batch, Num_Q_Blocks, Num_K_Blocks]
    """
    if triton is None:
        raise RuntimeError("Triton not available")

    # 1. 确保内存连续
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    gate = gate.contiguous()
    
    BATCH, HEAD, N_CTX_Q, HEAD_DIM = q.shape
    _, _, N_CTX_K, _ = k.shape # 支持 Q 和 K 长度不同
    
    BLOCK_M = 128
    BLOCK_N = 128
    
    expected_m = (N_CTX_Q + BLOCK_M - 1) // BLOCK_M
    expected_n = (N_CTX_K + BLOCK_N - 1) // BLOCK_N
    
    # 检查 Gate
    assert gate.shape[1] == expected_m, f"Gate M mismatch: {gate.shape[1]} vs {expected_m}"
    assert gate.shape[2] == expected_n, f"Gate N mismatch: {gate.shape[2]} vs {expected_n}"

    out = torch.empty_like(q)
    
    grid = (expected_m, BATCH, HEAD)
    
    _tgn_fwd_kernel[grid](
        q, k, v, gate, out,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        gate.stride(0), gate.stride(1), gate.stride(2),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        BATCH, HEAD, 
        N_CTX_Q=N_CTX_Q, # 传入 Q 长度
        N_CTX_K=N_CTX_K, # 传入 K 长度
        HEAD_DIM=HEAD_DIM,
        IS_CAUSAL=causal
    )
    
    return out

# =========================================================
# 3. Test
# =========================================================
def run_stability_test():
    print("\n>>> Running TGN Kernel Stability Test...")
    if triton is None:
        print("Skip: Triton not installed")
        return

    torch.manual_seed(42)
    device = "cuda"
    
    B, H, L, D = 1, 4, 1024, 64
    dtype = torch.float16
    
    q = torch.randn(B, H, L, D, device=device, dtype=dtype)
    k = torch.randn(B, H, L, D, device=device, dtype=dtype)
    v = torch.randn(B, H, L, D, device=device, dtype=dtype)
    
    n_blocks = (L + 127) // 128
    gate = (torch.rand(B, n_blocks, n_blocks, device=device) > 0.5).float()
    
    try:
        out = tgn_block_sparse_attention(q, k, v, gate)
        print(f"✅ Success! Output shape: {out.shape}")
        print(f"✅ Mean: {out.mean().item():.4f}")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    run_stability_test()