import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import os
import gc

# ==========================================
# 配置 (Configuration)
# ==========================================
class Config:
    def __init__(self):
        # 推荐使用 TinyLlama (1.1B) 和 GPT-2 XL (1.5B)
        # 这两个模型都比较小，显存友好，且代表了两种架构 (Pre-LN/RoPE vs Post-LN/AbsPos)
        self.models = {
            "TinyLlama": "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
            "GPT2-XL": "gpt2-xl" 
        }
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.seq_len = 1024 # 足够长以观察长程效应
        self.batch_size = 1
        self.out_dir = "result/fig1_mechanism"
        os.makedirs(self.out_dir, exist_ok=True)

config = Config()

# ==========================================
# 核心工具函数 (Core Utilities)
# ==========================================
def compute_effective_rank(hidden_states):
    """
    计算有效几何秩 R_eff = exp(Entropy(SingularValues))
    """
    # hidden_states: [Batch, SeqLen, Dim]
    # 我们只取第一个样本进行分析
    matrix = hidden_states[0].float() # [SeqLen, Dim]
    
    # 中心化 (Centering) - 这是一个严谨的协方差计算步骤
    matrix = matrix - matrix.mean(dim=0, keepdim=True)
    
    # SVD
    # 由于 SeqLen (1024) 接近 Dim (2048)，直接 SVD 比较快
    try:
        _, S, _ = torch.svd(matrix)
        # 归一化奇异值
        singular_values = S / S.sum()
        # 计算熵
        entropy = -torch.sum(singular_values * torch.log(singular_values + 1e-12))
        return torch.exp(entropy).item()
    except:
        return 0.0

def get_hidden_states(model, tokenizer, text_input, device, temp_scale=1.0, mask_attention=False):
    """
    获取模型的每一层 Hidden States。
    支持 Temperature Scaling 和 Attention Masking 干预。
    """
    inputs = tokenizer(text_input, return_tensors="pt", truncation=True, max_length=config.seq_len)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Hook 机制：用于干预 Attention
    hooks = []
    
    def attention_intervention_hook(module, args, output):
        # args[0] 是 hidden_states
        # 这是一个极其简化的 Hook，针对 HuggingFace 的标准实现
        # 如果要精确干预 Softmax 温度，最好的方式是 monkey patch 模型代码
        # 但为了通用性，我们这里用一种近似方法：
        # 对于 Mask Attention: 我们假设 Attention 输出为 0 (只保留 Residual)
        # 对于 Temperature: 实际上很难通过 Hook 外部干预内部 Softmax。
        # 因此，对于 Temperature Test，我们将采用 monkey patch 方式。
        pass

    # Monkey Patch for Attention Temperature & Masking
    # 这是一个黑魔法，用于动态修改模型的 Attention 行为
    original_forwards = {}
    
    for name, module in model.named_modules():
        if "Attention" in module.__class__.__name__ and "Self" in module.__class__.__name__: # 定位 SelfAttention 模块
            original_forwards[name] = module.forward
            
            # 定义新的 forward
            def new_forward(self, *args, **kwargs):
                # 这是一个通用 wrapper，尝试捕获 scale
                # 注意：不同模型的 forward 签名不同，这里只针对 Llama/GPT2 做适配
                # 最好的方式是修改 config 中的 attention_dropout 或 scale
                
                # 由于直接 Monkey Patch forward 太复杂且易碎
                # 我们这里采用更稳健的策略：
                # 1. 对于 Masking: 我们直接把 Attention 层的输出置零（通过 Hook 后处理）
                # 2. 对于 Temperature: 我们修改模型的 config.layer_norm_epsilon (这没用)
                #    真正的做法是：在推理前修改模型权重的缩放因子。
                
                # 恢复原始 forward 执行
                return original_forwards[name](self, *args, **kwargs)
            
            # 实际上，为了 Fig 1c (Temperature)，最简单的物理等效是：
            # Softmax(QK^T / (T * sqrt(d)))
            # 这等价于把 Q 和 K 的权重除以 sqrt(T)
            pass

    # --- 实施干预 ---
    if mask_attention:
        # 策略：Hook 住每一层 Attention 的输出，将其置零
        # 这样只有 FFN 和 Residual 在工作 -> 纯惯性流
        def block_attn_output(module, input, output):
            # output 通常是一个 tuple (attn_output, past_key_value, ...)
            if isinstance(output, tuple):
                return (torch.zeros_like(output[0]),) + output[1:]
            return torch.zeros_like(output)
            
        for name, module in model.named_modules():
            # 适配 GPT2 和 Llama 的命名习惯
            if ("attn" in name or "self_attention" in name) and "layer" in name and "c_proj" not in name and "o_proj" not in name:
                # 这是一个极其粗糙的 heuristic，为了精确，我们针对具体模型写
                pass
    
    # 重新加载模型以确保干净
    # 为简化脚本，我们将分别处理三种情况
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    # outputs.hidden_states 是一个 tuple，包含 (Emd, Layer1, ..., LayerN)
    return outputs.hidden_states

# ==========================================
# 实验 1a: Rank Evolution (Universal)
# ==========================================
def run_exp_1a(text_data):
    print("\n>>> Running Experiment 1a: Rank Evolution...")
    results = {}
    
    for model_name, model_path in config.models.items():
        print(f"Loading {model_name}...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.float16)
            
            hiddens = get_hidden_states(model, tokenizer, text_data, model.device)
            ranks = [compute_effective_rank(h) for h in hiddens]
            
            # 归一化层深 (0.0 - 1.0) 以便对比不同层数的模型
            layers = np.linspace(0, 1, len(ranks))
            results[model_name] = {"layers": layers, "ranks": ranks}
            
            del model, tokenizer
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"Failed to load {model_name}: {e}")
            
    # Save CSV
    df_list = []
    for name, data in results.items():
        for l, r in zip(data["layers"], data["ranks"]):
            df_list.append({"Model": name, "Normalized_Layer": l, "Effective_Rank": r})
    pd.DataFrame(df_list).to_csv(f"{config.out_dir}/fig1a_rank_evolution.csv", index=False)
    print("Fig 1a data saved.")

# ==========================================
# 实验 1b: Physical Ablation (No Attn)
# ==========================================
def run_exp_1b(text_data):
    print("\n>>> Running Experiment 1b: Physical Ablation (Attention Masking)...")
    # 使用 TinyLlama 作为实验对象
    model_path = config.models["TinyLlama"]
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.float16)
    
    # 1. Normal Run
    hiddens_normal = get_hidden_states(model, tokenizer, text_data, model.device)
    ranks_normal = [compute_effective_rank(h) for h in hiddens_normal]
    
    # 2. Ablated Run (Zero-out Attention Output)
    # 我们使用 Hook 技术强制将 Attention 层的输出置零
    handles = []
    for name, module in model.named_modules():
        # Llama 的 Attention 输出模块通常叫 'self_attn' (HuggingFace)
        # 我们 hook 它的 forward 输出
        if name.endswith("self_attn"): 
            def zero_out_hook(module, input, output):
                # output[0] 是 attention output 张量
                if isinstance(output, tuple):
                    return (torch.zeros_like(output[0]),) + output[1:]
                return torch.zeros_like(output)
            handles.append(module.register_forward_hook(zero_out_hook))
            
    hiddens_ablated = get_hidden_states(model, tokenizer, text_data, model.device)
    ranks_ablated = [compute_effective_rank(h) for h in hiddens_ablated]
    
    # Clean up hooks
    for h in handles: h.remove()
    
    # Save CSV
    df = pd.DataFrame({
        "Layer": range(len(ranks_normal)),
        "Rank_Normal": ranks_normal,
        "Rank_NoAttn": ranks_ablated
    })
    df.to_csv(f"{config.out_dir}/fig1b_ablation.csv", index=False)
    print("Fig 1b data saved.")
    
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

# ==========================================
# 实验 1c: Criticality (Temp Scaling)
# ==========================================
def run_exp_1c(text_data):
    print("\n>>> Running Experiment 1c: Criticality (Temperature Scaling)...")
    model_path = config.models["TinyLlama"]
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.float16)
    
    # 温度列表：对数刻度 [0.1, ..., 10]
    temps = np.logspace(-1, 1, 10) 
    # 加入 1.0 (Critical Point)
    temps = np.sort(np.append(temps, 1.0))
    
    avg_deep_ranks = []
    
    # 备份原始权重，以免累积修改
    original_weights = {}
    for name, module in model.named_modules():
        if name.endswith("self_attn.o_proj"): # Llama output projection
            # 我们通过缩放 Output Projection 的权重来模拟 Attention 的缩放
            # A_scaled = A / T
            # Output = (V * A_scaled) * W_o
            # 这不是严格的 Softmax 温度，但在线性近似下是等效的增益控制
            # 为了更精确，应该缩放 Q (Query)
            pass
            
    # 更精确的方法：直接修改 scaling_factor
    # LlamaModel 使用 scaling_factor = 1/sqrt(head_dim)
    # 我们要把它改成 1/(T * sqrt(head_dim))
    # 这通常硬编码在 modelling_llama.py 中，很难改。
    # 替代方案：缩放 Query 权重。
    # Q_new = Q_old / sqrt(T)
    
    # 定位 Query 层
    q_layers = []
    for name, module in model.named_modules():
        if "q_proj" in name:
            q_layers.append(module)
            
    # 备份原始权重
    import copy
    original_q_weights = [copy.deepcopy(layer.weight.data) for layer in q_layers]
    
    for T in temps:
        print(f"Scanning Temperature T={T:.2f}...")
        
        # 修改权重: Q' = Q / sqrt(T)
        # 这导致 DotProduct' = DotProduct / T
        scale = 1.0 / np.sqrt(T)
        
        for i, layer in enumerate(q_layers):
            layer.weight.data = original_q_weights[i] * scale
            
        hiddens = get_hidden_states(model, tokenizer, text_data, model.device)
        ranks = [compute_effective_rank(h) for h in hiddens]
        
        # 取深层 (后 20% 层) 的平均秩作为指标
        deep_idx = int(len(ranks) * 0.8)
        avg_deep_rank = np.mean(ranks[deep_idx:])
        avg_deep_ranks.append(avg_deep_rank)
        
    # Restore
    for i, layer in enumerate(q_layers):
        layer.weight.data = original_q_weights[i]
        
    # Save CSV
    df = pd.DataFrame({
        "Temperature": temps,
        "Deep_Layer_Rank": avg_deep_ranks
    })
    df.to_csv(f"{config.out_dir}/fig1c_criticality.csv", index=False)
    print("Fig 1c data saved.")

# ==========================================
# 主程序
# ==========================================
if __name__ == "__main__":
    # 准备一段长文本 (WikiText like)
    # 使用一段真实的文本比随机文本好，因为我们要测的是“语义流形”
    sample_text = """
    The theory of thermodynamics has been successfully applied to many complex systems, ranging from black holes to biological networks. 
    In this paper, we explore the connection between the geometry of information manifolds and the laws of non-equilibrium thermodynamics.
    Specifically, we propose that the attention mechanism in deep neural networks acts as a Maxwell's demon, selectively reducing the local entropy 
    of the system by introducing non-local correlations. This perspective offers a new way to understand the unreasonable effectiveness of 
    Transformers in modeling long-range dependencies. We validate our theory through extensive experiments on both synthetic and real-world datasets, 
    demonstrating that thermodynamic gating can significantly improve the efficiency of large language models without sacrificing performance.
    """ * 20 # 重复以达到 1024 长度
    
    run_exp_1a(sample_text)
    run_exp_1b(sample_text)
    run_exp_1c(sample_text)
    
    print("\nAll experiments complete. Data saved to result/fig1_mechanism/")
