import torch
import re
import argparse
import sys
import os
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_from_disk
# Ensure we can import from local directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from gate_tgn_architecture import GateTGN_Model
import builtins

# ------------------------------------------------------------------
# 1. 移植来自 run_upcycling_sidetuning.py 的权重手术工具
# ------------------------------------------------------------------
def transplant_qwen_attention(torch_attn_layer, qwen_attn_layer):
    """
    Helper function to transplant weights from Qwen2Attention to nn.MultiheadAttention.
    Handles potential Grouped Query Attention (GQA) by repeating K/V heads.
    Supports Slicing for Debug Dimensions (though we aim for full dimensions now).
    """
    try:
        q_weight = qwen_attn_layer.q_proj.weight.data
        k_weight = qwen_attn_layer.k_proj.weight.data
        v_weight = qwen_attn_layer.v_proj.weight.data
        o_weight = qwen_attn_layer.o_proj.weight.data
        
        # Target Dimensions
        target_embed_dim = torch_attn_layer.embed_dim
        
        # Check for GQA (Grouped Query Attention)
        num_heads = qwen_attn_layer.num_heads
        num_kv_heads = qwen_attn_layer.num_key_value_heads
        head_dim = qwen_attn_layer.head_dim
        
        if num_kv_heads != num_heads:
            # GQA: Repeat K and V heads
            n_rep = num_heads // num_kv_heads
            k_weight = k_weight.view(num_kv_heads, head_dim, -1).unsqueeze(1).expand(num_kv_heads, n_rep, head_dim, -1).reshape(num_heads * head_dim, -1)
            v_weight = v_weight.view(num_kv_heads, head_dim, -1).unsqueeze(1).expand(num_kv_heads, n_rep, head_dim, -1).reshape(num_heads * head_dim, -1)
            
            if qwen_attn_layer.k_proj.bias is not None:
                 k_bias = qwen_attn_layer.k_proj.bias.data.view(num_kv_heads, head_dim).unsqueeze(1).expand(num_kv_heads, n_rep, head_dim).reshape(num_heads * head_dim)
                 v_bias = qwen_attn_layer.v_proj.bias.data.view(num_kv_heads, head_dim).unsqueeze(1).expand(num_kv_heads, n_rep, head_dim).reshape(num_heads * head_dim)
            else:
                 k_bias = None
                 v_bias = None
        else:
             k_bias = qwen_attn_layer.k_proj.bias.data if qwen_attn_layer.k_proj.bias is not None else None
             v_bias = qwen_attn_layer.v_proj.bias.data if qwen_attn_layer.v_proj.bias is not None else None

        q_bias = qwen_attn_layer.q_proj.bias.data if qwen_attn_layer.q_proj.bias is not None else None
        
        # Slice to target dimensions (Robustness for any mismatch)
        src_dim = q_weight.shape[1]
        dim_slice = min(target_embed_dim, src_dim)
        
        q_w_sliced = q_weight[:dim_slice, :dim_slice]
        k_w_sliced = k_weight[:dim_slice, :dim_slice]
        v_w_sliced = v_weight[:dim_slice, :dim_slice]
        
        target_in_proj = torch_attn_layer.in_proj_weight.data
        target_in_proj[:dim_slice, :dim_slice].copy_(q_w_sliced)
        target_in_proj[target_embed_dim:target_embed_dim+dim_slice, :dim_slice].copy_(k_w_sliced)
        target_in_proj[2*target_embed_dim:2*target_embed_dim+dim_slice, :dim_slice].copy_(v_w_sliced)
        
        # Output Projection
        o_w_sliced = o_weight[:dim_slice, :dim_slice]
        torch_attn_layer.out_proj.weight.data[:dim_slice, :dim_slice].copy_(o_w_sliced)
        
        # Handle Biases
        if q_bias is not None:
             torch_attn_layer.in_proj_bias.data[:dim_slice].copy_(q_bias[:dim_slice])
             torch_attn_layer.in_proj_bias.data[target_embed_dim:target_embed_dim+dim_slice].copy_(k_bias[:dim_slice])
             torch_attn_layer.in_proj_bias.data[2*target_embed_dim:2*target_embed_dim+dim_slice].copy_(v_bias[:dim_slice])
            
        return True
    except Exception as e:
        print(f"Error transplanting attention: {e}")
        return False

def load_real_weights(tgn_model, mamba_path, qwen_path):
    """
    Evaluate 脚本专用的完整版手术工具。
    包含 Phase A (Mamba) 和 Phase B (Qwen)。
    """
    # ==========================================
    # 阶段 A: Mamba 骨架移植 (Backbone Surgery)
    # ==========================================
    print(f"\n[Surgery Phase A] Loading Falcon-Mamba Backbone from {mamba_path}...")
    try:
        # Load Mamba model to CPU
        # Falcon-Mamba 通常也是 AutoModelForCausalLM，底层是 Mamba 结构
        mamba_source = AutoModelForCausalLM.from_pretrained(mamba_path, torch_dtype=torch.float16, device_map="cpu", trust_remote_code=True)
        print("    [Success] Mamba source model loaded.")
        
        # 检查层数对齐
        src_layers = mamba_source.backbone.layers # Falcon-Mamba structure typically has 'backbone.layers'
        num_src_layers = len(src_layers)
        num_tgt_layers = tgn_model.num_layers
        
        print(f"    Transplanting Backbone Weights ({num_src_layers} layers -> {num_tgt_layers} layers)...")
        
        layer_indices = [int(i * (num_src_layers / num_tgt_layers)) for i in range(num_tgt_layers)]
        
        for i, src_idx in enumerate(layer_indices):
            tgt_block = tgn_model.layers[i]
            src_block = src_layers[src_idx] # HF MambaBlock
            
            # Falcon-Mamba (HF) -> Mamba-SSM (Our Implementation) Mapping
            with torch.no_grad():
                # 1. Norms (Auto-Detect)
                src_norm = None
                if hasattr(src_block, 'input_layernorm'):
                    src_norm = src_block.input_layernorm
                elif hasattr(src_block, 'norm'):
                    src_norm = src_block.norm
                elif hasattr(src_block, 'layer_norm'):
                    src_norm = src_block.layer_norm
                
                if src_norm is not None:
                    tgt_block.norm_m.weight.copy_(src_norm.weight)
                    if hasattr(src_norm, 'bias') and src_norm.bias is not None and tgt_block.norm_m.bias is not None:
                        tgt_block.norm_m.bias.copy_(src_norm.bias)
                else:
                    print(f"    [!] Warning: Could not find Norm layer in Mamba block {src_idx}.")
                
                # 2. Mamba Mixer
                if hasattr(src_block, 'mixer'):
                    src_mixer = src_block.mixer
                    tgt_mamba = tgt_block.mamba
                    
                    tgt_mamba.in_proj.weight.copy_(src_mixer.in_proj.weight)
                    if src_mixer.in_proj.bias is not None and tgt_mamba.in_proj.bias is not None:
                        tgt_mamba.in_proj.bias.copy_(src_mixer.in_proj.bias)
                        
                    tgt_mamba.conv1d.weight.copy_(src_mixer.conv1d.weight)
                    if src_mixer.conv1d.bias is not None and tgt_mamba.conv1d.bias is not None:
                        tgt_mamba.conv1d.bias.copy_(src_mixer.conv1d.bias)
                    
                    tgt_mamba.x_proj.weight.copy_(src_mixer.x_proj.weight)
                    if src_mixer.x_proj.bias is not None and tgt_mamba.x_proj.bias is not None:
                        tgt_mamba.x_proj.bias.copy_(src_mixer.x_proj.bias)
                        
                    tgt_mamba.dt_proj.weight.copy_(src_mixer.dt_proj.weight)
                    if src_mixer.dt_proj.bias is not None and tgt_mamba.dt_proj.bias is not None:
                        tgt_mamba.dt_proj.bias.copy_(src_mixer.dt_proj.bias)
                        
                    tgt_mamba.out_proj.weight.copy_(src_mixer.out_proj.weight)
                    if src_mixer.out_proj.bias is not None and tgt_mamba.out_proj.bias is not None:
                        tgt_mamba.out_proj.bias.copy_(src_mixer.out_proj.bias)
                        
                    tgt_mamba.A_log.copy_(src_mixer.A_log)
                    tgt_mamba.D.copy_(src_mixer.D)
                    
        print("    [Success] Mamba Backbone Transplanted.")
        
        del mamba_source
        import gc
        torch.cuda.empty_cache()
        gc.collect()
        
    except Exception as e:
        print(f"    [Warning] Mamba Backbone surgery failed: {e}")
        print("    Using Randomly Initialized Backbone (Eval will be garbage).")

    # ==========================================
    # 阶段 B: Qwen 旁路移植 (Sidecar Surgery)
    # ==========================================
    print(f"\n[Surgery Phase B] Loading Pre-trained Qwen weights from {qwen_path}...")
    try:
        # Load Qwen model to CPU
        qwen_model = AutoModelForCausalLM.from_pretrained(qwen_path, torch_dtype=torch.float16, device_map="cpu", trust_remote_code=True)
        print("    [Success] Qwen model loaded on CPU.")
        
        num_tgn_layers = tgn_model.num_layers
        num_qwen_layers = len(qwen_model.model.layers)
        
        print(f"    Transplanting weights (Layer Mapping: {num_qwen_layers} -> {num_tgn_layers})...")
        
        for i in range(num_tgn_layers):
            # Linear Interpolation Mapping
            qwen_idx = min(int(i * (num_qwen_layers / num_tgn_layers)), num_qwen_layers - 1)
            
            target_sidecar = tgn_model.layers[i].sidecar
            source_layer = qwen_model.model.layers[qwen_idx]
            
            # 1. Transplant Attention
            res = transplant_qwen_attention(target_sidecar.attn, source_layer.self_attn)
            if not res:
                 print(f"    [!] Failed to transplant Attention for Layer {i}")
            
            # 2. Transplant SwiGLU MLP
            def safe_copy(dest_layer, src_layer):
                d_out, d_in = dest_layer.weight.shape
                s_out, s_in = src_layer.weight.shape
                copy_out = min(d_out, s_out)
                copy_in = min(d_in, s_in)
                with torch.no_grad():
                    dest_layer.weight.data[:copy_out, :copy_in].copy_(src_layer.weight.data[:copy_out, :copy_in])
                    
            safe_copy(target_sidecar.gate_proj, source_layer.mlp.gate_proj)
            safe_copy(target_sidecar.up_proj, source_layer.mlp.up_proj)
            safe_copy(target_sidecar.down_proj, source_layer.mlp.down_proj)
            
            # 3. Transplant LayerNorm
            d_norm = target_sidecar.norm.weight.shape[0]
            s_norm = source_layer.input_layernorm.weight.shape[0]
            common_norm = min(d_norm, s_norm)
            target_sidecar.norm.weight.data[:common_norm].copy_(source_layer.input_layernorm.weight.data[:common_norm])

        print("    [Success] All Attention Sidecars Transplanted.")
        
        # Transplant LM Head & Embedding (Essential for generation)
        print("    Transplanting LM Head & Embeddings...")
        qwen_head = qwen_model.lm_head.weight.data
        common_vocab = min(tgn_model.lm_head.weight.shape[0], qwen_head.shape[0])
        common_dim = min(tgn_model.lm_head.weight.shape[1], qwen_head.shape[1])
        tgn_model.lm_head.weight.data[:common_vocab, :common_dim].copy_(qwen_head[:common_vocab, :common_dim])
        
        qwen_emb = qwen_model.model.embed_tokens.weight.data
        common_vocab_emb = min(tgn_model.embedding.weight.shape[0], qwen_emb.shape[0])
        common_dim_emb = min(tgn_model.embedding.weight.shape[1], qwen_emb.shape[1])
        tgn_model.embedding.weight.data[:common_vocab_emb, :common_dim_emb].copy_(qwen_emb[:common_vocab_emb, :common_dim_emb])
        
        print(f"    [Success] LM Head & Embeddings Transplanted (Vocab={common_vocab}, Dim={common_dim}).")
        
        del qwen_model
        import gc
        torch.cuda.empty_cache()
        gc.collect()
        
    except Exception as e:
        print(f"    [Error] Critical Failure in Weight Transplantation: {e}")
        import traceback
        traceback.print_exc()

# ------------------------------------------------------------------
# 2. 主评估逻辑
# ------------------------------------------------------------------
def evaluate_tgn(samples=50, adapter_path=None, force_gate=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 真实路径配置
    qwen_path = "/gz-data/Qwen3-8B"
    mamba_path = "/gz-data/falcon-mamba-7b"
    
    print("="*60)
    print(f"EVALUATING TGN MODEL")
    print(f"Adapter: {adapter_path if adapter_path else 'Fresh Upcycled (Full SOTA Config)'}")
    print(f"Force Gate: {force_gate if force_gate is not None else 'Adaptive (Thermodynamic)'}")
    print("="*60)
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(qwen_path, trust_remote_code=True)
    except:
        print("Warning: Could not load real tokenizer, using mock.")
        tokenizer = None # Handle gracefully if needed
        return

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # 构建模型 (Full SOTA Config)
    # Mamba-7B (4096) + Qwen-8B Sidecar (3584)
    vocab_size = len(tokenizer)
    mamba_dim = 4096
    qwen_dim = 3584
    intermediate_size = 18944
    
    print(f"Building Gate-TGN Model (D={mamba_dim}, Sidecar={qwen_dim}, Inter={intermediate_size})...")
    model = GateTGN_Model(d_model=mamba_dim, num_layers=24, vocab_size=vocab_size, sidecar_dim=qwen_dim, intermediate_size=intermediate_size).to(device)
    
    # 核心：加载真实权重 (Mamba + Qwen)
    load_real_weights(model, mamba_path, qwen_path)
    
    if adapter_path:
        print(f"Loading Adapter (Wormhole) weights from {adapter_path}...")
        try:
            adapter_state = torch.load(adapter_path, map_location=device)
            # Use strict=False because we only saved 'wormhole' params, not the whole model
            model.load_state_dict(adapter_state, strict=False)
            print("    [Success] Adapter weights loaded.")
        except Exception as e:
            print(f"    [Error] Could not load adapter: {e}")
        
    model.eval()

    # Dataset Loading
    try:
        # 尝试加载测试集 (假设和训练集在同一级目录)
        test_path = "/TGN/qwen/gsm8k_offline/test"
        if not os.path.exists(test_path):
             test_path = "/TGN/qwen/gsm8k_offline/train" # Fallback to train for sanity check if test missing
        
        dataset = load_from_disk(test_path)
        print(f"Loaded dataset from {test_path}")
        dataset = dataset.select(range(min(samples, len(dataset))))
    except:
        print("Could not load GSM8K dataset. Creating dummy dataset.")
        dataset = [{"question": "Janet buys 3 apples for $2 each. How much does she spend?", "answer": "#### 6"}] * samples
    
    score = 0
    print(f"\nStarting Evaluation on {len(dataset)} samples...")
    
    for i, data in enumerate(tqdm(dataset)):
        # Extract Gold Answer
        gold = "N/A"
        try:
            match = re.search(r"#### (\-?\d+)", data["answer"])
            if match:
                gold = match.group(1)
        except:
            pass
            
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Solve the math problem step by step and output the final answer number."},
            {"role": "user", "content": data['question']}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(device)
        
        with torch.no_grad():
            # Manual autoregressive loop
            input_ids = inputs.input_ids
            # Limit generation length
            for _ in range(256):
                logits, gate_values = model(input_ids)
                next_token_logits = logits[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
                
                if next_token.item() == tokenizer.eos_token_id:
                    break
                    
                input_ids = torch.cat([input_ids, next_token], dim=-1)
            
            generated_ids = input_ids[0][inputs.input_ids.shape[1]:]
            pred_str = tokenizer.decode(generated_ids, skip_special_tokens=True)
            
        # Post-processing
        numbers = re.findall(r"(\-?\d+)", pred_str)
        pred = numbers[-1] if numbers else "None"
        
        # Simple logging
        if i < 3:
            print(f"\n[Sample {i}] Q: {data['question'][:50]}...")
            print(f"Pred: {pred} (Gold: {gold})")
            print(f"Output: {pred_str[:100]}...")
        
        if str(pred) == str(gold):
            score += 1
            
    print(f"\n>>> TGN Accuracy: {score / len(dataset) * 100:.2f}% ({score}/{len(dataset)})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=str, default=None)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--force_gate", type=float, default=None)
    args = parser.parse_args()
    
    # 简单的全局变量 Hack，通知架构层
    import builtins
    if args.force_gate is not None:
        builtins.TGN_FORCE_GATE = args.force_gate
    else:
        if hasattr(builtins, "TGN_FORCE_GATE"):
            del builtins.TGN_FORCE_GATE
            
    evaluate_tgn(args.samples, args.adapter, args.force_gate)
