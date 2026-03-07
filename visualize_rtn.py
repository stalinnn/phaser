import torch
import torch.nn.functional as F
from demo_rtn_training import RTNModel, generate_needle_haystack
import matplotlib.pyplot as plt
import seaborn as sns
import io

def visualize_gate_activation(model, device):
    """
    可视化门控激活状态 (CT Scan)
    输入一条测试数据，查看模型在哪些位置开启了 Attention (Gate=1)
    """
    model.eval()
    
    # 1. 生成一条测试数据
    # 为了演示效果，我们手动构造一个更清晰的 Needle-Haystack
    seq_len = 128
    batch_size = 1
    vocab_size = 1000
    
    # 构造数据
    inputs = torch.randint(2, vocab_size, (batch_size, seq_len)).to(device)
    
    # 设定关键位置
    needle_pos = 20  # 藏针的位置 (很久以前)
    trigger_pos = 100 # 触发位置 (很久以后)
    needle_id = 100
    trigger_id = 1
    
    # 确保 Needle 和 Trigger 在不同的 Chunk
    # num_chunks = 2, seq_len = 128, chunk_size = 64
    # Needle at 20 (Chunk 0), Trigger at 100 (Chunk 1)
    
    inputs[0, needle_pos] = needle_id
    inputs[0, trigger_pos] = trigger_id
    
    # 关键修改：让 Needle 所在的 Chunk 也有点“热度”
    # 在 Needle 附近加一点点噪声，模拟上下文复杂性
    # 或者，我们不需要改数据，而是相信模型应该学会去查 Needle
    
    # 实际上，Trigger 所在的 Chunk 必须热 (因为要回答问题)
    # Needle 所在的 Chunk 未必热 (因为存进去的时候可能不知道它重要)
    # 但是！为了回答 Trigger，Attention 必须查 Needle。
    # 我们的 Gate 是控制“当前 Chunk 是否做 Query”。
    # 所以 Trigger Chunk 必须热 (Query)。
    # Needle Chunk 是否热 (Key/Value)？
    # 在我们的实现中：
    # if route_mask.sum() > 0:
    #    x_hot = x_reshaped[indices[:, 0], indices[:, 1]]
    #    out_attn_hot = self.attn_core(x_hot)
    
    # 等等！FlashAttention 的输入通常是 Q, K, V。
    # 如果我们只把 Hot Chunk 拿出来做 Attention，那它只能看到 Hot Chunk 内部！
    # 这就是问题所在！
    # 真正的稀疏 Attention 应该是：
    # Query 来自 Hot Chunk
    # Key/Value 来自 全局 (或者至少包含 Needle Chunk)
    
    # 让我们修改 demo_rtn_training.py 中的 Attention 逻辑来修复这个问题。
    pass
    
    print(f"\n=== Visualizing Gate Activation (Inference Mode) ===")
    print(f"Needle Position: {needle_pos} (ID: {needle_id})")
    print(f"Trigger Position: {trigger_pos} (ID: {trigger_id})")
    
    # 2. 前向传播
    # 我们需要 hook 住每一层的 gate 输出
    gate_activations = []
    
    def get_gate_hook(layer_idx):
        def hook(module, input, output):
            # output 是 (x, chunk_entropy)
            # 我们只关心 chunk_entropy [B, NumChunks, 1]
            chunk_entropy = output[1]
            gate_activations.append(chunk_entropy.detach().cpu())
        return hook
    
    handles = []
    for i, layer in enumerate(model.layers):
        handles.append(layer.register_forward_hook(get_gate_hook(i)))
        
    with torch.no_grad():
        _ = model(inputs)
        
    for h in handles:
        h.remove()
        
    # 3. 处理数据用于绘图
    # gate_activations: List of [1, NumChunks, 1]
    # Stack -> [NumLayers, NumChunks]
    # gate_activations elements are [1, NumChunks, 1]
    # cat on dim=0 will make it [NumLayers, NumChunks, 1] if batch_size=1
    # or stack will make it [NumLayers, 1, NumChunks, 1]
    
    # Debug: Check what's in gate_activations
    # print(f"DEBUG: Captured {len(gate_activations)} gate outputs")
    # if len(gate_activations) > 0:
    #     print(f"DEBUG: First gate shape: {gate_activations[0].shape}")

    # Correct way:
    # Each hook call appends a tensor of shape [B, NumChunks, 1]
    # We have batch_size=1, so [1, NumChunks, 1]
    
    # Debug
    # print(f"DEBUG: gates shape after stack: {gates.shape}")
    
    # If gate_activations elements were [NumChunks, 1] (maybe squeezed inside hook?)
    # Let's check hook.
    # Hook gets chunk_entropy [B, NumChunks, 1].
    # So stack gives [NumLayers, B, NumChunks, 1].
    
    # But wait, did we squeeze inside hook?
    # No: gate_activations.append(chunk_entropy.detach().cpu())
    
    # Maybe B was squeezed out if B=1? No.
    
    # Let's be safe.
    gates = torch.stack(gate_activations, dim=0) # [NumLayers, B, NumChunks, 1]
    
    if len(gates.shape) == 4 and gates.shape[1] == 1:
         gates = gates.squeeze(1) # [NumLayers, NumChunks, 1]
         
    if len(gates.shape) == 3 and gates.shape[-1] == 1:
        gates = gates.squeeze(-1) # [NumLayers, NumChunks]
        
    gates = gates.cpu().numpy()
    
    
    # Check shape
    if len(gates.shape) == 1:
        # Only 1 layer?
        gates = gates.reshape(1, -1)
    
    # If using stack:
    # gates = torch.stack(gate_activations, dim=0).squeeze(1).squeeze(-1).cpu().numpy()
    
    # 4. 打印 ASCII 热力图 (因为没有 GUI，我们用文本可视化)
    num_layers, num_chunks = gates.shape
    chunk_size = seq_len // num_chunks
    
    # 打印原始概率值以便调试
    print("\n[Raw Gate Probabilities]")
    print(gates)

    print(f"\n[Layer-wise Gate Activation Map]")
    print(f"X-axis: Sequence Chunks (Size={chunk_size}), Y-axis: Layers")
    print("-" * 60)
    
    # 打印表头 (Chunk Index)
    header = "L/C |"
    for c in range(num_chunks):
        # 标记特殊位置所在的 Chunk
        is_needle = (needle_pos // chunk_size) == c
        is_trigger = (trigger_pos // chunk_size) == c
        
        if is_needle: marker = "N"
        elif is_trigger: marker = "T"
        else: marker = " "
        
        header += f" {c}{marker} |"
    print(header)
    print("-" * 60)
    
    # 打印每一层的数据
    for l in range(num_layers):
        row = f"L{l}  |"
        for c in range(num_chunks):
            prob = gates[l, c]
            # 可视化概率：>0.5 为激活 (■), <0.5 为关闭 (.)
            if prob > 0.5:
                symbol = " ■ "  # Attention ON
            else:
                symbol = " . "  # Mamba ONLY
            row += f"{symbol}|"
        print(row)
        
    print("-" * 60)
    print("Legend: ■ = Attention Activated (Hot), . = Mamba Only (Cold)")
    print("N = Needle Chunk, T = Trigger Chunk")
    
    # 5. 验证结论
    # 检查 Needle 和 Trigger 所在的 Chunk 是否被激活
    needle_chunk = needle_pos // chunk_size
    trigger_chunk = trigger_pos // chunk_size
    
    # 取最后一层的激活状态作为最终判断
    last_layer_gates = gates[-1]
    
    is_needle_active = last_layer_gates[needle_chunk] > 0.5
    is_trigger_active = last_layer_gates[trigger_chunk] > 0.5
    
    print("\n[Physics Verification]")
    print(f"Needle Chunk Active?  {is_needle_active}")
    print(f"Trigger Chunk Active? {is_trigger_active}")
    
    if is_needle_active and is_trigger_active:
        print("\n>>> SUCCESS: The model correctly identified critical information regions!")
        print(">>> Physics: Phase transition occurred exactly where entropy/surprise was high.")
    else:
        print("\n>>> PARTIAL SUCCESS: The model is sparse, but might have missed critical regions.")
        print(">>> Suggestion: Tune sparsity_loss weight or training steps.")

if __name__ == "__main__":
    # 加载刚才训练好的模型 (假设还在显存里，或者重新初始化并训练一会儿)
    # 为了演示，我们重新快速训练一下 (Few steps)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Retraining briefly for visualization...")
    
    from demo_rtn_training import RTNModel, generate_needle_haystack
    import torch.optim as optim
    
    model = RTNModel(1000, 64, 2).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    
    # 快速训练 500 步让它学会
    for step in range(500):
        inputs, targets = generate_needle_haystack(16, 128, 1000)
        inputs, targets = inputs.to(device), targets.to(device)
        logits, avg_gate = model(inputs)
        
        # 增加 Task Loss 的权重，或者减少 Sparsity Loss 的权重
        # 现在的 Sparsity Loss 太强了，导致模型不敢开门
        loss = F.cross_entropy(logits.view(-1, 1000), targets.view(-1)) + 0.01 * (avg_gate - 0.1)**2
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step % 50 == 0:
            print(f"Retrain Step {step} | Loss: {loss.item():.4f} | Gate: {avg_gate.item():.2%}")
        
    visualize_gate_activation(model, device)
