import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import os
import sys

# Import components from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from holo_embedder import HyperbolicEmbedder

class HierarchicalDataset(Dataset):
    def __init__(self, json_file):
        with open(json_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
            
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        item = self.data[idx]
        return item['parent'], item['child'], float(item['label'])

def poincare_margin_loss(u, v, label, margin=2.0, c=1.0):
    """
    Robust Poincare Margin Loss using geoopt.
    """
    import geoopt
    import torch
    import torch.nn.functional as F
    
    manifold = geoopt.PoincareBall(c=c)
    
    # Calculate geodesic distance safely
    u_proj = manifold.projx(u)
    v_proj = manifold.projx(v)
    d_uv = manifold.dist(u_proj, v_proj)
    
    # 给正样本增加一个拉力下限，不要求完全重合，只要拉近到0.5以内即可，防止过度压缩
    margin_pos = 0.5
    margin_neg = 6.0
    
    # Push positive pairs closer (distance < margin_pos)
    # Push negative pairs away (distance > margin_neg)
    loss_pos = F.relu(d_uv - margin_pos)
    loss_neg = F.relu(margin_neg - d_uv)
    
    mask_pos = (label > 0).float()
    mask_neg = (label < 0).float()
    
    # Normalize by the number of active pairs to prevent loss scale from fluctuating
    num_pos = max(1, mask_pos.sum().item())
    num_neg = max(1, mask_neg.sum().item())
    
    # 增加正样本的权重，迫使模型学习父子连结，但不能太大否则会坍塌
    loss = 2.0 * (loss_pos * mask_pos).sum() / num_pos + (loss_neg * mask_neg).sum() / num_neg
    
    return loss

def train_batched_holo_embedder():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== 启动全息嵌入规模化微调 (Scale-Up Hyperbolic Fine-Tuning) on {device} ===")
    
    # We use english model for HotpotQA
    embedder = HyperbolicEmbedder("/gz-data/Qwen2.5-1.5B-Instruct").to(device)
    # 减小学习率，并使用 Riemannian Adam (因为普通的 Adam 会把参数拉出李代数/流形空间)
    import geoopt
    optimizer = geoopt.optim.RiemannianAdam(embedder.projection.parameters(), lr=1e-4)
    
    # Use the generated HotpotQA training pairs
    dataset = HierarchicalDataset("hotpotqa_train_pairs.json")
    # Reduce dataset size for faster training
    import random
    random.seed(42)
    dataset.data = random.sample(dataset.data, 2000)
    
    # Small batch size due to memory constraints of keeping Transformer on GPU
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    epochs = 12 # Push it further to properly separate the classes
    
    print(f"Dataset size: {len(dataset)}, Batches per epoch: {len(dataloader)}")
    print("[开始训练...]")
    
    for epoch in range(epochs):
        embedder.train()
        total_loss = 0.0
        
        for batch_idx, (batch_parents, batch_children, batch_labels) in enumerate(dataloader):
            optimizer.zero_grad()
            
            batch_labels = batch_labels.to(device)
            
            inp_p = embedder.tokenizer(list(batch_parents), padding=True, truncation=True, return_tensors="pt").to(device)
            inp_c = embedder.tokenizer(list(batch_children), padding=True, truncation=True, return_tensors="pt").to(device)
            
            u = embedder(inp_p.input_ids, inp_p.attention_mask)
            v = embedder(inp_c.input_ids, inp_c.attention_mask)
            
            loss = poincare_margin_loss(u, v, batch_labels, margin=4.0)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(embedder.projection.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 50 == 0:
                print(f"  Batch {batch_idx}/{len(dataloader)} | Loss: {loss.item():.4f}")
            
        avg_loss = total_loss / len(dataloader)
        
        # Anneal the learning rate slightly
        if epoch == 1:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 1e-5

        with torch.no_grad():
            r_u = torch.norm(u[0]).item()
            r_v = torch.norm(v[0]).item()
        print(f"Epoch {epoch:03d} | Avg Loss: {avg_loss:.4f} | R_父(sample): {r_u:.3f}, R_子(sample): {r_v:.3f}")

    # Save to the Scale-Up directory
    save_path = "scale_holo_projection_hotpotqa.pt"
    torch.save(embedder.projection.state_dict(), save_path)
    print(f"\n>>> 规模化微调完成！已保存至 {save_path}")

if __name__ == "__main__":
    train_batched_holo_embedder()
