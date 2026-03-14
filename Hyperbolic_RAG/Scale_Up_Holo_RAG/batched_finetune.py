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
    Robust Poincare Margin Loss with dynamic clamping.
    """
    def dist_sq(x, y):
        diff_sq = torch.sum((x - y)**2, dim=-1)
        norm_x_sq = torch.sum(x**2, dim=-1)
        norm_y_sq = torch.sum(y**2, dim=-1)
        
        norm_x_sq = torch.clamp(norm_x_sq, max=1.0 - 1e-5)
        norm_y_sq = torch.clamp(norm_y_sq, max=1.0 - 1e-5)
        
        arg = 1 + 2 * diff_sq / ((1 - norm_x_sq) * (1 - norm_y_sq))
        arg = torch.clamp(arg, min=1.0 + 1e-5)
        
        d = torch.acosh(arg)
        return d**2

    d_uv = dist_sq(u, v)
    
    loss = torch.where(
        label > 0,
        d_uv,
        F.relu(margin - d_uv)
    )
    
    # Hierarchy penalty: Root should be closer to origin
    norm_u = torch.norm(u, p=2, dim=-1)
    norm_v = torch.norm(v, p=2, dim=-1)
    
    hierarchy_penalty = F.relu(norm_u - norm_v + 0.3)
    loss = loss + 10.0 * torch.where(label > 0, hierarchy_penalty, torch.zeros_like(hierarchy_penalty))
    
    return loss.mean()

def train_batched_holo_embedder():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== 启动全息嵌入规模化微调 (Scale-Up Hyperbolic Fine-Tuning) on {device} ===")
    
    embedder = HyperbolicEmbedder("BAAI/bge-small-zh-v1.5").to(device)
    optimizer = optim.Adam(embedder.projection.parameters(), lr=1e-3)
    
    dataset = HierarchicalDataset("hierarchical_dataset.json")
    # Small batch size due to memory constraints of keeping Transformer on GPU
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    epochs = 300 # Increased epochs and added LR scheduling
    
    print(f"Dataset size: {len(dataset)}, Batches per epoch: {len(dataloader)}")
    print("[开始训练...]")
    
    for epoch in range(epochs):
        embedder.train()
        total_loss = 0.0
        
        for batch_parents, batch_children, batch_labels in dataloader:
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
            
        avg_loss = total_loss / len(dataloader)
        
        # Anneal the learning rate slightly
        if epoch == 100:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 5e-4
        if epoch == 200:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 1e-4

        if epoch % 20 == 0 or epoch == epochs - 1:
            with torch.no_grad():
                r_u = torch.norm(u[0]).item()
                r_v = torch.norm(v[0]).item()
            print(f"Epoch {epoch:03d} | Avg Loss: {avg_loss:.4f} | R_父(sample): {r_u:.3f}, R_子(sample): {r_v:.3f}")

    # Save to the Scale-Up directory
    save_path = "scale_holo_projection.pt"
    torch.save(embedder.projection.state_dict(), save_path)
    print(f"\n>>> 规模化微调完成！已保存至 {save_path}")

if __name__ == "__main__":
    train_batched_holo_embedder()
