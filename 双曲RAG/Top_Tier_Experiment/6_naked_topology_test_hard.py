import json
import torch
import numpy as np
import torch.nn as nn
import geoopt
from tqdm import tqdm
import os

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

class EuclideanGraph(nn.Module):
    def __init__(self, num_q, num_c, dim=4):
        super().__init__()
        self.q_emb = nn.Embedding(num_q, dim)
        self.c_emb = nn.Embedding(num_c, dim)
        nn.init.normal_(self.q_emb.weight, std=0.01)
        nn.init.normal_(self.c_emb.weight, std=0.01)

    def dist(self, q, c):
        return torch.norm(q - c, p=2, dim=-1)

class HyperbolicGraph(nn.Module):
    def __init__(self, num_q, num_c, dim=4):
        super().__init__()
        self.manifold = geoopt.PoincareBall(c=1.0)
        self.q_emb = geoopt.ManifoldParameter(torch.randn(num_q, dim) * 1e-3, manifold=self.manifold)
        self.c_emb = geoopt.ManifoldParameter(torch.randn(num_c, dim) * 1e-3, manifold=self.manifold)

    def dist(self, q, c):
        return self.manifold.dist(q, c)

def compute_mrr(dists, pos_indices):
    ranks = []
    for i, d in enumerate(dists):
        sorted_idx = np.argsort(d)
        rank_dict = {idx: rank + 1 for rank, idx in enumerate(sorted_idx)}
        if pos_indices[i]:
            ranks.append(min([rank_dict[idx] for idx in pos_indices[i]]))
    if not ranks:
        return 0.0
    return np.mean([1.0 / r for r in ranks])

def run_test():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Loading Graph Topology...")
    data_dir = "/phaser/Hyperbolic_RAG/Top_Tier_Experiment/data"
    with open(f"{data_dir}/metadata.json", "r") as f:
        metadata = json.load(f)

    queries = metadata["queries"]
    q_to_c_positives = metadata["q_to_c_positives"]

    num_queries = len(queries)
    try:
        num_contexts = np.load(f"{data_dir}/c_embs.npy", mmap_mode='r').shape[0]
    except:
        num_contexts = 2000000 
        
    edges = []
    q_pos_map = {i: [] for i in range(num_queries)}
    
    for i, q in enumerate(queries):
        q_id = q["id"]
        pos_c_ids = q_to_c_positives.get(q_id, [])
        for c_id in pos_c_ids:
            if isinstance(c_id, int): c_idx = c_id
            elif "_" in str(c_id) and str(c_id).split("_")[-1].isdigit(): c_idx = int(str(c_id).split("_")[-1])
            else:
                try: c_idx = int(c_id)
                except: continue
            if c_idx < num_contexts:
                edges.append((i, c_idx))
                q_pos_map[i].append(c_idx)

    test_q_indices = list(range(500))
    test_pos_indices = [q_pos_map[i] for i in test_q_indices]

    # EXTREME BOTTLENECK: Only 4 dimensions
    dim = 4
    batch_size = 512
    num_negatives = 50
    epochs = 40
    tau = 0.1
    
    print(f"Graph Stats: {num_queries} Queries, {num_contexts} Contexts, {len(edges)} Edges")

    def train_model(model_type="euclidean"):
        print(f"\n--- Training {model_type.upper()} Topology (Dim={dim}) ---")
        if model_type == "euclidean":
            model = EuclideanGraph(num_queries, num_contexts, dim).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        else:
            model = HyperbolicGraph(num_queries, num_contexts, dim).to(device)
            optimizer = geoopt.optim.RiemannianAdam(model.parameters(), lr=0.01)
            
        edge_arr = np.array(edges)
        best_mrr = 0
        
        for epoch in range(epochs):
            np.random.shuffle(edge_arr)
            total_loss = 0
            
            pbar = tqdm(range(0, len(edge_arr), batch_size), leave=False)
            for i in pbar:
                batch_edges = edge_arr[i:i+batch_size]
                q_idx = torch.tensor(batch_edges[:, 0], dtype=torch.long).to(device)
                pos_idx = torch.tensor(batch_edges[:, 1], dtype=torch.long).to(device)
                
                neg_idx = torch.randint(0, num_contexts, (len(batch_edges), num_negatives)).to(device)
                
                optimizer.zero_grad()
                
                q = model.q_emb(q_idx) if isinstance(model.q_emb, nn.Embedding) else model.q_emb[q_idx]
                pos_c = model.c_emb(pos_idx) if isinstance(model.c_emb, nn.Embedding) else model.c_emb[pos_idx]
                neg_c = model.c_emb(neg_idx) if isinstance(model.c_emb, nn.Embedding) else model.c_emb[neg_idx]
                
                pos_dist = model.dist(q, pos_c)
                q_exp = q.unsqueeze(1)
                neg_dist = model.dist(q_exp, neg_c)
                
                pos_logits = -pos_dist / tau
                neg_logits = -neg_dist / tau
                
                logits = torch.cat([pos_logits.unsqueeze(1), neg_logits], dim=1)
                labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)
                
                loss = nn.functional.cross_entropy(logits, labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
            with torch.no_grad():
                test_q = model.q_emb(torch.tensor(test_q_indices).to(device)) if isinstance(model.q_emb, nn.Embedding) else model.q_emb[torch.tensor(test_q_indices).to(device)]
                eval_c_num = min(num_contexts, 50000) 
                test_c = model.c_emb(torch.arange(eval_c_num).to(device)) if isinstance(model.c_emb, nn.Embedding) else model.c_emb[torch.arange(eval_c_num).to(device)]
                
                if model_type == "euclidean":
                    dists = torch.cdist(test_q, test_c, p=2).cpu().numpy()
                else:
                    dists = np.zeros((len(test_q), eval_c_num), dtype=np.float16)
                    for qi in range(len(test_q)):
                        dists[qi] = model.manifold.dist(test_q[qi].unsqueeze(0), test_c).cpu().numpy()
                
                filtered_pos = []
                for p_list in test_pos_indices:
                    filtered_pos.append([p for p in p_list if p < eval_c_num])
                    
                mrr = compute_mrr(dists, filtered_pos)
                if mrr > best_mrr: best_mrr = mrr
                
            print(f"Epoch {epoch+1:02d} | Loss: {total_loss/(len(edge_arr)//batch_size + 1):.4f} | Val MRR: {mrr:.4f}")
            
        return best_mrr

    mrr_euc = train_model("euclidean")
    mrr_hyp = train_model("hyperbolic")
    
    print("\n" + "="*50)
    print(f"🏆 NAKED TOPOLOGY EXTREME TEST (Dim={dim}) 🏆")
    print("="*50)
    print(f"Euclidean MRR:  {mrr_euc:.4f}")
    print(f"Hyperbolic MRR: {mrr_hyp:.4f}")
    print("="*50)
    
    if mrr_hyp > mrr_euc + 0.05:
        print("结论：在极限维度下，双曲空间完美展现了拓扑优势！这就证明了其理论基础是坚实的。")
    elif mrr_euc > mrr_hyp + 0.05:
        print("结论：令人意外！即便在极限维度下，欧氏空间依然获胜或持平，双曲优化难度带来的拖累大于其几何收益。")
    else:
        print("结论：两者在极限维度下性能相近。")

if __name__ == "__main__":
    run_test()
