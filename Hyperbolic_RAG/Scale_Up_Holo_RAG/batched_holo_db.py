import torch
import time
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from holo_math import PoincareMath

class BatchedHoloVectorDB:
    """
    Scale-up Holo Vector Database.
    Handles larger embeddings by batching the distance calculations to prevent CUDA OOM.
    """
    def __init__(self, c=1.0, device="cpu", batch_size=10000):
        self.c = c
        self.device = device
        self.batch_size = batch_size
        self.holo_math = PoincareMath(c=c)
        
        self.vectors = None
        self.metadata = []
        
    def add_texts(self, embeddings, texts):
        embeddings = embeddings.to(self.device)
        if self.vectors is None:
            self.vectors = embeddings
        else:
            self.vectors = torch.cat([self.vectors, embeddings], dim=0)
        self.metadata.extend(texts)
        # print(f"[BatchedHoloDB] Total database size: {len(self.metadata)}.")

    def search(self, query_embedding, top_k=3):
        if self.vectors is None or len(self.metadata) == 0:
            return []
            
        query_embedding = query_embedding.to(self.device)
        start_time = time.time()
        
        num_vectors = self.vectors.size(0)
        all_distances = []
        
        import geoopt
        manifold = geoopt.PoincareBall(c=self.c)
        
        # Batched distance computation
        for i in range(0, num_vectors, self.batch_size):
            batch_vectors = self.vectors[i:i+self.batch_size]
            batch_dists = manifold.dist(query_embedding, batch_vectors).squeeze()
            if batch_dists.dim() == 0:
                batch_dists = batch_dists.unsqueeze(0)
            all_distances.append(batch_dists)
            
        distances = torch.cat(all_distances, dim=0)
        
        k = min(top_k, len(self.metadata))
        topk_dists, topk_indices = torch.topk(distances, k, largest=False)
        
        search_time = (time.time() - start_time) * 1000
        # print(f"[BatchedHoloDB] Search completed in {search_time:.2f} ms.")
        
        results = []
        for i in range(k):
            idx = topk_indices[i].item()
            dist = topk_dists[i].item()
            text = self.metadata[idx]
            results.append((dist, text))
            
        return results

if __name__ == "__main__":
    db = BatchedHoloVectorDB(device="cpu")
    print("Batched Holo DB loaded.")
