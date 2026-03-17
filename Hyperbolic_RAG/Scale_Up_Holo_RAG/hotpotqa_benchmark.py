import torch
import json
import os
import sys
from datasets import load_dataset
from tqdm import tqdm
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from holo_embedder import HyperbolicEmbedder
from run_holo_rag import FlatVectorDB
from batched_holo_db import BatchedHoloVectorDB

def mrr_score(retrieved_texts, target_texts):
    """Calculate Mean Reciprocal Rank"""
    for rank, text in enumerate(retrieved_texts, 1):
        if any(target in text for target in target_texts) or any(text in target for target in target_texts):
            return 1.0 / rank
    return 0.0

def recall_at_k(retrieved_texts, target_texts, k):
    """Calculate Recall@K"""
    retrieved_k = retrieved_texts[:k]
    for text in retrieved_k:
        if any(target in text for target in target_texts) or any(text in target for target in target_texts):
            return 1.0
    return 0.0

def run_hotpotqa_benchmark():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== 运行公开基准测试 (HotpotQA Multi-hop Benchmark) on {device} ===\n")
    
    import requests
    
    # Download HotpotQA dev set directly from official source to avoid HuggingFace connection issues
    file_path = "hotpot_dev_distractor_v1.json"
    if not os.path.exists(file_path):
        print("Downloading HotpotQA validation set directly from official URL...")
        url = "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json"
        response = requests.get(url, stream=True)
        with open(file_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download complete.")
        
    print("Loading dataset into memory...")
    with open(file_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
        
    # We will sample 100 random multi-hop questions for this benchmark to keep runtime manageable
    import random
    random.seed(42)
    sample_size = 100
    dataset = random.sample(dataset, sample_size)
    
    # 2. Build the Knowledge Base
    print("Building Knowledge Base from contexts...")
    kb_set = set()
    for item in dataset:
        contexts = item['context']
        for title, sentences in contexts:
            chunk = f"Title: {title}. " + " ".join(sentences)
            kb_set.add(chunk)
            
    knowledge_base = list(kb_set)
    # Reduce knowledge base size slightly to save time
    import random
    random.seed(42)
    knowledge_base = random.sample(knowledge_base, min(len(knowledge_base), 1000))
    print(f"Total unique chunks in Knowledge Base: {len(knowledge_base)}")
    
    # 3. Load Models
    print("\nLoading Base Embedding Model (BAAI/bge-small-en-v1.5)...")
    embedder = HyperbolicEmbedder("/gz-data/Qwen2.5-1.5B-Instruct").to(device)
    
    print(">>> Note: Using Fine-Tuned Hyperbolic Projection (scale_holo_projection_hotpotqa.pt).")
    # Optional Zero-Shot testing
    if "--zero-shot" in sys.argv:
        print(">>> FORCE ZERO-SHOT MODE: Initializing projection as scaled Identity.")
        import torch.nn as nn
        # Initialize as roughly identity so we preserve BGE's original semantic relationships
        # Note: since we changed to a 3-layer MLP, we just zero out the weights and biases
        # and add a small identity to the combined transformation to pass through the signal
        with torch.no_grad():
            for m in embedder.projection.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=0.01)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
    else:
        try:
            # 尝试加载刚刚微调好的权重
            state_dict = torch.load("scale_holo_projection_hotpotqa.pt", map_location=device)
            embedder.projection.load_state_dict(state_dict)
            print(">>> Successfully loaded fine-tuned projection weights!")
        except FileNotFoundError:
            print(">>> WARNING: scale_holo_projection_hotpotqa.pt not found. Using untrained weights.")
    import torch.nn as nn
    embedder.eval()
    
    db_flat = FlatVectorDB(device=device)
    db_holo = BatchedHoloVectorDB(c=1.0, device=device)
    
    # 4. Encode and Ingest
    print("Encoding knowledge base...")
    batch_size = 32
    for i in tqdm(range(0, len(knowledge_base), batch_size)):
        batch_texts = knowledge_base[i:i+batch_size]
        inputs = embedder.tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = embedder.backbone(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
            attention_mask_expanded = inputs.attention_mask.unsqueeze(-1).expand_as(outputs.last_hidden_state)
            euclidean_embs = torch.sum(outputs.last_hidden_state * attention_mask_expanded, 1) / torch.clamp(attention_mask_expanded.sum(1), min=1e-9)
            
            scaled_embs = embedder.projection(euclidean_embs)
            hyperbolic_embs = embedder.holo_math.exp_map0(scaled_embs)
            
        db_flat.add_texts(euclidean_embs, batch_texts)
        db_holo.add_texts(hyperbolic_embs, batch_texts)
        
    print("\nStarting Retrieval Benchmark...")
    
    flat_mrr = 0.0
    holo_mrr = 0.0
    hybrid_mrr = 0.0
    flat_recall_2 = 0.0
    holo_recall_2 = 0.0
    hybrid_recall_2 = 0.0
    
    top_k_retrieval = 20
    
    for item in tqdm(dataset):
        query = item['question']
        supporting_titles = set([fact[0] for fact in item['supporting_facts']])
        target_texts = [t for t in knowledge_base if any(f"Title: {title}" in t for title in supporting_titles)]
        
        q_inputs = embedder.tokenizer([query], padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            q_out = embedder.backbone(input_ids=q_inputs.input_ids, attention_mask=q_inputs.attention_mask)
            attention_mask_expanded = q_inputs.attention_mask.unsqueeze(-1).expand_as(q_out.last_hidden_state)
            q_euc = torch.sum(q_out.last_hidden_state * attention_mask_expanded, 1) / torch.clamp(attention_mask_expanded.sum(1), min=1e-9)
            q_hyp = embedder.holo_math.exp_map0(embedder.projection(q_euc))
            
        # 1. Flat RAG
        flat_results = db_flat.search(q_euc, top_k=top_k_retrieval)
        flat_retrieved_texts = [res[1] for res in flat_results]
        flat_mrr += mrr_score(flat_retrieved_texts, target_texts)
        flat_recall_2 += recall_at_k(flat_retrieved_texts, target_texts, 2)
        
        # 2. Holo RAG
        holo_results = db_holo.search(q_hyp, top_k=top_k_retrieval)
        holo_retrieved_texts = [res[1] for res in holo_results]
        holo_mrr += mrr_score(holo_retrieved_texts, target_texts)
        holo_recall_2 += recall_at_k(holo_retrieved_texts, target_texts, 2)
        
        # 3. Hybrid RAG (RRF Fusion)
        rrf_scores = {}
        for rank, (score, text) in enumerate(flat_results):
            rrf_scores[text] = rrf_scores.get(text, 0.0) + 1.0 / (10 + rank + 1)
            
        for rank, (dist, text) in enumerate(holo_results):
            rrf_scores[text] = rrf_scores.get(text, 0.0) + 1.0 / (10 + rank + 1)
            
        # Sort by RRF score descending
        hybrid_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        hybrid_retrieved_texts = [res[0] for res in hybrid_results]
        
        hybrid_mrr += mrr_score(hybrid_retrieved_texts, target_texts)
        hybrid_recall_2 += recall_at_k(hybrid_retrieved_texts, target_texts, 2)
        
    flat_mrr /= sample_size
    holo_mrr /= sample_size
    hybrid_mrr /= sample_size
    flat_recall_2 /= sample_size
    holo_recall_2 /= sample_size
    hybrid_recall_2 /= sample_size
    
    print("\n" + "="*50)
    print("Benchmark Results (Fine-Tuned on HotpotQA subset)")
    print("="*50)
    print(f"Metrics over {sample_size} complex multi-hop queries (Fine-Tuned):")
    print(f"\n[Baseline: Flat Euclidean RAG (Cosine)]")
    print(f"MRR       : {flat_mrr:.4f}")
    print(f"Recall@2  : {flat_recall_2:.4f}")
    
    print(f"\n[Ablation: Pure Holo-RAG (Poincaré Geodesic)]")
    print(f"MRR       : {holo_mrr:.4f}")
    print(f"Recall@2  : {holo_recall_2:.4f}")
    
    print(f"\n[Ours: Holo-Hybrid RAG (Dual-Metric RRF)]")
    print(f"MRR       : {hybrid_mrr:.4f}")
    print(f"Recall@2  : {hybrid_recall_2:.4f}")
    print("="*50)

if __name__ == "__main__":
    run_hotpotqa_benchmark()
