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
    print(f"Total unique chunks in Knowledge Base: {len(knowledge_base)}")
    
    # 3. Load Models
    print("\nLoading Base Embedding Model (BAAI/bge-small-en-v1.5)...")
    embedder = HyperbolicEmbedder("BAAI/bge-small-en-v1.5").to(device)
    
    print(">>> Note: Using Zero-Shot Hyperbolic Projection (Untrained geometry).")
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
            euclidean_embs = outputs.last_hidden_state[:, 0, :]
            
            scaled_embs = embedder.projection(euclidean_embs)
            hyperbolic_embs = embedder.holo_math.exp_map0(scaled_embs)
            
        db_flat.add_texts(euclidean_embs, batch_texts)
        db_holo.add_texts(hyperbolic_embs, batch_texts)
        
    print("\nStarting Retrieval Benchmark...")
    
    flat_mrr = 0.0
    holo_mrr = 0.0
    flat_recall_2 = 0.0
    holo_recall_2 = 0.0
    
    top_k_retrieval = 10
    
    for item in tqdm(dataset):
        query = item['question']
        supporting_titles = set([fact[0] for fact in item['supporting_facts']])
        target_texts = [t for t in knowledge_base if any(f"Title: {title}" in t for title in supporting_titles)]
        
        q_inputs = embedder.tokenizer([query], padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            q_out = embedder.backbone(input_ids=q_inputs.input_ids, attention_mask=q_inputs.attention_mask)
            q_euc = q_out.last_hidden_state[:, 0, :]
            q_hyp = embedder.holo_math.exp_map0(embedder.projection(q_euc))
            
        flat_results = db_flat.search(q_euc, top_k=top_k_retrieval)
        flat_retrieved_texts = [res[1] for res in flat_results]
        flat_mrr += mrr_score(flat_retrieved_texts, target_texts)
        flat_recall_2 += recall_at_k(flat_retrieved_texts, target_texts, 2)
        
        holo_results = db_holo.search(q_hyp, top_k=top_k_retrieval)
        holo_retrieved_texts = [res[1] for res in holo_results]
        holo_mrr += mrr_score(holo_retrieved_texts, target_texts)
        holo_recall_2 += recall_at_k(holo_retrieved_texts, target_texts, 2)
        
    flat_mrr /= sample_size
    holo_mrr /= sample_size
    flat_recall_2 /= sample_size
    holo_recall_2 /= sample_size
    
    print("\n" + "="*50)
    print("Benchmark Results (Zero-Shot on HotpotQA subset)")
    print("="*50)
    print(f"Metrics over {sample_size} complex multi-hop queries:")
    print(f"\n[Baseline: Flat Euclidean RAG (Cosine)]")
    print(f"MRR       : {flat_mrr:.4f}")
    print(f"Recall@2  : {flat_recall_2:.4f}")
    
    print(f"\n[Ours: Holo-RAG (Poincaré Geodesic)]")
    print(f"MRR       : {holo_mrr:.4f}")
    print(f"Recall@2  : {holo_recall_2:.4f}")
    print("="*50)

if __name__ == "__main__":
    run_hotpotqa_benchmark()
