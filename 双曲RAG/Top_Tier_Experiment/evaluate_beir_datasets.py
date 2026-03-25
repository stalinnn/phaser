import os
import json
import torch
import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import cdist
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from sentence_transformers import SentenceTransformer
import sys

# 将刚开发的路由器加进路径
sys.path.append("/phaser/Geometric_RAG_Router")

import torch.nn as nn
import geoopt

class HoloEmbedder(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=1024, output_dim=64, c=1.0):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )
        self.manifold = geoopt.PoincareBall(c=c)
        
    def forward(self, x):
        v = self.mlp(x)
        return self.manifold.expmap0(v)

def rrf_scores(rank_dict_base, rank_dict_abl, k=60, w_base=0.5, w_abl=0.5):
    combined_scores = {}
    for c_idx in set(rank_dict_base.keys()).union(set(rank_dict_abl.keys())):
        r_base = rank_dict_base.get(c_idx, 100000)
        r_abl = rank_dict_abl.get(c_idx, 100000)
        combined_scores[c_idx] = w_base * (1.0 / (k + r_base)) + w_abl * (1.0 / (k + r_abl))
    return combined_scores

from curvature_detector import analyze_dataset_manifold

def evaluate_dataset(dataset_name, data_path, bge_model, holo_model, manifold_detector, device):
    # Only evaluate specific well-formed BEIR datasets to avoid clutter
    if dataset_name not in ["arguana", "fiqa", "hotpotqa", "nfcorpus", "nq", "trec-covid", "webis-touche2020", "scifact", "fever", "climate-fever", "quora", "scidocs", "dbpedia-entity"]:
        return None
        
    print(f"\n================ Evaluating {dataset_name} ================")
    
    try:
        corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")
    except Exception as e:
        try:
            print(f"Test split not found for {dataset_name}, trying dev split...")
            corpus, queries, qrels = GenericDataLoader(data_path).load(split="dev")
        except Exception as e:
            print(f"Skipping {dataset_name}, could not load: {e}")
            return None
            
    # Subsample if dataset is too large (for quick demo), BUT preserve topology!
    max_corpus = 5000 
    
    # 1. First gather all IDs that are actually relevant to our queries
    query_ids = list(queries.keys())
    # Limit queries to 100 for speed
    eval_query_ids = query_ids[:100]
    
    # 2. Extract ALL relevant corpus IDs for these specific queries to preserve Multi-hop logic!
    must_include_corpus_ids = set()
    for q_id in eval_query_ids:
        if q_id in qrels:
            must_include_corpus_ids.update(qrels[q_id].keys())
            
    # 3. Build the final corpus list: required docs + random padding
    must_include_corpus_ids = list(must_include_corpus_ids.intersection(set(corpus.keys())))
    remaining_slots = max(0, max_corpus - len(must_include_corpus_ids))
    
    # Get random other docs to pad up to max_corpus
    other_corpus_ids = list(set(corpus.keys()) - set(must_include_corpus_ids))
    import random
    random.shuffle(other_corpus_ids)
    padding_corpus_ids = other_corpus_ids[:remaining_slots]
    
    corpus_ids = must_include_corpus_ids + padding_corpus_ids
    
    # 4. Filter qrels to keep only queries that have relevant docs in our sampled corpus
    filtered_qrels = {}
    valid_query_ids = []
    for q_id in eval_query_ids:
        if q_id in qrels:
            relevant_docs = {c_id: score for c_id, score in qrels[q_id].items() if c_id in corpus_ids}
            if len(relevant_docs) > 0:
                filtered_qrels[q_id] = relevant_docs
                valid_query_ids.append(q_id)
                
    if not valid_query_ids:
        print(f"No valid queries found for {dataset_name} after filtering.")
        return None

    print(f"Evaluated Queries: {len(valid_query_ids)}, Evaluated Corpus: {len(corpus_ids)}")

    # Encode Corpus
    print("Encoding Corpus with BGE-M3...")
    corpus_texts = [(corpus[c_id].get("title", "") + " " + corpus[c_id].get("text", "")).strip() for c_id in corpus_ids]
    c_embs = bge_model.encode(corpus_texts, batch_size=32, show_progress_bar=True, convert_to_numpy=True)
    
    # Encode Queries
    print("Encoding Queries with BGE-M3...")
    query_texts = [queries[q_id] for q_id in valid_query_ids]
    q_embs = bge_model.encode(query_texts, batch_size=32, show_progress_bar=True, convert_to_numpy=True)
    
    print("Projecting to Hyperbolic Space...")
    with torch.no_grad():
        c_embs_tensor = torch.tensor(c_embs, dtype=torch.float32).to(device)
        q_embs_tensor = torch.tensor(q_embs, dtype=torch.float32).to(device)
        
        c_holo_list, q_holo_list = [], []
        for i in range(0, len(c_embs_tensor), 512):
            c_holo_list.append(holo_model(c_embs_tensor[i:i+512]))
        c_holo = torch.cat(c_holo_list, dim=0)
        
        for i in range(0, len(q_embs_tensor), 512):
            q_holo_list.append(holo_model(q_embs_tensor[i:i+512]))
        q_holo = torch.cat(q_holo_list, dim=0)
        
    print("Analyzing Manifold Curvature (Gromov Delta)...")
    # 修改：使用我们刚开发的独立产品：ManifoldCurvatureDetector
    manifold_stats = manifold_detector.analyze(c_embs)
    print(f"  -> Detected Topology: {manifold_stats['topology']}")
    print(f"  -> Mean Delta: {manifold_stats['mean_delta']:.4f}")
    print(f"  -> Recommended Mode: {manifold_stats['recommended_mode']}")
    w_hyp, w_euc = manifold_stats['fusion_weights']
    
    print("Computing Similarities...")
    # Baseline: Cosine
    cosine_dists = cdist(q_embs, c_embs, metric='cosine')
    cosine_sims = 1.0 - cosine_dists
    
    # Ablation: Hyperbolic Distance
    hyper_dists = np.zeros((len(q_embs), len(c_embs)), dtype=np.float16)
    with torch.no_grad():
        for i in range(len(q_holo)):
            q_vec = q_holo[i].unsqueeze(0)
            dists = holo_model.manifold.dist(q_vec, c_holo)
            hyper_dists[i] = dists.cpu().numpy().astype(np.float16)

    # Build results for BEIR
    results_base = {}
    results_ablation = {}
    results_hybrid = {}
    
    print("Ranking and Fusion...")
    for i, q_id in enumerate(tqdm(valid_query_ids)):
        # Base (1024D Euclidean)
        sims = cosine_sims[i]
        sorted_indices_base = np.argsort(-sims)
        rank_dict_base = {corpus_ids[idx]: rank+1 for rank, idx in enumerate(sorted_indices_base)}
        results_base[q_id] = {corpus_ids[idx]: float(sims[idx]) for idx in sorted_indices_base[:100]}
        
        # Ablation (64D Hyperbolic)
        hdists = hyper_dists[i]
        sorted_indices_abl = np.argsort(hdists)
        rank_dict_abl = {corpus_ids[idx]: rank+1 for rank, idx in enumerate(sorted_indices_abl)}
        # For BEIR evaluate, higher score is better, so use negative distance
        results_ablation[q_id] = {corpus_ids[idx]: float(-hdists[idx]) for idx in sorted_indices_abl[:100]}
        
        # Hybrid (RRF with Manifold Weights: 1024D Base + 64D Hyperbolic Adapter)
        # Note: If w_hyp is 0, this strictly falls back to 1024D Base rank ordering
        # To make it perfectly fallback when w_hyp=0, we use raw scores or weighted RRF
        if w_hyp == 0.0:
            results_hybrid[q_id] = results_base[q_id]
        else:
            hybrid_scores = rrf_scores(rank_dict_base, rank_dict_abl, k=60, w_base=w_euc, w_abl=w_hyp)
            results_hybrid[q_id] = hybrid_scores

    evaluator = EvaluateRetrieval()
    
    print("\n--- Baseline (Euclidean Cosine) ---")
    ndcg_b, _map_b, recall_b, p_b = evaluator.evaluate(filtered_qrels, results_base, evaluator.k_values)
    
    print("\n--- Ablation (Pure Hyperbolic) ---")
    ndcg_a, _map_a, recall_a, p_a = evaluator.evaluate(filtered_qrels, results_ablation, evaluator.k_values)
    
    print("\n--- Ours (Holo-Hybrid RRF) ---")
    ndcg_h, _map_h, recall_h, p_h = evaluator.evaluate(filtered_qrels, results_hybrid, evaluator.k_values)
    
    return {
        "Baseline": {"NDCG@10": ndcg_b["NDCG@10"], "Recall@100": recall_b["Recall@100"]},
        "Hyperbolic": {"NDCG@10": ndcg_a["NDCG@10"], "Recall@100": recall_a["Recall@100"]},
        "Hybrid": {"NDCG@10": ndcg_h["NDCG@10"], "Recall@100": recall_h["Recall@100"]}
    }

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading Models...")
    # Add offline mode for huggingface if network is unreachable
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    bge_model = SentenceTransformer("BAAI/bge-m3", device=device, local_files_only=True)
    bge_model.max_seq_length = 512 # Set max length to prevent OOM on long docs
    
    # 引入我们开发的几何路由器来分析每个数据集
    from geom_rag_router import ManifoldCurvatureDetector
    manifold_detector = ManifoldCurvatureDetector()
    
    holo_model = HoloEmbedder(input_dim=1024, hidden_dim=1024, output_dim=64, c=1.0).to(device)
    # 使用刚刚在多任务上训练出来的泛化性更强的权重
    holo_model.load_state_dict(torch.load("/phaser/Hyperbolic_RAG/Top_Tier_Experiment/models/holo_embedder_multitask.pth", map_location=device))
    holo_model.eval()

    base_dir = "/gz-data/datasets"
    
    # Evaluate downloaded datasets
    datasets = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
    results_file = "beir_results.json"
    # Force re-evaluate all datasets to test new sampling logic
    # if os.path.exists(results_file):
    #     with open(results_file, "r") as f:
    #         all_results = json.load(f)
    # else:
    #     all_results = {}
    all_results = {}
        
    for ds in datasets:
        if ds in all_results:
            print(f"Skipping {ds}, already evaluated.")
            continue
            
        ds_path = os.path.join(base_dir, ds)
        res = evaluate_dataset(ds, ds_path, bge_model, holo_model, manifold_detector, device)
        if res:
            all_results[ds] = res
            with open(results_file, "w") as f:
                json.dump(all_results, f, indent=4)
            print(f"\n[Intermediate Results for {ds}] saved to {results_file}.")
            print(f"  Baseline   -> NDCG@10: {res['Baseline']['NDCG@10']:.4f}, Recall@100: {res['Baseline']['Recall@100']:.4f}")
            print(f"  Hyperbolic -> NDCG@10: {res['Hyperbolic']['NDCG@10']:.4f}, Recall@100: {res['Hyperbolic']['Recall@100']:.4f}")
            print(f"  Hybrid     -> NDCG@10: {res['Hybrid']['NDCG@10']:.4f}, Recall@100: {res['Hybrid']['Recall@100']:.4f}")
            
    print("\n\n" + "="*50)
    print("FINAL BEIR ZERO-SHOT EVALUATION RESULTS")
    print("="*50)
    for ds, res in all_results.items():
        print(f"\nDataset: {ds}")
        print(f"  Baseline   -> NDCG@10: {res['Baseline']['NDCG@10']:.4f}, Recall@100: {res['Baseline']['Recall@100']:.4f}")
        print(f"  Hyperbolic -> NDCG@10: {res['Hyperbolic']['NDCG@10']:.4f}, Recall@100: {res['Hyperbolic']['Recall@100']:.4f}")
        print(f"  Hybrid     -> NDCG@10: {res['Hybrid']['NDCG@10']:.4f}, Recall@100: {res['Hybrid']['Recall@100']:.4f}")
        
if __name__ == "__main__":
    main()
