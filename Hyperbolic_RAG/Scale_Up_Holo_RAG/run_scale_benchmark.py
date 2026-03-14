import torch
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from holo_embedder import HyperbolicEmbedder
from run_holo_rag import FlatVectorDB
from batched_holo_db import BatchedHoloVectorDB

def run_scale_benchmark():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== 全息 RAG Scale-Up 基准测试 on {device} ===\n")
    
    # 1. Load Dataset
    with open("hierarchical_dataset.json", 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    # Extract unique texts for the Knowledge Base
    kb_set = set()
    for item in data:
        kb_set.add(item['parent'])
        kb_set.add(item['child'])
    knowledge_base = list(kb_set)
    
    print(f"Loaded knowledge base with {len(knowledge_base)} unique chunks.")
    
    # 2. Load Models
    print("Loading Base Embedding Model (BAAI/bge-small-zh-v1.5)...")
    embedder = HyperbolicEmbedder("BAAI/bge-small-zh-v1.5").to(device)
    
    try:
        embedder.projection.load_state_dict(torch.load("scale_holo_projection.pt"))
        print(">>> 成功加载规模化微调的全息投影矩阵 (Scale-Up Holo Projection Loaded!)")
    except FileNotFoundError:
        print(">>> 警告：未找到 scale_holo_projection.pt，请先运行 batched_finetune.py。")
        return
        
    embedder.eval()
    
    db_flat = FlatVectorDB(device=device)
    db_holo = BatchedHoloVectorDB(c=1.0, device=device)
    
    # 3. Batch Embed and Ingest
    print("Encoding knowledge base (this might take a moment)...")
    
    batch_size = 16
    for i in range(0, len(knowledge_base), batch_size):
        batch_texts = knowledge_base[i:i+batch_size]
        inputs = embedder.tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = embedder.backbone(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
            euclidean_embs = outputs.last_hidden_state[:, 0, :]
            
            scaled_embs = embedder.projection(euclidean_embs)
            hyperbolic_embs = embedder.holo_math.exp_map0(scaled_embs)
            
        db_flat.add_texts(euclidean_embs, batch_texts)
        db_holo.add_texts(hyperbolic_embs, batch_texts)
        
    print("Ingestion complete.")
    
    # 4. Define Complex Multihop Queries
    queries = [
        {
            "q": "探春为什么因为月钱没发而焦急？",
            "target": "贾府的经济状况逐渐衰退，入不敷出。"
        },
        {
            "q": "刘姥姥被门子拦住的深层原因是什么？",
            "target": "大观园内实行严格的封建等级制度。"
        },
        {
            "q": "平儿在凤姐面前战战兢兢说明了什么？",
            "target": "王熙凤掌管荣国府内务，手段严厉。"
        }
    ]
    
    print("\n" + "="*60)
    print("开始极限拷问 (The Ultimate Multihop Test)")
    print("="*60)
    
    for idx, query_obj in enumerate(queries, 1):
        q_text = query_obj["q"]
        target = query_obj["target"]
        
        print(f"\n[测试 {idx}] 提问: {q_text}")
        print(f"          理论应当召回的宏观父节点: {target}")
        print("-" * 60)
        
        q_inputs = embedder.tokenizer([q_text], padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            q_out = embedder.backbone(input_ids=q_inputs.input_ids, attention_mask=q_inputs.attention_mask)
            q_euc = q_out.last_hidden_state[:, 0, :]
            q_hyp = embedder.holo_math.exp_map0(embedder.projection(q_euc))
            
        # --- Flat RAG ---
        print("\n>>> 方案 A: 传统欧氏 RAG (Cosine Similarity)")
        flat_results = db_flat.search(q_euc, top_k=3)
        flat_found = False
        for rank, (score, text) in enumerate(flat_results, 1):
            if text == target: flat_found = True
            marker = "[命中]" if text == target else "      "
            print(f"{marker} Top {rank} | 相似度: {score:.4f} | {text}")
            
        # --- Holo RAG ---
        print("\n>>> 方案 B: 双曲全息 RAG (Poincaré Geodesic Distance)")
        holo_results = db_holo.search(q_hyp, top_k=3)
        holo_found = False
        for rank, (dist, text) in enumerate(holo_results, 1):
            if text == target: holo_found = True
            marker = "[命中]" if text == target else "      "
            print(f"{marker} Top {rank} | 双曲距离: {dist:.4f} | {text}")
            
        print("\n[单项结论]:")
        if flat_found and holo_found:
            print("平局：两者都召回了父节点。")
        elif not flat_found and holo_found:
            print("降维打击：只有全息 RAG 利用测地线跨越了字面障碍，成功召回了父节点！")
        elif flat_found and not holo_found:
            print("异常：传统 RAG 赢了（这通常说明微调崩了或数据没对齐）。")
        else:
            print("双输：两者都没找到，说明这个 Query 太难，超出了模型的隐式联想能力。")
        print("="*60)

if __name__ == "__main__":
    run_scale_benchmark()
