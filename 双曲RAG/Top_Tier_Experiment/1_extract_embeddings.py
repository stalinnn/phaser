import os
import json
import torch
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

def main():
    print("Loading BGE-M3 model...")
    # Use BAAI/bge-m3. We use GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer("BAAI/bge-m3", device=device)
    
    print("Loading HotpotQA dataset (validation split)...")
    # HotpotQA validation set has around 7405 examples. We will use a subset for quick experimentation.
    dataset = load_dataset("hotpot_qa", 'distractor', split="validation")
    
    # We will sample 1000 queries to make it tractable for the demo but still robust
    num_queries = len(dataset)
    print(f"Using full validation split: {num_queries} queries.")
    # dataset = dataset.select(range(num_queries))
    
    queries = []
    contexts = []
    q_to_c_positives = {}
    
    print("Processing dataset...")
    for i, item in enumerate(tqdm(dataset)):
        question = item['question']
        queries.append({"id": f"q_{i}", "text": question})
        
        # Extract supporting facts
        supporting_titles = item['supporting_facts']['title']
        
        pos_context_ids = []
        for title, sentences in zip(item['context']['title'], item['context']['sentences']):
            text = f"Title: {title}. " + " ".join(sentences)
            c_id = f"c_{len(contexts)}"
            contexts.append({"id": c_id, "text": text})
            
            if title in supporting_titles:
                pos_context_ids.append(c_id)
                
        q_to_c_positives[f"q_{i}"] = pos_context_ids

    print(f"Total queries: {len(queries)}")
    print(f"Total contexts (chunks): {len(contexts)}")
    
    # Extract embeddings
    print("Extracting query embeddings...")
    q_texts = [q['text'] for q in queries]
    q_embs = model.encode(q_texts, batch_size=256, show_progress_bar=True, convert_to_numpy=True)
    
    print("Extracting context embeddings...")
    c_texts = [c['text'] for c in contexts]
    c_embs = model.encode(c_texts, batch_size=256, show_progress_bar=True, convert_to_numpy=True)
    
    # Save data
    out_dir = "/phaser/Hyperbolic_RAG/Top_Tier_Experiment/data"
    os.makedirs(out_dir, exist_ok=True)
    
    np.save(os.path.join(out_dir, "q_embs.npy"), q_embs)
    np.save(os.path.join(out_dir, "c_embs.npy"), c_embs)
    
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump({
            "queries": queries,
            "contexts": contexts,
            "q_to_c_positives": q_to_c_positives
        }, f, ensure_ascii=False, indent=2)

    print(f"Data extraction complete! Saved to {out_dir}")

if __name__ == "__main__":
    main()
