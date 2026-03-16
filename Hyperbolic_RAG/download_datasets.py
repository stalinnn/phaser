import os
import requests
from datasets import load_dataset

def download_file(url, save_path):
    print(f"Downloading {url} to {save_path}...")
    if os.path.exists(save_path):
        print(f"{save_path} already exists. Skipping.")
        return
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Done.")
    except Exception as e:
        print(f"Failed to download {url}: {e}")

def main():
    base_dir = os.path.join(os.path.dirname(__file__), "datasets")
    os.makedirs(base_dir, exist_ok=True)
    
    print("=== 开始下载顶会级 RAG 实验所需的数据集 ===")
    
    # 1. HotpotQA
    hotpot_dir = os.path.join(base_dir, "HotpotQA")
    os.makedirs(hotpot_dir, exist_ok=True)
    # 下载 dev 集 (更常用于评估)
    download_file("http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json", os.path.join(hotpot_dir, "hotpot_dev_distractor_v1.json"))
    # 下载 train 集 (数据较大，视情况下载)
    download_file("http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json", os.path.join(hotpot_dir, "hotpot_train_v1.1.json"))
    
    # 2. 2WikiMultihopQA (经典多跳推理数据集)
    print("\nDownloading 2WikiMultihopQA raw files...")
    try:
        wiki_dir = os.path.join(base_dir, "2WikiMultihopQA")
        os.makedirs(wiki_dir, exist_ok=True)
        # Using the official Dropbox links provided by the authors of 2WikiMultihopQA
        download_file("https://www.dropbox.com/s/72ym6t70r9sw2a3/train.json?dl=1", os.path.join(wiki_dir, "train.json"))
        download_file("https://www.dropbox.com/s/d8i071tdv03a89r/dev.json?dl=1", os.path.join(wiki_dir, "dev.json"))
    except Exception as e:
        print(f"Failed to download 2WikiMultihopQA: {e}")

    # 3. MultiHop-RAG (最新的多跳 RAG 专用数据集)
    print("\nDownloading MultiHop-RAG raw files via HuggingFace Hub...")
    try:
        from huggingface_hub import hf_hub_download
        mhr_dir = os.path.join(base_dir, "MultiHopRAG")
        os.makedirs(mhr_dir, exist_ok=True)
        
        qa_file = hf_hub_download(repo_id="yixuantt/MultiHopRAG", filename="MultiHopRAG.json", repo_type="dataset")
        corpus_file = hf_hub_download(repo_id="yixuantt/MultiHopRAG", filename="corpus.json", repo_type="dataset")
        
        import shutil
        shutil.copy(qa_file, os.path.join(mhr_dir, "MultiHopRAG.json"))
        shutil.copy(corpus_file, os.path.join(mhr_dir, "corpus.json"))
        
        print("MultiHop-RAG downloaded successfully.")
    except Exception as e:
        print(f"Failed to download MultiHop-RAG: {e}")
        
    print("\n=== 数据集下载任务完成 ===")

if __name__ == "__main__":
    main()
