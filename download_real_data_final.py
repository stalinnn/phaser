import os
from datasets import load_dataset

def download_and_save():
    print("Downloading WikiText-103 locally...")
    # Download using local network (assuming you have VPN/Proxy if in CN)
    # If this fails locally, try setting HF_ENDPOINT=https://hf-mirror.com in your local terminal
    ds = load_dataset("wikitext", "wikitext-103-v1")
    
    save_path = "./data_cache/wikitext"
    os.makedirs(save_path, exist_ok=True)
    
    print(f"Saving to {save_path}...")
    ds.save_to_disk(save_path)
    print("Done! Now upload the 'data_cache' folder to your server.")

if __name__ == "__main__":
    download_and_save()
