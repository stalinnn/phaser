import os
from beir import util

def download_msmarco():
    dataset = "msmarco"
    out_dir = "/gz-data/datasets/beir"
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"Starting download of {dataset} to {out_dir}...")
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
    
    try:
        data_path = util.download_and_unzip(url, out_dir)
        print(f"Successfully downloaded and unzipped MS MARCO to: {data_path}")
    except Exception as e:
        print(f"Failed to download {dataset}. Error: {e}")

if __name__ == "__main__":
    download_msmarco()
