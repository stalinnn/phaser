import os
import urllib.request
import pandas as pd
from tqdm import tqdm

"""
Script to download ALL subjects from OpenNeuro ds003171
"""

BASE_URL = "https://s3.amazonaws.com/openneuro.org/ds003171"
SAVE_DIR = "./data/propofol_dataset"

def download_file(url, local_path):
    if os.path.exists(local_path):
        print(f"Skipping (exists): {local_path}")
        return

    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    # print(f"Downloading {url}")
    
    try:
        # Simple retrieval
        urllib.request.urlretrieve(url, local_path)
        print(f"  [OK] Saved to {local_path}")
    except Exception as e:
        print(f"  [Error] {url}: {e}")

def run_full_download():
    # 1. Read participants
    if not os.path.exists('participants.tsv'):
        print("Error: participants.tsv not found.")
        return
        
    df = pd.read_csv('participants.tsv', sep='\t')
    subjects = df['participant_id'].tolist()
    
    print(f"Found {len(subjects)} subjects: {subjects}")
    
    conditions = ['restawake', 'restlight', 'restdeep', 'restrecovery']
    
    total_files = len(subjects) * len(conditions)
    print(f"Starting download of {total_files} files...")
    
    for sub in tqdm(subjects):
        for cond in conditions:
            # Filename format: sub-02CB_task-restawake_run-01_bold.nii.gz
            # Note: The run number might vary? Assuming run-01 for now.
            # If run-01 fails, we might need run-02, but run-01 is standard.
            filename = f"{sub}_task-{cond}_run-01_bold.nii.gz"
            rel_path = f"{sub}/func/{filename}"
            
            url = f"{BASE_URL}/{rel_path}"
            local_path = f"{SAVE_DIR}/{rel_path}"
            
            download_file(url, local_path)

if __name__ == "__main__":
    run_full_download()
