import os
import requests
import pandas as pd
from tqdm import tqdm

"""
Critical Data Recovery Script
-----------------------------
Downloads REAL fMRI data for all 17 subjects from OpenNeuro ds003171.
We use the AWS S3 public bucket endpoint for OpenNeuro.

Target files per subject:
- func/sub-XX_task-restawake_run-01_bold.nii.gz
- func/sub-XX_task-restlight_run-01_bold.nii.gz
- func/sub-XX_task-restdeep_run-01_bold.nii.gz
- func/sub-XX_task-restrecovery_run-01_bold.nii.gz
"""

DATASET_ID = "ds003171"
VERSION = "1.0.2" # Latest version
BASE_URL = f"https://s3.amazonaws.com/openneuro.org/{DATASET_ID}"

# We read participants from local file if exists, otherwise hardcode
try:
    df = pd.read_csv('participants.tsv', sep='\t')
    subjects = df['participant_id'].tolist()
except:
    # Fallback list from inspection
    subjects = [
        "sub-02CB", "sub-04HD", "sub-04SG", "sub-08BC", "sub-08VR",
        "sub-10JR", "sub-13CA", "sub-16RA", "sub-17EK", "sub-17NA",
        "sub-19AK", "sub-19SA", "sub-22CY", "sub-22TK", "sub-25JK", # Typo in tsv? sub-2525JK -> sub-25JK likely
        "sub-27HF", "sub-30AQ"
    ]

# Typo fix for sub-2525JK if present
subjects = [s if s != "sub-2525JK" else "sub-25JK" for s in subjects]

TASKS = ["restawake", "restlight", "restdeep", "restrecovery"]
RUN = "01"

def download_file(url, local_path):
    if os.path.exists(local_path):
        # Check size to ensure not empty/corrupt? 
        if os.path.getsize(local_path) > 1000:
            print(f"  [Skip] Exists: {local_path}")
            return True
            
    print(f"  [Down] {url}")
    try:
        # Stream download
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            with open(local_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        return True
    except Exception as e:
        print(f"  [Fail] {e}")
        return False

def main():
    print(f"Starting FULL download for {len(subjects)} subjects...")
    print(f"Target: {os.path.abspath('./data/propofol_dataset')}")
    
    success_count = 0
    total_files = len(subjects) * len(TASKS)
    
    for sub in tqdm(subjects, desc="Subjects"):
        for task in TASKS:
            # Construct filename
            # Naming convention: sub-02CB_task-restawake_run-01_bold.nii.gz
            # Note: Version 1.0.2 structure on S3 might be flat or versioned.
            # AWS path: ds003171/sub-02CB/func/...
            
            fname = f"{sub}_task-{task}_run-{RUN}_bold.nii.gz"
            s3_key = f"{sub}/func/{fname}"
            url = f"{BASE_URL}/{s3_key}"
            
            local_path = f"data/propofol_dataset/{sub}/func/{fname}"
            
            if download_file(url, local_path):
                success_count += 1
                
    print(f"\nDownload Complete. Success: {success_count}/{total_files}")
    print("Now run 'python code/real_brain_data_analysis.py' for GENUINE results.")

if __name__ == "__main__":
    main()
