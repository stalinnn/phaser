import os
import urllib.request
from tqdm import tqdm

"""
Script to download minimal necessary data from OpenNeuro ds003171 (AWS S3)
Target: Subject sub-02CB
Files: 4 functional runs (Awake, Light, Deep, Recovery)
"""

# Base URL for OpenNeuro ds003171 on AWS S3
# Version 2.0.1
BASE_URL = "https://s3.amazonaws.com/openneuro.org/ds003171"

def download_file(url, local_path):
    if os.path.exists(local_path):
        print(f"Skipping (exists): {local_path}")
        return

    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    
    print(f"Downloading: {url}")
    print(f"To: {local_path}")
    
    try:
        with tqdm(unit='B', unit_scale=True, unit_divisor=1024, miniters=1, desc=os.path.basename(local_path)) as t:
            def update_to(b=1, bsize=1, tsize=None):
                if tsize is not None:
                    t.total = tsize
                t.update(b * bsize - t.n)
                
            urllib.request.urlretrieve(url, local_path, reporthook=update_to)
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        # Clean up partial file
        if os.path.exists(local_path):
            os.remove(local_path)

def run_download():
    # Target Subject
    sub = "sub-02CB"
    
    # We need functional data (BOLD)
    # The filenames usually follow BIDS format
    # Based on README/file structure inspection
    
    # Updated file list based on standard OpenNeuro BIDS structure for ds003171
    # We need to guess the exact filename. Usually:
    # sub-02CB_task-rest_condition-awake_bold.nii.gz
    # Let's try to list or just try standard naming.
    
    # Correct names from dataset browser:
    # sub-02CB_task-rest_acq-epi_run-01_bold.nii.gz ??
    # No, the README says "task-rest_condition-awake" etc.
    # Let's try the explicit names found in similar datasets.
    
    # Actually, S3 listing is tricky without awscli.
    # Let's try the specific paths.
    
    conditions = ['awake', 'light', 'deep', 'recovery']
    
    # Local directory to save
    save_dir = "./data/propofol_dataset"
    
    files_to_download = []
    
    for cond in conditions:
        # Construct BIDS filename
        # Note: Sometimes run-1 is included, sometimes not.
        # Let's try the most likely format for this dataset.
        filename = f"{sub}_task-rest_condition-{cond}_bold.nii.gz"
        remote_path = f"sub-02CB/func/{filename}"
        
        # S3 Path
        url = f"{BASE_URL}/{remote_path}"
        local_path = f"{save_dir}/{sub}/func/{filename}"
        
        files_to_download.append((url, local_path))
        
    print(f"Starting download for Subject {sub}...")
    print("Note: These files are large (~200MB each). Please wait.")
    
    for url, local_path in files_to_download:
        download_file(url, local_path)
        
    print("\nDownload complete!")

if __name__ == "__main__":
    run_download()
