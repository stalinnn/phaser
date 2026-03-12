import os
import urllib.request

BASE_URL = "https://s3.amazonaws.com/openneuro.org/ds003171"

def download(url, path):
    print(f"Downloading {url}...")
    try:
        urllib.request.urlretrieve(url, path)
        print(f"Saved to {path}")
    except Exception as e:
        print(f"Failed: {e}")

# Try to download the scans list for sub-02CB
# This file lists all .nii.gz files for the subject
target = "sub-02CB/sub-02CB_scans.tsv"
url = f"{BASE_URL}/{target}"
local = "code/sub-02CB_scans.tsv"

download(url, local)

# Also try session level if exists
target2 = "sub-02CB/ses-01/sub-02CB_ses-01_scans.tsv"
url2 = f"{BASE_URL}/{target2}"
local2 = "code/sub-02CB_ses-01_scans.tsv"

download(url2, local2)
