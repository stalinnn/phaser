import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from datasets import load_dataset
try:
    ds = load_dataset('wikitext', 'wikitext-103-v1', split='train', streaming=True)
    print("Successfully connected via HF Mirror!")
except Exception as e:
    print(f"Connection failed: {e}")
