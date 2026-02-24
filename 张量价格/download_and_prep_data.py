import requests
import gzip
import pandas as pd
import networkx as nx
import numpy as np
import os
import shutil
import pickle

DATA_URL = "https://snap.stanford.edu/data/soc-sign-bitcoinalpha.csv.gz"
RAW_FILE = "real_data/soc-sign-bitcoinalpha.csv.gz"
CLEAN_FILE = "real_data/bitcoin_alpha_graph.gpickle" # Saving as NetworkX object for speed

def download_data():
    if os.path.exists(CLEAN_FILE):
        print(f"Data already processed at {CLEAN_FILE}. Skipping download.")
        return

    if not os.path.exists(RAW_FILE):
        print(f"Downloading data from {DATA_URL}...")
        try:
            response = requests.get(DATA_URL, stream=True)
            if response.status_code == 200:
                with open(RAW_FILE, 'wb') as f:
                    response.raw.decode_content = True
                    shutil.copyfileobj(response.raw, f)
                print("Download complete.")
            else:
                print(f"Failed to download. Status code: {response.status_code}")
                return
        except Exception as e:
            print(f"Error downloading: {e}")
            return

    process_data()

def process_data():
    print("Processing raw data...")
    # The file is CSV-like: SOURCE, TARGET, RATING, TIME
    try:
        with gzip.open(RAW_FILE, 'rt') as f:
            df = pd.read_csv(f, names=['source', 'target', 'rating', 'time'])
        
        print(f"Raw edges: {len(df)}")
        
        def rating_to_weight(r):
            if r <= 0:
                return 1e-4 
            else:
                return r / 10.0 
        
        df['weight'] = df['rating'].apply(rating_to_weight)
        
        G = nx.from_pandas_edgelist(df, 'source', 'target', ['weight', 'rating'])
        
        largest_cc = max(nx.connected_components(G), key=len)
        G_giant = G.subgraph(largest_cc).copy()
        
        G_giant = nx.convert_node_labels_to_integers(G_giant)
        
        print(f"Giant Component: {G_giant.number_of_nodes()} nodes, {G_giant.number_of_edges()} edges.")
        
        # Save
        with open(CLEAN_FILE, 'wb') as f:
            pickle.dump(G_giant, f)
        print(f"Graph saved to {CLEAN_FILE}")
        
    except Exception as e:
        print(f"Error processing data: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    download_data()
