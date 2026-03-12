import requests
import json

def get_snapshot_files(dataset_id, tag):
    """
    Query OpenNeuro GraphQL API to get file tree.
    """
    url = 'https://openneuro.org/crn/graphql'
    query = """
    query getSnapshot($datasetId: ID!, $tag: String!) {
      snapshot(datasetId: $datasetId, tag: $tag) {
        files {
          id
          filename
          size
          urls
        }
      }
    }
    """
    
    variables = {
        "datasetId": dataset_id,
        "tag": tag
    }
    
    response = requests.post(url, json={'query': query, 'variables': variables})
    if response.status_code != 200:
        print("Error querying API")
        return []
        
    data = response.json()
    if 'data' not in data or 'snapshot' not in data['data']:
        print("Invalid response structure")
        print(data)
        return []
        
    return data['data']['snapshot']['files']

def find_func_files():
    ds_id = "ds003171"
    tag = "2.0.1" # The version we saw in dataset_description.json
    
    print(f"Fetching file list for {ds_id} v{tag}...")
    files = get_snapshot_files(ds_id, tag)
    
    print(f"Found {len(files)} files total.")
    
    # Filter for functional files of sub-02CB
    target_sub = "sub-02CB"
    func_files = [f for f in files if target_sub in f['filename'] and "bold.nii.gz" in f['filename']]
    
    print("\n--- Functional Files for sub-02CB ---")
    for f in func_files:
        print(f"Filename: {f['filename']}")
        # print(f"URL: {f['urls'][0]}") # URL might be internal, use filename construction
        
    # Generate a download script content based on findings
    if func_files:
        print("\n--- Download Commands ---")
        base_url = "https://s3.amazonaws.com/openneuro.org/ds003171"
        for f in func_files:
             # The API filename includes the path, e.g. "sub-02CB/func/..."
             # S3 URL is base + "/" + filename
             print(f"python code/download_direct.py \"{base_url}/{f['filename']}\" \"data/propofol_dataset/{f['filename']}\"")

if __name__ == "__main__":
    find_func_files()
