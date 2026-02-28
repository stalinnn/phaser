import requests
import os
import zipfile
import io

def download_ibm_double_pendulum():
    # IBM Double Pendulum Chaotic Dataset (Official Link from DAX)
    # 这是一个公开数据集，通常可以直接下载
    # URL 可能会变，这里使用一个常用的学术镜像或直接链接
    url = "https://dax-cdn.cdn.appdomain.cloud/dax-double-pendulum/1.0.0/double-pendulum.tar.gz"
    
    save_dir = "double_pendulum_data"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    print(f"Downloading dataset from {url}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # 解压
        print("Extracting...")
        import tarfile
        with tarfile.open(fileobj=io.BytesIO(response.content), mode="r:gz") as tar:
            tar.extractall(path=save_dir)
            
        print(f"Dataset downloaded and extracted to: {save_dir}")
        print("Please upload the CSV files in this directory to your cloud platform.")
        
    except Exception as e:
        print(f"Download failed: {e}")
        print("Try visiting: https://developer.ibm.com/exchanges/data/all/double-pendulum-chaotic/")

if __name__ == "__main__":
    download_ibm_double_pendulum()