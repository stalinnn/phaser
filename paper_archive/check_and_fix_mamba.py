import torch
import os
import subprocess
import sys

def run_cmd(cmd):
    print(f"Running: {cmd}")
    try:
        result = subprocess.run(cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error: {e.stderr}")

def check_mamba():
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device Name: {torch.cuda.get_device_name(0)}")

    print("\n--- Checking Mamba Installation ---")
    try:
        import mamba_ssm
        print(f"Mamba-SSM Version: {mamba_ssm.__version__}")
        from mamba_ssm import Mamba
        
        # Micro Benchmark
        print("Running Mamba Forward/Backward Check...")
        device = "cuda"
        model = Mamba(d_model=64, d_state=16, d_conv=4, expand=2).to(device)
        x = torch.randn(2, 128, 64).to(device)
        y = model(x)
        loss = y.sum()
        loss.backward()
        print("✅ SUCCESS: CUDA Mamba is working!")
        return True
        
    except Exception as e:
        print(f"❌ FAILURE: {e}")
        return False

def suggest_fix():
    print("\n--- SUGGESTED FIX ---")
    print("Your environment seems to be PyTorch 2.3.0 + CUDA 12.1.")
    print("The error 'torch.library' usually means 'causal-conv1d' was compiled for an older PyTorch.")
    
    print("\nPlease run the following commands to reinstall compatible versions:")
    print("-" * 60)
    print("pip uninstall -y mamba_ssm causal_conv1d")
    print("# Force reinstall from source or prebuilt wheel matching torch 2.3")
    print("pip install causal-conv1d>=1.2.0 mamba-ssm>=1.2.0 --no-cache-dir")
    print("-" * 60)

if __name__ == "__main__":
    success = check_mamba()
    if not success:
        suggest_fix()
