from huggingface_hub import snapshot_download
import os

# --- 加速配置 ---
# 1. 使用镜像站 (对于国内网络环境非常有效)
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# 2. 启用 HF_TRANSFER (需要安装: pip install hf_transfer)
#    这是一个基于 Rust 的高速下载器
#    注意：如果遇到 "cannot find the appropriate snapshot folder" 错误，请尝试注释掉下面这行
# os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
# ----------------

# 模型ID
model_id = "state-spaces/mamba2-1.3b"
# 本地保存路径
local_dir = os.path.join(os.getcwd(), "models", "mamba2-1.3b")

print(f"准备下载模型: {model_id}")
print(f"保存路径: {local_dir}")

# 确保目录存在
os.makedirs(local_dir, exist_ok=True)

try:
    # 下载模型
    # resume_download=True 支持断点续传
    # local_dir_use_symlinks=False 确保下载的是实际文件而不是符号链接（在Windows上这通常更好）
    snapshot_download(
        repo_id=model_id, 
        local_dir=local_dir, 
        local_dir_use_symlinks=False,
        resume_download=True
    )
    print("\n下载完成！")
except Exception as e:
    print(f"\n下载过程中出错: {e}")
    print("请确保已安装 huggingface_hub: pip install huggingface_hub")
