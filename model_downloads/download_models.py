# -*- coding: utf-8 -*-
"""
使用魔搭 ModelScope 下载 Mamba-7B、Qwen-7B、Llama-8B。
魔搭国内访问稳定，无需镜像。未在魔搭上的模型会回退到 HF-Mirror。
"""
import os
import time

# 魔搭提速：拉长单文件超时、提高并发
os.environ.setdefault("MODELSCOPE_DOWNLOAD_PARALLELS", "8")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# 7B/8B 大模型（不需要可注释）
MODELS = [
    # ("tiiuae/Falcon3-Mamba-7B-Instruct", "Falcon3-Mamba-7B-Instruct"),
    # ("Qwen/Qwen2-7B", "Qwen2-7B"),
    # ("meta-llama/Meta-Llama-3.1-8B", "Meta-Llama-3.1-8B"),
]

# 约 1B 小模型：Mamba 790M / Mamba2 1.3B、Qwen 1.5B、Llama 3.2 1B
MODELS_1B = [
    ("state-spaces/mamba-790m-hf", "Mamba-790M"),
    ("state-spaces/mamba2-1.3b", "Mamba2-1.3B"),           # 官方 Mamba2 1.3B
    ("Qwen/Qwen2.5-1.5B-Instruct", "Qwen2.5-1.5B-Instruct"),
    ("meta-llama/Llama-3.2-1B", "Llama-3.2-1B"),
]

# 魔搭 model_id（None 表示只用 HF-Mirror）
MODELSCOPE_IDS = {
    "tiiuae/Falcon3-Mamba-7B-Instruct": None,
    "Qwen/Qwen2-7B": "Qwen/Qwen2-7b",
    "meta-llama/Meta-Llama-3.1-8B": "LLM-Research/Meta-Llama-3.1-8B-Instruct",
    "state-spaces/mamba-790m-hf": None,
    "state-spaces/mamba2-1.3b": None,
    "Qwen/Qwen2.5-1.5B-Instruct": "Qwen/Qwen2.5-1.5B-Instruct",
    "meta-llama/Llama-3.2-1B": "LLM-Research/Llama-3.2-1B",
}


def download_via_modelscope(model_id: str, local_dir: str) -> bool:
    """用魔搭下载，返回是否成功。拉长超时、提高并发以减少 Read timed out 与提速。"""
    try:
        import modelscope.hub.constants as _ms_constants
        _ms_constants.API_FILE_DOWNLOAD_TIMEOUT = 3600  # 单文件 1 小时，避免 15GB 大文件 60s 超时
        from modelscope import snapshot_download as ms_download
    except ImportError:
        print("  未安装 modelscope，跳过魔搭。请运行: pip install modelscope")
        return False
    try:
        ms_download(model_id, local_dir=local_dir)
        return True
    except Exception as e:
        print(f"  魔搭下载失败: {e}")
        return False


def download_via_hf_mirror(repo_id: str, local_dir: str) -> bool:
    """用 HF-Mirror 逐个文件下载。"""
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "600")
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError:
        print("  未安装 huggingface_hub。请运行: pip install huggingface_hub")
        return False
    api = HfApi(endpoint=os.environ["HF_ENDPOINT"])
    try:
        files = api.list_repo_files(repo_id=repo_id)
        files = [f for f in files if not f.startswith(".git")]
    except Exception as e:
        print(f"  HF 获取文件列表失败: {e}")
        return False
    for i, f in enumerate(files, 1):
        print(f"  [{i}/{len(files)}] {f} ...")
        for attempt in range(10):
            try:
                hf_hub_download(repo_id=repo_id, filename=f, local_dir=local_dir)
                print(f"      -> 完成")
                break
            except Exception as e:
                if "cannot find the requested files" in str(e).lower() and ("meta-llama" in repo_id.lower() or "llama" in repo_id.lower()):
                    print("  [跳过] Llama 为受保护模型时镜像可能不提供。需 VPN + huggingface.co 登录后直连下载。")
                    return False
                print(f"      -> [重试 {attempt+1}/10] {str(e).split('(')[0]}")
                time.sleep(3)
        else:
            print(f"      -> 失败")
            return False
    return True


def main():
    all_models = MODELS + MODELS_1B
    print("=========== 下载源：魔搭 ModelScope（国内稳定），缺失时回退 HF-Mirror ===========\n")
    for repo_id, local_name in all_models:
        local_dir = os.path.join(MODELS_DIR, local_name)
        os.makedirs(local_dir, exist_ok=True)
        ms_id = MODELSCOPE_IDS.get(repo_id)
        print(f"[模型] {local_name} -> {local_dir}")

        if ms_id:
            print(f"  尝试魔搭: {ms_id}")
            if download_via_modelscope(ms_id, local_dir):
                print(f"  [完成] {local_name}\n")
                continue
            print("  魔搭未命中，改用 HF-Mirror ...")
        else:
            print("  使用 HF-Mirror（魔搭无此模型）...")

        if download_via_hf_mirror(repo_id, local_dir):
            print(f"  [完成] {local_name}\n")
        else:
            print(f"  [未完成] {local_name}\n")
    print("=========== 全部处理完毕 ===========")


if __name__ == "__main__":
    main()
