#!/usr/bin/env python3
"""
下载最新 Mamba 7B、Qwen-7B、Llama-8B 到本地。
使用 Hugging Face Hub，需先安装: pip install huggingface_hub

用法:
  python download_llm_models.py                    # 下载全部到 ./models
  python download_llm_models.py --model mamba7b   # 只下载 Mamba 7B
  python download_llm_models.py --model qwen7b --dir D:/models
  python download_llm_models.py --alt             # 使用替代模型（见下）

注意:
- Meta Llama 需在 https://huggingface.co/meta-llama/Llama-3.1-8B 同意许可并登录 HF。
- 大文件若超时，直接再次运行同一命令可断点续传。
- 国内可设镜像: set HF_ENDPOINT=https://hf-mirror.com
"""

import argparse
import os
from pathlib import Path

# 最新/常用模型 ID（Hugging Face）
MODELS = {
    "mamba7b": "tiiuae/falcon-mamba-7b",           # Falcon Mamba 7B（首个强 attention-free 7B）
    "qwen7b": "Qwen/Qwen2.5-7B",                   # Qwen2.5-7B（比 Qwen-7B 更新）
    "llama8b": "meta-llama/Llama-3.1-8B",          # Llama 3.1 8B（需在 HF 同意许可）
}

# 可选替代
ALT_MODELS = {
    "mamba7b_alt": "TRI-ML/mamba-7b-rw",           # 纯 Mamba 7B baseline
    "qwen7b_alt": "Qwen/Qwen-7B",                  # 原始 Qwen-7B
    "llama8b_alt": "meta-llama/Meta-Llama-3-8B",   # Llama 3 8B
}


def download_model(repo_id: str, local_dir: str, token: str = None):
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise SystemExit("请先安装: pip install huggingface_hub")

    token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    print(f"正在下载: {repo_id} -> {local_dir}")
    path = snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        token=token,
    )
    print(f"已保存到: {path}")
    return path


def main():
    p = argparse.ArgumentParser(description="下载 Mamba 7B / Qwen-7B / Llama-8B")
    p.add_argument("--model", choices=list(MODELS.keys()) + ["all"], default="all",
                   help="要下载的模型: mamba7b, qwen7b, llama8b, 或 all")
    p.add_argument("--dir", type=str, default="./models",
                   help="保存目录，各模型会放在子文件夹中 (默认: ./models)")
    p.add_argument("--token", type=str, default=None,
                   help="Hugging Face token（可选，也可用 HF_TOKEN 环境变量）")
    p.add_argument("--alt", action="store_true",
                   help="使用替代模型（mamba: TRI-ML/mamba-7b-rw, qwen: Qwen-7B, llama: Meta-Llama-3-8B）")
    args = p.parse_args()

    base = Path(args.dir)
    base.mkdir(parents=True, exist_ok=True)

    if args.alt:
        mapping = {
            "mamba7b": ALT_MODELS["mamba7b_alt"],
            "qwen7b": ALT_MODELS["qwen7b_alt"],
            "llama8b": ALT_MODELS["llama8b_alt"],
        }
    else:
        mapping = MODELS

    if args.model == "all":
        to_download = list(mapping.items())
    else:
        to_download = [(args.model, mapping[args.model])]

    for name, repo_id in to_download:
        local_dir = str(base / name)
        try:
            download_model(repo_id, local_dir, args.token)
        except Exception as e:
            print(f"下载失败 {repo_id}: {e}")
            if "llama" in name.lower():
                print("  -> Llama 需在 Hugging Face 同意许可并登录: https://huggingface.co/meta-llama/Llama-3.1-8B")


if __name__ == "__main__":
    main()
