# 大模型下载：Mamba-7B、Qwen-7B、Llama-8B

本目录用于将以下模型下载到本地 `models/` 目录。**默认使用魔搭 ModelScope（国内稳定）**，魔搭没有的模型自动回退到 HF-Mirror。

| 模型 | 魔搭 ModelScope | HF 回退 | 说明 |
|------|-----------------|---------|------|
| **Mamba 7B** | — | `tiiuae/Falcon3-Mamba-7B-Instruct` | Falcon3-Mamba 7B Instruct，32K 上下文 |
| **Qwen 7B** | `Qwen/Qwen2-7b` | `Qwen/Qwen2-7B` | 阿里 Qwen2 7B 基座 |
| **Llama 8B** | `LLM-Research/Meta-Llama-3.1-8B-Instruct` | `meta-llama/Meta-Llama-3.1-8B` | Meta Llama 3.1 8B Instruct |

## 环境与依赖

```bash
pip install -r requirements.txt
```

依赖：`modelscope`（魔搭）、`huggingface_hub`（回退 HF-Mirror 时使用）。

## 下载步骤

### 1. 安装依赖

```powershell
cd model_downloads
pip install -r requirements.txt
```

### 2. 运行下载

```powershell
python download_models.py
```

- **Qwen2-7B、Llama-3.1-8B**：优先从魔搭下载，国内无需镜像或 VPN。
- **Falcon3-Mamba-7B**：魔搭暂无，自动用 HF-Mirror 拉取。
- 若某模型在魔搭下载失败，会自动尝试 HF-Mirror。

### 3. 仅下载其中一个

编辑 `download_models.py` 中的 `MODELS` 列表，只保留需要的一项后重新运行。

### 3. 仅下载其中某一个

可编辑 `download_models.py` 中的 `MODELS` 列表，只保留需要的一项，再运行上述命令。

## 磁盘与网络

- 三个模型合计约 **20–30GB+**，请确保磁盘空间和稳定网络。
- 若下载中断，再次运行脚本会续传（已存在的文件会跳过）。

## 使用下载后的模型

以 Transformers 为例，指定本地路径即可：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# 例如使用本地 Qwen2-7B
model_path = "model_downloads/models/Qwen2-7B"
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_path)
```

Mamba 类模型可能需要 `trust_remote_code=True` 及额外依赖（如 `mamba-ssm`），请参考各模型在 Hugging Face 上的 README。
