# Scale-Up Holo-RAG: 从玩具模型到顶会标准的演进

本文档说明了 `Scale_Up_Holo_RAG` 目录在近期进行的重大升级。我们将系统的验证对象从人工编写的少量中文数据（以《红楼梦》为背景）切换到了真实的、国际学术界公认的大规模英文多跳推理数据集（HotpotQA），并彻底离线化以解决国内网络环境下的运行问题。

## 核心演进总结

**旧版 (玩具测试版) $\rightarrow$ 新版 (顶会打榜版)**
*   **数据集**: 人工编写的约100条中文造句 $\rightarrow$ 官方 HotpotQA 提取的约 20,000 条真实英文多跳推理对。
*   **底层模型**: `BAAI/bge-small-zh-v1.5` (需实时联网检查) $\rightarrow$ `BAAI/bge-small-en-v1.5` (纯本地化部署，防卡死)。
*   **训练规模**: 极小批量，验证 Loss 能否下降 $\rightarrow$ 近两万条数据，让双曲空间真正学到自然语言的层次物理结构。
*   **评估方式**: 零样本 (Zero-Shot) $\rightarrow$ 加载微调权重后进行打榜对比 (Ablation Study)。

---

## 关键文件更新详解

### 1. `download_datasets.py` (新增脚本)
**用途**：用于从官方渠道或 HuggingFace 自动下载论文所需的标准测试集。
*   下载了 `HotpotQA` (多跳推理基准)。
*   支持下载 `2WikiMultihopQA` 和 `MultiHop-RAG` (供后续扩展实验使用)。

### 2. `generate_hotpotqa_pairs.py` (新增脚本)
**用途**：自动构建用于双曲空间训练的“父子”节点对。
*   **逻辑机制**：将 HotpotQA 中的复杂 `Question`（问题）视作**宏观父节点**，将支撑答案的真实 `Context`（支持性证据句子）视作**微观子节点**。
*   **正负样本**：生成 `Label=1` 的正样本对，同时从同一问题的干扰项中抽取内容生成 `Label=-1` 的硬负样本 (Hard Negatives)。
*   **产出**：生成了包含近 20,000 条数据的 `hotpotqa_train_pairs.json`。这比旧版 `generate_dataset.py` (生成 `hierarchical_dataset.json`) 规模大了数百倍且更为真实。

### 3. `holo_embedder.py` (架构更新)
**用途**：核心全息嵌入器类。
*   **更新**：将其默认初始化的骨干模型路径从 HuggingFace 远程路径 `BAAI/bge-small-zh-v1.5` 修改为了本地绝对路径 `C:/Users/.../model_downloads/models/bge-small-en-v1.5`。
*   **意义**：彻底阻断了 `transformers` 库在国内网络下因为超时而导致脚本挂死/卡死的问题。实现了完全离线的实例化。

### 4. `batched_finetune.py` (训练规模化)
**用途**：执行双曲投影层的微调训练。
*   **切换语言环境**：指定加载本地的英文模型 `bge-small-en-v1.5`。
*   **切换数据集**：将 `DataLoader` 指向新生成的 `hotpotqa_train_pairs.json`。
*   **超参调整**：由于数据量激增，将训练轮数 (Epochs) 从 300 轮缩减至 30 轮，并调整了学习率衰减的时机。
*   **输出**：训练完成后将生成正式的微调权重 `scale_holo_projection_hotpotqa.pt`。

### 5. `hotpotqa_benchmark.py` (评估升级)
**用途**：对模型进行打榜评估，计算 MRR 与 Recall@K。
*   **更新**：添加了在初始化 `embedder` 后，尝试加载微调权重 `scale_holo_projection_hotpotqa.pt` 的代码块。
*   **意义**：在这之前，脚本只能测试 Zero-Shot 性能（即未经专属 `poincare_margin_loss` 建立起树状因果拓扑的随机状态），这正是为何之前的测试中双曲模型略逊于平直 Euclidean RAG。新版将展示微调后的“状态突变”。

---

## 实验逻辑链条与论文价值 (Ablation Study)

这一系列的升级，旨在闭环完成论文中最为关键的消融实验（Ablation Study）：

1.  **阶段一 (Zero-Shot)**：直接将双曲几何强加于预训练模型上而不加训练，性能不如普通向量检索 (已被旧版脚本验证)。这证明了双曲空间虽然具有树状潜力，但模型需要被“教会”如何利用它。
2.  **阶段二 (Scale-Up Fine-Tuning)**：利用 `batched_finetune.py` 与专属的 `poincare_margin_loss`，让模型在大规模真实数据 (`hotpotqa_train_pairs.json`) 中学习“宏观-微观”、“问题-证据”的向心与离心力物理法则。
3.  **阶段三 (Benchmark)**：微调后再次运行 `hotpotqa_benchmark.py`。预期的结果是各项检索指标（MRR, Recall）大幅跃升，甚至碾压纯平直空间 Baseline。

这一完整的链路，将是顶会论文中最坚实的实验证据。