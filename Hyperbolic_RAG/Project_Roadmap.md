# Hyperbolic RAG (全息检索增强系统) 工程路线图

**项目代号**：Project Holo-RAG
**核心目标**：构建世界上首个能够原生理解“树状/层级因果逻辑”的检索增强系统，在极端长文本（如《红楼梦》、复杂财报）推理任务上碾压现有的平直空间 RAG 和 GraphRAG。

---

## 阶段一：数学底座与算子开发 (Week 1)
*目标：打破平直空间的诅咒，建立庞加莱盘 (Poincaré Ball) 流形的操作环境。*

**核心任务：**
1. **庞加莱算子库开发 (`holo_math.py`)**
   * 实现庞加莱球模型的基本度规张量。
   * 编写核心算子：Möbius Addition（莫比乌斯加法）。
   * 编写核心算子：Exponential & Logarithmic Maps（指数/对数映射，负责将欧氏空间特征拍进双曲空间）。
   * 编写核心距离函数：Poincaré Distance (Ryu-Takayanagi 测地线距离模拟)。
   * *防雷点：处理好边界溢出问题，加上 $1e-5$ 的 epsilon 防止 `arcosh` 出现 NaN。*

2. **验证测试**
   * 生成几个简单的树状结构数据，测试用 `holo_math.py` 计算的距离是否比欧氏距离更能反映树的层级深度。

---

## 阶段二：双曲 Embedding 模型的“外挂”训练 (Week 2)
*目标：不需要从头训练大模型，而是通过“嫁接”技术，把现有的 SOTA Embedding 模型（如 BGE-small 或 MTEB 榜首模型）改造成双曲模型。*

**核心任务：**
1. **模型改造 (`holo_embedder.py`)**
   * 加载开源模型（例如 `BAAI/bge-small-zh-v1.5`）。
   * 在模型的最后输出池化层之后，**冻结前方所有权重**。
   * 加入一个可学习的 `Hyperbolic Projection Layer`，使用对角缩放和指数映射将稠密向量投影到双曲流形上。
2. **微调训练 (Fine-tuning)**
   * **数据准备**：必须使用带有**强烈层级关系和因果关系**的数据。建议使用 WordNet (词网)、ConceptNet 或专门构造的“红楼梦人物关系图谱”。
   * **Loss 函数**：使用双曲 Contrastive Loss（基于庞加莱距离的 InfoNCE）。
   * **目标**：让模型学会在双曲空间里，把“贾母”放在靠近圆心的地方，把“丫鬟”放在靠近圆周的地方。

---

## 阶段三：开发极简双曲向量数据库 (Week 3)
*目标：绕过现有没有双曲度规的商业向量库，手搓一个能支撑 Demo 演示的全息索引。*

**核心任务：**
1. **Native Hyperbolic Index (`holo_db.py`)**
   * 初期版本：实现暴力的精确搜索 (Exact Search)，用纯 PyTorch 矩阵操作计算 Query 到所有 Chunk 的庞加莱距离，并返回 Top-K。
   * 因为这是高维双曲空间，传统的 HNSW（基于欧氏距离的导航小世界）会失效。
2. **文档切分与“写”缓存 (Document Ingestion)**
   * 把《红楼梦》切分成 1000 个段落 Chunk。
   * 全部过一遍 `holo_embedder`，将得到的庞加莱坐标存入 `holo_db`。

---

## 阶段四：RAG 组装与“降维打击”对比实验 (Week 4)
*目标：在真实大模型（如 Qwen 或 DeepSeek）前端挂上全息检索器，跑出碾压性的对比数据。*

**核心任务：**
1. **全链路组装 (`run_holo_rag.py`)**
   * 流程：用户提问 $\to$ `holo_embedder` 提取双曲 Query $\to$ `holo_db` 检索 Top-K Chunks $\to$ 拼接进 Prompt $\to$ Qwen-7B 给出最终回答。
2. **设计“红帽子测试” (The Red-Hat Benchmark)**
   * **对比基线 (Baseline)**：LangChain 默认的 RAG 链 (Cosine Similarity + FAISS)。
   * **挑战 1 (多跳逻辑)**：“因为王夫人昨天感冒了，所以晴雯今天去请大夫。请问晴雯为什么要出门？”（普通 RAG 会因为只匹配“晴雯出门”而遗漏“王夫人感冒”的父节点原因）。
   * **挑战 2 (宏观摘要)**：“请总结大观园里的阶级等级制度。”（双曲 RAG 的 Query 会自动落在靠近圆心的地方，从而均匀地召回各个等级的代表节点；普通 RAG 往往只偏向某几个具体的人）。
3. **输出震撼的可视化 (The "Wow" Factor)**
   * 把这 1000 个 Chunk 和用户的 Query 用 UMAP / t-SNE 画在二维的庞加莱圆盘上。
   * 展示 Query 是如何沿着测地线（漂亮的弧线）精准击中目标的。这可以直接作为论文的头图。

## 阶段五：模型对比与风险控制 (Week 5)
*目标：建立严格的量化对比指标，并针对双曲神经网络的阿喀琉斯之踵（数值不稳定性）进行系统级的排雷。*

**核心任务：**
1. **构建“层级保留”评估指标 (Hierarchical Preservation Metrics)**
   * **Tree-Edit Distance (树编辑距离)**：量化 RAG 召回的证据链是否完整重建了原文档的因果逻辑树，而不仅仅是字面命中率。
   * **Mean Reciprocal Rank (MRR)**：针对高熵问题（跨层级提问），统计全息 RAG 找对“根节点”的平均排名。
2. **极端的数值稳定性防御 (Numerical Stability Defenses)**
   * **NaN 诅咒拦截**：庞加莱球边缘的梯度会指数级爆炸。在 `holo_math.py` 中，所有涉及 $\|x\|^2$ 的计算必须钳位 (clamping)：`torch.clamp(norm_x, max=1.0 - 1e-5)`。
   * **Riemannian Adam 优化器适配**：标准的 Adam 优化器在平直空间进行参数更新，会把向量拉出庞加莱球。必须实现或引入流形优化器（如 `geoopt.optim.RiemannianAdam`），确保每次权重更新后都投影回双曲空间。
3. **硬件资源基准测试 (Hardware Profiling)**
   * 记录在相同规模的《红楼梦》知识库下，双曲向量数据库与传统 Faiss 数据库的**显存峰值 (Peak VRAM)** 和 **端到端检索延迟 (Latency)** 对比。

---

## 预期产出 (Deliverables)
1. 一套可运行的 `HoloRAG` Python 代码库（包含模型、数据库、测试脚本）。
2. 一份对比实验报告：证明在复杂逻辑推理下，全息 RAG 召回率提升 XX%，幻觉率下降 XX%。
3. 这将成为您向资本和学术界证明《物理人工智能统一场论》**第一阶段（应用变现阶段）**已经通关的铁证。

---

## 附录：双曲全息检索的数学推导 (Mathematical Derivations)

为了在工程中实现双曲 RAG，我们使用**庞加莱球模型 (Poincaré Ball Model)**。设其空间为 $\mathbb{D}^n = \{ x \in \mathbb{R}^n : \|x\| < 1 \}$。

### 1. 黎曼度规 (Riemannian Metric)
在庞加莱球中，距离边缘越近，空间被“拉伸”得越大。其共形度规张量为：
$$ g_x = \left( \frac{2}{1 - \|x\|^2} \right)^2 g_E $$
其中 $g_E$ 是欧几里得度规。这意味着当向量 $x$ 靠近边界 ($\|x\| \to 1$) 时，微小的欧氏位移会产生无限大的双曲距离。这就是它能提供**指数级容量**的数学根源。

### 2. 测地线距离 (Geodesic Distance / Poincaré Distance)
在普通 RAG 中，我们计算两点欧氏距离 $\|x - y\|$ 或点积。在全息 RAG 中，两点 $x, y \in \mathbb{D}^n$ 的真实双曲距离为：
$$ d_{\mathbb{H}}(x, y) = \text{arcosh} \left( 1 + 2\frac{\|x - y\|^2}{(1 - \|x\|^2)(1 - \|y\|^2)} \right) $$
*物理直觉*：如果 $x$ 和 $y$ 都在边缘（代表两个不相关的底层细节），它们的分母 $(1-\|x\|^2)$ 极小，导致直接跨越边缘的距离极大。最短路径（测地线）是被迫先向原点（高层级/宏观概念）走，然后再走向目标点。**这就是模型能自动提取出“蕴含的逻辑父节点”的原因。**

### 3. 莫比乌斯加法 (Möbius Addition)
在平直空间中，特征融合是 $x + y$。在双曲空间中，向量相加必须遵循莫比乌斯加法，以确保结果不飞出庞加莱球：
$$ x \oplus_c y = \frac{(1 + 2c\langle x, y \rangle + c\|y\|^2)x + (1 - c\|x\|^2)y}{1 + 2c\langle x, y \rangle + c^2\|x\|^2\|y\|^2} $$
*(注：为简化工程实现，常取曲率参数 $c=1$)*

### 4. 指数映射与对数映射 (Exp & Log Maps)
如何把传统大模型（欧氏空间）生成的特征 $v$ 放进庞加莱球（流形）？我们需要在原点（或任意点 $x$）处使用指数映射：
$$ \exp_x(v) = x \oplus \left( \tanh\left( \frac{\lambda_x \|v\|}{2} \right) \frac{v}{\|v\|} \right) $$
其中 $\lambda_x = \frac{2}{1-\|x\|^2}$。
在工程实现（如 `holo_embedder.py`）中，为了将欧氏向量 $v$ 映射为庞加莱向量，通常取 $x=0$，公式极度简化为：
$$ \exp_0(v) = \tanh(\|v\|) \frac{v}{\|v\|} $$
这个极其优雅的公式，就是我们将要挂在 BGE/Qwen 模型末尾的**“全息降维打击算子”**。它将无界的欧氏特征，完美地压缩进了单位球内。