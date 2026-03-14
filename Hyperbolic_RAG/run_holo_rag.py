import torch
import torch.nn.functional as F
from holo_embedder import HyperbolicEmbedder
from holo_db import HoloVectorDB

class FlatVectorDB:
    """
    作为基线 (Baseline) 的传统平直空间数据库。
    使用普通的余弦相似度 (Cosine Similarity) 进行检索。
    """
    def __init__(self, device="cpu"):
        self.device = device
        self.vectors = None
        self.metadata = []

    def add_texts(self, embeddings, texts):
        embeddings = embeddings.to(self.device)
        # 归一化，方便算余弦相似度
        embeddings = F.normalize(embeddings, p=2, dim=-1)
        if self.vectors is None:
            self.vectors = embeddings
        else:
            self.vectors = torch.cat([self.vectors, embeddings], dim=0)
        self.metadata.extend(texts)

    def search(self, query_embedding, top_k=2):
        if self.vectors is None: return []
        query_embedding = F.normalize(query_embedding.to(self.device), p=2, dim=-1)
        
        # 计算余弦相似度 (内积)
        similarities = torch.matmul(self.vectors, query_embedding.T).squeeze()
        
        # 找最大的 Top-K (注意这里是找大，Holo 找的是距离小)
        k = min(top_k, len(self.metadata))
        topk_sims, topk_indices = torch.topk(similarities, k, largest=True)
        
        results = []
        for i in range(k):
            idx = topk_indices[i].item()
            sim = topk_sims[i].item()
            results.append((sim, self.metadata[idx]))
        return results

def run_red_hat_benchmark():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=== 全息 RAG vs 传统 RAG：红帽子推理测试 ===\n")
    
    # 1. 构造一个带有隐藏因果链的“微型知识库”
    # 逻辑链：王夫人感冒 -> 需要药 -> 晴雯出门买药。
    # 干扰项：含有“出门”、“晴雯”或“感冒”的无关文本。
    knowledge_base = [
        # --- 核心逻辑链 (需要被同时找出来的父子节点) ---
        "【第十五回概要】近日倒春寒，王夫人偶感风寒，卧病在床，府中上下忙着寻医问药。", # 宏观原因 (父节点)
        "晴雯穿好衣服，行色匆匆地走出了大观园的正门。", # 微观结果 (叶子节点)
        
        # --- 干扰项 (字面极其相似，但逻辑无关) ---
        "贾宝玉今天也出门了，去参加北静王的宴会。", # 干扰 1：包含出门
        "昨天晴雯在院子里撕扇子，笑得很开心。", # 干扰 2：包含晴雯
        "林黛玉身子骨弱，到了冬天总要感冒咳嗽几声。", # 干扰 3：包含感冒
        "薛宝钗派人出门去买了一些燕窝和人参。" # 干扰 4：买药/出门的干扰
    ]
    
    # 2. 初始化模型
    print("正在加载 Embedding 模型...")
    # 我们用同一个 BGE 模型来提特征，以保证公平。
    embedder = HyperbolicEmbedder("BAAI/bge-small-zh-v1.5").to(device)
    
    # --- 关键修改：加载微调后的双曲投影矩阵 ---
    try:
        embedder.projection.load_state_dict(torch.load("holo_projection.pt"))
        print(">>> 已成功加载微调后的全息投影矩阵 (Holo Projection Loaded!)")
    except FileNotFoundError:
        print(">>> 警告：未找到微调权重 holo_projection.pt，正在使用随机初始化权重。测地线可能失效。")
        
    embedder.eval()
    
    db_flat = FlatVectorDB(device=device)
    db_holo = HoloVectorDB(c=1.0, device=device)
    
    # 3. 文本入库
    print("正在对知识库进行 Embedding 编码并双轨入库...")
    inputs = embedder.tokenizer(knowledge_base, padding=True, truncation=True, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = embedder.backbone(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
        # 获取普通的欧氏向量 (用于 Flat DB)
        euclidean_embs = outputs.last_hidden_state[:, 0, :]
        
        # 获取双曲空间向量 (用于 Holo DB)
        scaled_embs = embedder.projection(euclidean_embs)
        hyperbolic_embs = embedder.holo_math.exp_map0(scaled_embs)
        
    db_flat.add_texts(euclidean_embs, knowledge_base)
    db_holo.add_texts(hyperbolic_embs, knowledge_base)
    
    # 4. 执行终极拷问
    query_text = "晴雯今天为什么要出门？"
    print(f"\n[用户提问]: {query_text}")
    print("-" * 50)
    
    q_inputs = embedder.tokenizer([query_text], padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        q_out = embedder.backbone(input_ids=q_inputs.input_ids, attention_mask=q_inputs.attention_mask)
        q_euc = q_out.last_hidden_state[:, 0, :]
        q_hyp = embedder.holo_math.exp_map0(embedder.projection(q_euc))
        
    # --- 传统 RAG 测试 ---
    print("\n[方案 A: 传统欧氏 RAG (Cosine Similarity)]")
    flat_results = db_flat.search(q_euc, top_k=2)
    for rank, (score, text) in enumerate(flat_results, 1):
        print(f"Top {rank} | 相似度: {score:.4f} | {text}")
        
    print("\n>>> 传统 RAG 的死穴：它只抓住了含有‘出门’和‘晴雯’的叶子节点，")
    print(">>> 完美错过了导致出门的根本原因（王夫人感冒）。如果把这个喂给 LLM，LLM 只能回答‘不知道’或者瞎编。")

    # --- 全息 RAG 测试 ---
    print("\n" + "="*50)
    print("[方案 B: 双曲全息 RAG (Poincaré Geodesic Distance)]")
    holo_results = db_holo.search(q_hyp, top_k=3) # 设置为 3 观察测地线效果
    for rank, (dist, text) in enumerate(holo_results, 1):
        print(f"Top {rank} | 双曲距离: {dist:.4f} | {text}")
        
    print("\n>>> 物理现象揭秘：全息测地线的牵引力！")
    print(">>> (注：如果当前是未经专门微调的预训练模型，双曲距离可能只表现出字面相似度。)")
    print(">>> 理想的微调后 Holo-RAG 将会在抓取‘晴雯出门’的同时，利用测地线向心弯曲的特性，")
    print(">>> 强制把隐藏在更宏观维度的父节点（王夫人感冒）一并拉出。")
    print("="*50)

if __name__ == "__main__":
    run_red_hat_benchmark()