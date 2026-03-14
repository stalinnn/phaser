import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from holo_embedder import HyperbolicEmbedder
from holo_db import HoloVectorDB

class HoloRAGChatEngine:
    """
    端到端全息检索增强生成 (Holo-RAG) 聊天引擎。
    包含：
    1. 双曲全息检索器 (Holo-DB + Holo-Embedder)
    2. 生成式大语言模型 (如 Qwen-Chat)
    """
    def __init__(self, embedder_path="BAAI/bge-small-zh-v1.5", llm_path="Qwen/Qwen1.5-0.5B-Chat", device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"Initializing Holo-RAG Engine on {self.device}...")
        
        # 1. 初始化双曲检索模块
        print(">> 正在加载并升维 Embedding 模型...")
        self.embedder = HyperbolicEmbedder(embedder_path).to(self.device)
        try:
            self.embedder.projection.load_state_dict(torch.load("holo_projection.pt"))
            print("   [成功] 全息投影矩阵 (Holo Projection) 已挂载！测地线追踪开启。")
        except FileNotFoundError:
            print("   [警告] 未找到微调的 holo_projection.pt，正在使用随机曲率。")
        self.embedder.eval()
        
        self.holo_db = HoloVectorDB(c=1.0, device=self.device)
        
        # 2. 初始化本地 LLM 模块 (为了演示速度，默认使用极小的 Qwen-0.5B)
        print(f">> 正在加载本地 LLM ({llm_path})...")
        self.llm_tokenizer = AutoTokenizer.from_pretrained(llm_path, trust_remote_code=True)
        self.llm_model = AutoModelForCausalLM.from_pretrained(
            llm_path, 
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,
            device_map="auto"
        )
        self.llm_model.eval()
        print(">> Holo-RAG 引擎启动完毕！\n")

    def ingest_knowledge(self, knowledge_list):
        """将知识块注入全息数据库"""
        print(f"正在将 {len(knowledge_list)} 条知识刻录进庞加莱球...")
        inputs = self.embedder.tokenizer(knowledge_list, padding=True, truncation=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.embedder.backbone(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
            euclidean_embs = outputs.last_hidden_state[:, 0, :]
            # 全息升维
            hyperbolic_embs = self.embedder.holo_math.exp_map0(self.embedder.projection(euclidean_embs))
        
        self.holo_db.add_texts(hyperbolic_embs, knowledge_list)

    def ask(self, query_text, top_k=2):
        """
        全流程：提问 -> 全息检索 -> 构建 Prompt -> LLM 生成
        """
        # 1. 全息检索
        q_inputs = self.embedder.tokenizer([query_text], padding=True, truncation=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            q_out = self.embedder.backbone(input_ids=q_inputs.input_ids, attention_mask=q_inputs.attention_mask)
            q_euc = q_out.last_hidden_state[:, 0, :]
            q_hyp = self.embedder.holo_math.exp_map0(self.embedder.projection(q_euc))
            
        retrieved_results = self.holo_db.search(q_hyp, top_k=top_k)
        
        # 拼接召回上下文
        context_str = "\n".join([f"- {text} (双曲距离: {dist:.2f})" for dist, text in retrieved_results])
        
        # 2. 构建给 LLM 的 Prompt
        prompt = (
            "你是一个具备高级逻辑推理能力的AI助手。请根据以下我通过全息检索系统（能够捕捉深层因果关系）为你提供的背景信息，回答用户的问题。\n\n"
            f"【全息上下文信息】:\n{context_str}\n\n"
            f"【用户问题】: {query_text}\n"
            "请直接给出答案，并简要说明你的逻辑链。"
        )
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        
        # 3. LLM 生成
        text_input = self.llm_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.llm_tokenizer([text_input], return_tensors="pt").to(self.llm_model.device)
        
        print("\n[Holo-RAG 思考中...]")
        with torch.no_grad():
            generated_ids = self.llm_model.generate(
                model_inputs.input_ids,
                max_new_tokens=200,
                temperature=0.3,
                top_p=0.8,
                pad_token_id=self.llm_tokenizer.eos_token_id
            )
            
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        response = self.llm_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        print("="*60)
        print(f"用户提问: {query_text}")
        print("-" * 60)
        print("全息召回知识:")
        print(context_str)
        print("-" * 60)
        print(f"LLM 终极回答:\n{response}")
        print("="*60)
        return response

if __name__ == "__main__":
    # 模拟真实商业/学术 Demo 流程
    # 注意：这里会下载 Qwen1.5-0.5B-Chat (大约 1GB)。如果网络慢，可以换成更小的模型。
    engine = HoloRAGChatEngine()
    
    # 依然使用我们经典的“红帽子”多跳逻辑数据集
    knowledge_base = [
        "【第十五回概要】近日倒春寒，王夫人偶感风寒，卧病在床，府中上下忙着寻医问药。",
        "晴雯穿好衣服，行色匆匆地走出了大观园的正门。",
        "贾宝玉今天也出门了，去参加北静王的宴会。",
        "昨天晴雯在院子里撕扇子，笑得很开心。",
        "林黛玉身子骨弱，到了冬天总要感冒咳嗽几声。",
        "薛宝钗派人出门去买了一些燕窝和人参。"
    ]
    
    engine.ingest_knowledge(knowledge_base)
    
    # 执行测试
    engine.ask("晴雯今天为什么要出门？")