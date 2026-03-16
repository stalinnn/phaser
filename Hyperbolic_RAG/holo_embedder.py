import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from holo_math import PoincareMath

class HyperbolicEmbedder(nn.Module):
    """
    全息嵌入器 (Holo-Embedder)
    作用：将传统的预训练欧氏空间 Embedding 模型（如 BERT / BGE）升级为双曲模型。
    原理：冻结底层 Transformer 权重，在池化层后接入可学习的全息投影层，将特征映射进庞加莱球。
    """
    def __init__(self, model_name="C:/Users/29478.000/Desktop/系统科学金融理论/model_downloads/models/bge-small-en-v1.5", c=1.0):
        super().__init__()
        # 1. 加载骨干网络
        print(f"Loading backbone model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.backbone.config.hidden_size
        
        # 2. 冻结骨干网络权重 (省显存、防崩溃)
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # 3. 双曲几何引擎
        self.holo_math = PoincareMath(c=c)
        
        # 4. 全息投影层 (Holographic Projection Layer)
        # 这是一个简单的线性变换，负责缩放和旋转欧氏特征，以便更好地拍进庞加莱球
        # 这一层是我们要微调 (Fine-tune) 的唯一参数！
        self.projection = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        # 初始化为较小的值，确保初始映射落在球心附近，防止一开始就撞墙 (NaN)
        nn.init.normal_(self.projection.weight, mean=0, std=0.01)

    def forward(self, input_ids, attention_mask):
        """
        前向传播：文本 -> 欧氏向量 -> 全息投影 -> 庞加莱坐标
        """
        # Step 1: 提取欧氏空间的特征 (利用冻结的 Transformer)
        with torch.no_grad():
            outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
            # 使用 CLS token 的表示作为句子/段落的 Embedding
            # Shape: [Batch_size, Hidden_size]
            euclidean_emb = outputs.last_hidden_state[:, 0, :]
            
        # Step 2: 线性缩放与对齐 (可学习部分)
        # 这一步决定了概念的层级。宏观概念会被投射为短向量，微观概念会被投射为长向量。
        scaled_emb = self.projection(euclidean_emb)
        
        # Step 3: 全息降维打击 (指数映射)
        # 将欧氏向量优雅地“拍进”庞加莱球内部
        hyperbolic_emb = self.holo_math.exp_map0(scaled_emb)
        
        return hyperbolic_emb

# =====================================================================
# 简单的单元测试
# =====================================================================
if __name__ == "__main__":
    # 为了跑通测试且不下载几个G的模型，我们可以换一个小一点的模型，或者直接用 BGE-small
    # 注意：这里需要联网下载 BAAI/bge-small-zh-v1.5 的权重 (约 100MB)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 初始化
    try:
        model = HyperbolicEmbedder("BAAI/bge-small-zh-v1.5").to(device)
        model.eval() # 测试模式
        
        # 造几句话 (体现宏观与微观)
        texts = [
            "红楼梦是一部描写封建家族兴衰的中国古典古典名著。", # 宏观/树根
            "贾宝玉是红楼梦的核心男主角。",                     # 中层/树干
            "今天贾宝玉在院子里摔了那块通灵宝玉，惹得黛玉哭了。"  # 微观/树叶
        ]
        
        # 预处理
        inputs = model.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
        
        # 前向传播，获取双曲向量
        with torch.no_grad():
            h_embs = model(inputs.input_ids, inputs.attention_mask)
            
        print("\n[全息 Embedding 提取成功]")
        print(f"Output Shape: {h_embs.shape} (Batch, Dim)")
        
        # 计算距离圆心的半径 (范数)
        norms = torch.norm(h_embs, p=2, dim=-1)
        print("\n[向量在庞加莱球中的半径 (深度)]")
        for i, text in enumerate(texts):
            print(f"R={norms[i].item():.4f} | {text[:20]}...")
            
        print("\n>>> 说明：因为投影层刚被随机初始化，此时的半径分布是随机的。")
        print(">>> 接下来在阶段三，我们需要用《红楼梦》的大纲-细节数据来微调它，让‘宏观=短半径’，‘微观=长半径’成为物理定律！")
        
    except Exception as e:
        print(f"下载或初始化失败，请检查网络或更换轻量级模型。错误信息: {e}")