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
    def __init__(self, model_name="/gz-data/Qwen2.5-1.5B-Instruct", c=1.0):
        super().__init__()
        # 1. 加载骨干网络
        print(f"Loading backbone model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.backbone = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.backbone.config.hidden_size
        
        # 2. 冻结骨干网络权重 (省显存、防崩溃)
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # 3. 双曲几何引擎
        self.holo_math = PoincareMath(c=c)
        
        # 4. 全息投影层 (Holographic Projection Layer)
        # 强制降维：将 1536 维的欧氏特征压缩到 64 维双曲空间
        self.target_dim = 64
        self.projection = nn.Sequential(
            nn.Linear(self.hidden_size, 512),
            nn.GELU(),
            nn.Linear(512, self.target_dim)
        )
        
        # 初始化 - 使用极小的方差，让初始映射点全部聚集在庞加莱球心 (欧氏零点) 附近
        for m in self.projection.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, input_ids, attention_mask):
        """
        前向传播：文本 -> 欧氏向量 -> 全息投影 -> 庞加莱坐标
        """
        # Step 1: 提取欧氏空间的特征 (利用冻结的 Transformer)
        with torch.no_grad():
            outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            # Use mean pooling for causal LMs to get a good sentence representation
            attention_mask_expanded = attention_mask.unsqueeze(-1).expand_as(outputs.last_hidden_state)
            euclidean_emb = torch.sum(outputs.last_hidden_state * attention_mask_expanded, 1) / torch.clamp(attention_mask_expanded.sum(1), min=1e-9)
            
        # Step 2: 非线性降维与特征提取
        # 给模型添加残差连接 (Residual Connection) 以保底基础语义
        # 因为前1536维不好直接残差，我们可以让 projection 尽量保持原始距离比例
        scaled_emb = self.projection(euclidean_emb)
        
        # Step 3: 安全地映射进庞加莱球 (利用 geoopt 库)
        import geoopt
        manifold = geoopt.PoincareBall(c=1.0)
        # 放大特征的方差，让不同层级的节点在球内分布得更开
        # 用 0.5 缩放确保即使极端长也能安全地装进球里
        scaled_emb = scaled_emb * 0.5
        
        # 限制最大范数，防止映射到无穷远
        norm = torch.norm(scaled_emb, p=2, dim=-1, keepdim=True)
        scaled_emb = torch.where(norm > 5.0, scaled_emb / norm * 5.0, scaled_emb)
        
        hyperbolic_emb = manifold.expmap0(scaled_emb)
        
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