import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# 极简配置
VOCAB_SIZE = 20
D_MODEL = 64
SEQ_LEN = 32 # 短序列
BATCH_SIZE = 4

class ToyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.pos = nn.Embedding(SEQ_LEN, D_MODEL)
        self.attn = nn.MultiheadAttention(D_MODEL, 2, batch_first=True)
        self.head = nn.Linear(D_MODEL, VOCAB_SIZE)
        
    def forward(self, x):
        B, L = x.shape
        pos = torch.arange(L, device=x.device).unsqueeze(0)
        h = self.emb(x) + self.pos(pos)
        
        # Causal Mask
        mask = torch.triu(torch.ones(L, L, device=x.device) * float('-inf'), diagonal=1)
        h, _ = self.attn(h, h, h, attn_mask=mask)
        
        return self.head(h)

def get_batch():
    X = torch.zeros(BATCH_SIZE, SEQ_LEN, dtype=torch.long)
    Y = torch.zeros(BATCH_SIZE, dtype=torch.long)
    
    for i in range(BATCH_SIZE):
        # 简单任务：输入 [K, V, ..., K, 0] -> 输出 V
        k = np.random.randint(1, VOCAB_SIZE)
        v = np.random.randint(1, VOCAB_SIZE)
        
        X[i, 0] = k
        X[i, 1] = v
        X[i, -2] = k # Query
        X[i, -1] = 0 # Placeholder
        
        Y[i] = v
        
        # 填充随机噪声
        X[i, 2:-2] = torch.randint(1, VOCAB_SIZE, (SEQ_LEN-4,))
        
    return X, Y

def test():
    model = ToyTransformer()
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    print("Start Debugging...")
    for step in range(500):
        X, Y = get_batch()
        logits = model(X)
        
        # 预测倒数第二个位置的下一个词
        pred = logits[:, -2, :] 
        loss = F.cross_entropy(pred, Y)
        
        optim.zero_grad()
        loss.backward()
        optim.step()
        
        if step % 50 == 0:
            acc = (pred.argmax(dim=-1) == Y).float().mean()
            print(f"Step {step} | Loss: {loss.item():.4f} | Acc: {acc:.1%}")
            
            if step == 0:
                print(f"Sample Input: {X[0].tolist()}")
                print(f"Sample Target: {Y[0].item()}")
                print(f"Sample Pred: {pred.argmax(dim=-1)[0].item()}")

if __name__ == "__main__":
    test()
