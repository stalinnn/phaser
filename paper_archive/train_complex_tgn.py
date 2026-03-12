import torch
import torch.nn as nn
import torch.nn.functional as F
from complex_layers import ComplexLinear, ComplexLayerNorm, ComplexReLU
import math

class ComplexRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=4096):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, inv_freq)
        
        # freqs shape: (max_seq_len, dim/2)
        # We need complex rotation e^{i*theta}
        # cos + i*sin
        self.register_buffer("cos_cached", freqs.cos())
        self.register_buffer("sin_cached", freqs.sin())

    def forward(self, x, seq_len):
        # x shape: (B, H, T, D)
        # We need to apply rotation to the last dim D
        # But x is complex64.
        # RoPE on complex numbers: q_new = q * e^{i*theta}
        # This is just element-wise multiplication by a phasor!
        
        cos = self.cos_cached[:seq_len, :].unsqueeze(0).unsqueeze(0) # (1, 1, T, D/2)
        sin = self.sin_cached[:seq_len, :].unsqueeze(0).unsqueeze(0)
        
        # Expand to full dim D: [cos, cos] to match [real, imag] or similar?
        # Actually, for complex numbers, we have D dimensions (real) + D dimensions (imag).
        # Standard RoPE rotates pairs of real numbers.
        # Complex RoPE rotates the complex number itself!
        # So we just need theta for each complex dimension.
        
        # If D is the number of complex channels:
        # We generate D frequencies.
        
        # Let's adjust initialization to match input D
        # If input x has D complex channels, we need D frequencies.
        pass

class ComplexRoPE(nn.Module):
    def __init__(self, dim, max_len=4096):
        super().__init__()
        # Generate frequencies for each dimension
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim).float() / dim))
        t = torch.arange(max_len).float()
        freqs = torch.outer(t, inv_freq) # (T, D)
        
        # Create complex phasor: e^{i * theta}
        self.register_buffer("phasor", torch.polar(torch.ones_like(freqs), freqs)) # Complex64

    def forward(self, x):
        # x: (B, H, T, D) - Complex64
        T = x.shape[2]
        phasor = self.phasor[:T, :].unsqueeze(0).unsqueeze(0) # (1, 1, T, D)
        
        # Apply rotation: q * e^{i * theta}
        # This is relative position encoding for complex networks!
        return x * phasor

class ComplexAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head
        
        self.w_q = ComplexLinear(d_model, d_model)
        self.w_k = ComplexLinear(d_model, d_model)
        self.w_v = ComplexLinear(d_model, d_model)
        self.w_o = ComplexLinear(d_model, d_model)
        
        # RoPE
        self.rope = ComplexRoPE(self.head_dim)

    def forward(self, x, mask=None):
        B, T, C = x.shape
        
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)
        
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        
        # Apply Complex RoPE
        q = self.rope(q)
        k = self.rope(k)
        
        # Attention: Q * K_conj
        score = torch.matmul(q, k.conj().transpose(-2, -1))
        score = score / math.sqrt(self.head_dim)
        score_mag = score.abs()
        
        if mask is not None:
            score_mag = score_mag.masked_fill(mask == 0, -1e9)
            
        attn_weights = F.softmax(score_mag, dim=-1)
        attn_weights_c = torch.complex(attn_weights, torch.zeros_like(attn_weights))
        
        out = torch.matmul(attn_weights_c, v)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        
        return self.w_o(out)

class AntiMatterGate(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate_net = ComplexLinear(dim, dim)
        
    def forward(self, x):
        g = self.gate_net(x)
        g_norm = g.abs() + 1e-8
        g_phasor = g / torch.complex(g_norm, torch.zeros_like(g_norm))
        return x * g_phasor

class ComplexTGNLayer(nn.Module):
    def __init__(self, d_model, n_head, d_ff):
        super().__init__()
        self.attn = ComplexAttention(d_model, n_head)
        self.norm1 = ComplexLayerNorm(d_model)
        self.ff1 = ComplexLinear(d_model, d_ff)
        self.relu = ComplexReLU()
        self.ff2 = ComplexLinear(d_ff, d_model)
        self.norm2 = ComplexLayerNorm(d_model)
        self.gate = AntiMatterGate(d_model)
        
    def forward(self, x, mask=None):
        norm_x = self.norm1(x)
        x = x + self.attn(norm_x, mask)
        
        norm_x = self.norm2(x)
        out = self.ff1(norm_x)
        out = self.relu(out)
        out = self.ff2(out)
        out = self.gate(out)
        x = x + out
        
        return x

class ComplexTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, n_layer, n_head, d_ff, max_len=512):
        super().__init__()
        self.d_model = d_model
        
        self.emb = nn.Embedding(vocab_size, d_model)
        self.emb_imag = nn.Embedding(vocab_size, d_model)
        
        # Remove Absolute Positional Embedding (Replaced by RoPE)
        # self.pos_emb = nn.Embedding(max_len, d_model)
        # self.pos_emb_imag = nn.Embedding(max_len, d_model)
        
        self.layers = nn.ModuleList([
            ComplexTGNLayer(d_model, n_head, d_ff) for _ in range(n_layer)
        ])
        
        self.out_head = ComplexLinear(d_model, vocab_size)
        
    def forward(self, x):
        B, T = x.shape
        
        # No adding positional embedding here!
        x_real = self.emb(x)
        x_imag = self.emb_imag(x)
        x_complex = torch.complex(x_real, x_imag)
        
        mask = torch.tril(torch.ones(T, T, device=x.device)).unsqueeze(0).unsqueeze(0)
        
        for layer in self.layers:
            x_complex = layer(x_complex, mask)
            
        logits_complex = self.out_head(x_complex)
        logits = logits_complex.abs()
        return logits

def train_wikitext_demo():
    # Configuration for a small but real model
    model_config = {
        'vocab_size': 10000, 
        'd_model': 256,
        'n_layer': 4,
        'n_head': 4,
        'd_ff': 1024,
        'max_len': 512
    }
    
    train_config = {
        'batch_size': 32,
        'seq_len': 128
    }
    
    print(f"Initializing Native Complex-TGN Transformer with config: {model_config}")
    model = ComplexTransformer(**model_config)
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = model.to(device)
    
    # Count parameters
    params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters (Complex64): {params/1e6:.2f}M")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    print("Starting training loop (Simulated)...")
    
    import time
    start_time = time.time()
    
    for step in range(100):
        # Generate random data
        inputs = torch.randint(0, model_config['vocab_size'], (train_config['batch_size'], train_config['seq_len'])).to(device)
        targets = torch.randint(0, model_config['vocab_size'], (train_config['batch_size'], train_config['seq_len'])).to(device)
        
        logits = model(inputs)
        
        # Flatten for loss
        loss = criterion(logits.view(-1, model_config['vocab_size']), targets.view(-1))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step % 10 == 0:
            elapsed = time.time() - start_time
            print(f"Step {step} | Loss: {loss.item():.4f} | Time: {elapsed:.2f}s")
            start_time = time.time()

if __name__ == "__main__":
    train_wikitext_demo()
