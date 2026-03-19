import torch
import torch.nn as nn
import torch.nn.functional as F
import math

try:
    from mamba_ssm import Mamba
except ImportError:
    print("Warning: mamba_ssm not installed. Using fallback (not recommended for exact repro).")
    Mamba = None

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, max_seq_len):
        super().__init__()
        self.c_attn = nn.Linear(d_model, 3 * d_model)
        self.c_proj = nn.Linear(d_model, d_model)
        self.n_heads = n_heads
        self.d_model = d_model
        self.register_buffer("bias", torch.tril(torch.ones(max_seq_len, max_seq_len))
                                    .view(1, 1, max_seq_len, max_seq_len))

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.d_model, dim=2)
        k = k.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        q = q.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        v = v.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)

class TGNBlock(nn.Module):
    """
    Thermodynamic Gated Network (TGN) Block.
    Combines a low-cost inertial channel (Mamba) with a dynamic geometric channel (Attention).
    """
    def __init__(self, d_model, n_heads, max_seq_len, d_ff=None, chunk_size=128):
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model
            
        self.chunk_size = chunk_size
        self.ln1 = nn.LayerNorm(d_model)
        
        # Inertial Channel: Real Mamba
        if Mamba is not None:
            # mamba-ssm needs layer_idx for certain initializations in its native block
            # But the core Mamba mixer just takes d_model
            self.inertial = Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
        else:
            self.inertial = nn.GRU(d_model, d_model, batch_first=True)
            
        # Geometric Channel: Attention
        self.attn = CausalSelfAttention(d_model, n_heads, max_seq_len)
        
        # Thermodynamic Gate
        self.gate_proj = nn.Linear(d_model, 1)
        self.gate_proj.bias.data.fill_(0.0) 
        
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff), 
            nn.GELU(), 
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x, mode='standard', fixed_mask=None, hard_gate_threshold=0.05):
        B, L, D = x.shape
        x_norm = self.ln1(x)
        
        # 1. Inertial Path (Always on, acts as the adiabatic baseline)
        if isinstance(self.inertial, nn.GRU):
            h_inertial, _ = self.inertial(x_norm)
        else:
            h_inertial = self.inertial(x_norm)
            
        # Add small eps to prevent dead gradients in deep blocks
        h_inertial = h_inertial + 1e-5 * x_norm
            
        # 2. Gating Decision
        if mode in ['standard', 'soft']:
            gate_score = torch.sigmoid(self.gate_proj(h_inertial)) # [B, L, 1]
            
            # --- HARD GATING WITH STE ---
            if mode == 'standard':
                hard_gate = (gate_score > hard_gate_threshold).float()
                if self.training:
                    # Straight-Through Estimator: forward is hard, backward is soft
                    # gate_score = hard_gate.detach() - gate_score.detach() + gate_score
                    # We MUST use the STE trick safely to not interfere with standard Pytorch autograd.
                    # This formulation ensures forward pass uses hard_gate, but backward pass sees gate_score gradients.
                    gate_score = gate_score + (hard_gate - gate_score).detach()
                else:
                    # Thresholding for actual sparse execution
                    gate_score = hard_gate
        elif mode == 'random' and fixed_mask is not None:
            gate_score = fixed_mask
        elif mode == 'jamba':
            gate_score = torch.ones((B, L, 1), device=x.device)
        else: # mamba
            gate_score = torch.zeros((B, L, 1), device=x.device)
            
        # Protect gate from total zero
        gate_score = gate_score.clamp(min=1e-5)
        
        # 3. Geometric Path (Anti-dissipative work)
        attn_out = self.attn(x_norm)
        
        # 4. Thermodynamic Mixing
        #out = (1 - gate_score) * h_inertial + gate_score * attn_out
        out = h_inertial + gate_score * attn_out
        # RESIDUAL CONNECTION!
        x = x + out
        
        # 5. FFN with its own Residual
        x = x + self.mlp(self.ln2(x))
        
        return x, gate_score.mean()

class UniversalModel(nn.Module):
    """
    A unified model container that can act as Transformer, Mamba, Jamba, or TGN 
    for fair scientific comparisons.
    """
    def __init__(self, vocab_size, d_model, n_layers, n_heads, max_seq_len, model_type='tgn'):
        super().__init__()
        self.model_type = model_type
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        
        self.blocks = nn.ModuleList()
        for i in range(n_layers):
            if model_type == 'transformer':
                # Pure Transformer Baseline
                class TransformerBlock(nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.ln1 = nn.LayerNorm(d_model)
                        self.attn = CausalSelfAttention(d_model, n_heads, max_seq_len)
                        self.ln2 = nn.LayerNorm(d_model)
                        self.mlp = nn.Sequential(nn.Linear(d_model, 4*d_model), nn.GELU(), nn.Linear(4*d_model, d_model))
                    def forward(self, x):
                        x = x + self.attn(self.ln1(x))
                        x = x + self.mlp(self.ln2(x))
                        return x, torch.tensor(1.0)
                self.blocks.append(TransformerBlock())
                
            elif model_type == 'mamba':
                # Pure Mamba Baseline
                class MambaBlockWrapper(nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.ln1 = nn.LayerNorm(d_model)
                        if Mamba is not None:
                            self.mixer = Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
                        else:
                            self.mixer = nn.GRU(d_model, d_model, batch_first=True)
                        self.ln2 = nn.LayerNorm(d_model)
                        self.mlp = nn.Sequential(nn.Linear(d_model, 4*d_model), nn.GELU(), nn.Linear(4*d_model, d_model))
                    def forward(self, x):
                        if isinstance(self.mixer, nn.GRU):
                            h, _ = self.mixer(self.ln1(x))
                        else:
                            h = self.mixer(self.ln1(x))
                        x = x + h
                        x = x + self.mlp(self.ln2(x))
                        return x, torch.tensor(0.0)
                self.blocks.append(MambaBlockWrapper())
                
            elif model_type == 'jamba':
                # Jamba Baseline: Static Interleaving (e.g., 1:7 ratio in paper, or 1:1 if 2 layers)
                # For n_layers=12, we can interleave every 8th layer. 
                # If n_layers=2, this defaults to 1 Mamba + 1 Attention.
                if i % 8 == 1 or (n_layers == 2 and i == 1):
                    # Attention Layer
                    class TransformerBlock(nn.Module):
                        def __init__(self):
                            super().__init__()
                            self.ln1 = nn.LayerNorm(d_model)
                            self.attn = CausalSelfAttention(d_model, n_heads, max_seq_len)
                            self.ln2 = nn.LayerNorm(d_model)
                            self.mlp = nn.Sequential(nn.Linear(d_model, 4*d_model), nn.GELU(), nn.Linear(4*d_model, d_model))
                        def forward(self, x):
                            x = x + self.attn(self.ln1(x))
                            x = x + self.mlp(self.ln2(x))
                            return x, torch.tensor(1.0)
                    self.blocks.append(TransformerBlock())
                else:
                    # Mamba Layer
                    class MambaBlockWrapper(nn.Module):
                        def __init__(self):
                            super().__init__()
                            self.ln1 = nn.LayerNorm(d_model)
                            if Mamba is not None:
                                self.mixer = Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
                            else:
                                self.mixer = nn.GRU(d_model, d_model, batch_first=True)
                            self.ln2 = nn.LayerNorm(d_model)
                            self.mlp = nn.Sequential(nn.Linear(d_model, 4*d_model), nn.GELU(), nn.Linear(4*d_model, d_model))
                        def forward(self, x):
                            if isinstance(self.mixer, nn.GRU):
                                h, _ = self.mixer(self.ln1(x))
                            else:
                                h = self.mixer(self.ln1(x))
                            x = x + h
                            x = x + self.mlp(self.ln2(x))
                            return x, torch.tensor(0.0)
                    self.blocks.append(MambaBlockWrapper())
                    
            elif model_type == 'tgn':
                # TGN: Thermodynamic Gated Network
                self.blocks.append(TGNBlock(d_model, n_heads, max_seq_len))
                
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.emb.weight = self.head.weight # Weight tying

    def forward(self, idx, targets=None, mode='standard', target_sparsity=None, hard_gate_threshold=0.05):
        B, T = idx.size()
        x = self.emb(idx) + self.pos_emb(torch.arange(T, device=idx.device))
        
        total_g = 0
        count = 0
        for block in self.blocks:
            mask = None
            if mode == 'random' and self.model_type == 'tgn':
                mask = torch.bernoulli(torch.full((B, T, 1), target_sparsity, device=idx.device))
                
            if self.model_type == 'tgn':
                x, g = block(x, mode=mode, fixed_mask=mask, hard_gate_threshold=hard_gate_threshold)
            else:
                x, g = block(x)
                
            total_g += g
            count += 1
            
        x = self.ln_f(x)
        logits = self.head(x)
        avg_gate = total_g / count if count > 0 else torch.tensor(0.0)
        
        loss = None
        if targets is not None:
            # Shift targets for language modeling if targets shape matches idx
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = targets[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
        return logits, loss, avg_gate
