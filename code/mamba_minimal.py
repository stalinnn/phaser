import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_inner = int(expand * d_model)
        self.dt_rank = math.ceil(d_model / 16)
        
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=True,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        
        # S4 params
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

    def forward(self, x):
        # x: (B, L, D)
        B, L, D = x.shape
        x_and_res = self.in_proj(x)  # (B, L, 2*d_inner)
        (x, res) = x_and_res.split(split_size=[self.d_inner, self.d_inner], dim=-1)

        x = x.transpose(1, 2)
        x = self.conv1d(x)[:, :, :L]
        x = x.transpose(1, 2)
        x = F.silu(x)

        y = self.ssm(x)
        y = y * F.silu(res)
        return self.out_proj(y)

    def ssm(self, x):
        # Selective Scan Simulation (PyTorch Recurrent)
        # Slower than CUDA kernel but functionally identical for validation
        B, L, D = x.shape
        x_dbl = self.x_proj(x)  # (B, L, dt_rank + 2*d_state)
        d_state = x_dbl.shape[-1] - self.dt_rank - x_dbl.shape[-1]//2 # Roughly infer state dim
        
        # Actually we know d_state from init, but let's be explicit
        d_state = (x_dbl.shape[-1] - self.dt_rank) // 2
        
        (delta, B_mat, C_mat) = x_dbl.split(
            split_size=[self.dt_rank, d_state, d_state], dim=-1
        )
        delta = F.softplus(self.dt_proj(delta))  # (B, L, D)

        y = torch.zeros_like(x)
        h = torch.zeros(B, D, d_state, device=x.device)
        
        A = -torch.exp(self.A_log.float())  # (D, N)
        
        # Scan loop
        for t in range(L):
            dt = delta[:, t, :] # (B, D)
            dA = torch.exp(torch.einsum('bd,dn->bdn', dt, A)) # (B, D, N)
            dB = torch.einsum('bd,bn->bdn', dt, B_mat[:, t, :]) # (B, D, N)
            
            h = h * dA + dB * x[:, t, :].unsqueeze(-1) # (B, D, N)
            y[:, t, :] = torch.einsum('bdn,bn->bd', h, C_mat[:, t, :]) # (B, D)
            
        return y + x * self.D

class MambaModel(nn.Module):
    def __init__(self, vocab_size, d_model, n_layer):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([MambaBlock(d_model) for _ in range(n_layer)])
        self.norm_f = nn.LayerNorm(d_model)

    def forward(self, x):
        h = self.embedding(x)
        for layer in self.layers:
            h = h + layer(h)
        return self.norm_f(h)
