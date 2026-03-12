import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ComplexLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Use complex64 (equivalent to two float32s)
        self.weight = nn.Parameter(torch.randn(out_features, in_features, dtype=torch.complex64))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.complex64))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # Initialization for complex weights
        # We initialize magnitude using Kaiming and phase uniformly
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        scale = 1 / math.sqrt(fan_in)
        
        # Magnitude
        mag = torch.rand_like(self.weight.real) * scale
        # Phase
        phase = torch.rand_like(self.weight.real) * 2 * math.pi
        
        with torch.no_grad():
            self.weight.copy_(torch.polar(mag, phase))
            if self.bias is not None:
                self.bias.zero_()

    def forward(self, x):
        # x is complex64
        return F.linear(x, self.weight, self.bias)

class ComplexLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        # Gamma and Beta are complex parameters
        self.gamma = nn.Parameter(torch.ones(normalized_shape, dtype=torch.complex64))
        self.beta = nn.Parameter(torch.zeros(normalized_shape, dtype=torch.complex64))

    def forward(self, x):
        # Mean and Variance are calculated on the magnitude squared (real-valued)
        mag_sq = x.abs().pow(2)
        mean = mag_sq.mean(dim=-1, keepdim=True)
        var = mag_sq.var(dim=-1, keepdim=True, unbiased=False)
        
        std = torch.sqrt(var + self.eps)
        
        # Normalize: divide complex vector by real scalar std
        x_norm = (x - torch.complex(mean, torch.zeros_like(mean))) / torch.complex(std, torch.zeros_like(std))
        
        return x_norm * self.gamma + self.beta

class ComplexReLU(nn.Module):
    def forward(self, x):
        # Apply ReLU to real and imag parts separately (Split Activation)
        # This is one of many ways to define Complex ReLU
        return torch.complex(F.relu(x.real), F.relu(x.imag))
