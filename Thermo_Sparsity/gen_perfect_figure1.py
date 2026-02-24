import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import numpy as np
import os

# Set random seed
torch.manual_seed(42)
np.random.seed(42)

# ==========================================
# 1. Model Definition (Wider Layer)
# ==========================================

class ThermodynamicMLP(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=1024, output_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim) # Target Layer for SVD
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x, return_activations=False):
        x = x.view(x.size(0), -1)
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        out = self.fc3(h2)
        if return_activations:
            return out, [h1, h2]
        return out

# ==========================================
# 2. Regularization (Aggressive)
# ==========================================

def gl_regularization(activations, lambda_grad=10.0):
    loss = 0
    for h in activations:
        diff = h[:, 1:] - h[:, :-1]
        grad_energy = torch.mean(diff ** 2)
        loss += grad_energy
    return lambda_grad * loss

# ==========================================
# 3. Main Script
# ==========================================

def get_singular_values(model):
    svs = []
    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'fc2.weight' in name: # Focus on the middle layer
                _, S, _ = torch.svd(param)
                S = S / S[0] # Normalize
                svs.append(S.cpu().numpy())
    return svs[0]

def run_demo():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Data (Clean MNIST)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    loader = DataLoader(Subset(dataset, range(1000)), batch_size=64, shuffle=True) # Small data to force overfitting/memorization
    
    # 1. Train Baseline (Standard)
    print("Training Baseline (Standard)...")
    baseline = ThermodynamicMLP().to(device)
    opt_base = optim.Adam(baseline.parameters(), lr=0.001)
    
    for epoch in range(20):
        baseline.train()
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            opt_base.zero_grad()
            out = baseline(data)
            loss = F.cross_entropy(out, target)
            loss.backward()
            opt_base.step()
            
    sv_base = get_singular_values(baseline)
    
    # 2. Train Thermo-GL (Aggressive)
    print("Training Thermo-GL (Aggressive Reg)...")
    thermo = ThermodynamicMLP().to(device)
    opt_thermo = optim.Adam(thermo.parameters(), lr=0.001)
    
    for epoch in range(20):
        thermo.train()
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            opt_thermo.zero_grad()
            out, acts = thermo(data, return_activations=True)
            loss = F.cross_entropy(out, target)
            # Add STRONG GL Regularization
            loss += gl_regularization(acts, lambda_grad=5.0) 
            loss.backward()
            opt_thermo.step()
            
    sv_thermo = get_singular_values(thermo)
    
    # 3. Plot
    plt.figure(figsize=(10, 6))
    plt.plot(sv_base, label='Baseline (Standard)', linewidth=2, alpha=0.8)
    plt.plot(sv_thermo, label='Thermo-GL (Structured)', linewidth=3, alpha=0.9)
    plt.yscale('log')
    plt.title('Ideal Low-Rank Emergence (Aggressive GL Reg)')
    plt.xlabel('Singular Value Index')
    plt.ylabel('Normalized Singular Value (Log Scale)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    path = 'Thermo_Sparsity/thermo_low_rank_ideal.png'
    plt.savefig(path)
    print(f"Saved ideal figure to {path}")

if __name__ == "__main__":
    run_demo()
