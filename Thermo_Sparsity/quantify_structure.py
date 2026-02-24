import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import os

# Reuse model definition from previous script
class ThermodynamicMLP(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=512, output_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x, return_activations=False):
        x = x.view(x.size(0), -1)
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        out = self.fc3(h2)
        if return_activations:
            return out, [h1, h2]
        return out

def gl_regularization(activations, lambda_grad=0.01):
    loss = 0
    for h in activations:
        diff = h[:, 1:] - h[:, :-1]
        grad_energy = torch.mean(diff ** 2)
        loss += grad_energy
    return lambda_grad * loss

def l1_regularization(activations, lambda_l1=0.001):
    loss = 0
    for h in activations:
        loss += torch.mean(torch.abs(h))
    return lambda_l1 * loss

def train(model, loader, optimizer, device, reg_type=None):
    model.train()
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output, activations = model(data, return_activations=True)
        ce_loss = F.cross_entropy(output, target)
        loss = ce_loss
        if reg_type == 'gl':
            loss += gl_regularization(activations, lambda_grad=0.1)
        elif reg_type == 'l1':
            loss += l1_regularization(activations, lambda_l1=0.001)
        loss.backward()
        optimizer.step()

def calculate_spatial_autocorrelation(activations):
    """
    Quantifies the 'Structureness' (Magnetic Domain quality).
    Lag-1 Autocorrelation along the neuron dimension.
    Range: [-1, 1]. Higher means more structured/smooth. Near 0 means random.
    """
    # h: [Batch, Hidden_Dim]
    # We calculate per sample, then mean
    scores = []
    for h in activations:
        # h_i and h_{i+1}
        h_current = h[:, :-1]
        h_next = h[:, 1:]
        
        # Center the data
        h_current = h_current - h_current.mean(dim=1, keepdim=True)
        h_next = h_next - h_next.mean(dim=1, keepdim=True)
        
        # Covariance
        cov = torch.sum(h_current * h_next, dim=1)
        
        # Variances
        var_current = torch.sum(h_current**2, dim=1)
        var_next = torch.sum(h_next**2, dim=1)
        
        # Correlation
        corr = cov / (torch.sqrt(var_current * var_next) + 1e-8)
        scores.append(corr.mean().item())
        
    return np.mean(scores)

def run_metric_experiment():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(Subset(train_dataset, range(2000)), batch_size=64, shuffle=True)
    test_loader = DataLoader(Subset(test_dataset, range(1000)), batch_size=64, shuffle=False)
    
    # Define models
    models = {
        'Baseline': (ThermodynamicMLP().to(device), None),
        'L1-Lasso': (ThermodynamicMLP().to(device), 'l1'),
        'Thermo-GL': (ThermodynamicMLP().to(device), 'gl')
    }
    
    results = {}
    
    print("\nTraining and Measuring Structure Score...")
    print(f"{'Model':<15} | {'Structure Score (Autocorr)':<25}")
    print("-" * 45)
    
    for name, (model, reg_type) in models.items():
        opt = optim.Adam(model.parameters(), lr=0.001)
        
        # Quick Train
        for epoch in range(5):
            train(model, train_loader, opt, device, reg_type)
            
        # Measure Structure
        model.eval()
        all_acts = []
        with torch.no_grad():
            for data, _ in test_loader:
                data = data.to(device)
                _, acts = model(data, return_activations=True)
                all_acts.append(acts[1]) # Use Layer 2
        
        # Calculate metric
        full_acts = torch.cat(all_acts, dim=0)
        score = calculate_spatial_autocorrelation([full_acts])
        results[name] = score
        
        print(f"{name:<15} | {score:.4f}")
        
    print("-" * 45)
    print("Interpretation: Thermo-GL should have significantly higher score,")
    print("indicating strong spatial coherence (domains).")

if __name__ == "__main__":
    run_metric_experiment()
