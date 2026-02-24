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
# 1. Model Definition: MLP with GL Regularization
# ==========================================

class ThermodynamicMLP(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=512, output_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x, return_activations=False):
        x = x.view(x.size(0), -1)
        
        # Layer 1
        h1_pre = self.fc1(x)
        h1 = F.relu(h1_pre)
        
        # Layer 2
        h2_pre = self.fc2(h1)
        h2 = F.relu(h2_pre)
        
        # Output
        out = self.fc3(h2)
        
        if return_activations:
            return out, [h1, h2]
        return out

# ==========================================
# 2. Regularization Functions
# ==========================================

def gl_regularization(activations, lambda_grad=0.01):
    """
    Calculates the Gradient Energy (Dirichlet Energy) of the activations.
    This encourages neurons to form 'domains' (clusters of similar activity).
    """
    loss = 0
    for h in activations:
        # h: [Batch, Hidden_Dim]
        # Topological Assumption: Neurons are arranged in a 1D ring manifold.
        # We enforce periodic boundary conditions for smoothness.
        diff = h[:, 1:] - h[:, :-1]
        grad_energy = torch.mean(diff ** 2)
        loss += grad_energy
    return lambda_grad * loss

def l1_regularization(activations, lambda_l1=0.01):
    """
    Standard L1 Regularization (Lasso)
    Encourages random sparsity (individual neurons turn off).
    """
    loss = 0
    for h in activations:
        loss += torch.mean(torch.abs(h))
    return lambda_l1 * loss

# ==========================================
# 3. Experiment Utils
# ==========================================

def get_singular_values(model):
    """Compute SVD of the weight matrices to measure Rank"""
    svs = []
    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'weight' in name and len(param.shape) == 2:
                _, S, _ = torch.svd(param)
                # Normalize singular values
                S = S / S[0]
                svs.append(S.cpu().numpy())
    return svs

def train(model, loader, optimizer, device, reg_type=None):
    model.train()
    total_loss = 0
    
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        
        output, activations = model(data, return_activations=True)
        ce_loss = F.cross_entropy(output, target)
        
        loss = ce_loss
        if reg_type == 'gl':
            loss += gl_regularization(activations, lambda_grad=0.1)
        elif reg_type == 'l1':
            loss += l1_regularization(activations, lambda_l1=0.01)
            
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    return total_loss / len(loader)

def test(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    return 100. * correct / total

# ==========================================
# 4. Main Experiment
# ==========================================

def run_experiment():
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists('figures'): os.makedirs('figures')
    print(f"Using device: {device}")
    
    # Data
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    
    # Check if data exists
    if not os.path.exists('./data'):
        os.makedirs('./data')
        
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    # === OPTIMIZATION: Add Label Noise to force overfitting ===
    # This destroys SGD's implicit bias and highlights the power of GL regularization
    noise_ratio = 0.2
    num_train = 2000
    indices = torch.randperm(len(train_dataset))[:num_train]
    
    # Corrupt labels manually
    targets = train_dataset.targets[indices].clone()
    num_corrupt = int(noise_ratio * num_train)
    corrupt_indices = torch.randperm(num_train)[:num_corrupt]
    random_labels = torch.randint(0, 10, (num_corrupt,))
    targets[corrupt_indices] = random_labels
    train_dataset.targets[indices] = targets
    
    print(f"Applied {noise_ratio*100}% label noise to training set.")
    # ==========================================================
    
    train_set = Subset(train_dataset, indices)
    test_set = Subset(test_dataset, range(1000))
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False)
    
    # 1. Train Baseline
    print("Training Baseline Model...")
    baseline = ThermodynamicMLP().to(device)
    opt_base = optim.Adam(baseline.parameters(), lr=0.001)
    
    for epoch in range(10):
        train(baseline, train_loader, opt_base, device, reg_type=None)
        
    acc_base = test(baseline, test_loader, device)
    print(f"Baseline Accuracy: {acc_base:.2f}%")
    sv_base = get_singular_values(baseline)
    
    # 2. Train Thermo-Reg Model (With GL)
    print("\nTraining Thermodynamic Model (with GL Reg)...")
    thermo_model = ThermodynamicMLP().to(device)
    opt_thermo = optim.Adam(thermo_model.parameters(), lr=0.001)
    
    for epoch in range(10):
        train(thermo_model, train_loader, opt_thermo, device, reg_type='gl')
        
    acc_thermo = test(thermo_model, test_loader, device)
    print(f"Thermo Model Accuracy: {acc_thermo:.2f}%")
    sv_thermo = get_singular_values(thermo_model)

    # 3. Train L1-Reg Model (Control Group)
    print("\nTraining L1-Reg Model (Standard Sparsity)...")
    l1_model = ThermodynamicMLP().to(device)
    opt_l1 = optim.Adam(l1_model.parameters(), lr=0.001)
    
    for epoch in range(10):
        train(l1_model, train_loader, opt_l1, device, reg_type='l1')
        
    acc_l1 = test(l1_model, test_loader, device)
    print(f"L1 Model Accuracy: {acc_l1:.2f}%")
    sv_l1 = get_singular_values(l1_model)
    
    # 4. Visualization: SVD Spectrum Decay
    plt.figure(figsize=(10, 5))
    layer_idx = 1 # fc2 (Layer 2 weights)
    
    if layer_idx < len(sv_base):
        plt.plot(sv_base[layer_idx], label='Baseline', alpha=0.7, linewidth=2, linestyle='--')
        plt.plot(sv_l1[layer_idx], label='L1 Regularization', alpha=0.7, linewidth=2)
        plt.plot(sv_thermo[layer_idx], label='Thermodynamic (GL)', alpha=0.7, linewidth=3)
        
        plt.yscale('log')
        plt.title('Emergence of Low-Rankness: Singular Value Spectrum (Layer 2)')
        plt.xlabel('Singular Value Index')
        plt.ylabel('Normalized Singular Value (Log Scale)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        save_path = 'figures/thermo_low_rank_emergence.png'
        plt.savefig(save_path)
        print(f"\nPlot saved to {save_path}")
    else:
        print("Error: Could not extract singular values for plotting.")

    # 5. Visualization: Activation Map (Magnetic Domains)
    # Extract one batch
    data_iter = iter(test_loader)
    data, _ = next(data_iter)
    data = data.to(device)
    
    _, acts_base = baseline(data, return_activations=True)
    _, acts_thermo = thermo_model(data, return_activations=True)
    _, acts_l1 = l1_model(data, return_activations=True)
    
    # Visualize hidden layer 1 for the first 100 neurons of the first sample
    h_base = acts_base[0][0, :100].detach().cpu().numpy()
    h_thermo = acts_thermo[0][0, :100].detach().cpu().numpy()
    h_l1 = acts_l1[0][0, :100].detach().cpu().numpy()
    
    plt.figure(figsize=(18, 4))
    
    plt.subplot(1, 3, 1)
    plt.bar(range(100), h_base, color='gray')
    plt.title('Baseline (Noisy/High Entropy)')
    plt.ylim(0, max(h_base.max(), h_thermo.max(), h_l1.max())*1.1)
    
    plt.subplot(1, 3, 2)
    plt.bar(range(100), h_l1, color='blue')
    plt.title('L1 Lasso (Random Sparsity)')
    plt.ylim(0, max(h_base.max(), h_thermo.max(), h_l1.max())*1.1)
    
    plt.subplot(1, 3, 3)
    plt.bar(range(100), h_thermo, color='green')
    plt.title('Thermodynamic (Structured/Domains)')
    plt.ylim(0, max(h_base.max(), h_thermo.max(), h_l1.max())*1.1)
    
    save_path_act = 'figures/thermo_magnetic_domains.png'
    plt.savefig(save_path_act)
    print(f"Activation plot saved to {save_path_act}")

if __name__ == "__main__":
    run_experiment()
