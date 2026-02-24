import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import os
import math

# Set random seed
torch.manual_seed(42)
np.random.seed(42)

# ==========================================
# 1. Data Preparation
# ==========================================

def get_dataloaders(batch_size=64):
    # Standard MNIST transformation
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Test set: We will manually rotate in the test loop
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Check if data exists, if not, download
    if not os.path.exists('./data'):
        os.makedirs('./data')

    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform_train)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform_test)
    
    # Use smaller subsets for speed and clear visualization
    # 2000 training samples
    train_subset = Subset(train_dataset, range(2000))
    # 500 test samples
    test_subset = Subset(test_dataset, range(500))

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

# ==========================================
# 2. Models
# ==========================================

class BaselineCNN(nn.Module):
    """Standard CNN"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x, return_feats=False):
        # x: [B, 1, 28, 28]
        x = F.relu(F.max_pool2d(self.conv1(x), 2)) # -> [B, 16, 14, 14]
        x = F.relu(F.max_pool2d(self.conv2(x), 2)) # -> [B, 32, 7, 7]
        x = x.view(-1, 32 * 7 * 7)
        feat = F.relu(self.fc1(x)) # Feature vector [B, 128]
        out = self.fc2(feat)
        
        if return_feats:
            return out, feat
        return out

class NeuralGaugeField(nn.Module):
    """
    Robust NGF Implementation with Group Pooling
    """
    def __init__(self, num_rotations=8):
        super().__init__()
        self.num_rotations = num_rotations
        self.encoder = BaselineCNN()
        
    def forward(self, x, return_feats=False):
        # x: [B, 1, 28, 28]
        B, C, H, W = x.shape
        
        # 1. Generate Rotated Inputs (Group Lifting)
        logits_list = []
        feats_list = []
        
        for i in range(self.num_rotations):
            angle = i * (360.0 / self.num_rotations)
            
            if angle == 0:
                x_rot = x.clone()
            else:
                angle_rad = -math.radians(angle)
                theta = torch.tensor([
                    [math.cos(angle_rad), -math.sin(angle_rad), 0],
                    [math.sin(angle_rad),  math.cos(angle_rad), 0]
                ], device=x.device, dtype=x.dtype)
                theta = theta.unsqueeze(0).repeat(B, 1, 1)
                grid = F.affine_grid(theta, x.size(), align_corners=False)
                x_rot = F.grid_sample(x, grid, mode='bilinear', align_corners=False)
            
            # Pass through shared encoder
            logits, feats = self.encoder(x_rot, return_feats=True)
            logits_list.append(logits.unsqueeze(1)) # [B, 1, 10]
            feats_list.append(feats.unsqueeze(1))   # [B, 1, 128]
            
        # 2. Gauge Pooling (Max over group orbit)
        all_logits = torch.cat(logits_list, dim=1)
        all_feats = torch.cat(feats_list, dim=1)
        
        # Max-Pool over angles for logits
        output, max_indices = torch.max(all_logits, dim=1)
        
        # For features, we select the feature corresponding to the max logit
        # This represents the "canonical pose" feature
        # max_indices: [B, 10] (class-wise max). We need sample-wise max.
        # Let's approximate by taking max over logits sum or max logit value
        # Simpler: just max pool features element-wise (robust feature extraction)
        feats, _ = torch.max(all_feats, dim=1)
        
        if return_feats:
            return output, feats
        return output

# ==========================================
# 3. Experiment Loop
# ==========================================

def train(model, loader, optimizer, device):
    model.train()
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

def extract_features_for_tsne(model, loader, device, target_angles=[0, 45, 90]):
    """Extract features for specific rotation angles"""
    model.eval()
    features = []
    labels = []
    angles_list = []
    
    # Only take one batch for visualization to keep plot clean
    data_iter = iter(loader)
    data, target = next(data_iter)
    data, target = data.to(device), target.to(device)
    
    with torch.no_grad():
        for angle in target_angles:
            # Rotate batch
            if angle == 0:
                data_rot = data.clone()
            else:
                B = data.shape[0]
                angle_rad = -math.radians(angle)
                theta = torch.tensor([
                    [math.cos(angle_rad), -math.sin(angle_rad), 0],
                    [math.sin(angle_rad),  math.cos(angle_rad), 0]
                ], device=device, dtype=data.dtype)
                theta = theta.unsqueeze(0).repeat(B, 1, 1)
                grid = F.affine_grid(theta, data.size(), align_corners=False)
                data_rot = F.grid_sample(data, grid, mode='bilinear', align_corners=False)
            
            # Extract features
            _, feats = model(data_rot, return_feats=True)
            
            features.append(feats.cpu().numpy())
            labels.append(target.cpu().numpy())
            # Mark which angle this batch belongs to
            angles_list.append(np.full(target.shape[0], angle))
            
    return np.concatenate(features), np.concatenate(labels), np.concatenate(angles_list)

def run_experiment():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    train_loader, test_loader = get_dataloaders()
    
    # 1. Train Models
    baseline = BaselineCNN().to(device)
    ngf = NeuralGaugeField(num_rotations=8).to(device)
    
    opt_base = optim.Adam(baseline.parameters(), lr=0.001)
    opt_ngf = optim.Adam(ngf.parameters(), lr=0.001)
    
    print("\nTraining Baseline (Standard CNN)...")
    for epoch in range(3): # Short training for demo
        train(baseline, train_loader, opt_base, device)
        print(f"Epoch {epoch+1} done.")
        
    print("\nTraining NGF (Neural Gauge Field)...")
    for epoch in range(3):
        train(ngf, train_loader, opt_ngf, device)
        print(f"Epoch {epoch+1} done.")

    # 2. t-SNE Visualization
    print("\nGenerating t-SNE visualization...")
    
    # Extract features from both models on 0, 45, 90 degree inputs
    angles_to_test = [0, 45, 90]
    
    feats_base, labels_base, angles_base = extract_features_for_tsne(baseline, test_loader, device, angles_to_test)
    feats_ngf, labels_ngf, angles_ngf = extract_features_for_tsne(ngf, test_loader, device, angles_to_test)
    
    # Compute t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    
    # Process Baseline
    tsne_base = tsne.fit_transform(feats_base)
    
    # Process NGF
    tsne_ngf = tsne.fit_transform(feats_ngf)
    
    # 3. Plotting
    plt.figure(figsize=(16, 7))
    
    # Plot Baseline
    plt.subplot(1, 2, 1)
    colors = ['r', 'g', 'b'] # Colors for angles
    markers = ['o', 's', '^']
    
    for i, angle in enumerate(angles_to_test):
        mask = angles_base == angle
        # Only plot first 3 digits (classes) to avoid clutter
        subset_mask = mask & (labels_base < 3) 
        plt.scatter(tsne_base[subset_mask, 0], tsne_base[subset_mask, 1], 
                   c=colors[i], label=f'{angle}° Input', alpha=0.6, s=40)
        
    plt.title('Baseline CNN Feature Space\n(Misaligned Manifolds)', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot NGF
    plt.subplot(1, 2, 2)
    for i, angle in enumerate(angles_to_test):
        mask = angles_ngf == angle
        subset_mask = mask & (labels_ngf < 3)
        plt.scatter(tsne_ngf[subset_mask, 0], tsne_ngf[subset_mask, 1], 
                   c=colors[i], label=f'{angle}° Input', alpha=0.6, s=40)
        
    plt.title('Neural Gauge Field Feature Space\n(Aligned Manifolds)', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_path = os.path.join('figures', 'ngf_manifold_alignment.png')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    print(f"Visualization saved to {output_path}")

if __name__ == "__main__":
    run_experiment()
