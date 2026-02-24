import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
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
    
    # Test set: We will manually rotate in the test loop to ensure fairness
    # So here we just load upright images
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform_train)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform_test)
    
    # Use smaller subsets
    train_subset = Subset(train_dataset, range(2000))  # 2000 training samples
    test_subset = Subset(test_dataset, range(500))     # 500 test samples

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

    def forward(self, x):
        # x: [B, 1, 28, 28]
        x = F.relu(F.max_pool2d(self.conv1(x), 2)) # -> [B, 16, 14, 14]
        x = F.relu(F.max_pool2d(self.conv2(x), 2)) # -> [B, 32, 7, 7]
        x = x.view(-1, 32 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class NeuralGaugeField(nn.Module):
    """
    Robust NGF Implementation:
    Input-side Group Lifting (Rotated Inputs -> Shared Encoder -> MaxPool)
    """
    def __init__(self, num_rotations=8):
        super().__init__()
        self.num_rotations = num_rotations
        
        # Shared Feature Extractor (Same architecture as Baseline)
        self.encoder = BaselineCNN()
        # Remove the last classification layer from encoder to use it as feature extractor
        # Actually, let's just use the full BaselineCNN and pool the LOGITS.
        # This is "TI-Pooling" (Transformation Invariant Pooling) on the output.
        
    def forward(self, x):
        # x: [B, 1, 28, 28]
        B, C, H, W = x.shape
        
        # 1. Generate Rotated Inputs
        # We process each rotation angle.
        logits_list = []
        
        for i in range(self.num_rotations):
            angle = i * (360.0 / self.num_rotations)
            
            if angle == 0:
                x_rot = x
            else:
                # Rotate input image
                # Use TF.rotate or affine_grid. Since we are in forward, use affine_grid
                angle_rad = -math.radians(angle)
                theta = torch.tensor([
                    [math.cos(angle_rad), -math.sin(angle_rad), 0],
                    [math.sin(angle_rad),  math.cos(angle_rad), 0]
                ], device=x.device, dtype=x.dtype)
                theta = theta.unsqueeze(0).repeat(B, 1, 1)
                grid = F.affine_grid(theta, x.size(), align_corners=False)
                x_rot = F.grid_sample(x, grid, mode='bilinear', align_corners=False)
            
            # 2. Pass through Shared Encoder
            # logits: [B, 10]
            logits = self.encoder(x_rot)
            logits_list.append(logits.unsqueeze(1)) # [B, 1, 10]
            
        # 3. Gauge Pooling (Max over rotations)
        # Stack: [B, Angles, 10]
        all_logits = torch.cat(logits_list, dim=1)
        
        # Max-Pool over Angles: "Which angle gave the strongest confidence?"
        # output: [B, 10]
        output, _ = torch.max(all_logits, dim=1)
        
        return output

# ==========================================
# 3. Experiment Loop
# ==========================================

def train(model, loader, optimizer):
    model.train()
    for data, target in loader:
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

def evaluate_rotation_robustness(model, loader, model_name):
    model.eval()
    
    # We test on 8 fixed angles to get a curve
    angles = [0, 45, 90, 135, 180, 225, 270, 315]
    accuracies = []
    
    print(f"\nEvaluating {model_name} on rotations...")
    
    for angle in angles:
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in loader:
                B = data.shape[0]
                # Manually rotate the batch by 'angle'
                if angle != 0:
                    angle_rad = -math.radians(angle)
                    theta = torch.tensor([
                        [math.cos(angle_rad), -math.sin(angle_rad), 0],
                        [math.sin(angle_rad),  math.cos(angle_rad), 0]
                    ], device=data.device, dtype=data.dtype)
                    theta = theta.unsqueeze(0).repeat(B, 1, 1)
                    grid = F.affine_grid(theta, data.size(), align_corners=False)
                    data = F.grid_sample(data, grid, mode='bilinear', align_corners=False)
                
                output = model(data)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += B
        
        acc = 100. * correct / total
        accuracies.append(acc)
        print(f"Angle {angle:>3}°: {acc:.2f}%")
        
    return angles, accuracies

def run_experiment():
    train_loader, test_loader = get_dataloaders()
    
    # 1. Train Baseline
    baseline = BaselineCNN()
    opt_base = optim.Adam(baseline.parameters(), lr=0.001)
    
    print("\nTraining Baseline CNN (Upright)...")
    for epoch in range(5):
        train(baseline, train_loader, opt_base)
        print(f"Epoch {epoch+1} done.")
        
    # 2. Train NGF
    # NGF uses a BaselineCNN inside. To be fair, we initialize a fresh one
    # and train it from scratch.
    # Note: NGF is computationally heavier (8x passes), so it learns "slower" in wall-clock
    # but sees "more" augmented data effectively.
    ngf = NeuralGaugeField(num_rotations=8)
    opt_ngf = optim.Adam(ngf.parameters(), lr=0.001)
    
    print("\nTraining NGF (Upright)...")
    for epoch in range(5):
        train(ngf, train_loader, opt_ngf)
        print(f"Epoch {epoch+1} done.")
        
    # 3. Evaluate
    angles, acc_base = evaluate_rotation_robustness(baseline, test_loader, "Baseline")
    angles, acc_ngf = evaluate_rotation_robustness(ngf, test_loader, "NGF")
    
    # 4. Visualization
    plt.figure(figsize=(10, 6))
    plt.plot(angles, acc_base, 'o--', color='gray', label='Baseline CNN', linewidth=2)
    plt.plot(angles, acc_ngf, 'o-', color='green', label='Neural Gauge Field (NGF)', linewidth=3)
    
    plt.xlabel('Test Rotation Angle (Degrees)')
    plt.ylabel('Accuracy (%)')
    plt.title('Zero-Shot Geometric Generalization: NGF vs Baseline')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xticks(angles)
    plt.ylim(0, 100)
    
    # Calculate Mean Accuracy across all angles
    mean_base = np.mean(acc_base)
    mean_ngf = np.mean(acc_ngf)
    print(f"\nMean Accuracy (0-360): Baseline={mean_base:.1f}%, NGF={mean_ngf:.1f}%")
    
    output_path = os.path.join('figures', 'ngf_rotation_experiment.png')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    run_experiment()
