import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import math

# Set seeds
torch.manual_seed(42)
np.random.seed(42)

# ==========================================
# 1. Synthetic 3D Dataset Generation
# ==========================================

class GeometricShapes3D(Dataset):
    """
    Synthetic 3D Point Cloud Dataset: Cube, Sphere, Tetrahedron
    """
    def __init__(self, num_samples=1000, num_points=1024, mode='train'):
        self.num_samples = num_samples
        self.num_points = num_points
        self.mode = mode
        self.data, self.labels = self._generate_data()

    def _generate_data(self):
        data = []
        labels = []
        
        for _ in range(self.num_samples):
            # Equal probability for 3 classes
            shape_type = np.random.randint(0, 3)
            points = self._sample_shape(shape_type)
            
            # For training, we keep them upright (or small noise)
            # For testing, we will rotate them manually in the eval loop
            if self.mode == 'train':
                # Add small jitter noise
                points += np.random.normal(0, 0.02, points.shape)
                
            data.append(points)
            labels.append(shape_type)
            
        return torch.tensor(np.array(data), dtype=torch.float32), torch.tensor(np.array(labels), dtype=torch.long)

    def _sample_shape(self, shape_type):
        # 0: Cube, 1: Sphere, 2: Tetrahedron
        if shape_type == 0:
            # Cube: uniform sampling from faces
            points = np.random.rand(self.num_points, 3) * 2 - 1
            # Project to surface (simple approximation: push to nearest face)
            # Actually, let's just sample inside a cube to be simple, 
            # or surface? Surface is better for shape.
            # Simple surface sampling:
            # Fix one coordinate to +/- 1, randomize others
            # To be simple and fast: just use random points inside [-1, 1]
            # But "Solid Cube" vs "Solid Sphere" is hard to distinguish by just distribution?
            # Let's use Surface sampling.
            return self._sample_cube_surface()
        elif shape_type == 1:
            # Sphere: normalize random gaussian
            vec = np.random.randn(self.num_points, 3)
            vec /= np.linalg.norm(vec, axis=1)[:, np.newaxis]
            return vec
        elif shape_type == 2:
            # Tetrahedron: barycentric coordinates from vertices
            verts = np.array([[1,1,1], [1,-1,-1], [-1,1,-1], [-1,-1,1]]) / math.sqrt(3)
            # Sample faces
            return self._sample_tetra_surface(verts)
            
    def _sample_cube_surface(self):
        points = []
        for _ in range(self.num_points):
            axis = np.random.randint(0, 3)
            sign = np.random.choice([-1, 1])
            pt = np.random.rand(3) * 2 - 1
            pt[axis] = sign
            points.append(pt)
        return np.array(points)

    def _sample_tetra_surface(self, verts):
        # 4 faces
        faces = [[0,1,2], [0,1,3], [0,2,3], [1,2,3]]
        points = []
        for _ in range(self.num_points):
            face_idx = np.random.randint(0, 4)
            face = faces[face_idx]
            # Barycentric sampling
            r1, r2 = np.random.rand(), np.random.rand()
            if r1 + r2 > 1:
                r1 = 1 - r1
                r2 = 1 - r2
            a, b, c = verts[face[0]], verts[face[1]], verts[face[2]]
            pt = a + r1*(b-a) + r2*(c-a)
            points.append(pt)
        return np.array(points)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# ==========================================
# 2. Models: PointNet (Baseline) vs NGF-PointNet
# ==========================================

class PointNetBaseline(nn.Module):
    def __init__(self, k=3):
        super().__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        # x: [B, N, 3] -> transpose to [B, 3, N]
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0] # Global Max Pooling
        x = x.view(-1, 1024)
        
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        return x

class GaugePointNet(nn.Module):
    """
    NGF for Point Cloud:
    Instead of T-Net (which learns a single matrix), we implement 
    Group Pooling over discrete SO(3) rotations.
    """
    def __init__(self, k=3, num_rotations=4):
        super().__init__()
        self.encoder = PointNetBaseline(k)
        self.num_rotations = num_rotations
        
        # Pre-compute some rotation matrices (e.g. around Z axis, or random)
        self.rotations = self._generate_rotations(num_rotations)

    def _generate_rotations(self, n):
        # Generate N rotations around Z axis + Y axis
        rots = []
        # Identity
        rots.append(torch.eye(3))
        # Z-rotations
        for i in range(1, n):
            # Simplification: Just rotate around Z for this demo
            angle = i * (2 * math.pi / n)
            c, s = math.cos(angle), math.sin(angle)
            R = torch.tensor([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=torch.float32)
            rots.append(R)
        return torch.stack(rots) # [N, 3, 3]

    def forward(self, x):
        # x: [B, N, 3]
        B = x.shape[0]
        
        # We need to apply ALL rotations to x, pass through encoder, and pooling
        # Since PointNet is small, we can batch this.
        
        # Move rotations to device
        rots = self.rotations.to(x.device) # [R, 3, 3]
        
        # Expand x: [B, 1, N, 3]
        x_exp = x.unsqueeze(1) 
        
        # Expand rots: [1, R, 3, 3]
        rots_exp = rots.unsqueeze(0)
        
        # Apply rotation: x @ R^T
        # [B, 1, N, 3] @ [1, R, 3, 3] (transposed) -> [B, R, N, 3]
        # Actually easier: iterate
        
        logits_list = []
        for i in range(self.num_rotations):
            R = rots[i].to(x.device)
            # x: [B, N, 3], R: [3, 3] -> x @ R.T : [B, N, 3]
            x_rot = torch.matmul(x, R.t())
            logits = self.encoder(x_rot) # [B, k]
            logits_list.append(logits.unsqueeze(1))
            
        all_logits = torch.cat(logits_list, dim=1) # [B, R, k]
        
        # Gauge Pooling: Max over rotations
        output, _ = torch.max(all_logits, dim=1)
        return output

# ==========================================
# 3. Experiment Utilities
# ==========================================

def get_rotation_matrix(angle_deg, axis='z'):
    angle = math.radians(angle_deg)
    c, s = math.cos(angle), math.sin(angle)
    if axis == 'z':
        return torch.tensor([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=torch.float32)
    elif axis == 'y':
        return torch.tensor([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=torch.float32)
    elif axis == 'x':
        return torch.tensor([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=torch.float32)

def evaluate_3d_accuracy(model, loader, device):
    model.eval()
    # Test on rotations around Z axis from 0 to 360
    angles = [0, 45, 90, 135, 180, 225, 270, 315]
    accuracies = []
    
    with torch.no_grad():
        for angle in angles:
            correct = 0
            total = 0
            R = get_rotation_matrix(angle, 'z').to(device)
            
            for data, target in loader:
                data, target = data.to(device), target.to(device)
                # Rotate input
                data_rot = torch.matmul(data, R.t())
                
                out = model(data_rot)
                pred = out.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += data.shape[0]
            
            accuracies.append(100. * correct / total)
    return angles, accuracies

# ==========================================
# 4. Main Run
# ==========================================

def run_experiment():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Generate Data
    print("Generating Synthetic 3D Data (Cube, Sphere, Tetrahedron)...")
    train_ds = GeometricShapes3D(num_samples=1000, mode='train')
    test_ds = GeometricShapes3D(num_samples=200, mode='test')
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)
    
    # 2. Train Baseline
    baseline = PointNetBaseline(k=3).to(device)
    opt_base = optim.Adam(baseline.parameters(), lr=0.001)
    
    print("\nTraining Baseline PointNet (Upright)...")
    for epoch in range(10):
        baseline.train()
        total_loss = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            opt_base.zero_grad()
            out = baseline(data)
            loss = F.cross_entropy(out, target)
            loss.backward()
            opt_base.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")
        
    # 3. Train NGF
    # We use 8 discrete rotations in the gauge group
    ngf = GaugePointNet(k=3, num_rotations=8).to(device)
    opt_ngf = optim.Adam(ngf.parameters(), lr=0.001)
    
    print("\nTraining Gauge-Equivariant PointNet (Upright)...")
    for epoch in range(10):
        ngf.train()
        total_loss = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            opt_ngf.zero_grad()
            out = ngf(data)
            loss = F.cross_entropy(out, target)
            loss.backward()
            opt_ngf.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")
        
    # 4. Evaluate
    print("\nEvaluating 3D Geometric Generalization...")
    angles, acc_base = evaluate_3d_accuracy(baseline, test_loader, device)
    angles, acc_ngf = evaluate_3d_accuracy(ngf, test_loader, device)
    
    # 5. Plot
    plt.figure(figsize=(8, 6))
    plt.plot(angles, acc_base, 'o--', color='gray', label='Baseline PointNet', linewidth=2)
    plt.plot(angles, acc_ngf, '*-', color='red', label='Gauge PointNet (Ours)', linewidth=3)
    plt.xlabel('Z-Axis Rotation Angle (°)')
    plt.ylabel('Accuracy (%)')
    plt.title('3D Geometric Generalization (Synthetic Shapes)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 100)
    plt.xticks(angles)
    
    out_file = os.path.join('figures', 'ngf_3d_pointcloud_result.png')
    os.makedirs('figures', exist_ok=True)
    plt.savefig(out_file)
    print(f"Results saved to {out_file}")
    
    print(f"\nMean Acc (0-360°):")
    print(f"  Baseline: {np.mean(acc_base):.1f}%")
    print(f"  NGF (Ours): {np.mean(acc_ngf):.1f}%")

if __name__ == "__main__":
    run_experiment()
