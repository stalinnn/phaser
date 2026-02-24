import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

def gl_loss(gate_activations, kappa=0.1, alpha=-1.0, beta=1.0):
    """
    Calculate Ginzburg-Landau Free Energy Loss
    
    Args:
        gate_activations: Tensor of shape [Batch, H, W] representing the 2D grid of neuron gates.
                          Values should be roughly in [0, 1] (sigmoid output).
        kappa: Gradient cost coefficient (Transport Stiffness)
        alpha: Local potential quadratic coefficient (Entropy control)
        beta: Local potential quartic coefficient (Saturation control)
        
    Returns:
        Scalar loss
    """
    # 1. Local Potential Energy: f_local = alpha * |Psi|^2 + (beta/2) * |Psi|^4
    # We want a double-well potential to encourage 0 or 1 (binary gating).
    # A typical double well is (psi^2 - 1)^2 = psi^4 - 2psi^2 + 1. 
    # Here we use a form that works well for Sigmoid outputs [0, 1]: Psi^2 * (1-Psi)^2
    # This has minima at 0 and 1.
    psi = gate_activations
    potential_energy = torch.mean(psi**2 * (1 - psi)**2)
    
    # 2. Gradient Energy (Transport Cost): f_grad = (kappa/2) * |nabla Psi|^2
    # Calculate finite differences along H and W dimensions
    # diff_h: [Batch, H-1, W]
    diff_h = psi[:, 1:, :] - psi[:, :-1, :]
    # diff_w: [Batch, H, W-1]
    diff_w = psi[:, :, 1:] - psi[:, :, :-1]
    
    gradient_energy = torch.mean(diff_h**2) + torch.mean(diff_w**2)
    
    return potential_energy + kappa * gradient_energy

class ThermodynamicLayer(nn.Module):
    def __init__(self, input_dim, grid_h, grid_w):
        super().__init__()
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.hidden_dim = grid_h * grid_w
        
        # Linear transformation for features
        self.fc = nn.Linear(input_dim, self.hidden_dim)
        
        # Gating mechanism: A separate path to determine "activation"
        # In TGN, this monitors "Surprise" or Error, here simplified as a learned gate.
        self.gate_fc = nn.Linear(input_dim, self.hidden_dim)
        
    def forward(self, x):
        # x: [Batch, Input_Dim]
        
        # 1. Compute Feature activations (Pre-gate)
        features = torch.relu(self.fc(x))
        
        # 2. Compute Gate values (Order Parameter Psi)
        # Sigmoid to constrain between 0 and 1
        gates_flat = torch.sigmoid(self.gate_fc(x))
        
        # 3. Apply Gate
        gated_features = features * gates_flat
        
        # Reshape gates to 2D grid for GL Loss calculation
        gates_grid = gates_flat.view(-1, self.grid_h, self.grid_w)
        
        return gated_features, gates_grid

class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, grid_h, grid_w, num_classes):
        super().__init__()
        self.tgn_layer = ThermodynamicLayer(input_dim, grid_h, grid_w)
        self.output_head = nn.Linear(grid_h * grid_w, num_classes)
        
    def forward(self, x):
        features, gates_grid = self.tgn_layer(x)
        logits = self.output_head(features)
        return logits, gates_grid

def run_experiment():
    print("Loading Digits dataset...")
    digits = load_digits()
    X = digits.data
    y = digits.target
    
    # Standardize
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Convert to Tensor
    X_train = torch.FloatTensor(X_train)
    y_train = torch.LongTensor(y_train)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.LongTensor(y_test)
    
    # Parameters
    INPUT_DIM = 64 # 8x8 images
    GRID_H = 16    # Hidden layer arranged as 16x16 grid (256 neurons)
    GRID_W = 16
    NUM_CLASSES = 10
    EPOCHS = 200
    LR = 0.01
    
    # --- Experiment A: Baseline (No GL Loss) ---
    print("\nTraining Baseline Model (No GL Loss)...")
    model_base = SimpleClassifier(INPUT_DIM, GRID_H, GRID_W, NUM_CLASSES)
    optimizer_base = optim.Adam(model_base.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    
    loss_history_base = []
    
    for epoch in range(EPOCHS):
        optimizer_base.zero_grad()
        logits, _ = model_base(X_train)
        loss = criterion(logits, y_train)
        loss.backward()
        optimizer_base.step()
        loss_history_base.append(loss.item())
        if epoch % 50 == 0:
            print(f"Epoch {epoch}: Loss {loss.item():.4f}")

    # --- Experiment B: RTN (With GL Loss) ---
    print("\nTraining RTN Model (With Ginzburg-Landau Loss)...")
    model_rtn = SimpleClassifier(INPUT_DIM, GRID_H, GRID_W, NUM_CLASSES)
    optimizer_rtn = optim.Adam(model_rtn.parameters(), lr=LR)
    
    loss_history_rtn = []
    
    GL_WEIGHT = 2.0  # Weight for the physical constraint
    KAPPA = 5.0      # Strength of gradient penalty (stiffness)
    
    for epoch in range(EPOCHS):
        optimizer_rtn.zero_grad()
        logits, gates_grid = model_rtn(X_train)
        
        # Task Loss
        task_loss = criterion(logits, y_train)
        
        # Physics Loss (GL)
        phys_loss = gl_loss(gates_grid, kappa=KAPPA)
        
        # Total Loss
        total_loss = task_loss + GL_WEIGHT * phys_loss
        
        total_loss.backward()
        optimizer_rtn.step()
        loss_history_rtn.append(total_loss.item())
        if epoch % 50 == 0:
            print(f"Epoch {epoch}: Total {total_loss.item():.4f} (Task {task_loss.item():.4f} + Phys {phys_loss.item():.4f})")

    # --- Visualization ---
    print("\nGenerating Visualization...")
    
    # Get average gate activation patterns for a few test samples
    with torch.no_grad():
        _, gates_base = model_base(X_test[:50]) # Take 50 samples
        _, gates_rtn = model_rtn(X_test[:50])
        
        # Average across the batch to see the "Structure" of the layer
        avg_gate_base = torch.mean(gates_base, dim=0).numpy()
        avg_gate_rtn = torch.mean(gates_rtn, dim=0).numpy()
        
        # Also pick a SINGLE sample to see individual activation
        single_gate_base = gates_base[0].numpy()
        single_gate_rtn = gates_rtn[0].numpy()
        
        # Print stats to understand the scale
        print(f"Baseline Max: {single_gate_base.max():.4f}, Mean: {single_gate_base.mean():.4f}")
        print(f"RTN Max: {single_gate_rtn.max():.4f}, Mean: {single_gate_rtn.mean():.4f}")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot Baseline Average
    im0 = axes[0, 0].imshow(avg_gate_base, cmap='viridis', vmin=0, vmax=1)
    axes[0, 0].set_title("Baseline: Average Gate Activity\n(No Physical Constraint)")
    fig.colorbar(im0, ax=axes[0, 0])
    
    # Plot RTN Average (Auto-scale to see structure)
    im1 = axes[0, 1].imshow(avg_gate_rtn, cmap='viridis')
    axes[0, 1].set_title(f"RTN: Average Gate Activity\n(With GL Loss, Auto-scaled)")
    fig.colorbar(im1, ax=axes[0, 1])
    
    # Plot Baseline Single Sample
    im2 = axes[1, 0].imshow(single_gate_base, cmap='viridis', vmin=0, vmax=1)
    axes[1, 0].set_title("Baseline: Single Sample Gate\n(Noisy / Unstructured)")
    fig.colorbar(im2, ax=axes[1, 0])
    
    # Plot RTN Single Sample (Auto-scale to see structure)
    im3 = axes[1, 1].imshow(single_gate_rtn, cmap='viridis')
    axes[1, 1].set_title("RTN: Single Sample Gate\n(Domain Formation / Structured - Auto-scaled)")
    fig.colorbar(im3, ax=axes[1, 1])
    
    plt.tight_layout()
    output_path = os.path.join('figures', 'gl_domain_experiment.png')
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    plt.savefig(output_path)
    print(f"Visualization saved to {output_path}")

if __name__ == "__main__":
    run_experiment()
