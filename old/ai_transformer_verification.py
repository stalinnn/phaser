import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

"""
AI as a Model Organism: Transformer Learns Riemannian Geometry
------------------------------------------------------------
FIXED VERSION: WITH POSITIONAL ENCODING
"""

def generate_kuramoto_data(n_samples=2000, n_nodes=20):
    print("Generating Physics Data...")
    X = np.zeros((n_samples, n_nodes))
    Y = np.zeros((n_samples, n_nodes)) 
    
    # Topology: 1D Ring
    adj = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        adj[i, (i+1)%n_nodes] = 1
        adj[i, (i-1)%n_nodes] = 1
        
    COUPLING = 5.0
    dt = 0.2
    
    for i in range(n_samples):
        theta = np.random.uniform(-np.pi, np.pi, n_nodes)
        d_theta = np.zeros(n_nodes)
        for u in range(n_nodes):
            for v in range(n_nodes):
                if adj[u,v]:
                    d_theta[u] += COUPLING * np.sin(theta[v] - theta[u])
        
        X[i] = theta
        Y[i] = d_theta * dt 
        
    return torch.FloatTensor(X), torch.FloatTensor(Y), adj

class PhysicsTransformer(nn.Module):
    def __init__(self, n_nodes, d_model=64):
        super().__init__()
        self.embedding = nn.Linear(1, d_model)
        
        # POSITIONAL ENCODING (Crucial for Geometry)
        self.pos_embedding = nn.Parameter(torch.randn(1, n_nodes, d_model))
        
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        
        self.output = nn.Linear(d_model, 1)
        self.d_model = d_model
        
    def forward(self, x):
        # x: [batch, n_nodes] -> [batch, n_nodes, 1]
        x = x.unsqueeze(-1)
        
        # Embed + Pos
        h = self.embedding(x) + self.pos_embedding
        
        # Attention
        Q = self.query(h)
        K = self.key(h)
        V = self.value(h)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_model)
        attn_weights = torch.softmax(scores, dim=-1)
        
        context = torch.matmul(attn_weights, V)
        pred = self.output(context).squeeze(-1)
        
        return pred, attn_weights

def run_ai_experiment():
    N_NODES = 20
    EPOCHS = 1000 # More training
    
    X, Y, ADJ = generate_kuramoto_data(n_nodes=N_NODES)
    
    model = PhysicsTransformer(n_nodes=N_NODES)
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    loss_fn = nn.MSELoss()
    
    print("Training Transformer to learn Physics...")
    
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        pred, _ = model(X)
        loss = loss_fn(pred, Y)
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

    print("Extracting Attention Map...")
    test_state = torch.zeros(1, N_NODES) 
    _, attn_map = model(test_state)
    attn_map = attn_map[0].detach().numpy() 
    
    # Ground Truth: Green's Function
    physical_truth = np.zeros((N_NODES, N_NODES))
    for i in range(N_NODES):
        for j in range(N_NODES):
            dist = min(abs(i-j), N_NODES - abs(i-j))
            physical_truth[i,j] = np.exp(-dist / 1.5)
    
    # Normalize
    # Clip small values for cleaner visualization
    attn_map[attn_map < 0.01] = 0
    physical_truth = physical_truth / physical_truth.max()
    attn_map = attn_map / attn_map.max()

    os.makedirs('figures', exist_ok=True)
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    sns.heatmap(attn_map, cmap='viridis', square=True)
    plt.title("AI Learned Attention (Empirical)")
    
    plt.subplot(1, 2, 2)
    sns.heatmap(physical_truth, cmap='viridis', square=True)
    plt.title("Physical Geometry (Analytical)")
    
    plt.tight_layout()
    plt.savefig('figures/ai_transformer_geometry.png', dpi=300)
    
    # Simple metric: Diagonal dominance check
    diag_strength = np.mean(np.diag(attn_map))
    off_diag_strength = np.mean(attn_map) - diag_strength/N_NODES
    ratio = diag_strength / (off_diag_strength + 1e-6)
    
    print(f"\n>>> Local Geometry Strength: {ratio:.2f} (Should be > 1.0) <<<")
    
if __name__ == "__main__":
    run_ai_experiment()