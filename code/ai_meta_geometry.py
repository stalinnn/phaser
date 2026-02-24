import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm import tqdm

"""
EXP 60: Geometric Meta-Learning (Time-Series Refinement)
------------------------------------------------------
Hypothesis:
Transformer can "infer" the topology from the correlation structure of time-series data
and approximate the Oracle GNN (which knows the true topology).

Task:
Input: Time series history of node states X_{t-H:t}
Output: Next state X_{t+1}
Domain: Diffusion dynamics on Random Graphs (Train) -> 2D Grid (Test)
"""

# --- Configuration ---
TRAIN_NODES_MIN = 15
TRAIN_NODES_MAX = 30
N_SAMPLES = 4000
EPOCHS = 15
HISTORY_LEN = 5  # Length of time series history input

def generate_diffusion_trajectory(G, steps=20, dt=0.1):
    """
    Simulates diffusion: dx/dt = -L * x
    Returns a trajectory of shape [Steps, N]
    """
    N = len(G.nodes)
    L = nx.laplacian_matrix(G).toarray()
    # Normalize
    eigenvalues = np.linalg.eigvalsh(L)
    max_eig = np.max(np.abs(eigenvalues))
    if max_eig > 1e-5:
        L = L / max_eig
    
    # Initial state: random heat spots
    x = np.random.randn(N)
    
    traj = [x]
    curr_x = x
    
    # Euler integration
    for _ in range(steps):
        dx = -L @ curr_x
        curr_x = curr_x + dx * dt
        traj.append(curr_x)
        
    return np.array(traj), L

def create_trajectory_dataset(n_samples, history_len):
    print(f"Generating Meta-Learning Dataset ({n_samples} trajectories)...")
    
    data_list = []
    graph_types = ['ER', 'BA', 'WS']
    
    for i in tqdm(range(n_samples)):
        N = np.random.randint(TRAIN_NODES_MIN, TRAIN_NODES_MAX)
        g_type = np.random.choice(graph_types)
        
        if g_type == 'ER':
            G = nx.erdos_renyi_graph(N, p=0.3)
        elif g_type == 'BA':
            m = min(3, N-1)
            G = nx.barabasi_albert_graph(N, m=m)
        elif g_type == 'WS':
            G = nx.connected_watts_strogatz_graph(N, k=4, p=0.3)
            
        # Ensure connected (optional, but good for diffusion)
        if not nx.is_connected(G):
             G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
             N = len(G.nodes)
        
        # Generate trajectory
        traj, L = generate_diffusion_trajectory(G, steps=history_len+1)
        
        # Input: first 'history_len' steps -> [N, History]
        # Target: step 'history_len+1' -> [N]
        # Transpose to [N, History]
        x_in = traj[:history_len].T 
        y_target = traj[history_len]
        
        # Store Adjacency for GNN Baseline
        adj = nx.adjacency_matrix(G).toarray()
        # Add self-loops to adj for GNN
        adj = adj + np.eye(N)
        # Normalize Adj (Simple mean aggregation)
        deg = np.sum(adj, axis=1, keepdims=True)
        adj_norm = adj / (deg + 1e-9)
        
        data_list.append({
            'x': torch.FloatTensor(x_in),       # [N, H]
            'y': torch.FloatTensor(y_target),   # [N]
            'adj': torch.FloatTensor(adj_norm), # [N, N]
            'L': L
        })
        
    return data_list

# --- Models ---

class OracleGCN(nn.Module):
    """
    Baseline: Knows the TRUE topology.
    Simple Graph Conv: H = A * X * W
    """
    def __init__(self, input_dim, hidden_dim=32):
        super().__init__()
        self.proj1 = nn.Linear(input_dim, hidden_dim)
        self.proj2 = nn.Linear(hidden_dim, 1)
        self.act = nn.ReLU()
        
    def forward(self, x, adj):
        # x: [B, N, In]
        # adj: [B, N, N]
        
        # Layer 1
        h = self.proj1(x) # [B, N, Hid]
        h = torch.matmul(adj, h) # Diffusion step: A * H
        h = self.act(h)
        
        # Layer 2
        h = self.proj2(h) # [B, N, 1]
        h = torch.matmul(adj, h)
        
        return h.squeeze(-1), None # No "Attention Map" to return

class MetaTransformer(nn.Module):
    """
    Ours: Must INFER topology from data correlations.
    """
    def __init__(self, input_dim, d_model=32):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        
        # Standard QKV
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        
        self.output = nn.Linear(d_model, 1)
        self.d_model = d_model
        
    def forward(self, x, adj=None):
        # x: [B, N, In]
        # adj: Ignored (Blind to topology)
        
        b, n, _ = x.shape
        h = self.embedding(x) # [B, N, d]
        
        # Self-Attention (Spatial)
        Q = self.query(h)
        K = self.key(h)
        V = self.value(h)
        
        # Scores: [B, N, N]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_model)
        attn_weights = torch.softmax(scores, dim=-1)
        
        context = torch.matmul(attn_weights, V)
        pred = self.output(context).squeeze(-1) # [B, N]
        
        return pred, attn_weights

def run_experiment():
    # 1. Data
    train_data = create_trajectory_dataset(N_SAMPLES, HISTORY_LEN)
    
    # 2. Setup
    gcn = OracleGCN(input_dim=HISTORY_LEN)
    tf = MetaTransformer(input_dim=HISTORY_LEN)
    
    opt_gcn = optim.Adam(gcn.parameters(), lr=0.001)
    opt_tf = optim.Adam(tf.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    
    print("\n--- Training Comparison ---")
    print("Transformer (Blind) vs GCN (Oracle Topology)")
    
    tf_losses = []
    gcn_losses = []
    
    # Training Loop
    for epoch in range(EPOCHS):
        epoch_loss_tf = 0
        epoch_loss_gcn = 0
        np.random.shuffle(train_data)
        
        # Batch size 1 for simplicity with variable graphs
        for sample in train_data:
            x = sample['x'].unsqueeze(0)   # [1, N, H]
            y = sample['y'].unsqueeze(0)   # [1, N]
            adj = sample['adj'].unsqueeze(0) # [1, N, N]
            
            # Train GCN
            opt_gcn.zero_grad()
            pred_gcn, _ = gcn(x, adj)
            loss_gcn = loss_fn(pred_gcn, y)
            loss_gcn.backward()
            opt_gcn.step()
            epoch_loss_gcn += loss_gcn.item()
            
            # Train Transformer
            opt_tf.zero_grad()
            pred_tf, _ = tf(x) # NO ADJ
            loss_tf = loss_fn(pred_tf, y)
            loss_tf.backward()
            opt_tf.step()
            epoch_loss_tf += loss_tf.item()
            
        avg_tf = epoch_loss_tf / len(train_data)
        avg_gcn = epoch_loss_gcn / len(train_data)
        tf_losses.append(avg_tf)
        gcn_losses.append(avg_gcn)
        
        print(f"Epoch {epoch+1}: TF Loss={avg_tf:.6f} | GCN Loss={avg_gcn:.6f}")

    # 3. Test on Unseen Grid
    print("\nTesting on Unseen 5x5 Grid...")
    ROWS, COLS = 5, 5
    G_grid = nx.grid_2d_graph(ROWS, COLS)
    G_grid = nx.convert_node_labels_to_integers(G_grid)
    
    # Generate test samples (same graph, different diffusions)
    N_TEST = 200
    test_inputs = []
    test_targets = []
    
    # Grid Adjacency
    adj_grid = nx.adjacency_matrix(G_grid).toarray() + np.eye(ROWS*COLS)
    deg = np.sum(adj_grid, axis=1, keepdims=True)
    adj_grid_norm = adj_grid / deg
    adj_grid_torch = torch.FloatTensor(adj_grid_norm).unsqueeze(0)
    
    # Theoretical Green's Function (for visualization)
    L_grid = nx.laplacian_matrix(G_grid).toarray()
    L_damped = L_grid + 0.1 * np.eye(ROWS*COLS)
    Greens_true = np.linalg.inv(L_damped)
    
    for _ in range(N_TEST):
        traj, _ = generate_diffusion_trajectory(G_grid, steps=HISTORY_LEN+1)
        test_inputs.append(traj[:HISTORY_LEN].T)
        test_targets.append(traj[HISTORY_LEN])
        
    x_test = torch.FloatTensor(np.array(test_inputs)) # [N_TEST, N, H]
    y_test = torch.FloatTensor(np.array(test_targets)) # [N_TEST, N]
    
    # Evaluate
    gcn.eval()
    tf.eval()
    
    loss_sum_tf = 0
    loss_sum_gcn = 0
    accum_attn = 0
    
    with torch.no_grad():
        for i in range(N_TEST):
            xt = x_test[i].unsqueeze(0)
            yt = y_test[i].unsqueeze(0)
            
            # GCN (Cheating with Adj)
            pg, _ = gcn(xt, adj_grid_torch)
            loss_sum_gcn += loss_fn(pg, yt).item()
            
            # TF (Blind)
            pt, attn = tf(xt)
            loss_sum_tf += loss_fn(pt, yt).item()
            accum_attn += attn[0].numpy()
            
    final_loss_tf = loss_sum_tf / N_TEST
    final_loss_gcn = loss_sum_gcn / N_TEST
    
    print(f"Final Test Loss on Grid:")
    print(f"Oracle GCN (Upper Bound): {final_loss_gcn:.6f}")
    print(f"Transformer (Inferred):   {final_loss_tf:.6f}")
    
    # 4. Visualization
    avg_attn = accum_attn / N_TEST
    avg_attn_norm = avg_attn / np.max(avg_attn)
    Greens_norm = Greens_true / np.max(Greens_true)
    
    os.makedirs('figures', exist_ok=True)
    plt.figure(figsize=(15, 6))
    
    # Learned Attention
    plt.subplot(1, 3, 1)
    sns.heatmap(avg_attn_norm, cmap='magma', xticklabels=False, yticklabels=False)
    plt.title(f'Transformer Learned Topology\n(Loss: {final_loss_tf:.4f})')
    
    # True Physics
    plt.subplot(1, 3, 2)
    sns.heatmap(Greens_norm, cmap='viridis', xticklabels=False, yticklabels=False)
    plt.title('True Green\'s Function\n(Target Physics)')
    
    # Field Reconstruction
    center_node = 12
    attn_field = avg_attn_norm[center_node].reshape(ROWS, COLS)
    
    plt.subplot(1, 3, 3)
    plt.imshow(attn_field, cmap='magma', interpolation='nearest')
    plt.colorbar(label='Influence')
    plt.title(f'Reconstructed 2D Locality\n(Node {center_node} Receptive Field)')
    
    plt.tight_layout()
    plt.savefig('figures/ai_generalization_proof.png', dpi=300)
    print("Saved comparison to figures/ai_generalization_proof.png")

if __name__ == "__main__":
    run_experiment()
