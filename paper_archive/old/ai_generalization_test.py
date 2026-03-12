import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import os

"""
AI Experiment 2.0: Zero-Shot Topological Generalization
-------------------------------------------------------
Critique: Previous experiment was overfitting to a fixed topology.
Response: Train on CHAOS (Random Graphs), Test on ORDER (Ring).

Hypothesis:
To predict dynamics on unknown random graphs, the AI must learn a universal law:
"Attention ~ Dynamical Correlation ~ Effective Geometry".
If this holds, it should zero-shot recover the geometry of a Ring lattice.
"""

# -----------------------------------------------------------------------------
# 1. Dynamic Data Generator (Variable Topology)
# -----------------------------------------------------------------------------
def generate_trajectory(adj_matrix, n_steps=50, dt=0.05, coupling=3.0):
    n_nodes = adj_matrix.shape[0]
    # Start from random phases
    theta = np.random.uniform(-np.pi, np.pi, n_nodes)
    trajectory = np.zeros((n_steps, n_nodes))
    
    # Run Kuramoto dynamics
    for t in range(n_steps):
        d_theta = np.zeros(n_nodes)
        # Compute interaction sin(theta_j - theta_i)
        # Vectorized for speed
        # diff[i, j] = theta[j] - theta[i]
        diff = theta[None, :] - theta[:, None] 
        interaction = np.sin(diff)
        
        # Apply adjacency mask
        d_theta = np.sum(interaction * adj_matrix, axis=1) * coupling
        
        theta = theta + d_theta * dt
        # Normalize to [-pi, pi]
        theta = np.mod(theta + np.pi, 2*np.pi) - np.pi
        trajectory[t] = theta
        
    return trajectory

def get_batch(batch_size=32, n_nodes=20, seq_len=20):
    """
    Generates a batch where EACH SAMPLE has a DIFFERENT RANDOM TOPOLOGY.
    """
    batch_X = []
    batch_Y = []
    
    for _ in range(batch_size):
        # 1. Create Random GEOMETRIC Graph (RGG) - Matches Physics better
        # Nodes are dropped in 2D, connected by distance radius
        # This teaches the AI that "Geometry Matters" without showing it a Ring.
        G = nx.random_geometric_graph(n_nodes, radius=0.4)
        
        # Ensure connectedness (simple fix: reconnect if disconnected)
        # Or just use whatever comes out, robustness is good.
        adj = nx.to_numpy_array(G)
        
        # 2. Generate Dynamics
        # We run for seq_len + 1 steps to get (History) -> (Target)
        traj = generate_trajectory(adj, n_steps=seq_len+1)
        
        # Input: [0...T-1], Target: [T] (Predict next step from history)
        # Actually, simpler: Input [t], Predict [t+1]. 
        # But to infer topology, we might need history. Let's try single step first.
        # If single step fails, we need history (Transformer over time).
        # Let's use history to be safe. "Context" allows inferring causality.
        
        batch_X.append(traj[:-1]) # Shape [T, N]
        batch_Y.append(traj[1:])  # Shape [T, N]
        
    return torch.FloatTensor(np.array(batch_X)), torch.FloatTensor(np.array(batch_Y))

# -----------------------------------------------------------------------------
# 2. The Universal Transformer (Temporal)
# -----------------------------------------------------------------------------
class UniversalPhysicsTransformer(nn.Module):
    def __init__(self, n_nodes, d_model=64):
        super().__init__()
        # Input is a single scalar state per node? No, let's treat nodes as tokens.
        # But we have time dimension.
        # Strategy: Embed Time -> Apply Attention over Nodes.
        
        self.embedding = nn.Linear(1, d_model)
        
        # Encoder to process time history per node
        # We squash time [T] into feature vector [d_model]
        # Simple approach: MLP on the last state? 
        # No, "Dynamics from history".
        
        self.d_model = d_model
        
        # Self-Attention (The Brain)
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        
        self.output = nn.Linear(d_model, 1)

    def forward(self, x):
        # x shape: [Batch, Time, Nodes]
        # We want to predict x[t+1] for the last step? 
        # Or seq-to-seq? Let's do seq-to-seq.
        
        # For simplicity in this demo, let's just use the LAST state x[:,-1,:]
        # and try to predict next state.
        # WAIT: If input is only current state, and topology is unknown, 
        # it is IMPOSSIBLE to predict interaction.
        # The topology must be latent or inferred from history.
        # BUT: For this "Generalization" test, let's see if it can learn
        # "Generalized Interaction Rule" if we give it the state.
        # Actually, without Graph Input, x(t) -> x(t+1) is ill-posed for variable graphs.
        
        # Correction: The original paper's claim is about "Attention ~ Geometry".
        # In a real biological system, the topology is FIXED but UNKNOWN to the observer.
        # The organism learns the fixed topology over evolutionary time.
        
        # To satisfy the user's "Zero-Shot" requirement:
        # We train on "Many Small Worlds" to learn the *Physics of Synchronization*.
        # Then we test on "One Big Ring".
        
        # Let's assume the Transformer gets a short history window to "infer" who is connected.
        # Input: [Batch, Nodes, 1] (Just state) is not enough.
        # Let's use [Batch, Nodes, Window_Size].
        
        B, T, N = x.shape
        # Transpose to [B, N, T] -> Treat history as features
        x_feat = x.permute(0, 2, 1) 
        
        # Embed history [T] -> [d_model]
        # This implies: "Identify neighbors by how their history matches mine"
        h = nn.Linear(T, self.d_model)(x_feat) 
        
        # Attention over Nodes
        Q = self.query(h)
        K = self.key(h)
        V = self.value(h)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_model)
        attn_weights = torch.softmax(scores, dim=-1)
        
        context = torch.matmul(attn_weights, V)
        
        # Predict change
        pred = self.output(context).squeeze(-1)
        
        return pred, attn_weights

# -----------------------------------------------------------------------------
# 3. Experiment Runner
# -----------------------------------------------------------------------------
def run_generalization_experiment():
    N_NODES = 20
    BATCH_SIZE = 16
    BATCHES = 1000 # Increase training time
    SEQ_LEN = 20 # Longer history to infer causality
    
    model = UniversalPhysicsTransformer(n_nodes=N_NODES, d_model=32)
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    loss_fn = nn.MSELoss()
    
    print("Step 1: Training on RANDOM CHAOS (Erdos-Renyi Graphs)...")
    loss_history = []
    
    for i in range(BATCHES):
        # Generate a batch of DIFFERENT random graphs
        X, Y = get_batch(BATCH_SIZE, N_NODES, SEQ_LEN)
        
        # X: [B, T, N], Y: [B, T, N]
        # We only predict the LAST step Y[:,-1,:] using history X
        target = Y[:, -1, :]
        input_seq = X 
        
        optimizer.zero_grad()
        pred, _ = model(input_seq)
        
        # We predict the NEXT state.
        # Target is the actual next state.
        loss = loss_fn(pred, target)
        loss.backward()
        optimizer.step()
        
        if i % 20 == 0:
            print(f"Batch {i}, Loss: {loss.item():.6f}")
            
    # -------------------------------------------------------------------------
    # Step 2: The ZERO-SHOT Test on ORDER (Ring Lattice)
    # -------------------------------------------------------------------------
    print("\nStep 2: Zero-Shot Testing on RING LATTICE (Unseen Topology)...")
    
    # Create the Ring
    G_test = nx.watts_strogatz_graph(N_NODES, k=4, p=0.0) # p=0 -> Perfect Ring
    adj_test = nx.to_numpy_array(G_test)
    
    # Generate data on Ring
    traj_test = generate_trajectory(adj_test, n_steps=SEQ_LEN+1)
    X_test = torch.FloatTensor(traj_test[:-1]).unsqueeze(0) # [1, T, N]
    
    # Inference
    _, attn_map = model(X_test)
    attn_map = attn_map[0].detach().numpy()
    
    # Plotting
    os.makedirs('figures', exist_ok=True)
    plt.figure(figsize=(12, 5))
    
    # 1. AI Attention
    plt.subplot(1, 2, 1)
    sns.heatmap(attn_map, cmap='viridis', square=True)
    plt.title("Zero-Shot AI Attention (on Ring)")
    plt.xlabel("Key Node")
    plt.ylabel("Query Node")
    
    # 2. Ground Truth Adjacency
    plt.subplot(1, 2, 2)
    sns.heatmap(adj_test, cmap='Greys', square=True)
    plt.title("True Hidden Topology (Ring)")
    plt.xlabel("Node Index")
    
    plt.tight_layout()
    plt.savefig('figures/ai_generalization_proof.png', dpi=300)
    print("Saved proof to figures/ai_generalization_proof.png")
    
    # Metric: Does attention correlate with adjacency?
    # We ignore the diagonal (self-attention) for this check
    np.fill_diagonal(attn_map, 0)
    np.fill_diagonal(adj_test, 0)
    corr = np.corrcoef(attn_map.flatten(), adj_test.flatten())[0,1]
    print(f"\n>>> GENERALIZATION SCORE (Correlation): {corr:.4f} <<<")
    
    if corr > 0.3:
        print("SUCCESS: AI inferred geometry from dynamics zero-shot!")
    else:
        print("FAILURE: AI could not generalize.")

if __name__ == "__main__":
    run_generalization_experiment()
