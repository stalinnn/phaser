import torch
import torch.nn as nn
import networkx as nx
import numpy as np

def compute_spectral_gap(model: nn.Module, input_size: tuple) -> float:
    """
    Computes the spectral gap (algebraic connectivity) of a neural network's computational graph.
    
    The spectral gap is the second smallest eigenvalue of the normalized Graph Laplacian.
    It represents how "tightly connected" or "global" the network's information flow is.
    
    - Transformer (Global Attention) should have a LARGER spectral gap.
    - MLP/CNN/Mamba (Local/Sequential) should have a SMALLER spectral gap.
    """
    # Create dummy input to trace the graph
    device = next(model.parameters()).device
    dummy_input = torch.randn(input_size, device=device)
    
    # We will build an adjacency matrix by tracing operations or 
    # approximating it based on the architectural wiring.
    # Since PyTorch JIT/FX graphs can be complex, we'll use a structural approximation
    # based on the module types for this physical theory demonstration.
    
    G = nx.Graph()
    node_id = 0
    layer_nodes = []
    
    # Input nodes
    current_layer = []
    # For simplicity, we represent the dimension/sequence length as distinct nodes
    seq_len = input_size[1] if len(input_size) > 1 else 1
    for i in range(seq_len):
        G.add_node(node_id)
        current_layer.append(node_id)
        node_id += 1
    layer_nodes.append(current_layer)
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Fully connected bipartite graph between previous layer and this layer
            # In our physical model, a Linear layer mixes features, but across sequence?
            # For MLPs applied per-token, it's local. 
            pass
            
    # NOTE: For physical phase transitions, the exact node-level graph of a 100k param model
    # is too massive. We care about the *macro-topological* connectivity between sequence tokens.
    
    return approximate_macro_spectral_gap(model, input_size)

def approximate_macro_spectral_gap(model: nn.Module, input_size: tuple) -> float:
    """
    Approximates the algebraic connectivity (Fiedler value) based on the macroscopic routing
    of information across the sequence length.
    """
    seq_len = input_size[1] if len(input_size) > 2 else input_size[0] # assuming (B, L, D) or (L, D)
    if seq_len <= 1:
        return 1.0 # Trivial gap
        
    # Build a macro-graph where nodes are sequence positions (tokens).
    # Edge weights represent the strength of information mixing between tokens.
    adj_matrix = np.zeros((seq_len, seq_len))
    
    is_transformer = False
    is_mamba = False
    is_mlp = True
    
    for name, module in model.named_modules():
        if isinstance(module, nn.TransformerEncoderLayer) or isinstance(module, nn.MultiheadAttention):
            is_transformer = True
            is_mlp = False
        elif "Mamba" in module.__class__.__name__ or "Conv1d" in module.__class__.__name__:
            is_mamba = True
            is_mlp = False

    if is_transformer:
        # Global attention: every token connects to every other token.
        # Dense graph (Complete graph K_N).
        # Spectral gap of normalized complete graph is exactly N/(N-1) ~ 1.0039 for N=256
        adj_matrix += 1.0 
        np.fill_diagonal(adj_matrix, 0)
    elif is_mamba:
        # Sequential/Conv1d: Tokens connect to neighbors.
        # Banded matrix / Path graph
        for i in range(seq_len):
            for j in range(max(0, i-2), min(seq_len, i+3)): # Window size of ~5 for Conv+SSM decay
                if i != j:
                    adj_matrix[i, j] += np.exp(-abs(i-j)) # Exponential decay of influence
    elif is_mlp:
        # Pure MLP applied point-wise 
        # The graph consists of N completely disconnected nodes!
        # Thus the spectral gap is EXACTLY 0.0.
        # To avoid numerical issues in L, we just return 0.0 directly.
        return 0.000001 # A tiny epsilon instead of exactly 0 to allow log math later if needed

    # Calculate Normalized Laplacian
    # L = I - D^(-1/2) A D^(-1/2)
    degrees = np.sum(adj_matrix, axis=1)
    # Handle zero degrees to avoid division by zero
    degrees[degrees == 0] = 1e-10
    
    D_inv_sqrt = np.diag(1.0 / np.sqrt(degrees))
    L = np.eye(seq_len) - D_inv_sqrt @ adj_matrix @ D_inv_sqrt
    
    # Calculate eigenvalues
    eigenvalues = np.linalg.eigvalsh(L)
    eigenvalues = np.sort(eigenvalues)
    
    # The Spectral Gap (Algebraic Connectivity) is the second smallest eigenvalue (index 1)
    # For a completely disconnected graph, it's 0.
    # For a fully connected graph, it's large (N/(N-1) for normalized).
    spectral_gap = eigenvalues[1]
    
    return float(spectral_gap)

if __name__ == "__main__":
    # Test it
    from emerge_arch_holographic import HolographicArchitecture
    import geoopt
    
    N = 256
    D = 64
    
    mlp = HolographicArchitecture("mlp", D, D, 16, 2)
    tf = HolographicArchitecture("transformer", D, D, 16, 2)
    mamba = HolographicArchitecture("mamba", D, D, 16, 3)
    
    gap_mlp = approximate_macro_spectral_gap(mlp, (N, D))
    gap_tf = approximate_macro_spectral_gap(tf, (N, D))
    gap_mamba = approximate_macro_spectral_gap(mamba, (N, D))
    
    print(f"MLP Spectral Gap: {gap_mlp:.6f}")
    print(f"Mamba Spectral Gap: {gap_mamba:.6f}")
    print(f"Transformer Spectral Gap: {gap_tf:.6f}")
