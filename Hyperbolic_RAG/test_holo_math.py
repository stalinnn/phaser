import torch
from holo_math import poincare_distance, exp_map_0

def main():
    print("==================================================")
    print("Project Holo-RAG - Phase 1 Basic Experiment")
    print("Testing Poincaré Distance vs Euclidean Distance")
    print("==================================================\n")

    # Define a simple tree structure in Euclidean space
    # Root represents the most abstract concept
    root = torch.tensor([0.0, 0.0])
    
    # Branch A: Concepts related to "Economics"
    node_A1 = torch.tensor([3.0, 1.0])   # Parent in branch A
    node_A2 = torch.tensor([6.0, 2.0])   # Child in branch A
    
    # Branch B: Concepts related to "Physics"
    node_B1 = torch.tensor([3.0, -1.0])  # Parent in branch B
    node_B2 = torch.tensor([6.0, -2.0])  # Child in branch B
    
    # Notice that node_A1 and node_B1 are geometrically very close in Euclidean space.
    # In Euclidean space, their distance is 2.0.
    # But in a logical tree, to go from A1 to B1, you must traverse up to the Root and down.
    # Let's see how Poincare geometry handles this.
    
    # 1. Calculate Euclidean Distances
    eucl_A1_B1 = torch.norm(node_A1 - node_B1).item()
    eucl_A1_root = torch.norm(node_A1 - root).item()
    eucl_A1_A2 = torch.norm(node_A1 - node_A2).item()
    
    print("--- 1. Euclidean Distances (The Flat Space Illusion) ---")
    print(f"Dist(A1, B1) [Across different branches]: {eucl_A1_B1:.4f}")
    print(f"Dist(A1, Root) [Moving up hierarchy]   : {eucl_A1_root:.4f}")
    print(f"Dist(A1, A2) [Moving down same branch] : {eucl_A1_A2:.4f}")
    print("Observation: In Euclidean space, A1 and B1 seem very related (distance 2.0), ")
    print("even though they belong to completely different branches.\n")
    
    # 2. Map vectors to Poincaré Ball
    root_poin = exp_map_0(root)
    node_A1_poin = exp_map_0(node_A1)
    node_A2_poin = exp_map_0(node_A2)
    node_B1_poin = exp_map_0(node_B1)
    node_B2_poin = exp_map_0(node_B2)
    
    # 3. Calculate Poincaré Distances
    poin_A1_B1 = poincare_distance(node_A1_poin, node_B1_poin).item()
    poin_A1_root = poincare_distance(node_A1_poin, root_poin).item()
    poin_B1_root = poincare_distance(node_B1_poin, root_poin).item()
    poin_A1_A2 = poincare_distance(node_A1_poin, node_A2_poin).item()
    
    print("--- 2. Poincaré Distances (The Holographic Truth) ---")
    print(f"Dist(A1, B1) [Across different branches]: {poin_A1_B1:.4f}")
    print(f"Dist(A1, Root) [Moving up hierarchy]   : {poin_A1_root:.4f}")
    print(f"Dist(B1, Root) [Moving up hierarchy]   : {poin_B1_root:.4f}")
    print(f"Dist(A1, A2) [Moving down same branch] : {poin_A1_A2:.4f}")
    
    # 4. Verify Tree-like property (Ryu-Takayanagi approximation)
    print("\n--- 3. Tree-like Metric Property Verification ---")
    tree_path_dist = poin_A1_root + poin_B1_root
    print(f"Path distance via Root (A1 -> Root -> B1): {tree_path_dist:.4f}")
    print(f"Direct Poincare distance (A1 -> B1)      : {poin_A1_B1:.4f}")
    
    ratio = poin_A1_B1 / tree_path_dist
    print(f"Ratio (Direct / Path via Root)           : {ratio:.4f}")
    if ratio > 0.9:
        print("Success! The direct Poincaré distance strongly approximates the tree traversal path.")
        print("This means the metric naturally forces 'geodesics' to bend towards the origin (Root).")
        print("It inherently understands the hierarchy!")
    else:
        print("Hmm, the approximation is not as strong as expected.")
        
    print("\n--- 4. Extreme Leaf Nodes ---")
    poin_A2_B2 = poincare_distance(node_A2_poin, node_B2_poin).item()
    poin_A2_root = poincare_distance(node_A2_poin, root_poin).item()
    poin_B2_root = poincare_distance(node_B2_poin, root_poin).item()
    tree_path_dist_leaf = poin_A2_root + poin_B2_root
    print(f"Dist(A2, B2) [Deep leaf nodes, different branches]: {poin_A2_B2:.4f}")
    print(f"Path distance via Root (A2 -> Root -> B2)       : {tree_path_dist_leaf:.4f}")
    print(f"Euclidean Dist(A2, B2)                           : {torch.norm(node_A2 - node_B2).item():.4f}")
    print("Observation: The deeper the nodes (closer to boundary), the more strictly")
    print("the shortest path behaves like a tree traversal through the root.")

if __name__ == "__main__":
    main()
