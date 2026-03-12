import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import NMF
from scipy.special import softmax
from scipy.optimize import minimize
import os

"""
FINANCIAL INVERSE ATTENTION: 
RECONSTRUCTING THE 'QUERY' (SENTIMENT) AND 'KEY' (FUNDAMENTALS)
---------------------------------------------------------------

Goal: To prove that financial crises are a 'Topological Collapse' of the 
Investor Attention Mechanism (Query), not just a change in Fundamentals (Key).

Methodology:
1. Model Market Interactions as an Attention Matrix: A ~ Softmax(Q @ K.T)
2. Assume 'Keys' (K) represent slowly-changing fundamental asset characteristics.
   We estimate K_static from a 'Stable Period' (2006).
3. Solve the Inverse Problem for 'Queries' (Q_t) at each time step t:
   Find Q_t that minimizes || A_t - Softmax(Q_t @ K_static.T) ||
4. Analyze the Entropy/Structure of Q_t.
   Hypothesis: In crisis, Q_t collapses to a single mode (Panic/Herding), 
   even though K (Fundamentals) remains distinct.

This separates 'Sentiment Topology' from 'Fundamental Value'.
"""

def download_data():
    tickers = [
        "MSFT", "AMZN", "JPM", "XOM", "GE", 
        "JNJ", "PFE", "C", "KO", "PEP",
        "WMT", "MRK", "INTC", "CSCO", "IBM",
        "GS", "BAC", "AIG", "AXP", "MCD" 
    ]
    start_date = "2006-01-01"
    end_date = "2009-06-01"
    
    print("Downloading data for Inverse Attention Analysis...")
    try:
        data = yf.download(tickers, start=start_date, end=end_date, progress=False)['Close']
        # Forward fill to handle missing data
        data = data.ffill().dropna()
        return data
    except Exception as e:
        print(f"Data download failed: {e}")
        return None

def get_interaction_matrix(returns, window=60):
    """
    Computes a sliding window interaction matrix.
    We use Correlation Matrix shifted to [0, 1] range to simulate 'Attention Probabilities'.
    """
    # Pearson correlation
    corr = returns.rolling(window=window).corr()
    
    # Transform to "Attention-like" weights (0 to 1, rows sum to 1)
    # Correlation is [-1, 1]. We map to [0, 1] via (x+1)/2 or exp(x).
    # Softmax is natural for Attention.
    
    # We need to handle the MultiIndex of rolling correlation
    return corr

def inverse_solve_q(A_target, K_fixed, learning_rate=0.01, steps=100):
    """
    Given Target Attention A (N x N) and Fixed Keys K (N x d),
    Find Query Q (N x d) that approximates A ~ Softmax(Q K.T).
    
    Since this is a simple convex optimization, we can use Gradient Descent.
    
    Loss = KL_Divergence(A_target || Softmax(Q K.T)) or MSE.
    Since A_target might not be a strict probability distribution (it's correlation),
    we first normalize A_target row-wise.
    """
    N, d = K_fixed.shape
    
    # Normalize A_target to be a probability matrix (Attention map)
    # Clip negative correlations for this physical analogy
    A_pos = np.clip(A_target, 0, None) 
    row_sums = A_pos.sum(axis=1, keepdims=True) + 1e-9
    A_prob = A_pos / row_sums
    
    # Initialize Q randomly
    Q = np.random.normal(0, 0.1, (N, d))
    
    # Simple Gradient Descent (or just use scipy.optimize for stability)
    # For speed in python loop, let's use a closed form approx or simple iterative update?
    # Actually, let's use NMF as a proxy first. 
    # If A = Q K^T (before softmax), then Q = A (K^T)^+
    # But we have Softmax. 
    # Log-Space: log(A) ~ Q K^T - RowNorms
    # This is close to Matrix Factorization of log(A).
    
    # Strategy:
    # 1. Compute Logits L = log(A_prob + epsilon)
    # 2. We want L_ij = q_i^T k_j + C_i
    # 3. Center L row-wise: L_centered = L - mean(L, axis=1)
    # 4. Minimize || L_centered - Q K^T ||^2
    # 5. Q = L_centered @ K @ inv(K^T K)
    
    epsilon = 1e-9
    L = np.log(A_prob + epsilon)
    # Remove the row-dependent normalization constant (Softmax denominator)
    # L_ij = Q_i K_j - log(sum_k exp(Q_i K_k))
    # If we center the rows, the constant term vanishes approx.
    L_centered = L - L.mean(axis=1, keepdims=True)
    
    # Linear Regression: Q * K.T = L_centered
    # Q = L_centered * K * (K.T * K)^-1
    
    # Ridge regression for stability
    lambda_reg = 0.1
    K_inv = np.linalg.inv(K_fixed.T @ K_fixed + lambda_reg * np.eye(d))
    Q_est = L_centered @ K_fixed @ K_inv
    
    return Q_est

def run_inverse_attention():
    data = download_data()
    if data is None:
        return

    returns = np.log(data / data.shift(1)).dropna()
    
    # 1. Define Phases
    stable_start = "2006-01-01"
    stable_end = "2006-06-01" # 6 months of stability
    
    # 2. Extract Static Keys (Fundamentals) from Stable Phase
    print("Extracting Static Keys (Fundamentals) from Stable Phase...")
    stable_returns = returns.loc[stable_start:stable_end]
    corr_stable = stable_returns.corr().values
    
    # Use SVD/NMF to get Embeddings
    # We assume d=3 latent factors (Sector, Beta, Momentum?)
    d_model = 3
    
    # NMF for interpretability (Non-negative features)
    # Shift correlation to be positive
    corr_stable_pos = np.clip(corr_stable, 0, 1)
    nmf = NMF(n_components=d_model, init='random', random_state=42)
    W_stable = nmf.fit_transform(corr_stable_pos)
    H_stable = nmf.components_
    
    # We define Keys K = H_stable.T (Feature vectors of assets)
    # We assume these features (what sector you are in) don't change fast.
    K_static = H_stable.T 
    
    # 3. Time-Evolution of Query (Sentiment)
    print("Inverting Attention Mechanism over time...")
    dates = []
    q_entropies = [] # How diverse is the Query?
    q_similarities = [] # How similar are all investors?
    
    window = 60
    step = 5
    
    # Sliding window
    for t in range(window, len(returns), step):
        current_date = returns.index[t]
        window_returns = returns.iloc[t-window:t]
        
        # Get Current Interaction Matrix
        corr_t = window_returns.corr().values
        
        # Invert to find Q_t
        Q_t = inverse_solve_q(corr_t, K_static)
        
        # Analysis of Q_t
        # 1. Similarity: Are all rows of Q_t the same? (Everyone looking for same thing)
        # Calculate cosine similarity matrix of Q_t rows
        norm_Q = np.linalg.norm(Q_t, axis=1, keepdims=True)
        Q_normalized = Q_t / (norm_Q + 1e-9)
        cosine_sim = np.dot(Q_normalized, Q_normalized.T)
        avg_similarity = np.mean(cosine_sim[np.triu_indices(len(Q_t), k=1)])
        
        q_similarities.append(avg_similarity)
        dates.append(current_date)
    
    # 4. Plotting
    print("Plotting results...")
    
    # Align dates for plotting
    df_res = pd.DataFrame({
        'Date': dates,
        'Query_Homogeneity': q_similarities
    })
    df_res.set_index('Date', inplace=True)
    
    # Get Market Index for comparison (Mean return)
    market_cum = returns.mean(axis=1).cumsum()
    market_cum = market_cum.reindex(df_res.index)
    
    # Plot
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    color = 'tab:red'
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Investor Attention Homogeneity (Q-Collapse)', color=color)
    ax1.plot(df_res.index, df_res['Query_Homogeneity'], color=color, linewidth=2, label='Attention Homogeneity (Panic)')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)
    
    # Add crisis shading
    ax1.axvspan(pd.Timestamp('2008-09-01'), pd.Timestamp('2009-01-01'), color='grey', alpha=0.3, label='2008 Crisis')
    
    ax2 = ax1.twinx()  
    color = 'tab:blue'
    ax2.set_ylabel('Market Cumulative Return', color=color)  
    ax2.plot(market_cum.index, market_cum, color=color, linestyle='--', alpha=0.5, label='Market Return')
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title("Mechanism of Crisis: The Collapse of Investor 'Query' Diversity\n(Inverse Attention Analysis)")
    fig.tight_layout()
    plt.savefig('figures/inverse_attention_proof.png', dpi=300)
    print("Saved proof to figures/inverse_attention_proof.png")

if __name__ == "__main__":
    run_inverse_attention()
