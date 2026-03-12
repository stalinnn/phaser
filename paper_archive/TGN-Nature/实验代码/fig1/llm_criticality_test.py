import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import os
from tqdm import tqdm
import argparse

"""
Experiment 5.6: Theoretical Prediction Validation - Temperature Phase Transition
-------------------------------------------------------------------------------
Theoretical Prediction:
1. Geometric Rank R_eff(T) should exhibit an S-shaped phase transition curve.
2. Model Perplexity PPL(T) should be minimized exactly at the Critical Point (Inflection Point).

This script performs Inference-time Intervention by scanning the softmax temperature T
and measuring R_eff and Perplexity.
"""

def effective_rank(matrix):
    # matrix: [Seq, Hidden]
    # Center
    m = matrix - np.mean(matrix, axis=0)
    # Standardize
    # std = np.std(m, axis=0, keepdims=True)
    # m = m / (std + 1e-9)
    # Covariance
    cov = m.T @ m / (m.shape[0] - 1)
    # SVD
    try:
        S = np.linalg.eigvalsh(cov)
        S = S[::-1]
        S = np.maximum(S, 0)
    except:
        return 1.0
    # Normalize
    S_sum = np.sum(S)
    if S_sum < 1e-9: return 1.0
    p = S / S_sum
    # Entropy
    h = -np.sum(p * np.log(p + 1e-12))
    return np.exp(h)

def calculate_correlations():
    print("Loading GPT-2...")
    model_name = 'gpt2'
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.eval()
    
    # We need to hook into the model to change attention temperature?
    # Or just use the 'temperature' parameter in generation?
    # NO. The 'temperature' in generation only affects the final softmax over vocabulary (sampling).
    # It DOES NOT affect the internal Attention Mechanism's temperature (1/sqrt(d)).
    
    # To test the theory, we must scale the Attention Weights internally.
    # We can do this by monkey-patching the forward pass or using a hook?
    # For simplicity, we can simulate the effect by analyzing the output representations 
    # of a pre-trained model? No, that's fixed T=1.
    
    # To rigorously test the prediction, we need to intervene.
    # However, re-training is expensive. 
    # Can we use 'output_attentions=True' and manually re-weight? No, that's post-hoc.
    
    # Wait, the theory predicts that the PRE-TRAINED model (at T=1) sits at the critical point.
    # If we perturb T during inference, performance should degrade.
    # AND rank should change.
    
    # Let's try to monkey-patch the Attention module.
    # GPT2Attention uses 'scale_attn_weights' and 'attn_dropout'.
    # We can wrap the model to inject a custom scale factor.
    
    # GPT2Attention code snippet logic:
    # w = torch.matmul(q, k)
    # w = w / (float(v.size(-1)) ** 0.5)  <-- This is the fixed scale
    # w = w / temperature_override        <-- We want to add this
    
    print("Injecting Temperature Control Hook...")
    
    # Global variable for current T
    CURRENT_TEMP = 1.0
    
    original_forward = model.transformer.h[0].attn.forward
    
    # We need to patch the class method, but it's bound.
    # Easier way: Just iterate over all layers and replace the forward method?
    # Or even better: GPT2Attention has no 'temperature' arg in forward.
    # We must patch the class or the instances.
    
    from transformers.models.gpt2.modeling_gpt2 import GPT2Attention
    
    # Save original method
    _old_attn_forward = GPT2Attention.forward
    
    def _new_attn_forward(self, hidden_states, layer_past=None, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, use_cache=False, output_attentions=False):
        # We need to copy the code from transformers source to inject T?
        # That's brittle.
        # Hack: The scaling happens inside _attn.
        # Let's patch _attn instead.
        return _old_attn_forward(self, hidden_states, layer_past, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask, use_cache, output_attentions)

    # Patching _attn is safer as it contains the logic
    # def _attn(self, query, key, value, attention_mask=None, head_mask=None):
    #     w = torch.matmul(query, key)
    #     if self.scale_attn_weights:
    #         w = w / (float(value.size(-1)) ** 0.5)
    #     ...
    
    # Let's iterate through layers and monkey patch their _attn method instance
    for block in model.transformer.h:
        # Bind the current block's _attn to a variable to avoid closure issues?
        # Actually, we can just define a new function that calls the original logic but modifies Q or K before?
        # If we divide Q by T, then Q*K becomes (Q*K)/T. Perfect.
        
        old_attn_method = block.attn._attn
        
        def make_new_attn(old_method):
            def new_attn(query, key, value, attention_mask=None, head_mask=None):
                # Scale query by 1/T
                # Note: sqrt(d) scaling is already inside old_method if scale_attn_weights=True (default)
                # We want total scale to be 1/(T*sqrt(d)).
                # So we need to divide query by T.
                query_scaled = query / CURRENT_TEMP
                return old_method(query_scaled, key, value, attention_mask, head_mask)
            return new_attn
            
        block.attn._attn = make_new_attn(old_attn_method)
        
    print("Hook injected. Scanning Temperature...")
    
    temps = np.logspace(np.log10(0.2), np.log10(5.0), 15) # 0.2 to 5.0
    ppls = []
    ranks = []
    
    # Use Wikitext-2 validation subset (short)
    # Just use a fixed long text for stability
    text = "The theory of relativity usually encompasses two interrelated theories by Albert Einstein: special relativity and general relativity. Special relativity applies to all physical phenomena in the absence of gravity. General relativity explains the law of gravitation and its relation to other forces of nature. It applies to the cosmological and astrophysical realm, including astronomy."
    # Repeat to make it long enough
    text = text * 5
    encodings = tokenizer(text, return_tensors='pt')
    input_ids = encodings.input_ids
    
    for t_val in tqdm(temps):
        CURRENT_TEMP = t_val
        
        # 1. Measure Perplexity (Loss)
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids, output_hidden_states=True)
            loss = outputs.loss
            ppl = torch.exp(loss).item()
            ppls.append(ppl)
            
            # 2. Measure Geometric Rank
            # Use last layer hidden states
            hidden_states = outputs.hidden_states[-1][0].numpy() # [Seq, D]
            r = effective_rank(hidden_states)
            ranks.append(r)
            
    # Plotting
    os.makedirs('figures', exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color = 'tab:blue'
    ax1.set_xlabel('Attention Temperature $T$', fontsize=12)
    ax1.set_ylabel('Geometric Rank $R_{eff}$', color=color, fontsize=12)
    ax1.plot(temps, ranks, color=color, marker='o', linewidth=2, label='Geometric Rank')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xscale('log')
    ax1.grid(True, alpha=0.3)
    
    # Annotate Phase Transition
    ax1.axvline(1.0, color='gray', linestyle='--')
    ax1.text(1.0, np.mean(ranks), ' Critical Point\n (T=1.0)', ha='right')
    
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:red'
    ax2.set_ylabel('Perplexity (Performance)', color=color, fontsize=12)  # we already handled the x-label with ax1
    ax2.plot(temps, ppls, color=color, marker='s', linestyle='--', linewidth=2, label='Perplexity')
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title('Theoretical Verification: Transformer at the Edge of Chaos', fontsize=14)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    
    plt.savefig('figures/llm_criticality_test.png', dpi=300)
    print("Saved phase transition plot to figures/llm_criticality_test.png")

if __name__ == "__main__":
    calculate_correlations()
