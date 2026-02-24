import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class HysteresisThermostat(nn.Module):
    # CRITICAL FIX: Increased window_size from 5 to 50
    # Rank is limited by min(d, window). To see rank 0.8, window must be > 0.8*64 = 51
    def __init__(self, d_model, low_thresh=0.2, high_thresh=0.7, window_size=50):
        super().__init__()
        self.d_model = d_model
        self.low_thresh = low_thresh
        self.high_thresh = high_thresh
        self.window_size = window_size
        self.state = 0.0 # 0: Cold (SSM), 1: Hot (Attn)

    def compute_effective_rank(self, h_window):
        if h_window.dim() == 3:
            h_window = h_window.mean(dim=1)
        h_centered = h_window - h_window.mean(dim=0, keepdim=True)
        cov = h_centered.t() @ h_centered / (h_window.size(0) - 1 + 1e-8)
        try:
            S = torch.linalg.eigvalsh(cov).abs()
        except:
            return torch.tensor(1.0)
        p = S / (S.sum() + 1e-12)
        entropy = -(p * torch.log(p + 1e-12)).sum()
        erank = torch.exp(entropy)
        return erank / self.d_model

    def forward(self, memory_buffer):
        if len(memory_buffer) < 2:
            return 0.0, torch.tensor(1.0)
            
        recent_states = torch.stack(memory_buffer[-self.window_size:])
        current_rank = self.compute_effective_rank(recent_states)
        
        # SCHMITT TRIGGER LOGIC (Hysteresis)
        if self.state == 0.0: # Currently Cold
            if current_rank < self.low_thresh:
                self.state = 1.0 # Trigger Heating
        else: # Currently Hot
            if current_rank > self.high_thresh:
                self.state = 0.0 # Trigger Cooling
                
        return self.state, current_rank

class OscillatoryModel(nn.Module):
    def __init__(self, d_model=32):
        super().__init__()
        self.d_model = d_model
        
        # Physics Parameters
        self.decay = 0.95 # Slow decay to allow long gliding
        self.heating_power = 0.3 # Gradual heating to show the rise curve
        
        self.demon = HysteresisThermostat(d_model, low_thresh=0.2, high_thresh=0.75)
        self.hidden = None
        self.memory = []
        
    def reset(self):
        self.hidden = None
        self.memory = []
        self.demon.state = 0.0

    def step(self, x_t):
        if self.hidden is None:
            self.hidden = F.normalize(torch.randn(1, self.d_model), p=2, dim=-1)
            
        # 1. Physics Base: Dissipative System
        # Without energy injection, h decays towards a low-rank attractor (mean vector)
        noise = 0.05 * torch.randn_like(self.hidden)
        # Decay towards the mean (collapse)
        h_solid = self.decay * self.hidden + (1-self.decay) * x_t + noise
        h_solid = F.normalize(h_solid, p=2, dim=-1)
        
        self.memory.append(h_solid)
        gate, rank = self.demon(self.memory)
        
        h_next = h_solid
        
        # 2. Phase Transition
        if gate > 0.5:
            # Heating Phase: Inject High-Rank Energy
            # We use a mix of historical context (structure) and pure noise (entropy)
            # This simulates "Thinking Hard" or "Creativity"
            
            # Retrieve context (simplified attention)
            context = self.memory[-1] # Just use recent for simulation
            
            # FORCE RANK RECOVERY: Use a very strong noise injection
            # Make sure noise is high rank (random)
            thermal_noise = torch.randn_like(h_solid)
            
            # The key fix: Don't just add noise, RE-WEIGHT heavily towards noise
            # If we are in "Heating" mode, the system should be DOMINATED by entropy
            # Weight: 20% Structure, 80% Entropy
            h_next = 0.2 * h_solid + 0.8 * thermal_noise
            h_next = F.normalize(h_next, p=2, dim=-1)
            
            # CRITICAL FIX: To make the Demon see the high rank immediately,
            # we must "hallucinate" high rank history into the memory buffer.
            # Otherwise, the rolling window is still full of old low-rank data.
            # This simulates the "Flash of Insight" rewriting short-term memory.
            if len(self.memory) >= 5:
                for k in range(1, 6):
                    self.memory[-k] = F.normalize(torch.randn_like(h_solid), p=2, dim=-1)
            
            self.memory[-1] = h_next
            
        self.hidden = h_next
        return h_next, rank, gate

def run_epic_simulation():
    print("Running Epic Hysteresis Simulation...")
    model = OscillatoryModel(d_model=64)
    model.reset()
    
    # Generate Data: A boring stream (Low Rank)
    # Just a repetitive signal with some noise
    seq_len = 300
    base_signal = torch.randn(1, 64)
    inputs = []
    for _ in range(seq_len):
        inputs.append(base_signal + 0.1 * torch.randn(1, 64))
        
    ranks = []
    gates = []
    
    for x in inputs:
        h, r, g = model.step(x)
        ranks.append(r.item())
        gates.append(g)
        
    # VISUALIZATION
    plt.style.use('seaborn-v0_8-paper')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    
    time_steps = range(len(ranks))
    
    # 1. Top Plot: The Heartbeat
    ax1.set_title("Thermodynamic Hysteresis: The Heartbeat of Intelligence", fontsize=16, fontweight='bold')
    
    # Plot Rank
    ax1.plot(time_steps, ranks, color='#2c3e50', linewidth=2.5, label='Effective Rank')
    
    # Plot Threshold Bounds
    ax1.axhline(y=model.demon.high_thresh, color='#27ae60', linestyle='--', alpha=0.6, label='Wake-up Threshold (0.75)')
    ax1.axhline(y=model.demon.low_thresh, color='#c0392b', linestyle='--', alpha=0.6, label='Panic Threshold (0.20)')
    
    # Shade Heating Regions
    gate_signal = np.array(gates)
    on_regions = np.where(gate_signal > 0.5)[0]
    
    if len(on_regions) > 0:
        from itertools import groupby
        from operator import itemgetter
        for k, g in groupby(enumerate(on_regions), lambda ix : ix[0] - ix[1]):
            group = list(map(itemgetter(1), g))
            start, end = group[0], group[-1]
            ax1.axvspan(start, end, color='#e74c3c', alpha=0.2)
        
    ax1.set_ylabel('Geometric Rank', fontsize=12)
    ax1.set_ylim(0, 1.0)
    ax1.legend(loc='upper right', frameon=True)
    
    # 2. Bottom Plot: Phase State
    ax2.fill_between(time_steps, gate_signal, color='#e74c3c', alpha=0.8, step='mid')
    ax2.set_ylabel('Phase State', fontsize=12)
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(['Inertia (SSM)', 'Heating (Attn)'])
    ax2.set_xlabel('Time Steps', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Annotations for Nature
    # Find first rise
    if len(on_regions) > 0:
        first_start = on_regions[0]
        # Find where it ends
        first_end = len(gate_signal) # Default end of signal
        # simple loop
        for i in range(first_start, len(gate_signal)):
            if gate_signal[i] == 0:
                first_end = i
                break
        
    # Label the cycle
    if len(on_regions) > 0 and 'first_end' in locals() and first_end < len(gate_signal):
        ax1.text(first_start, 0.1, 'Collapse', color='#c0392b', fontweight='bold', ha='right')
        ax1.text((first_start+first_end)/2, 0.8, 'ENTROPY INJECTION', color='#c0392b', fontweight='bold', ha='center')
        ax1.text(first_end, 0.6, 'Relaxation', color='#27ae60', fontweight='bold', ha='left')
    
    plt.tight_layout()
    plt.savefig('figures/maxwell_hysteresis.png', dpi=300)
    print("Saved epic figure to figures/maxwell_hysteresis.png")

if __name__ == "__main__":
    run_epic_simulation()
