"""
EXP 102: Cross-Modal Validation on Sleep-EEG (Sleep-EDFx)
---------------------------------------------------------
Goal: Validate the "Geometric Rank Collapse" hypothesis on a second biological modality (EEG).
Dataset: Sleep-EDF Database Expanded (PhysioNet) via MNE-Python.

Hypothesis:
    If Attention (Long-range correlation) corresponds to Consciousness,
    then the Effective Rank of the EEG covariance matrix should be:
    High in Wake (W) > Low in Deep Sleep (N3/N4).

Dependencies:
    pip install mne matplotlib numpy scipy pandas
"""

import numpy as np
import matplotlib.pyplot as plt
import os
try:
    import mne
    from mne.datasets.sleep_physionet.age import fetch_data
except ImportError:
    print("MNE-Python not found. Please install: pip install mne")
    exit()

def compute_effective_rank(data_matrix):
    """
    Computes Effective Rank of the covariance matrix of channels.
    data_matrix: [n_channels, n_times]
    """
    # 1. Z-score normalization per channel
    # Avoid division by zero
    std = np.std(data_matrix, axis=1, keepdims=True)
    std[std == 0] = 1.0
    X = (data_matrix - np.mean(data_matrix, axis=1, keepdims=True)) / std

    # 2. Covariance Matrix (n_channels x n_channels)
    # Using numpy.corrcoef is equivalent to covariance of z-scored data
    C = np.corrcoef(X)
    
    # Handle NaNs if any channel is flat
    if np.isnan(C).any():
        C = np.nan_to_num(C)

    # 3. Singular Values (Eigenvalues of Symmetric Matrix)
    # For correlation matrix, eigenvalues are sufficient
    try:
        s = np.linalg.eigvalsh(C)
        s = s[s > 1e-10] # Filter numerical noise
    except np.linalg.LinAlgError:
        return np.nan

    # 4. Normalize to probability distribution
    p = s / np.sum(s)

    # 5. Shannon Entropy
    entropy = -np.sum(p * np.log(p))

    # 6. Effective Rank
    return np.exp(entropy)

def run_sleep_analysis(n_subjects=2):
    # Fetch data (will download if not present)
    # Using just 2 subjects for demonstration/fast validation
    print(f"Fetching {n_subjects} subjects from Sleep-EDFx...")
    files = fetch_data(subjects=range(n_subjects), recording=[1])

    ranks = {'W': [], 'N1': [], 'N2': [], 'N3': [], 'REM': []}
    
    # Mapping annotations to simple stages
    # Sleep-EDF uses: 'Sleep stage W', 'Sleep stage 1', ...
    event_id = {
        'Sleep stage W': 1,
        'Sleep stage 1': 2,
        'Sleep stage 2': 3,
        'Sleep stage 3': 4,
        'Sleep stage 4': 4, # Merge N3/N4
        'Sleep stage R': 5
    }
    
    stage_map = {1: 'W', 2: 'N1', 3: 'N2', 4: 'N3', 5: 'REM'}

    for subj_idx, (raw_fname, annot_fname) in enumerate(files):
        print(f"Processing Subject {subj_idx}...")
        
        # Load raw data
        raw = mne.io.read_raw_edf(raw_fname, preload=True, verbose=False)
        annot = mne.read_annotations(annot_fname)
        raw.set_annotations(annot, emit_warning=False)
        
        # Standardize channel names and pick EEG
        # Sleep-EDF channel names are like 'EEG Fpz-Cz', 'EEG Pz-Oz'
        # We only have 2 EEG channels in this dataset! 
        # Wait, Sleep-EDFx only has 2 EEG channels (Fpz-Cz, Pz-Oz).
        # This is a limitation for "High-Dimensional" Rank analysis.
        # Rank max is 2. This might be too small to show "Collapse" effectively.
        # Let's check if we can use EOG as well to increase "system dimension"?
        # Or maybe this dataset is NOT suitable for RANK analysis due to low channel count.
        
        # ALTERNATIVE: Use High-Density EEG if available?
        # For Sleep-EDFx, we strictly only have 2 EEG + 1 EOG.
        # Let's include EOG to get 3 dim. 
        # Rank range: 1.0 to 3.0. 
        # It's weak evidence but better than nothing.
        
        raw.pick_types(eeg=True, eog=True, stim=False, exclude=[])
        print(f"  Channels: {raw.ch_names}")
        
        # Filter
        raw.filter(0.5, 30.0, verbose=False)

        # Epoching
        # Create fixed length events to analyze continuous data in chunks
        # OR better: leverage the annotations directly
        events, _ = mne.events_from_annotations(raw, event_id=event_id, chunk_duration=30.)
        
        # Create epochs
        tmin, tmax = 0., 30. - 1./raw.info['sfreq']
        # The event_id dictionary needs to map description strings to integers, 
        # BUT events_from_annotations returns integer event codes that correspond 
        # to the keys in its returned event_id dictionary.
        # We need to correctly map the events found in the file to our target stages.
        
        # 1. Get events from annotations
        events, event_id_found = mne.events_from_annotations(raw, chunk_duration=30., event_id=event_id)
        
        print(f"  Events found: {event_id_found}")
        
        # 2. Create Epochs using the events we just extracted
        # Note: We must pass event_id=event_id (the mapping) to Epochs 
        # to correctly label them.
        
        # DEBUG: Print epoch construction info
        # print(f"  Constructing epochs with events shape: {events.shape}")
        
        epochs = mne.Epochs(raw, events, event_id=event_id, tmin=tmin, tmax=tmax, 
                            baseline=None, verbose=False, preload=True)
                            
        # DEBUG: Check if epochs are actually found for each stage
        # print(f"  Epochs found per condition: {epochs.event_id}")

        # Compute rank per epoch
        for stage_code, stage_name in stage_map.items():
            if stage_name not in ranks: continue
            
            # Check if this stage exists in the found epochs
            # The event_id keys in epochs are the description strings (e.g. 'Sleep stage W')
            # But stage_map has stage codes. 
            # We need to map stage_name back to the event description in event_id dict.
            
            # Find key in event_id that has value == stage_code
            target_event_desc = None
            for k, v in event_id.items():
                if v == stage_code:
                    target_event_desc = k
                    break
            
            if target_event_desc and target_event_desc in epochs.event_id:
                stage_epochs = epochs[target_event_desc]
                
                # Compute rank for each 30s window
                stage_ranks = []
                data = stage_epochs.get_data() # [n_epochs, n_channels, n_times]
                
                for epoch_data in data:
                    r = compute_effective_rank(epoch_data)
                    stage_ranks.append(r)
                
                # Average for this subject
                if len(stage_ranks) > 0:
                    avg_r = np.mean(stage_ranks)
                    ranks[stage_name].append(avg_r)
                    print(f"  {stage_name}: {avg_r:.3f} (n={len(stage_ranks)})")
            else:
                pass
                # print(f"  No epochs found for {stage_name}")

    # --- Plotting ---
    print("\nResults Summary (Mean Rank):")
    means = []
    stds = []
    labels = ['W', 'N1', 'N2', 'N3', 'REM']
    
    # Nature-style Color Palette
    # W: Red (Awake/High Energy)
    # N1/N2: Light Blues (Transition)
    # N3: Dark Blue (Deep Sleep/Low Rank)
    # REM: Purple or Teal (Paradoxical)
    nature_colors = ['#E64B35', '#4DBBD5', '#3C5488', '#00A087', '#8491B4'] # Red, Light Blue, Dark Blue, Teal, Grey

    for l in labels:
        vals = ranks[l]
        m = np.mean(vals) if len(vals) > 0 else 0
        s = np.std(vals) if len(vals) > 0 else 0
        means.append(m)
        stds.append(s)
        print(f"  {l}: {m:.3f} +/- {s:.3f}")

    # Set Nature style
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    plt.rcParams['font.size'] = 7
    plt.rcParams['axes.linewidth'] = 0.5
    
    plt.figure(figsize=(3.5, 3), dpi=300) # Single column width ~89mm
    
    # Create bar plot
    bars = plt.bar(labels, means, yerr=stds, capsize=3, 
            color=nature_colors, edgecolor='black', linewidth=0.5, 
            error_kw={'elinewidth':0.5})
            
    plt.ylabel('Effective Rank', fontsize=7)
    plt.title('EEG Geometric Rank (Sleep-EDFx)', fontsize=8)
    plt.tick_params(axis='both', which='major', labelsize=6, width=0.5)
    
    # Remove top and right spines
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    plt.tight_layout()
    os.makedirs('figures', exist_ok=True)
    plt.savefig('figures/sleep_eeg_rank_validation.png', dpi=300)
    print("Plot saved to figures/sleep_eeg_rank_validation.png")

if __name__ == "__main__":
    run_sleep_analysis()
