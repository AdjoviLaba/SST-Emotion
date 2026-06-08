import numpy as np
import scipy.ndimage
import os

def map_to_grid(features):
    """
    features: shape (5, 62)
    Returns: shape (5, 32, 32)
    """
    # 9x9 grid mapping
    grid_map = np.array([
        [-1, -1, -1,  0,  1,  2, -1, -1, -1],
        [-1, -1,  3, -1, -1, -1,  4, -1, -1],
        [-1,  5,  6,  7,  8,  9, 10, 11, -1],
        [-1, 12, 13, 14, 15, 16, 17, 18, -1],
        [19, 20, 21, 22, 23, 24, 25, 26, 27],
        [-1, 28, 29, 30, 31, 32, 33, 34, -1],
        [-1, 35, 36, 37, 38, 39, 40, 41, -1],
        [-1, -1, 42, 43, 44, 45, 46, -1, -1],
        [-1, -1, -1, 47, 48, 49, -1, -1, -1],
    ])
    
    mapped = np.zeros((5, 9, 9))
    for b in range(5):
        for i in range(9):
            for j in range(9):
                ch = grid_map[i, j]
                if ch != -1:
                    mapped[b, i, j] = features[b, ch]
    
    # Zoom from 9x9 to 32x32
    # output shape (5, 32, 32)
    resized = scipy.ndimage.zoom(mapped, (1, 32/9, 32/9), order=1)
    return resized

def main():
    print("Loading raw data...")
    # Update paths to match the user's workspace
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data = np.load(os.path.join(base_dir, 'SEED/DatasetCaricatoNoImage/arr_0.npy'))
    labels = np.load(os.path.join(base_dir, 'SEED/LabelsNoImage/arr_0.npy'))
    subjects = np.load(os.path.join(base_dir, 'SEED/SubjectsNoImage/arr_0.npy'))
    
    out_dir = os.path.abspath(os.path.join(base_dir, '../SEED_input_data'))
    print(f"Saving preprocessed data to: {out_dir}")
    
    for split in ['train', 'test']:
        for type_dir in ['feature', 'temporal', 'label']:
            os.makedirs(os.path.join(out_dir, split, type_dir), exist_ok=True)
            for i in range(15):
                os.makedirs(os.path.join(out_dir, split, type_dir, f'subject_{i}'), exist_ok=True)
                
    unique_subs = np.unique(subjects)
    
    # Window size for temporal
    W = 5 
    
    for sub in unique_subs:
        print(f"Processing Subject {sub}...")
        sub_mask = (subjects == sub)
        sub_data = data[sub_mask]     # (N, 5, 62)
        sub_labels = labels[sub_mask] # (N,)
        
        # Z-score normalization per subject
        mean = np.mean(sub_data, axis=0, keepdims=True)
        std = np.std(sub_data, axis=0, keepdims=True)
        std[std == 0] = 1e-8 # Prevent division by zero
        sub_data = (sub_data - mean) / std
        
        N = len(sub_data)
        
        # 1. Apply spatial mapping to all samples for this subject
        spatial_data = []
        for i in range(N):
            spatial_data.append(map_to_grid(sub_data[i]))
        spatial_data = np.array(spatial_data) # (N, 5, 32, 32)
        
        # 2. Build temporal stream
        # We need to take 5 consecutive frames. 
        # To maintain the same N, we will zero-pad the first 4 frames.
        temporal_data = np.zeros((N, 5, 5, 32, 32)) # (N, W, bands, H, W)
        for i in range(N):
            for w in range(W):
                idx = i - (W - 1) + w
                if idx >= 0:
                    temporal_data[i, w] = spatial_data[idx]
        
        # Reshape to expected format: 
        # specInput: [N, 32, 32, 5, 1]
        spec_input = np.transpose(spatial_data, (0, 2, 3, 1)) # (N, 32, 32, 5)
        spec_input = np.expand_dims(spec_input, axis=-1)      # (N, 32, 32, 5, 1)
        
        # temInput: [N, 32, 32, 25, 1]
        # Transpose to (N, 32, 32, 5_time, 5_bands)
        temp_input = np.transpose(temporal_data, (0, 3, 4, 1, 2))
        # Reshape to (N, 32, 32, 25)
        temp_input = temp_input.reshape((N, 32, 32, 25))
        temp_input = np.expand_dims(temp_input, axis=-1)      # (N, 32, 32, 25, 1)
        
        # 3. Split into 3 sections
        chunk_size = N // 3
        sections = [
            (0, chunk_size),
            (chunk_size, 2*chunk_size),
            (2*chunk_size, N)
        ]
        
        for j, (start, end) in enumerate(sections):
            sec_spec = spec_input[start:end]
            sec_temp = temp_input[start:end]
            sec_lbl = sub_labels[start:end]
            
            # 20/80 train/test split: fine-tune on 20% (~3 trials), evaluate on 80% (~12 trials)
            split_idx = int(0.2 * len(sec_spec))
            
            train_spec = sec_spec[:split_idx]
            train_temp = sec_temp[:split_idx]
            train_lbl  = sec_lbl[:split_idx]
            
            test_spec = sec_spec[split_idx:]
            test_temp = sec_temp[split_idx:]
            test_lbl  = sec_lbl[split_idx:]
            
            # Save files
            np.save(os.path.join(out_dir, 'train', 'feature', f'subject_{sub}', f'section_{j}_data.npy'), train_spec)
            np.save(os.path.join(out_dir, 'train', 'temporal', f'subject_{sub}', f'section_{j}_data.npy'), train_temp)
            np.save(os.path.join(out_dir, 'train', 'label', f'subject_{sub}', f'section_{j}_label.npy'), train_lbl)
            
            np.save(os.path.join(out_dir, 'test', 'feature', f'subject_{sub}', f'section_{j}_data.npy'), test_spec)
            np.save(os.path.join(out_dir, 'test', 'temporal', f'subject_{sub}', f'section_{j}_data.npy'), test_temp)
            np.save(os.path.join(out_dir, 'test', 'label', f'subject_{sub}', f'section_{j}_label.npy'), test_lbl)
            
    print("Preprocessing complete!")

if __name__ == '__main__':
    main()
