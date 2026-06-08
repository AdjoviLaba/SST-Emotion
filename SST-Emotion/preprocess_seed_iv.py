import os
import glob
import numpy as np
import scipy.io as sio
import scipy.ndimage

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
    resized = scipy.ndimage.zoom(mapped, (1, 32/9, 32/9), order=1)
    return resized

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    seed_iv_dir = os.path.join(base_dir, 'SEED_IV')
    out_dir = os.path.abspath(os.path.join(base_dir, '../SEED_IV_input_data'))
    
    print("Pre-creating output directories...")
    for split in ['train', 'test']:
        for type_dir in ['feature', 'temporal', 'label']:
            os.makedirs(os.path.join(out_dir, split, type_dir), exist_ok=True)
            for i in range(15):
                os.makedirs(os.path.join(out_dir, split, type_dir, f'subject_{i}'), exist_ok=True)
                
    # Labels from ReadMe.txt
    session_labels = {
        1: [1,2,3,0,2,0,0,1,0,1,2,1,1,1,2,3,2,2,3,3,0,3,0,3],
        2: [2,1,3,0,0,2,0,2,3,3,2,3,2,0,1,1,2,1,0,3,0,1,3,1],
        3: [1,2,2,1,3,3,3,1,1,2,1,0,2,3,3,0,2,3,0,0,2,0,1,0]
    }
    
    W = 5 # Temporal window size
    
    # Process each subject (1 to 15)
    for sub in range(1, 16):
        print(f"Processing Subject {sub}...")
        
        # Load all 3 sessions' data
        session_data = []
        session_lbls = []
        
        for sess in [1, 2, 3]:
            sess_dir = os.path.join(seed_iv_dir, 'eeg_feature_smooth', str(sess))
            # Find the file starting with "{sub}_"
            pattern = os.path.join(sess_dir, f"{sub}_*.mat")
            files = glob.glob(pattern)
            if not files:
                raise FileNotFoundError(f"Could not find .mat file for subject {sub}, session {sess} in {sess_dir}")
            
            mat_path = files[0]
            mat_data = sio.loadmat(mat_path)
            
            labels_list = session_labels[sess]
            
            sess_feats = []
            sess_lbls_list = []
            
            for trial in range(1, 25):
                key = f"de_LDS{trial}"
                # Shape: (62, T, 5)
                trial_data = mat_data[key]
                T = trial_data.shape[1]
                
                # Transpose to (T, 5, 62)
                trial_data_t = np.transpose(trial_data, (1, 2, 0))
                sess_feats.append(trial_data_t)
                sess_lbls_list.append(np.full((T,), labels_list[trial - 1]))
                
            session_data.append(np.concatenate(sess_feats, axis=0)) # (N_sess, 5, 62)
            session_lbls.append(np.concatenate(sess_lbls_list, axis=0)) # (N_sess,)
            
        # Z-score normalization per subject across all sessions
        all_sub_data = np.concatenate(session_data, axis=0) # (N, 5, 62)
        mean = np.mean(all_sub_data, axis=0, keepdims=True)
        std = np.std(all_sub_data, axis=0, keepdims=True)
        std[std == 0] = 1e-8
        
        # Normalize each session using the global subject stats
        normalized_sessions = []
        for sess_feats in session_data:
            normalized_sessions.append((sess_feats - mean) / std)
            
        # Apply spatial mapping, temporal windowing, and saving per session (section)
        for j, (sec_data, sec_labels) in enumerate(zip(normalized_sessions, session_lbls)):
            N_sec = len(sec_data)
            
            # 1. Apply spatial mapping
            spatial_data = []
            for i in range(N_sec):
                spatial_data.append(map_to_grid(sec_data[i]))
            spatial_data = np.array(spatial_data) # (N_sec, 5, 32, 32)
            
            # 2. Build temporal stream
            temporal_data = np.zeros((N_sec, 5, 5, 32, 32))
            for i in range(N_sec):
                for w in range(W):
                    idx = i - (W - 1) + w
                    if idx >= 0:
                        temporal_data[i, w] = spatial_data[idx]
                        
            # Reshape to expected format
            # specInput: (N_sec, 32, 32, 5, 1)
            spec_input = np.transpose(spatial_data, (0, 2, 3, 1))
            spec_input = np.expand_dims(spec_input, axis=-1)
            
            # temInput: (N_sec, 32, 32, 25, 1)
            temp_input = np.transpose(temporal_data, (0, 3, 4, 1, 2))
            temp_input = temp_input.reshape((N_sec, 32, 32, 25))
            temp_input = np.expand_dims(temp_input, axis=-1)
            
            # 80/20 train/test split per subject section
            split_idx = int(0.8 * N_sec)
            
            train_spec = spec_input[:split_idx]
            train_temp = temp_input[:split_idx]
            train_lbl  = sec_labels[:split_idx]
            
            test_spec = spec_input[split_idx:]
            test_temp = temp_input[split_idx:]
            test_lbl  = sec_labels[split_idx:]
            
            # Save files (use 0-based subject index to match SEED evaluation)
            np.save(os.path.join(out_dir, 'train', 'feature', f'subject_{sub - 1}', f'section_{j}_data.npy'), train_spec)
            np.save(os.path.join(out_dir, 'train', 'temporal', f'subject_{sub - 1}', f'section_{j}_data.npy'), train_temp)
            np.save(os.path.join(out_dir, 'train', 'label', f'subject_{sub - 1}', f'section_{j}_label.npy'), train_lbl)
            
            np.save(os.path.join(out_dir, 'test', 'feature', f'subject_{sub - 1}', f'section_{j}_data.npy'), test_spec)
            np.save(os.path.join(out_dir, 'test', 'temporal', f'subject_{sub - 1}', f'section_{j}_data.npy'), test_temp)
            np.save(os.path.join(out_dir, 'test', 'label', f'subject_{sub - 1}', f'section_{j}_label.npy'), test_lbl)
            
    print("SEED-IV preprocessing complete!")

if __name__ == '__main__':
    main()
