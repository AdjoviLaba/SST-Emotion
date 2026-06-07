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

    resized = scipy.ndimage.zoom(mapped, (1, 32/9, 32/9), order=1)
    return resized


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    feat_dir = os.path.join(base_dir, 'SEED_EEG', 'ExtractedFeatures_1s')
    out_dir = os.path.abspath(os.path.join(base_dir, '../SEED_input_data'))

    # Labels: -1/0/1 → 0/1/2 (same 15 clips shown every session)
    label_mat = sio.loadmat(os.path.join(feat_dir, 'label.mat'))
    trial_labels = label_mat['label'][0] + 1  # shape (15,), values 0/1/2

    print(f"Saving preprocessed data to: {out_dir}")
    for split in ['train', 'test']:
        for type_dir in ['feature', 'temporal', 'label']:
            for i in range(15):
                os.makedirs(os.path.join(out_dir, split, type_dir, f'subject_{i}'), exist_ok=True)

    W = 5

    # Group files by subject ID (prefix before first '_')
    all_files = glob.glob(os.path.join(feat_dir, '*.mat'))
    all_files = [f for f in all_files if os.path.basename(f) != 'label.mat']

    subject_files = {}
    for f in all_files:
        sub_id = int(os.path.basename(f).split('_')[0])
        subject_files.setdefault(sub_id, []).append(f)

    for sub in sorted(subject_files.keys()):
        print(f"Processing Subject {sub}...")
        sessions = sorted(subject_files[sub])  # sort by date in filename
        assert len(sessions) == 3, f"Expected 3 sessions for subject {sub}, got {len(sessions)}"

        # Load all sessions
        session_data = []
        session_lbls = []

        for sess_path in sessions:
            mat = sio.loadmat(sess_path)
            sess_feats = []
            sess_lbls_list = []

            for trial in range(1, 16):
                key = f"de_LDS{trial}"
                trial_data = mat[key]       # shape (62, T, 5)
                T = trial_data.shape[1]
                trial_data_t = np.transpose(trial_data, (1, 2, 0))  # (T, 5, 62)
                sess_feats.append(trial_data_t)
                sess_lbls_list.append(np.full((T,), trial_labels[trial - 1]))

            session_data.append(np.concatenate(sess_feats, axis=0))
            session_lbls.append(np.concatenate(sess_lbls_list, axis=0))

        # Z-score normalization per subject across all sessions
        all_sub_data = np.concatenate(session_data, axis=0)
        mean = np.mean(all_sub_data, axis=0, keepdims=True)
        std = np.std(all_sub_data, axis=0, keepdims=True)
        std[std == 0] = 1e-8

        normalized_sessions = [(s - mean) / std for s in session_data]

        # Each session becomes one section (j=0,1,2)
        for j, (sec_data, sec_labels) in enumerate(zip(normalized_sessions, session_lbls)):
            N_sec = len(sec_data)

            # Spatial mapping
            spatial_data = np.array([map_to_grid(sec_data[i]) for i in range(N_sec)])  # (N, 5, 32, 32)

            # Temporal windowing
            temporal_data = np.zeros((N_sec, 5, 5, 32, 32))
            for i in range(N_sec):
                for w in range(W):
                    idx = i - (W - 1) + w
                    if idx >= 0:
                        temporal_data[i, w] = spatial_data[idx]

            # specInput: (N, 32, 32, 5, 1)
            spec_input = np.transpose(spatial_data, (0, 2, 3, 1))
            spec_input = np.expand_dims(spec_input, axis=-1)

            # temInput: (N, 32, 32, 25, 1)
            temp_input = np.transpose(temporal_data, (0, 3, 4, 1, 2))
            temp_input = temp_input.reshape((N_sec, 32, 32, 25))
            temp_input = np.expand_dims(temp_input, axis=-1)

            # 20% train / 80% test split
            split_idx = int(0.2 * N_sec)

            sub_idx = sub - 1  # 0-based index
            for prefix, spec, temp, lbl in [
                ('train', spec_input[:split_idx],  temp_input[:split_idx],  sec_labels[:split_idx]),
                ('test',  spec_input[split_idx:],  temp_input[split_idx:],  sec_labels[split_idx:]),
            ]:
                np.save(os.path.join(out_dir, prefix, 'feature',  f'subject_{sub_idx}', f'section_{j}_data.npy'),  spec)
                np.save(os.path.join(out_dir, prefix, 'temporal', f'subject_{sub_idx}', f'section_{j}_data.npy'),  temp)
                np.save(os.path.join(out_dir, prefix, 'label',    f'subject_{sub_idx}', f'section_{j}_label.npy'), lbl)

    print("SEED preprocessing complete!")


if __name__ == '__main__':
    main()
