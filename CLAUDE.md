# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SST-EmotionNet is an EEG-based emotion recognition system implementing the Spatial-Spectral-Temporal Attention 3D Dense Network. It uses Reptile meta-learning with Leave-One-Subject-Out (LOSO) cross-validation across 15 subjects on the SEED (3-class) and SEED-IV (4-class) datasets.

## Environment

- Python 3.7.7, TensorFlow-GPU 2.1.0, Keras 2.3.1, CUDA 10.1 / CuDNN 7.6.5
- Dependencies: `numpy==1.16.2`, `scipy==1.4.1`, `tensorflow_gpu==2.1.0`, `Keras==2.3.1`

## Commands

**Preprocess raw data** (must be run before training):
```bash
python preprocess.py
```

**Train the model** (requires a config file):
```bash
python run.py -c ./config/SEED.ini
```

**Plot results** (after training completes):
```bash
python plot_results.py
```

**Test the EEG-to-grid mapping utility:**
```bash
python test_mapping.py
```

## Data Pipeline

1. Raw SEED data lives in `SEED/DatasetCaricatoNoImage/arr_0.npy` with shape `(N, 5, 62)` — N samples, 5 frequency bands (delta/theta/alpha/beta/gamma), 62 EEG channels.
2. `preprocess.py` maps 62 channels onto a 9×9 topographic grid (with -1 for empty slots), resizes it to 32×32, constructs temporal windows of W=5, then does per-subject z-score normalization. Output is stored in `../SEED_input_data/` with an 80/20 train/test split per subject per session (3 sections each).
3. `run.py` reads the preprocessed `.npy` files via paths defined in the config `.ini` file.

## Architecture

The model (`model/model.py`) has two parallel streams merged at the end:

- **Spatial-Spectral Stream**: Input `[N, 32, 32, 5, 1]` — 5 frequency band maps stacked as a 3D volume.
- **Spatial-Temporal Stream**: Input `[N, 32, 32, 25, 1]` — 5 time steps × 5 frequency bands flattened to 25.

Each stream passes through `__create_dense_net`, which builds a 3D DenseNet with:
- `__conv_block`: decomposed 3D convolutions — `(3,3,1)` then `(1,1,3)` — rather than a single `(3,3,3)` kernel.
- `__dense_block`: feature reuse via concatenation (standard DenseNet pattern).
- `__transition_block`: `Conv3D(1×1×1)` + `AveragePooling3D(2×2×2)` for compression.
- `Attention_block`: after each transition, computes separate spatial and temporal attention weights via sigmoid-gated dense layers applied to the channel-wise mean.

After both streams produce feature vectors via `GlobalAveragePooling3D`, they are concatenated and passed through `Dense(50) → Dropout(0.5) → Dense(nb_class)`.

## Training Loop (`run.py`)

For each of the 15 test subjects (LOSO):
1. **Reptile meta-training**: 50 iterations, each sampling a random (subject, session) task from the 14 training subjects, running 3 inner gradient steps, then updating meta-weights as `meta_w += meta_lr * (task_w - meta_w)`.
2. **Fine-tuning**: Load meta-weights, concatenate all 3 sessions of the test subject's training split, shuffle, and train with `EarlyStopping(patience=15)` and `ModelCheckpoint` saving the best val_accuracy model.
3. Results saved to `result/all_meta_result.txt` and per-subject training history to `result/Sub_{i}_history.json`.

## Configuration (`config/SEED.ini`)

Key parameters:
- `input_width=32` — spatial map side length
- `specInput_length=5` — number of frequency bands (spectral depth)
- `temInput_length=25` — temporal depth (5 time steps × 5 bands)
- `depth_spec` / `depth_tem` — DenseNet depth for each stream
- `gr_spec` / `gr_tem` — growth rate for each stream
- `nb_dense_block` — number of dense blocks per stream
- `nb_class=3` — emotion classes (3 for SEED, 4 for SEED-IV)

## Known Issues

- `model.py:151` applies the initial `Conv3D` twice to `img_input` directly instead of chaining the second conv on the output of the first — this is a bug in the original code.
- `test_mapping.py` uses a naive sequential grid assignment (not anatomically correct); `preprocess.py` uses the proper topographic mapping.
