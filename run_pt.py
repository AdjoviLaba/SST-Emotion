"""
run_pt.py — PyTorch entry point for the meta-learning benchmark.

Usage:
    python run_pt.py -c config/SEED.ini --meta_algorithm reptile
    python run_pt.py -c config/SEED.ini --meta_algorithm fomaml --smoke_test

Algorithms: reptile | fomaml | maml | anil | none
"""

import argparse
import configparser
import copy
import csv
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import meta as meta_registry
from model.model_pt import build_model


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

SEED = 42

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Device detection — report where we are running
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    if torch.cuda.is_available():
        dev = torch.device("cuda")
    elif torch.backends.mps.is_available():
        # Test if Conv3D is supported on MPS (emulated or older versions lack it)
        try:
            test_x = torch.randn(1, 1, 3, 3, 3, device="mps")
            test_conv = torch.nn.Conv3d(1, 1, 3).to("mps")
            _ = test_conv(test_x)
            dev = torch.device("mps")
        except Exception:
            print("[device] MPS is available but does not support Conv3D. Falling back to CPU.")
            dev = torch.device("cpu")
    else:
        dev = torch.device("cpu")
    print(f"[device] Using {dev}")
    if dev.type == "cpu":
        print("[device] WARNING: running on CPU — a full LOSO sweep will be slow.")
    return dev


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def read_config(config_path: str) -> dict:
    conf = configparser.ConfigParser()
    conf.optionxform = str  # Preserve case!
    conf.read(config_path)

    cfg = {}
    for section in conf.sections():
        for key, val in conf.items(section):
            cfg[key] = val
            cfg[key.lower()] = val

    # Inject meta hyperparameters with defaults (override in .ini if needed)
    for k, v in [
        ("meta_iterations", "300"),
        ("meta_lr", "0.1"),
        ("inner_steps", "3"),
        ("patience", "15"),
        ("seed", str(SEED)),
        ("nb_subjects", "15")
    ]:
        cfg.setdefault(k, v)
        cfg.setdefault(k.lower(), v)

    return cfg


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_npy_task(root_spec: str, root_temp: str, root_label: str,
                  subject: int, session: int) -> tuple:
    """Load one (subject, session) task and return tensors ready for use."""
    spec = np.load(os.path.join(root_spec,
                                f"subject_{subject}/section_{session}_data.npy"))
    temp = np.load(os.path.join(root_temp,
                                f"subject_{subject}/section_{session}_data.npy"))
    labels = np.load(os.path.join(root_label,
                                  f"subject_{subject}/section_{session}_label.npy"))
    
    spec_t = torch.from_numpy(spec).float()
    temp_t = torch.from_numpy(temp).float()
    
    # Convert from Keras shape (N, H, W, D, 1) to PyTorch shape (N, 1, D, H, W)
    if spec_t.dim() == 5 and spec_t.shape[-1] == 1:
        spec_t = spec_t.squeeze(-1).permute(0, 3, 1, 2).unsqueeze(1)
    elif spec_t.dim() == 4:
        spec_t = spec_t.permute(0, 3, 1, 2).unsqueeze(1)
        
    if temp_t.dim() == 5 and temp_t.shape[-1] == 1:
        temp_t = temp_t.squeeze(-1).permute(0, 3, 1, 2).unsqueeze(1)
    elif temp_t.dim() == 4:
        temp_t = temp_t.permute(0, 3, 1, 2).unsqueeze(1)
        
    labels_t = torch.from_numpy(labels).long()
    return spec_t, temp_t, labels_t


def make_loader(spec: torch.Tensor, temp: torch.Tensor,
                labels: torch.Tensor, batch_size: int,
                shuffle: bool = True) -> DataLoader:
    ds = TensorDataset(spec, temp, labels)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def load_subject_splits(cfg: dict, subject: int):
    """
    Load fine-tune (train split) and test split for a given subject.
    Concatenates all 3 sessions; shuffles fine-tune set.

    Prints sample counts — rule: always verify before training.
    Expected on SEED: ~678 fine-tune / ~2716 test per subject.
    """
    ft_specs, ft_temps, ft_labels = [], [], []
    ev_specs, ev_temps, ev_labels = [], [], []

    for sess in range(3):
        s, t, l = load_npy_task(
            cfg["train_specinput_root_path"],
            cfg["train_tempinput_root_path"],
            cfg["train_label_root_path"],
            subject, sess)
        ft_specs.append(s); ft_temps.append(t); ft_labels.append(l)

        s, t, l = load_npy_task(
            cfg["test_specinput_root_path"],
            cfg["test_tempinput_root_path"],
            cfg["test_label_root_path"],
            subject, sess)
        ev_specs.append(s); ev_temps.append(t); ev_labels.append(l)

    ft_spec = torch.cat(ft_specs); ft_temp = torch.cat(ft_temps)
    ft_label = torch.cat(ft_labels)
    ev_spec = torch.cat(ev_specs); ev_temp = torch.cat(ev_temps)
    ev_label = torch.cat(ev_labels)

    # Shuffle fine-tune
    idx = torch.randperm(ft_spec.size(0))
    ft_spec, ft_temp, ft_label = ft_spec[idx], ft_temp[idx], ft_label[idx]

    print(f"  [data] Subject {subject}: fine-tune={ft_spec.size(0)}, "
          f"test={ev_spec.size(0)}  "
          f"ratio={ft_spec.size(0) / (ft_spec.size(0) + ev_spec.size(0)):.2f}")

    ratio = ft_spec.size(0) / (ft_spec.size(0) + ev_spec.size(0))
    if ratio > 0.35:
        print(f"  [data] WARNING: fine-tune ratio {ratio:.2f} > 0.35 — "
              f"expected ~0.20. Check preprocess.py split setting.")

    batch_size = int(cfg["batch_size"])
    ft_loader = make_loader(ft_spec, ft_temp, ft_label, batch_size, shuffle=True)
    ev_loader = make_loader(ev_spec, ev_temp, ev_label, batch_size, shuffle=False)
    return ft_loader, ev_loader


# ---------------------------------------------------------------------------
# Result logging
# ---------------------------------------------------------------------------

def init_result_files(result_path: str, algo: str) -> tuple:
    os.makedirs(result_path, exist_ok=True)
    csv_path = os.path.join(result_path, f"{algo}_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["subject", "accuracy", "f1", "kappa", "wall_time"])
        writer.writeheader()
    return csv_path


def append_result(csv_path: str, row: dict) -> None:
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["subject", "accuracy", "f1", "kappa", "wall_time"])
        writer.writerow(row)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(args) -> None:
    cfg = read_config(args.c)
    device = get_device()
    set_seed(int(cfg["seed"]))

    result_path = cfg.get("result_path", "./result")
    model_save_path = cfg.get("model_save_path", "./output_model")
    os.makedirs(model_save_path, exist_ok=True)

    algo = args.meta_algorithm.lower()
    csv_path = init_result_files(result_path, algo)

    nb_subjects = int(cfg["nb_subjects"])
    subject_range = range(1) if args.smoke_test else range(nb_subjects)

    print(f"\n{'='*60}")
    print(f"  Algorithm : {algo.upper()}")
    print(f"  Subjects  : {'SMOKE TEST (subject 0 only)' if args.smoke_test else nb_subjects}")
    print(f"  Seed      : {cfg['seed']}")
    print(f"  Device    : {device}")
    print(f"  Config    : {args.c}")
    print(f"{'='*60}\n")

    # Log all hyperparameters to disk for reproducibility
    hparam_path = os.path.join(result_path, f"{algo}_hparams.json")
    with open(hparam_path, "w") as f:
        json.dump({**cfg, "meta_algorithm": algo, "device": str(device)}, f, indent=2)

    all_accuracies = []

    for test_subject in subject_range:
        print(f"\n{'='*50}")
        print(f"  LOSO: test subject {test_subject}")
        print(f"{'='*50}")

        set_seed(int(cfg["seed"]) + test_subject)  # per-subject seed for reproducibility

        # Build fresh model for each subject
        model = build_model(cfg, device)
        learner = meta_registry.build(algo, model, cfg)

        train_subjects = [i for i in range(nb_subjects) if i != test_subject]

        def get_task_fn(subj, sess):
            return load_npy_task(
                cfg["train_specinput_root_path"],
                cfg["train_tempinput_root_path"],
                cfg["train_label_root_path"],
                subj, sess)

        # --- Meta-training ---
        t_meta_start = time.time()
        print("Meta-training...")
        learner.meta_train(get_task_fn, train_subjects)
        meta_state = copy.deepcopy(learner.model.state_dict())
        print(f"  Meta-training done in {time.time() - t_meta_start:.1f}s")

        # --- Fine-tune & evaluate ---
        ft_loader, ev_loader = load_subject_splits(cfg, test_subject)
        metrics = learner.fine_tune_and_eval(ft_loader, ev_loader, meta_state)

        print(f"\n  Subject {test_subject}: "
              f"acc={metrics['accuracy']:.4f}  "
              f"f1={metrics['f1']:.4f}  "
              f"kappa={metrics['kappa']:.4f}  "
              f"wall={metrics['wall_time']:.1f}s")

        # Save history
        hist_path = os.path.join(result_path, f"{algo}_Sub{test_subject}_history.json")
        with open(hist_path, "w") as f:
            json.dump(metrics["history"], f)

        row = {
            "subject": test_subject,
            "accuracy": f"{metrics['accuracy']:.6f}",
            "f1": f"{metrics['f1']:.6f}",
            "kappa": f"{metrics['kappa']:.6f}",
            "wall_time": f"{metrics['wall_time']:.1f}",
        }
        append_result(csv_path, row)
        all_accuracies.append(metrics["accuracy"])

    # --- Aggregate summary ---
    if len(all_accuracies) > 1:
        mean_acc = np.mean(all_accuracies)
        std_acc = np.std(all_accuracies)
        print(f"\n{'='*60}")
        print(f"  {algo.upper()} LOSO summary")
        print(f"  Subjects evaluated : {len(all_accuracies)}")
        print(f"  Mean accuracy      : {mean_acc:.4f} ± {std_acc:.4f}")
        print(f"{'='*60}\n")

        summary_path = os.path.join(result_path, f"{algo}_summary.json")
        with open(summary_path, "w") as f:
            json.dump({
                "algorithm": algo,
                "n_subjects": len(all_accuracies),
                "mean_accuracy": mean_acc,
                "std_accuracy": std_acc,
                "per_subject": all_accuracies,
            }, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PyTorch meta-learning benchmark for EEG emotion recognition.")
    parser.add_argument("-c", type=str, required=True,
                        help="Path to .ini config file.")
    parser.add_argument("--meta_algorithm", type=str, default="reptile",
                        choices=["reptile", "fomaml", "maml", "anil", "none"],
                        help="Meta-learning algorithm.")
    parser.add_argument("--smoke_test", action="store_true",
                        help="Run only subject 0 to verify the pipeline.")
    args = parser.parse_args()
    run(args)
