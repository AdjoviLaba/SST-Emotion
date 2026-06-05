"""
Abstract base class for meta-learning algorithms.

All algorithms share the same interface so run_pt.py never needs to know
which algorithm is running.

Contract
--------
meta_train(get_task_fn, train_subjects)
    Updates self.model weights in-place to the meta-initialization.
    get_task_fn(subject, session) → (spec, temp, labels) as torch.Tensors on device.

fine_tune_and_eval(ft_loader, eval_loader) → dict
    Fine-tunes from the current model weights and returns metrics.
    Resets model back to meta-init weights before returning so LOSO subjects
    are independent.
"""

import copy
import time
from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class MetaLearner(ABC):

    def __init__(self, model: nn.Module, cfg: dict):
        self.model = model
        self.cfg = cfg
        self.device = next(model.parameters()).device

    @abstractmethod
    def meta_train(
        self,
        get_task_fn: Callable[[int, int], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        train_subjects: List[int],
    ) -> None:
        """Mutate self.model with meta-trained weights."""

    def fine_tune_and_eval(
        self,
        ft_loader: DataLoader,
        eval_loader: DataLoader,
        meta_state: dict,
    ) -> Dict:
        """
        Fine-tune a copy of the model from meta_state, evaluate, return metrics.
        Does NOT modify self.model.
        """
        import copy
        from sklearn.metrics import f1_score, cohen_kappa_score

        ft_model = copy.deepcopy(self.model)
        ft_model.load_state_dict(meta_state)
        ft_model.to(self.device)

        lr = float(self.cfg["lr"])
        nb_epoch = int(self.cfg["nbEpoch"])
        patience = int(self.cfg.get("patience", 15))

        optimizer = torch.optim.Adam(ft_model.parameters(), lr=lr,
                                     betas=(0.9, 0.999), eps=1e-8)
        criterion = nn.CrossEntropyLoss()

        best_val_loss = float("inf")
        best_state = copy.deepcopy(ft_model.state_dict())
        epochs_no_improve = 0
        history = {"train_loss": [], "val_loss": [], "val_acc": []}

        t0 = time.time()
        for epoch in range(nb_epoch):
            ft_model.train()
            for spec, temp, labels in ft_loader:
                spec, temp, labels = spec.to(self.device), temp.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                out = ft_model(spec, temp)
                loss = criterion(out, labels)
                loss.backward()
                optimizer.step()

            ft_model.eval()
            val_loss, val_correct, val_total = 0.0, 0, 0
            with torch.no_grad():
                for spec, temp, labels in eval_loader:
                    spec, temp, labels = spec.to(self.device), temp.to(self.device), labels.to(self.device)
                    out = ft_model(spec, temp)
                    val_loss += criterion(out, labels).item() * labels.size(0)
                    val_correct += (out.argmax(1) == labels).sum().item()
                    val_total += labels.size(0)

            val_loss /= val_total
            val_acc = val_correct / val_total
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            if hasattr(ft_loader.dataset, "__len__"):
                # estimate train loss cheaply from last batch
                history["train_loss"].append(float(loss.item()))

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = copy.deepcopy(ft_model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    break

        # Evaluate best checkpoint
        ft_model.load_state_dict(best_state)
        ft_model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for spec, temp, labels in eval_loader:
                spec, temp = spec.to(self.device), temp.to(self.device)
                preds = ft_model(spec, temp).argmax(1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.numpy())

        accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
        f1 = f1_score(all_labels, all_preds, average="macro")
        kappa = cohen_kappa_score(all_labels, all_preds)

        return {
            "accuracy": accuracy,
            "f1": f1,
            "kappa": kappa,
            "history": history,
            "wall_time": time.time() - t0,
        }
