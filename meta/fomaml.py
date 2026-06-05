"""
FOMAML: First-Order MAML (Finn et al. 2017, first-order variant).

Hand-coded without learn2learn for compatibility with Python 3.11+.

Inner loop: K SGD/Adam steps on support set (deepcopy of meta-model).
Outer step: gradient of query loss at adapted params, copied to meta-model
            (first-order approximation — ignores second-order terms through
            the inner update).
"""

import copy
from typing import Callable, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from .base import MetaLearner


class FOMAML(MetaLearner):

    def meta_train(
        self,
        get_task_fn: Callable[[int, int], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        train_subjects: List[int],
    ) -> None:
        meta_iterations = int(self.cfg.get("meta_iterations", 300))
        inner_lr = float(self.cfg["lr"])
        meta_lr = float(self.cfg.get("meta_lr", 0.01))
        inner_steps = int(self.cfg.get("inner_steps", 3))
        batch_size = int(self.cfg["batch_size"])

        meta_optimizer = torch.optim.Adam(self.model.parameters(), lr=meta_lr,
                                          betas=(0.9, 0.999), eps=1e-8)
        criterion = nn.CrossEntropyLoss()
        rng = np.random.default_rng(int(self.cfg.get("seed", 42)))

        for iteration in range(meta_iterations):
            subj = int(rng.choice(train_subjects))
            sess = int(rng.integers(0, 3))

            spec, temp, labels = get_task_fn(subj, sess)
            spec, temp, labels = (spec.to(self.device), temp.to(self.device),
                                   labels.to(self.device))

            n = spec.size(0)
            perm = torch.randperm(n, device=self.device)
            half = max(1, n // 2)
            s_idx, q_idx = perm[:half], perm[half:]

            spec_s, temp_s, lab_s = spec[s_idx], temp[s_idx], labels[s_idx]
            spec_q, temp_q, lab_q = spec[q_idx], temp[q_idx], labels[q_idx]

            # Clone meta model for task-level inner update
            task_model = copy.deepcopy(self.model)
            task_model.train()
            inner_opt = torch.optim.SGD(task_model.parameters(), lr=inner_lr)

            for _ in range(inner_steps):
                idx = torch.randperm(spec_s.size(0), device=self.device)[:batch_size]
                inner_opt.zero_grad()
                loss = criterion(task_model(spec_s[idx], temp_s[idx]), lab_s[idx])
                loss.backward()
                inner_opt.step()

            # Outer gradient at adapted params (first-order approx)
            for p in task_model.parameters():
                p.grad = None
            query_loss = criterion(task_model(spec_q, temp_q), lab_q)
            query_loss.backward()

            # Assign task gradient to meta-model, then step
            meta_optimizer.zero_grad()
            for meta_p, task_p in zip(self.model.parameters(), task_model.parameters()):
                meta_p.grad = task_p.grad

            meta_optimizer.step()

            if (iteration + 1) % 50 == 0:
                print(f"  [FOMAML] iter {iteration + 1}/{meta_iterations}  "
                      f"query_loss={query_loss.item():.4f}")
