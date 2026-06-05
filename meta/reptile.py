"""
Reptile meta-learning (Nichol & Schulman 2018).

Update rule: θ ← θ + ε * (φ_i − θ)
where φ_i are the weights after K inner gradient steps on task i.

No support/query split needed — uses the full task batch for inner updates.
"""

import copy
from typing import Callable, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from .base import MetaLearner


class Reptile(MetaLearner):

    def meta_train(
        self,
        get_task_fn: Callable[[int, int], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        train_subjects: List[int],
    ) -> None:
        meta_iterations = int(self.cfg.get("meta_iterations", 300))
        meta_lr = float(self.cfg.get("meta_lr", 0.1))
        inner_lr = float(self.cfg["lr"])
        inner_steps = int(self.cfg.get("inner_steps", 3))
        batch_size = int(self.cfg["batch_size"])

        criterion = nn.CrossEntropyLoss()
        meta_state = copy.deepcopy(self.model.state_dict())

        rng = np.random.default_rng(int(self.cfg.get("seed", 42)))

        for iteration in range(meta_iterations):
            subj = int(rng.choice(train_subjects))
            sess = int(rng.integers(0, 3))

            spec, temp, labels = get_task_fn(subj, sess)
            spec, temp, labels = spec.to(self.device), temp.to(self.device), labels.to(self.device)

            # Clone meta state into task model
            task_model = copy.deepcopy(self.model)
            task_model.load_state_dict(meta_state)
            task_model.train()
            optimizer = torch.optim.Adam(task_model.parameters(), lr=inner_lr,
                                         betas=(0.9, 0.999), eps=1e-8)

            n = spec.size(0)
            for _ in range(inner_steps):
                # Mini-batch inner step
                idx = torch.randperm(n, device=self.device)[:batch_size]
                optimizer.zero_grad()
                out = task_model(spec[idx], temp[idx])
                loss = criterion(out, labels[idx])
                loss.backward()
                optimizer.step()

            # Reptile outer update: θ ← θ + ε*(φ - θ)
            task_state = task_model.state_dict()
            for key in meta_state:
                meta_state[key] = meta_state[key] + meta_lr * (
                    task_state[key].float() - meta_state[key].float()
                )

            if (iteration + 1) % 50 == 0:
                print(f"  [Reptile] iter {iteration + 1}/{meta_iterations}")

        self.model.load_state_dict(meta_state)
