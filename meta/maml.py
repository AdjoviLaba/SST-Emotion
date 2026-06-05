"""
MAML: Model-Agnostic Meta-Learning (Finn et al. 2017), full second-order.

Uses torch.func.functional_call (PyTorch >= 2.0) to run the model with
custom parameter dicts, allowing gradients to flow through inner-loop
updates back to the meta-initialization (second-order).

Inner update rule:
    φ_i = θ - α * ∇_θ L_support(f_θ)   [K steps, create_graph=True]

Outer update:
    θ ← θ - β * ∇_θ L_query(f_{φ_i})   [second-order gradient]

Warning: ~2-3× slower and ~2× more memory than FOMAML due to the retained
computation graph. Estimate wall-clock before running a full sweep.
"""

from typing import Callable, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.func import functional_call

from .base import MetaLearner


class MAML(MetaLearner):

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

        # Buffers (BatchNorm running stats) stay fixed; only parameters are adapted
        buffers = dict(self.model.named_buffers())

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

            # Leaf copies of meta-params — inner updates happen here
            task_params = {n: p.clone() for n, p in self.model.named_parameters()}

            # Inner loop: retain computation graph for second-order gradients
            self.model.train()
            for _ in range(inner_steps):
                idx = torch.randperm(spec_s.size(0), device=self.device)[:batch_size]
                out = functional_call(self.model, (task_params, buffers),
                                      (spec_s[idx], temp_s[idx]))
                loss = criterion(out, lab_s[idx])
                grads = torch.autograd.grad(loss, task_params.values(),
                                            create_graph=True)  # retain graph for 2nd order
                task_params = {n: p - inner_lr * g
                               for (n, p), g in zip(task_params.items(), grads)}

            # Outer loss through adapted params — second-order gradient back to self.model
            out_q = functional_call(self.model, (task_params, buffers),
                                    (spec_q, temp_q))
            query_loss = criterion(out_q, lab_q)

            meta_optimizer.zero_grad()
            query_loss.backward()   # second-order gradients reach self.model.parameters()
            meta_optimizer.step()

            if (iteration + 1) % 50 == 0:
                print(f"  [MAML] iter {iteration + 1}/{meta_iterations}  "
                      f"query_loss={query_loss.item():.4f}")
