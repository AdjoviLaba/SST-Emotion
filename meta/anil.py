"""
ANIL: Almost No Inner Loop (Raghu et al. 2019).

Key difference from FOMAML: the inner loop adapts ONLY the classification
head. The feature extractor is frozen during inner updates but receives
outer-loop gradients (through the query forward pass).

First-order implementation: inner loop uses create_graph=False so second-
order terms through the inner update are ignored. The outer gradient for
the head uses the first-order approximation (gradient at adapted params).

Manual parameter update is used instead of learn2learn because the
current version lacks a standalone ANIL module.
"""

from typing import Callable, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import MetaLearner


def _apply_head(feat: torch.Tensor, params: list, training: bool = False) -> torch.Tensor:
    """Apply Sequential [Linear(d,50), Dropout(0.5), Linear(50,C)] with custom params."""
    # params order: [weight_0, bias_0, weight_1, bias_1]
    h = F.linear(feat, params[0], params[1])
    h = F.dropout(h, p=0.5, training=training)
    h = F.linear(h, params[2], params[3])
    return h


class ANIL(MetaLearner):

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

            # detach().clone() produces a true leaf tensor; .requires_grad_(True)
            # enables gradient tracking for the first-order outer update.
            head_params_0 = [p.detach().clone().requires_grad_(True)
                              for p in self.model.head.parameters()]
            head_params = head_params_0  # head_params will be reassigned each step

            # ---- Inner loop: adapt head only (features frozen via no_grad) ----
            for _ in range(inner_steps):
                idx = torch.randperm(spec_s.size(0), device=self.device)[:batch_size]
                with torch.no_grad():
                    feat_s = self.model.get_features(spec_s[idx], temp_s[idx])
                out_s = _apply_head(feat_s, head_params, training=True)
                support_loss = criterion(out_s, lab_s[idx])

                # create_graph=False → first-order approximation
                grads = torch.autograd.grad(support_loss, head_params,
                                            create_graph=False)
                head_params = [p - inner_lr * g for p, g in zip(head_params, grads)]

            # ---- Outer loop: feature extractor gets gradients via query forward ----
            self.model.train()
            feat_q = self.model.get_features(spec_q, temp_q)      # grad flows here
            out_q = _apply_head(feat_q, head_params, training=False)
            query_loss = criterion(out_q, lab_q)

            meta_optimizer.zero_grad()
            query_loss.backward()

            # Assign first-order head gradient to the actual model head params
            for meta_p, init_p in zip(self.model.head.parameters(), head_params_0):
                meta_p.grad = init_p.grad  # gradient at adapted point (1st-order approx)

            meta_optimizer.step()

            if (iteration + 1) % 50 == 0:
                print(f"  [ANIL] iter {iteration + 1}/{meta_iterations}  "
                      f"query_loss={query_loss.item():.4f}")
