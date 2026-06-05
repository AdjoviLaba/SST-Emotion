"""
No-Meta ablation: skip meta-training entirely, fine-tune from random init.

Useful for measuring the contribution of meta-learning vs. a vanilla
fine-tuning baseline.
"""

from typing import Callable, List, Tuple

import torch

from .base import MetaLearner


class NoMeta(MetaLearner):

    def meta_train(
        self,
        get_task_fn: Callable[[int, int], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        train_subjects: List[int],
    ) -> None:
        # No meta-training — model stays at random initialization.
        print("  [NoMeta] skipping meta-training (ablation baseline)")
