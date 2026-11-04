"""DINO Self-Supervised Distillation Loss with Centering and Sharpening."""

from typing import Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from core.registry import LOSSES


@LOSSES.register()
class DINOLoss(nn.Module):
    """Self-distillation Cross-Entropy Loss for DINO.

    Computes cross-entropy loss between sharpened and centered teacher softmax probabilities
    and student softmax distributions across all global and local multi-crop views.

    Args:
        out_dim: Output dimension of projection head (e.g., 16384).
        ncrops: Total number of crop views (2 global + N local).
        warmup_teacher_temp: Initial temperature for teacher sharpening.
        teacher_temp: Final converged temperature for teacher sharpening (e.g., 0.04 or 0.07).
        warmup_teacher_temp_epochs: Number of epochs to warm up teacher temperature.
        nepochs: Total training epochs.
        student_temp: Fixed temperature for student softmax (default: 0.1).
        center_momentum: Momentum factor for EMA center updates (default: 0.9).
    """

    def __init__(
        self,
        out_dim: int,
        ncrops: int,
        warmup_teacher_temp: float,
        teacher_temp: float,
        warmup_teacher_temp_epochs: int,
        nepochs: int,
        student_temp: float = 0.1,
        center_momentum: float = 0.9,
    ) -> None:
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops

        # Register center as buffer
        self.register_buffer("center", torch.zeros(1, out_dim))

        # Schedule teacher temperature warmup
        if nepochs < warmup_teacher_temp_epochs:
            self.teacher_temp_schedule = np.linspace(warmup_teacher_temp, teacher_temp, nepochs)
        else:
            self.teacher_temp_schedule = np.concatenate((
                np.linspace(warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs),
                np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp,
            ))

    def forward(
        self,
        student_output: torch.Tensor,
        teacher_output: torch.Tensor,
        epoch: int,
    ) -> torch.Tensor:
        """Computes cross-entropy between student and teacher softmax outputs.

        Args:
            student_output: Concatenated student logits across all ncrops views.
            teacher_output: Concatenated teacher logits across 2 global views.
            epoch: Current training epoch index.

        Returns:
            Scalar cross-entropy distillation loss.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # Teacher centering and sharpening
        epoch_idx = min(epoch, len(self.teacher_temp_schedule) - 1)
        temp = self.teacher_temp_schedule[epoch_idx]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0.0
        n_loss_terms = 0

        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # Skip matching identical views (student global i vs teacher global i)
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss = total_loss + loss.mean()
                n_loss_terms += 1

        total_loss = total_loss / n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output: torch.Tensor) -> None:
        """Updates the running center vector using Exponential Moving Average (EMA).

        Args:
            teacher_output: Teacher logits of shape (B * 2, out_dim).
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        batch_center = batch_center / len(teacher_output)

        # EMA update
        self.center = self.center * self.center_momentum + batch_center * (1.0 - self.center_momentum)
