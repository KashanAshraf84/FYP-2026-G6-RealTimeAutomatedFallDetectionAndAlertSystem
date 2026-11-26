"""
Fall Detection System - Focal Loss for Class Imbalance
======================================================
Addresses severe class imbalance in fall detection:
- Normal everyday activities (walking, sitting, standing) represent 90-95% of data.
- Falls are rare (<5% of frames).
- Stumbling/unstable postures (warning) represent a transitional minority.

Focal Loss formulation:
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

Parameters:
    gamma: Focusing parameter (typically 2.0).
           Down-weights well-classified easy examples and forces the model
           to focus on hard, ambiguous, and rare fall events.
    alpha: Class weight vector to balance cross-class representation.
    label_smoothing: Prevents overconfidence on edge-case postures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, List


class FocalLoss(nn.Module):
    """
    Multi-Class Focal Loss with optional class weights and label smoothing.
    """

    def __init__(
        self,
        alpha: Optional[Union[List[float], torch.Tensor]] = None,
        gamma: float = 2.0,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ):
        """
        Args:
            alpha: Class weights (list or tensor of shape (num_classes,)).
                   If None, uniform weighting is used.
            gamma: Focusing parameter (>= 0). Default is 2.0.
            reduction: 'mean', 'sum', or 'none'.
            label_smoothing: Label smoothing factor in [0.0, 1.0).
        """
        super().__init__()
        self.gamma = float(gamma)
        self.reduction = reduction
        self.label_smoothing = float(label_smoothing)

        if alpha is not None:
            if isinstance(alpha, (list, tuple)):
                self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32))
            elif isinstance(alpha, torch.Tensor):
                self.register_buffer("alpha", alpha.float())
            else:
                raise TypeError(f"alpha must be list or Tensor, got {type(alpha)}")
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute multi-class Focal Loss.

        Args:
            logits: Predicted raw scores, shape (batch_size, num_classes).
            targets: Ground truth class indices, shape (batch_size,).

        Returns:
            Computed scalar loss (if reduction is 'mean' or 'sum') or per-sample loss.
        """
        num_classes = logits.size(-1)

        # 1. Compute log-probabilities for numerical stability
        log_p = F.log_softmax(logits, dim=-1)
        p = torch.exp(log_p)

        # 2. Gather probabilities for target classes
        target_indices = targets.view(-1, 1)
        log_pt = log_p.gather(dim=-1, index=target_indices).squeeze(-1)
        pt = p.gather(dim=-1, index=target_indices).squeeze(-1)

        # 3. Compute focal modulating factor: (1 - pt)^gamma
        focal_weight = torch.pow(1.0 - pt, self.gamma)

        # 4. Apply alpha weighting
        if self.alpha is not None:
            if self.alpha.device != logits.device:
                self.alpha = self.alpha.to(logits.device)
            alpha_t = self.alpha.gather(dim=0, index=targets)
            focal_loss = -alpha_t * focal_weight * log_pt
        else:
            focal_loss = -focal_weight * log_pt

        # 5. Apply label smoothing if configured
        if self.label_smoothing > 0.0:
            smooth_loss = -log_p.mean(dim=-1)
            focal_loss = (1.0 - self.label_smoothing) * focal_loss + self.label_smoothing * smooth_loss

        # 6. Reduction
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        elif self.reduction == "none":
            return focal_loss
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")

    def __repr__(self) -> str:
        return (
            f"FocalLoss(gamma={self.gamma}, "
            f"alpha={self.alpha.tolist() if self.alpha is not None else None}, "
            f"reduction='{self.reduction}', label_smoothing={self.label_smoothing})"
        )
