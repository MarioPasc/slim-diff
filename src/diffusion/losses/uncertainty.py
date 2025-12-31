"""Uncertainty-weighted loss (Kendall et al.).

Implements learnable uncertainty weights for multi-task learning
where each output channel (image, mask) has its own uncertainty.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class UncertaintyWeightedLoss(nn.Module):
    """Uncertainty-weighted multi-task loss.

    Implements the homoscedastic uncertainty weighting from:
    "Multi-Task Learning Using Uncertainty to Weigh Losses for Scene
    Geometry and Semantics" (Kendall et al., 2018)

    For each task i with loss L_i:
        weighted_loss_i = exp(-log_var_i) * L_i + 0.5 * log_var_i

    Where log_var_i = log(sigma_i^2) is a learnable parameter.
    """

    def __init__(
        self,
        n_tasks: int = 2,
        initial_log_vars: list[float] | None = None,
        learnable: bool = True,
        clamp_range: tuple[float, float] = (-5.0, 5.0),
    ) -> None:
        """Initialize the loss module.

        Args:
            n_tasks: Number of tasks (default 2 for image + mask).
            initial_log_vars: Initial values for log variance parameters.
            learnable: Whether log_vars are learnable.
            clamp_range: Min/max values for clamping log_vars.
        """
        super().__init__()
        self.n_tasks = n_tasks
        self.learnable = learnable
        self.clamp_range = clamp_range

        # Initialize log variance parameters
        if initial_log_vars is None:
            initial_log_vars = [0.0] * n_tasks

        # Create learnable parameters
        log_vars = torch.tensor(initial_log_vars, dtype=torch.float32)
        if learnable:
            self.log_vars = nn.Parameter(log_vars)
        else:
            self.register_buffer("log_vars", log_vars)

        logger.info(
            f"UncertaintyWeightedLoss: "
            f"n_tasks={n_tasks}, "
            f"initial_log_vars={initial_log_vars}, "
            f"learnable={learnable}, "
            f"clamp_range={clamp_range}"
        )

    def forward(
        self,
        losses: list[torch.Tensor] | torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute uncertainty-weighted total loss.

        Args:
            losses: List of task losses or tensor of shape (n_tasks,).

        Returns:
            Tuple of (total_loss, details_dict) where details_dict contains
            individual weighted losses and log_vars.
        """
        if isinstance(losses, torch.Tensor):
            losses = [losses[i] for i in range(losses.shape[0])]

        if len(losses) != self.n_tasks:
            raise ValueError(
                f"Expected {self.n_tasks} losses, got {len(losses)}"
            )

        total_loss = torch.tensor(0.0, device=losses[0].device)
        details = {}

        for i, loss_i in enumerate(losses):
            # Clamp log_var to prevent unbounded growth
            log_var_i = torch.clamp(self.log_vars[i], min=self.clamp_range[0], max=self.clamp_range[1])

            # Weighted loss: exp(-log_var) * loss + 0.5 * log_var
            precision = torch.exp(-log_var_i)
            weighted_loss = precision * loss_i + 0.5 * log_var_i

            total_loss = total_loss + weighted_loss

            details[f"loss_{i}"] = loss_i.detach()
            details[f"weighted_loss_{i}"] = weighted_loss.detach()
            details[f"log_var_{i}"] = log_var_i.detach()
            details[f"sigma_{i}"] = torch.exp(0.5 * log_var_i).detach()

        details["total_loss"] = total_loss.detach()

        return total_loss, details

    def get_log_vars(self) -> torch.Tensor:
        """Get current log variance values (unclamped).

        Note: These are the raw parameter values. For values that match
        the optimization (clamped to [-5, 5]), use get_log_vars_clamped().

        Returns:
            Tensor of log variance values.
        """
        return self.log_vars.detach()

    def get_weights(self) -> torch.Tensor:
        """Get current effective weights (precisions).

        Returns:
            Tensor of weights exp(-log_var).
        """
        return torch.exp(-self.log_vars).detach()

    def get_log_vars_clamped(self) -> torch.Tensor | None:
        """Get clamped log variance values used in forward pass.

        Returns the same clamped values [-5, 5] that are used during
        loss computation to ensure logging matches optimization.

        Returns:
            Clamped log variance tensor if learnable, None otherwise.
        """
        if not self.learnable:
            return None
        return torch.clamp(self.log_vars.detach(), self.clamp_range[0], self.clamp_range[1])


class SimpleWeightedLoss(nn.Module):
    """Simple fixed-weight multi-task loss.

    Alternative to uncertainty weighting with fixed weights.
    """

    def __init__(
        self,
        weights: list[float] | None = None,
        n_tasks: int = 2,
    ) -> None:
        """Initialize the loss module.

        Args:
            weights: Fixed weights for each task.
            n_tasks: Number of tasks.
        """
        super().__init__()
        self.n_tasks = n_tasks

        if weights is None:
            weights = [1.0] * n_tasks

        self.register_buffer("weights", torch.tensor(weights, dtype=torch.float32))

    def forward(
        self,
        losses: list[torch.Tensor] | torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute weighted total loss.

        Args:
            losses: List of task losses.

        Returns:
            Tuple of (total_loss, details_dict).
        """
        if isinstance(losses, torch.Tensor):
            losses = [losses[i] for i in range(losses.shape[0])]

        total_loss = torch.tensor(0.0, device=losses[0].device)
        details = {}

        for i, loss_i in enumerate(losses):
            weighted_loss = self.weights[i] * loss_i
            total_loss = total_loss + weighted_loss

            details[f"loss_{i}"] = loss_i.detach()
            details[f"weighted_loss_{i}"] = weighted_loss.detach()
            details[f"weight_{i}"] = self.weights[i]

        details["total_loss"] = total_loss.detach()

        return total_loss, details
