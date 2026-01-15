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


class GroupUncertaintyWeightedLoss(nn.Module):
    """Group-level uncertainty-weighted multi-task loss.

    Extension of UncertaintyWeightedLoss that groups related losses
    together under a single uncertainty parameter per group.

    Example grouping for JS-DDPM with FFL:
        Group 0: mse_image + mse_mask (spatial losses)
        Group 1: ffl (frequency loss)

    Each group has one learnable log_var parameter.

    For each group g:
        group_loss_g = sum(intra_weight_i * loss_i) for i in group_g
        weighted_g = exp(-log_var_g) * group_loss_g + 0.5 * log_var_g
    total_loss = sum(weighted_g)
    """

    def __init__(
        self,
        n_groups: int = 2,
        group_membership: list[int] | None = None,
        initial_log_vars: list[float] | None = None,
        learnable: bool = True,
        clamp_range: tuple[float, float] = (-5.0, 5.0),
        intra_group_weights: list[float] | None = None,
    ) -> None:
        """Initialize the loss module.

        Args:
            n_groups: Number of loss groups.
            group_membership: List mapping each loss to its group index.
                             E.g., [0, 0, 1] means losses 0,1 are group 0,
                             loss 2 is group 1.
            initial_log_vars: Initial values for each group's log variance.
            learnable: Whether log_vars are learnable.
            clamp_range: Min/max values for clamping log_vars.
            intra_group_weights: Fixed weights for losses within same group.
                                E.g., [1.0, 1.0, 1.0] for equal weighting.
        """
        super().__init__()
        self.n_groups = n_groups
        self.learnable = learnable
        self.clamp_range = clamp_range

        # Default group membership: [0, 0, 1] for [mse_img, mse_mask, ffl]
        if group_membership is None:
            group_membership = [0, 0, 1]
        self.group_membership = group_membership

        # Default intra-group weights: equal weighting
        if intra_group_weights is None:
            intra_group_weights = [1.0] * len(group_membership)
        self.register_buffer(
            "intra_group_weights",
            torch.tensor(intra_group_weights, dtype=torch.float32),
        )

        # Initialize log variance parameters (one per GROUP)
        if initial_log_vars is None:
            initial_log_vars = [0.0] * n_groups

        log_vars = torch.tensor(initial_log_vars, dtype=torch.float32)
        if learnable:
            self.log_vars = nn.Parameter(log_vars)
        else:
            self.register_buffer("log_vars", log_vars)

        logger.info(
            f"GroupUncertaintyWeightedLoss: n_groups={n_groups}, "
            f"membership={self.group_membership}, "
            f"initial_log_vars={initial_log_vars}, "
            f"intra_group_weights={intra_group_weights}"
        )

    def forward(
        self,
        losses: list[torch.Tensor] | torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute group uncertainty-weighted total loss.

        Args:
            losses: List of individual task losses.

        Returns:
            Tuple of (total_loss, details_dict).
        """
        if isinstance(losses, torch.Tensor):
            losses = [losses[i] for i in range(losses.shape[0])]

        n_losses = len(losses)
        if n_losses != len(self.group_membership):
            raise ValueError(
                f"Expected {len(self.group_membership)} losses, got {n_losses}"
            )

        device = losses[0].device

        # Aggregate losses per group with intra-group weights
        group_losses = [
            torch.tensor(0.0, device=device) for _ in range(self.n_groups)
        ]

        for i, (loss_i, group_idx) in enumerate(
            zip(losses, self.group_membership)
        ):
            group_losses[group_idx] = (
                group_losses[group_idx] + self.intra_group_weights[i] * loss_i
            )

        # Apply uncertainty weighting per group
        total_loss = torch.tensor(0.0, device=device)
        details = {}

        for g in range(self.n_groups):
            log_var_g = torch.clamp(
                self.log_vars[g],
                min=self.clamp_range[0],
                max=self.clamp_range[1],
            )

            precision_g = torch.exp(-log_var_g)
            weighted_group_loss = precision_g * group_losses[g] + 0.5 * log_var_g

            total_loss = total_loss + weighted_group_loss

            details[f"group_{g}_loss"] = group_losses[g].detach()
            details[f"group_{g}_weighted_loss"] = weighted_group_loss.detach()
            details[f"log_var_group_{g}"] = log_var_g.detach()
            details[f"sigma_group_{g}"] = torch.exp(0.5 * log_var_g).detach()
            details[f"precision_group_{g}"] = precision_g.detach()

        # Also log individual losses for debugging
        for i, loss_i in enumerate(losses):
            details[f"loss_{i}"] = loss_i.detach()

        details["total_loss"] = total_loss.detach()

        return total_loss, details

    def get_log_vars(self) -> torch.Tensor:
        """Get current log variance values (per group, unclamped).

        Returns:
            Tensor of log variance values.
        """
        return self.log_vars.detach()

    def get_log_vars_clamped(self) -> torch.Tensor | None:
        """Get clamped log variance values used in forward pass.

        Returns:
            Clamped log variance tensor if learnable, None otherwise.
        """
        if not self.learnable:
            return None
        return torch.clamp(
            self.log_vars.detach(), self.clamp_range[0], self.clamp_range[1]
        )

    def get_weights(self) -> torch.Tensor:
        """Get current effective weights (precisions) per group.

        Returns:
            Tensor of weights exp(-log_var).
        """
        return torch.exp(-self.log_vars).detach()
