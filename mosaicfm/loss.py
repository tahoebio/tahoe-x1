# Copyright (C) Vevo Therapeutics 2024-2025. All rights reserved.
from typing import Any

import torch
import torch.nn.functional as F
from torchmetrics import Metric
from torchmetrics.functional.regression import spearman_corrcoef


def masked_mse_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Compute the masked mean squared error loss between input and target.

    This function calculates MSE loss only for positions where the mask is True,
    allowing for computation over variable-length sequences or partially masked data.
    The loss is normalized by the sum of the mask to account for different numbers
    of valid positions across batches.

    Args:
        input: Predicted values with shape (batch_size, seq_len) or (batch_size, seq_len, dim).
        target: Target values with same shape as input.
        mask: Boolean mask with shape matching input/target. True indicates valid positions
            to include in loss computation.

    Returns:
        Scalar tensor containing the masked MSE loss.
    """
    mask = mask.float()
    loss = F.mse_loss(input * mask, target * mask, reduction="sum")
    return loss / mask.sum()


def criterion_neg_log_bernoulli(
    input: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Compute the masked negative log-likelihood of Bernoulli distribution.

    This function treats the problem as binary classification where target values > 0
    are considered positive class. The input should contain probabilities (0-1 range)
    and the function computes the negative log-likelihood of observing the targets
    under a Bernoulli distribution parameterized by the input probabilities.

    Args:
        input: Predicted probabilities with shape (batch_size, seq_len). Values should
            be in [0, 1] range representing probability of positive class.
        target: Target values with same shape as input. Values > 0 are treated as
            positive class, values <= 0 as negative class.
        mask: Boolean mask with shape matching input/target. True indicates valid
            positions to include in loss computation.

    Returns:
        Scalar tensor containing the masked negative log-likelihood loss.
    """
    mask = mask.float()
    bernoulli = torch.distributions.Bernoulli(probs=input)
    masked_log_probs = bernoulli.log_prob((target > 0).float()) * mask
    return -masked_log_probs.sum() / mask.sum()


def masked_relative_error(
    input: torch.Tensor,
    target: torch.Tensor,
    mask: torch.LongTensor,
) -> torch.Tensor:
    """Compute the masked mean absolute relative error between input and target.

    This function calculates the relative error |input - target| / (target + eps)
    only for positions where the mask is True. The small epsilon value (1e-6)
    prevents division by zero for targets close to zero.

    Args:
        input: Predicted values with arbitrary shape.
        target: Target values with same shape as input.
        mask: Boolean mask with same shape as input/target. Only positions where
            mask is True are included in the error computation.

    Returns:
        Scalar tensor containing the mean relative error over masked positions.

    Raises:
        AssertionError: If no positions in the mask are True.
    """
    assert mask.any()
    loss = torch.abs(input[mask] - target[mask]) / (target[mask] + 1e-6)
    return loss.mean()


class MaskedMseMetric(Metric):
    """Torchmetrics-compatible metric for computing masked mean squared error.

    This metric accumulates MSE loss over multiple batches while handling variable-length
    sequences or partially masked data. It maintains running sums of the MSE loss and
    mask counts to compute the final average.

    The metric is designed for single-cell genomics applications where sequences may
    have different lengths or certain positions may be padded/masked.

    Attributes:
        name: Human-readable name for the metric (used for logging).
        sum_mse: Running sum of masked MSE values across all batches.
        sum_mask: Running sum of mask counts (number of valid positions).

    Examples:
        >>> metric = MaskedMseMetric(name="Gene_MSE")
        >>> preds = torch.randn(32, 100)  # batch_size=32, seq_len=100
        >>> targets = torch.randn(32, 100)
        >>> mask = torch.randint(0, 2, (32, 100)).bool()  # random mask
        >>> metric.update(preds, targets, mask)
        >>> mse_value = metric.compute()
    """

    def __init__(self, name: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.name = name
        self.add_state(
            "sum_mse",
            default=torch.tensor(0.0, dtype=torch.float32),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "sum_mask",
            default=torch.tensor(0.0, dtype=torch.float32),
            dist_reduce_fx="sum",
        )
        # Type annotations for state variables
        self.sum_mse: torch.Tensor
        self.sum_mask: torch.Tensor

    def update(
        self,
        preds: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> None:
        """Update the metric state with a new batch of predictions and targets.

        Args:
            preds: Predicted values with shape (batch_size, seq_len) or broader.
            target: Target values with same shape as preds.
            mask: Boolean mask with same shape as preds/target. True indicates
                valid positions to include in metric computation.

        Raises:
            ValueError: If preds and target have different shapes.
        """
        if preds.shape != target.shape:
            raise ValueError("preds and target must have the same shape")
        mask = mask.float()
        self.sum_mse += torch.nn.functional.mse_loss(
            preds * mask,
            target * mask,
            reduction="sum",
        )
        self.sum_mask += mask.sum()

    def compute(self) -> torch.Tensor:
        """Compute the final masked MSE metric value.

        Returns:
            Scalar tensor containing the average MSE loss across all batches
            and valid positions.
        """
        return self.sum_mse / self.sum_mask


class MaskedSpearmanMetric(Metric):
    """Torchmetrics-compatible metric for computing masked Spearman correlation
    coefficient.

    This metric computes Spearman rank correlation between predictions and targets
    for each example in the batch, handling variable-length sequences through masking.
    The correlation is computed separately for each sequence and then averaged.

    Spearman correlation is particularly useful for gene expression prediction tasks
    where the rank order of expression values may be more important than absolute
    values, making it robust to scaling differences.

    Attributes:
        name: Human-readable name for the metric (used for logging).
        sum_spearman: Running sum of Spearman correlation coefficients.
        num_examples: Running count of processed examples.

    Examples:
        >>> metric = MaskedSpearmanMetric(name="Gene_Spearman")
        >>> preds = torch.randn(32, 100)  # batch_size=32, seq_len=100
        >>> targets = torch.randn(32, 100)
        >>> mask = torch.randint(0, 2, (32, 100)).bool()  # random mask
        >>> metric.update(preds, targets, mask)
        >>> correlation = metric.compute()
    """

    def __init__(self, name: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.name = name
        self.add_state(
            "sum_spearman",
            default=torch.tensor(0.0, dtype=torch.float32),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "num_examples",
            default=torch.tensor(0.0, dtype=torch.float32),
            dist_reduce_fx="sum",
        )
        # Type annotations for state variables
        self.sum_spearman: torch.Tensor
        self.num_examples: torch.Tensor

    def update(
        self,
        preds: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> None:
        """Update the metric state with a new batch of predictions and targets.

        This method computes Spearman correlation for each example in the batch
        separately, using only the non-masked positions for each sequence.

        Args:
            preds: Predicted values with shape (batch_size, seq_len).
            target: Target values with same shape as preds.
            mask: Boolean mask with same shape as preds/target. True indicates
                valid positions to include in correlation computation.

        Raises:
            ValueError: If preds and target have different shapes.

        Note:
            Tensors are moved to CPU for Spearman correlation computation as
            required by the torchmetrics implementation.
        """
        if preds.shape != target.shape:
            raise ValueError("preds and target must have the same shape")
        for pred_i, target_i, mask_i in zip(preds, target, mask):
            non_mask_preds = pred_i[mask_i].to("cpu")
            non_mask_targets = target_i[mask_i].to("cpu")
            self.sum_spearman += spearman_corrcoef(non_mask_preds, non_mask_targets)
            self.num_examples += 1

    def compute(self) -> torch.Tensor:
        """Compute the final average Spearman correlation coefficient.

        Returns:
            Scalar tensor containing the average Spearman correlation across
            all processed examples.
        """
        return self.sum_spearman / self.num_examples
