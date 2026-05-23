"""
Validation utilities for expanded models.
"""

from typing import Dict, List, Optional, Tuple, Any, Callable
import logging

import torch
from torch import nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def validate_model_output(
    model: nn.Module,
    input_ids: torch.Tensor,
    expected_output: Optional[torch.Tensor] = None,
    tolerance: float = 1e-5,
) -> Dict[str, Any]:
    """
    Validate that a model produces expected outputs.

    Args:
        model: Model to validate
        input_ids: Input token IDs
        expected_output: Expected output for comparison
        tolerance: Numerical tolerance for comparison

    Returns:
        Validation results
    """
    model.eval()

    with torch.no_grad():
        try:
            outputs = model(input_ids)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs

            results = {
                "success": True,
                "output_shape": tuple(logits.shape),
                "has_nan": torch.isnan(logits).any().item(),
                "has_inf": torch.isinf(logits).any().item(),
                "max_value": logits.max().item(),
                "min_value": logits.min().item(),
            }

            if expected_output is not None:
                diff = torch.abs(logits - expected_output)
                results["max_diff"] = diff.max().item()
                results["mean_diff"] = diff.mean().item()
                results["matches_expected"] = diff.max().item() < tolerance

            return results

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }


class CatastrophicForgettingDetector:
    """
    Monitor KL divergence between expanded model and frozen base.

    Helps detect when expanded model deviates too much from
    original capabilities.
    """

    def __init__(
        self,
        base_model: nn.Module,
        threshold: float = 0.1,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize the detector.

        Args:
            base_model: The original frozen base model
            threshold: KL divergence threshold for alerting
            device: Device to run on
        """
        self.base_model = base_model
        self.threshold = threshold
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Freeze base model
        self.base_model.eval()
        for param in self.base_model.parameters():
            param.requires_grad = False

        self.base_model.to(self.device)

        self.kl_history: List[float] = []

    def check(
        self,
        expanded_model: nn.Module,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[bool, float]:
        """
        Check if expanded model has diverged from base.

        Args:
            expanded_model: The expanded model to check
            batch: Input batch

        Returns:
            (is_acceptable, kl_divergence)
        """
        expanded_model.eval()

        with torch.no_grad():
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}

            # Get outputs from both models
            base_outputs = self.base_model(**batch)
            expanded_outputs = expanded_model(**batch)

            base_logits = base_outputs.logits
            expanded_logits = expanded_outputs.logits

            # Compute KL divergence
            base_probs = F.softmax(base_logits, dim=-1)
            expanded_log_probs = F.log_softmax(expanded_logits, dim=-1)

            kl = F.kl_div(
                expanded_log_probs.view(-1, expanded_log_probs.shape[-1]),
                base_probs.view(-1, base_probs.shape[-1]),
                reduction="batchmean",
            )

            kl_value = kl.item()
            self.kl_history.append(kl_value)

            is_acceptable = kl_value < self.threshold

            if not is_acceptable:
                logger.warning(
                    f"High KL divergence detected: {kl_value:.4f} "
                    f"(threshold: {self.threshold})"
                )

            return is_acceptable, kl_value

    def get_report(self) -> Dict[str, Any]:
        """Get a report of KL divergence history."""
        if not self.kl_history:
            return {"message": "No checks performed yet"}

        import statistics

        return {
            "num_checks": len(self.kl_history),
            "mean_kl": statistics.mean(self.kl_history),
            "max_kl": max(self.kl_history),
            "min_kl": min(self.kl_history),
            "threshold": self.threshold,
            "violations": sum(1 for kl in self.kl_history if kl > self.threshold),
        }


def check_for_catastrophic_forgetting(
    original_model: nn.Module,
    expanded_model: nn.Module,
    eval_dataset: Any,
    tokenizer: Any,
    metric_fn: Optional[Callable] = None,
) -> Dict[str, Any]:
    """
    Check if expanded model has suffered catastrophic forgetting.

    Compares performance on a evaluation dataset between original
    and expanded models.

    Args:
        original_model: Original pretrained model
        expanded_model: Expanded model
        eval_dataset: Evaluation dataset
        tokenizer: Tokenizer
        metric_fn: Optional custom metric function

    Returns:
        Comparison results
    """
    original_model.eval()
    expanded_model.eval()

    # Default metric: perplexity
    if metric_fn is None:
        def metric_fn(model, batch):
            with torch.no_grad():
                outputs = model(**batch)
                return outputs.loss.item() if hasattr(outputs, "loss") else 0.0

    original_scores = []
    expanded_scores = []

    for batch in eval_dataset:
        original_scores.append(metric_fn(original_model, batch))
        expanded_scores.append(metric_fn(expanded_model, batch))

    import statistics

    original_mean = statistics.mean(original_scores)
    expanded_mean = statistics.mean(expanded_scores)

    return {
        "original_score": original_mean,
        "expanded_score": expanded_mean,
        "absolute_diff": abs(expanded_mean - original_mean),
        "relative_diff": abs(expanded_mean - original_mean) / abs(original_mean) if original_mean != 0 else 0,
        "forgotten": expanded_mean > original_mean * 1.2,  # 20% degradation threshold
    }
